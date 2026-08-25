from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from uod_rg24_contracts import ArtifactManifestV1, QCSummaryV1
from uod_rg24_preprocessing import HarmonizationResult

from ..errors import SerializationError
from ..hashing import sha256_bytes
from ..json_codec import canonical_json_bytes
from ..models import JSON_MEDIA_TYPE, ArtifactPayload
from .layouts import OUTPUT_LAYOUTS, OUTPUT_NAMES, ArtifactLayout

QC_SCHEMA_ID = "artifact-qc/1.0"
QC_KIND = "qcSummary"
MANIFEST_SCHEMA_ID = "artifact-manifest/1.0"
MANIFEST_KIND = "artifactManifest"


class _ParquetWriteOptions(TypedDict):
    compression: Literal["zstd"]
    compression_level: int
    data_page_version: Literal["2.0"]
    row_group_size: int
    use_dictionary: Literal[False]
    version: Literal["2.6"]
    write_statistics: Literal[True]


PARQUET_WRITE_OPTIONS: Final[_ParquetWriteOptions] = cast(
    _ParquetWriteOptions,
    MappingProxyType(
        {
            "compression": "zstd",
            "compression_level": 3,
            "data_page_version": "2.0",
            "row_group_size": 65_536,
            "use_dictionary": False,
            "version": "2.6",
            "write_statistics": True,
        }
    ),
)
_SAMPLE_REPORT_SCHEMA_IDS: Final[frozenset[str]] = frozenset(
    OUTPUT_LAYOUTS[name].schema_id for name in OUTPUT_NAMES
)


def _parquet_payload(
    layout: ArtifactLayout,
    rows: list[dict[str, object]],
    *,
    sample_order_sha256: str,
) -> ArtifactPayload:
    try:
        table = pa.Table.from_pylist(rows, schema=layout.arrow_schema)
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            **PARQUET_WRITE_OPTIONS,
        )
        data = sink.getvalue().to_pybytes()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise SerializationError(
            f"Could not serialize {layout.output_name} as canonical Parquet."
        ) from exc
    return ArtifactPayload.from_bytes(
        output_name=layout.output_name,
        kind=layout.kind,
        schema_id=layout.schema_id,
        media_type=layout.media_type,
        data=data,
        row_count=len(rows),
        sample_order_sha256=sample_order_sha256,
    )


def _validate_result(result: HarmonizationResult) -> tuple[str, ...]:
    roles = tuple(result.qc.cohort_roles)
    if not roles or len(set(roles)) != len(roles):
        raise SerializationError("Harmonization result has invalid cohort roles.")
    if list(roles) != sorted(roles):
        raise SerializationError("Harmonization cohort roles are not canonical.")
    if len(result.retained_cohort) != len(set(result.retained_cohort)):
        raise SerializationError("Retained cohort contains duplicate entity IDs.")
    if result.retained_cohort != sorted(result.retained_cohort):
        raise SerializationError("Retained cohort is not in canonical order.")
    expected_order_hash = sha256_bytes(
        "".join(f"{patient_id}\n" for patient_id in result.retained_cohort).encode()
    )
    if expected_order_hash != result.sample_order_sha256:
        raise SerializationError("Retained cohort does not match sample-order hash.")
    if set(result.aligned_sample_ids) != set(roles):
        raise SerializationError("Aligned-ID roles do not match cohort roles.")
    if any(
        len(result.aligned_sample_ids[role]) != len(result.retained_cohort)
        for role in roles
    ):
        raise SerializationError(
            "Aligned-ID vectors do not match retained cohort length."
        )
    return roles


def serialize_harmonization_reports(
    result: HarmonizationResult,
) -> dict[str, ArtifactPayload]:
    """Serialize all five fixed sample-report outputs."""

    roles = _validate_result(result)
    sample_map_rows: list[dict[str, object]] = [
        {
            "entityId": row.entity_id,
            "patientId": row.patient_id,
            "modality": row.modality,
            "role": row.role,
            "sourceSampleId": row.source_sample_id,
            "selected": row.selected,
            "selectionReason": row.selection_reason,
            "rejectedReason": row.rejected_reason,
            "replicateGroup": row.replicate_group,
            "sourceArtifactId": row.source_artifact_id,
            "mappingStatus": row.mapping_status,
        }
        for row in sorted(
            result.sample_map,
            key=lambda item: (
                item.patient_id,
                item.role,
                item.source_sample_id or "",
                item.source_artifact_id,
            ),
        )
    ]

    availability_rows: list[dict[str, object]] = []
    for row in sorted(result.availability, key=lambda item: item.patient_id):
        if set(row.role_availability) != set(roles) or set(row.selected_samples) != set(
            roles
        ):
            raise SerializationError(
                "Availability role maps do not match cohort roles."
            )
        for role in roles:
            availability_rows.append(
                {
                    "entityId": row.entity_id,
                    "patientId": row.patient_id,
                    "role": role,
                    "available": row.role_availability[role],
                    "selectedSourceSampleId": row.selected_samples[role],
                    "exactSampleOverlap": row.exact_sample_overlap,
                    "retained": row.retained,
                }
            )

    retained_rows: list[dict[str, object]] = [
        {
            "position": position,
            "entityId": patient_id,
            "patientId": patient_id,
        }
        for position, patient_id in enumerate(result.retained_cohort)
    ]
    drop_rows: list[dict[str, object]] = [
        {
            "entityId": row.patient_id,
            "patientId": row.patient_id,
            "role": row.role,
            "reason": row.reason,
            "sourceSampleId": row.source_sample_id,
        }
        for row in sorted(
            result.drop_report,
            key=lambda item: (
                item.patient_id,
                item.role,
                item.reason,
                item.source_sample_id or "",
            ),
        )
    ]
    aligned_rows: list[dict[str, object]] = [
        {
            "position": position,
            "entityId": patient_id,
            "patientId": patient_id,
            "role": role,
            "sourceSampleId": result.aligned_sample_ids[role][position],
        }
        for position, patient_id in enumerate(result.retained_cohort)
        for role in roles
    ]
    row_sets: dict[str, list[dict[str, object]]] = {
        "sampleMap": sample_map_rows,
        "availability": availability_rows,
        "retainedCohort": retained_rows,
        "dropReport": drop_rows,
        "alignedSampleIds": aligned_rows,
    }
    return {
        name: _parquet_payload(
            OUTPUT_LAYOUTS[name],
            row_sets[name],
            sample_order_sha256=result.sample_order_sha256,
        )
        for name in OUTPUT_NAMES
    }


def serialize_manifest(
    manifest: ArtifactManifestV1,
    *,
    expected_sample_order_sha256: str | None = None,
) -> ArtifactPayload:
    sample_order_sha256 = manifest.data.sample_order_sha256
    if (
        manifest.data.schema_id in _SAMPLE_REPORT_SCHEMA_IDS
        and sample_order_sha256 is None
    ):
        raise SerializationError(
            "Sample-report manifests require data.sampleOrderSha256."
        )
    if (
        expected_sample_order_sha256 is not None
        and sample_order_sha256 != expected_sample_order_sha256
    ):
        raise SerializationError(
            "Manifest sample-order hash does not match the serialized report."
        )
    data = canonical_json_bytes(manifest)
    return ArtifactPayload.from_bytes(
        output_name="manifest",
        kind=MANIFEST_KIND,
        schema_id=MANIFEST_SCHEMA_ID,
        media_type=JSON_MEDIA_TYPE,
        data=data,
        row_count=manifest.data.row_count,
        sample_order_sha256=sample_order_sha256,
    )


def serialize_qc_sidecars(
    result: HarmonizationResult,
    *,
    artifact_ids: Mapping[str, str],
    generated_at: datetime,
) -> dict[str, ArtifactPayload]:
    if set(artifact_ids) != set(OUTPUT_NAMES):
        raise SerializationError("QC artifact IDs must match the five output names.")
    counts = {
        "candidateCount": result.qc.candidate_count,
        "validCandidateCount": result.qc.valid_candidate_count,
        "selectedCandidateCount": result.qc.selected_candidate_count,
        "rejectedCandidateCount": result.qc.rejected_candidate_count,
        "availablePatientCount": result.qc.available_patient_count,
        "retainedPatientCount": result.qc.retained_patient_count,
        "droppedPatientCount": result.qc.dropped_patient_count,
        "patientLevelFallbackCount": result.qc.patient_level_fallback_count,
        "barcodeDerivedCount": result.qc.barcode_derived_count,
    }
    output: dict[str, ArtifactPayload] = {}
    for output_name in OUTPUT_NAMES:
        qc = QCSummaryV1.model_validate(
            {
                "schemaVersion": QC_SCHEMA_ID,
                "artifactId": artifact_ids[output_name],
                "generatedAt": generated_at,
                "counts": counts,
                "metrics": {},
                "warnings": result.qc.warnings,
                "extensions": {
                    "outputName": output_name,
                    "sampleOrderSha256": result.sample_order_sha256,
                    "cohortRoles": result.qc.cohort_roles,
                    "ignoredSidecarRoles": result.qc.ignored_sidecar_roles,
                },
            }
        )
        data = canonical_json_bytes(qc)
        output[output_name] = ArtifactPayload.from_bytes(
            output_name=f"{output_name}.qc",
            kind=QC_KIND,
            schema_id=QC_SCHEMA_ID,
            media_type=JSON_MEDIA_TYPE,
            data=data,
        )
    return output
