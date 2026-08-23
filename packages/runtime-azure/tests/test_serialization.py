from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
import json
from datetime import datetime
from io import BytesIO
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from uod_rg24_contracts import ArtifactManifestV1
from uod_rg24_preprocessing import HarmonizationResult
from uod_rg24_runtime_azure import (
    OUTPUT_LAYOUTS,
    OUTPUT_NAMES,
    OUTPUT_SCHEMA_IDS,
    PARQUET_WRITE_OPTIONS,
    SerializationError,
    serialize_harmonization_reports,
    serialize_manifest,
    serialize_qc_sidecars,
    sha256_bytes,
)

EXPECTED_ARROW_FIELDS = {
    "sampleMap": (
        ("entityId", pa.string(), False),
        ("patientId", pa.string(), False),
        ("modality", pa.string(), False),
        ("role", pa.string(), False),
        ("sourceSampleId", pa.string(), True),
        ("selected", pa.bool_(), False),
        ("selectionReason", pa.string(), True),
        ("rejectedReason", pa.string(), True),
        ("replicateGroup", pa.string(), False),
        ("sourceArtifactId", pa.string(), False),
        ("mappingStatus", pa.string(), False),
    ),
    "availability": (
        ("entityId", pa.string(), False),
        ("patientId", pa.string(), False),
        ("role", pa.string(), False),
        ("available", pa.bool_(), False),
        ("selectedSourceSampleId", pa.string(), True),
        ("exactSampleOverlap", pa.bool_(), False),
        ("retained", pa.bool_(), False),
    ),
    "retainedCohort": (
        ("position", pa.int64(), False),
        ("entityId", pa.string(), False),
        ("patientId", pa.string(), False),
    ),
    "dropReport": (
        ("entityId", pa.string(), False),
        ("patientId", pa.string(), False),
        ("role", pa.string(), False),
        ("reason", pa.string(), False),
        ("sourceSampleId", pa.string(), True),
    ),
    "alignedSampleIds": (
        ("position", pa.int64(), False),
        ("entityId", pa.string(), False),
        ("patientId", pa.string(), False),
        ("role", pa.string(), False),
        ("sourceSampleId", pa.string(), True),
    ),
}


def _schema_fields(schema: pa.Schema) -> tuple[tuple[str, pa.DataType, bool], ...]:
    return tuple((field.name, field.type, field.nullable) for field in schema)


def _row_order_key(output_name: str, row: dict[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key in OUTPUT_LAYOUTS[output_name].sort_keys:
        value = row[key]
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(f"{value:020d}")
        else:
            values.append("" if value is None else str(value))
    return tuple(values)


def test_registry_and_exact_arrow_contracts_are_fixed() -> None:
    assert OUTPUT_NAMES == (
        "sampleMap",
        "availability",
        "retainedCohort",
        "dropReport",
        "alignedSampleIds",
    )
    assert dict(OUTPUT_SCHEMA_IDS) == {
        "sampleMap": "sample-map/1.0",
        "availability": "sample-availability/1.0",
        "retainedCohort": "retained-cohort/1.0",
        "dropReport": "sample-drop-report/1.0",
        "alignedSampleIds": "aligned-sample-ids/1.0",
    }
    for name, expected_fields in EXPECTED_ARROW_FIELDS.items():
        schema = OUTPUT_LAYOUTS[name].arrow_schema
        assert _schema_fields(schema) == expected_fields
        assert schema.metadata == {
            b"uod.output_name": name.encode(),
            b"uod.kind": OUTPUT_LAYOUTS[name].kind.encode(),
            b"uod.schema_id": OUTPUT_SCHEMA_IDS[name].encode(),
        }


def test_parquet_writer_policy_is_exact_and_immutable() -> None:
    assert type(PARQUET_WRITE_OPTIONS) is type(MappingProxyType({}))
    assert dict(PARQUET_WRITE_OPTIONS) == {
        "compression": "zstd",
        "compression_level": 3,
        "data_page_version": "2.0",
        "row_group_size": 65_536,
        "use_dictionary": False,
        "version": "2.6",
        "write_statistics": True,
    }


def test_reports_are_byte_repeatable_ordered_parquet_with_exact_metadata(
    harmonization_result: HarmonizationResult,
) -> None:
    first = serialize_harmonization_reports(harmonization_result)
    second = serialize_harmonization_reports(harmonization_result)
    assert tuple(first) == OUTPUT_NAMES
    assert {name: item.data for name, item in first.items()} == {
        name: item.data for name, item in second.items()
    }

    for name, payload in first.items():
        parquet_file = pq.ParquetFile(BytesIO(payload.data))
        table = parquet_file.read()
        layout = OUTPUT_LAYOUTS[name]
        rows = table.to_pylist()

        assert table.schema.equals(layout.arrow_schema, check_metadata=True)
        assert _schema_fields(table.schema) == EXPECTED_ARROW_FIELDS[name]
        assert rows == sorted(rows, key=lambda row: _row_order_key(name, row))
        assert payload.schema_id == layout.schema_id
        assert payload.kind == layout.kind
        assert payload.row_count == table.num_rows
        assert payload.sample_order_sha256 == harmonization_result.sample_order_sha256

        metadata = parquet_file.metadata
        assert metadata.format_version == "2.6"
        assert metadata.num_row_groups == 1
        row_group = metadata.row_group(0)
        assert row_group.num_rows == table.num_rows
        for column_index in range(metadata.num_columns):
            column = row_group.column(column_index)
            assert column.compression == "ZSTD"
            assert column.is_stats_set
            assert "RLE_DICTIONARY" not in column.encodings
            assert "PLAIN_DICTIONARY" not in column.encodings


def test_availability_is_long_form_and_aligned_ids_follow_positions(
    harmonization_result: HarmonizationResult,
) -> None:
    reports = serialize_harmonization_reports(harmonization_result)
    availability = pq.read_table(BytesIO(reports["availability"].data)).to_pylist()
    assert len(availability) == len(harmonization_result.availability) * 2
    assert {(row["patientId"], row["role"]) for row in availability} == {
        (row.patient_id, role)
        for row in harmonization_result.availability
        for role in ("mrna", "protein")
    }

    aligned = pq.read_table(BytesIO(reports["alignedSampleIds"].data)).to_pylist()
    assert aligned == [
        {
            "position": 0,
            "entityId": "TCGA-AA-0001",
            "patientId": "TCGA-AA-0001",
            "role": "mrna",
            "sourceSampleId": "TCGA-AA-0001-01A",
        },
        {
            "position": 0,
            "entityId": "TCGA-AA-0001",
            "patientId": "TCGA-AA-0001",
            "role": "protein",
            "sourceSampleId": "TCGA-AA-0001-01A",
        },
    ]


def test_retained_positions_are_contiguous_from_zero(
    harmonization_result: HarmonizationResult,
) -> None:
    retained = ["PATIENT-0001", "PATIENT-0002", "PATIENT-0003"]
    sample_order_sha256 = sha256_bytes(
        "".join(f"{patient_id}\n" for patient_id in retained).encode()
    )
    expanded = harmonization_result.model_copy(
        update={
            "retained_cohort": retained,
            "aligned_sample_ids": {
                "mrna": ["MRNA-1", "MRNA-2", "MRNA-3"],
                "protein": ["PROTEIN-1", "PROTEIN-2", "PROTEIN-3"],
            },
            "sample_order_sha256": sample_order_sha256,
        }
    )
    payload = serialize_harmonization_reports(expanded)["retainedCohort"]
    rows = pq.read_table(BytesIO(payload.data)).to_pylist()
    assert rows == [
        {"position": index, "entityId": patient_id, "patientId": patient_id}
        for index, patient_id in enumerate(retained)
    ]


def test_zero_row_drop_report_preserves_full_arrow_schema(
    harmonization_result: HarmonizationResult,
) -> None:
    empty = harmonization_result.model_copy(update={"drop_report": []})
    payload = serialize_harmonization_reports(empty)["dropReport"]
    table = pq.read_table(BytesIO(payload.data))
    assert table.num_rows == 0
    assert payload.row_count == 0
    assert table.schema.equals(
        OUTPUT_LAYOUTS["dropReport"].arrow_schema, check_metadata=True
    )
    assert _schema_fields(table.schema) == EXPECTED_ARROW_FIELDS["dropReport"]


def test_invalid_parallel_aligned_ids_fail_before_serialization(
    harmonization_result: HarmonizationResult,
) -> None:
    invalid = harmonization_result.model_copy(
        update={"aligned_sample_ids": {"mrna": [], "protein": []}}
    )
    with pytest.raises(SerializationError, match="retained cohort length"):
        serialize_harmonization_reports(invalid)


def test_qc_is_one_canonical_json_sidecar_per_output(
    harmonization_result: HarmonizationResult,
    utc_now: datetime,
) -> None:
    artifact_ids = {name: f"art_{name}" for name in OUTPUT_NAMES}
    first = serialize_qc_sidecars(
        harmonization_result, artifact_ids=artifact_ids, generated_at=utc_now
    )
    second = serialize_qc_sidecars(
        harmonization_result, artifact_ids=artifact_ids, generated_at=utc_now
    )
    assert {name: payload.data for name, payload in first.items()} == {
        name: payload.data for name, payload in second.items()
    }
    for name, payload in first.items():
        value = json.loads(payload.data)
        assert value["schemaVersion"] == "artifact-qc/1.0"
        assert value["artifactId"] == artifact_ids[name]
        assert value["extensions"]["outputName"] == name
        assert payload.schema_id == "artifact-qc/1.0"


@pytest.mark.parametrize("schema_id", tuple(OUTPUT_SCHEMA_IDS.values()))
def test_fixed_sample_report_manifest_requires_sample_order_hash(
    artifact_manifest: ArtifactManifestV1,
    schema_id: str,
) -> None:
    manifest = artifact_manifest.model_copy(
        update={
            "data": artifact_manifest.data.model_copy(
                update={"schema_id": schema_id, "sample_order_sha256": None}
            )
        }
    )
    with pytest.raises(SerializationError, match="require data.sampleOrderSha256"):
        serialize_manifest(manifest)


def test_manifest_hash_must_match_serialized_report_when_expected(
    artifact_manifest: ArtifactManifestV1,
    harmonization_result: HarmonizationResult,
) -> None:
    report = serialize_harmonization_reports(harmonization_result)["sampleMap"]
    manifest = artifact_manifest.model_copy(
        update={
            "data": artifact_manifest.data.model_copy(
                update={"sample_order_sha256": report.sample_order_sha256}
            )
        }
    )
    payload = serialize_manifest(
        manifest,
        expected_sample_order_sha256=report.sample_order_sha256,
    )
    assert payload.sample_order_sha256 == report.sample_order_sha256
    assert json.loads(payload.data)["data"]["sampleOrderSha256"] == (
        report.sample_order_sha256
    )

    with pytest.raises(SerializationError, match="does not match"):
        serialize_manifest(
            manifest,
            expected_sample_order_sha256="f" * 64,
        )


def test_manifest_uses_canonical_json(artifact_manifest: ArtifactManifestV1) -> None:
    payload = serialize_manifest(artifact_manifest)
    assert payload.data.endswith(b"\n")
    assert json.loads(payload.data)["artifactId"] == "art_output"
    assert payload.schema_id == "artifact-manifest/1.0"
    assert payload.sample_order_sha256 == artifact_manifest.data.sample_order_sha256
