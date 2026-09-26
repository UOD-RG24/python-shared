from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from datetime import UTC, datetime
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from uod_rg24_preprocessing.ingestion import (
    RegistrationParameters,
    plan_wide_matrix_registration,
)
from uod_rg24_runtime_azure import (
    CanonicalMatrixError,
    build_canonical_matrix,
    order_hash,
    serialize_canonical_matrix,
    serialize_registration_qc,
)

GENERATED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
SOURCE_SHA = "c" * 64


def _plan(header: list[str], value_scale: str = "log2(FPKM+1)", **overrides: object):
    payload: dict[str, object] = {"valueScale": value_scale}
    payload.update(overrides)
    return plan_wide_matrix_registration(
        header, RegistrationParameters.model_validate(payload)
    )


def _csv(rows: list[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode()


MRNA_HEADER = ["patient_id", "mrna__1", "mrna__10"]
MRNA_CSV = _csv(
    [
        "patient_id,mrna__1,mrna__10",
        "TCGA-05-4244,0.1513244503266472,0.2116149063789633",
        "TCGA-05-4249,0.2683547471853125,0.2301811167998683",
        "TCGA-05-4250,0.0,1.5",
    ]
)


class TestBuildCanonicalMatrix:
    def test_builds_the_canonical_matrix_from_a_live_shaped_upload(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        assert result.row_count == 3
        assert result.feature_count == 2
        assert result.value_count == 6
        assert result.table.column_names == [
            "entityId",
            "patientId",
            "mrna__1",
            "mrna__10",
        ]

    def test_entity_id_equals_patient_id_as_fa2_requires(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        assert result.table.column(0).to_pylist() == result.table.column(1).to_pylist()
        assert result.table.column(0).to_pylist()[0] == "TCGA-05-4244"

    def test_declares_the_metadata_fa2_validates(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        metadata = result.table.schema.metadata
        assert metadata[b"uod_rg24.schemaId"] == b"omics-wide-matrix/1.0"
        assert metadata[b"uod_rg24.modality"] == b"mrna"
        assert metadata[b"uod_rg24.valueScale"] == b"log2(FPKM+1)"
        assert metadata[b"uod_rg24.entityIdColumn"] == b"entityId"
        assert metadata[b"uod_rg24.featureNamespace"] == b"doubleUnderscore"

    def test_feature_values_are_float64_and_non_nullable(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        for index in (2, 3):
            assert result.table.schema.field(index).type == pa.float64()
            assert result.table.schema.field(index).nullable is False

    def test_preserves_source_entity_and_feature_order(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        assert result.table.column(0).to_pylist() == [
            "TCGA-05-4244",
            "TCGA-05-4249",
            "TCGA-05-4250",
        ]
        assert result.table.column_names[2:] == ["mrna__1", "mrna__10"]

    def test_order_hashes_match_the_shared_construction(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        expected_samples = sha256(
            b"TCGA-05-4244\nTCGA-05-4249\nTCGA-05-4250\n"
        ).hexdigest()
        expected_features = sha256(b"mrna__1\nmrna__10\n").hexdigest()
        assert result.sample_order_sha256 == expected_samples
        assert result.feature_order_sha256 == expected_features

    def test_order_hash_helper_matches_fa3(self) -> None:
        assert order_hash(["a", "b"]) == sha256(b"a\nb\n").hexdigest()

    def test_reports_the_observed_value_range(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        assert result.minimum == pytest.approx(0.0)
        assert result.maximum == pytest.approx(1.5)

    def test_reads_tab_separated_sources(self) -> None:
        source = _csv(["patient_id\tmrna__1", "TCGA-05-4244\t1.0", "TCGA-05-4249\t2.0"])

        result = build_canonical_matrix(
            source, _plan(["patient_id", "mrna__1"]), delimiter="\t"
        )

        assert result.row_count == 2


class TestValueScalePolicy:
    def test_accepts_negative_discrete_copy_number(self) -> None:
        source = _csv(
            ["patient_id,cna__1,cna__10", "TCGA-05-4244,-2,1", "TCGA-05-4249,0,-4"]
        )

        result = build_canonical_matrix(
            source, _plan(["patient_id", "cna__1", "cna__10"], "discreteCopyNumber")
        )

        assert result.minimum == pytest.approx(-4.0)
        assert result.maximum == pytest.approx(1.0)

    def test_rejects_negative_values_under_a_non_negative_scale(self) -> None:
        source = _csv(["patient_id,mrna__1", "TCGA-05-4244,-0.15"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"], "FPKM"))

        assert excinfo.value.code == "INVALID_VALUE_DOMAIN"

    def test_rejects_fractional_values_under_a_discrete_scale(self) -> None:
        source = _csv(["patient_id,cna__1,cna__2", "TCGA-05-4244,1.5,2"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(
                source, _plan(["patient_id", "cna__1", "cna__2"], "discreteCopyNumber")
            )

        assert excinfo.value.code == "INVALID_VALUE_DOMAIN"


class TestValidationFailures:
    def test_rejects_missing_values(self) -> None:
        source = _csv(
            [
                "patient_id,mrna__1,mrna__10",
                "TCGA-05-4244,1.0,2.0",
                "TCGA-05-4249,3.0,",
            ]
        )

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(MRNA_HEADER))

        assert excinfo.value.code == "INVALID_VALUE_DOMAIN"

    def test_rejects_a_wholly_empty_feature_column(self) -> None:
        source = _csv(["patient_id,mrna__1,mrna__10", "TCGA-05-4244,1.0,"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(MRNA_HEADER))

        assert excinfo.value.code == "INVALID_VALUE_DOMAIN"

    def test_rejects_non_finite_values(self) -> None:
        source = _csv(["patient_id,mrna__1", "TCGA-05-4244,inf"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "INVALID_VALUE_DOMAIN"

    def test_rejects_a_non_numeric_feature(self) -> None:
        source = _csv(["patient_id,mrna__1", "TCGA-05-4244,high"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "INCOMPATIBLE_FEATURE_SCHEMA"

    def test_rejects_duplicate_entity_identifiers(self) -> None:
        source = _csv(["patient_id,mrna__1", "TCGA-05-4244,1.0", "TCGA-05-4244,2.0"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "DUPLICATE_SAMPLE_ID"

    def test_rejects_an_empty_entity_identifier(self) -> None:
        source = _csv(["patient_id,mrna__1", '" ",1.0'])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "SCHEMA_VALIDATION_FAILED"

    def test_rejects_a_source_whose_columns_drifted_from_the_plan(self) -> None:
        plan = _plan(MRNA_HEADER)
        source = _csv(["patient_id,mrna__1,mrna__99", "TCGA-05-4244,1.0,2.0"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, plan)

        assert excinfo.value.code == "SCHEMA_VALIDATION_FAILED"

    def test_rejects_a_source_with_no_rows(self) -> None:
        source = _csv(["patient_id,mrna__1"])

        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(source, _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "SCHEMA_VALIDATION_FAILED"

    def test_rejects_unreadable_bytes(self) -> None:
        with pytest.raises(CanonicalMatrixError) as excinfo:
            build_canonical_matrix(b"\x00\x01\x02", _plan(["patient_id", "mrna__1"]))

        assert excinfo.value.code == "SCHEMA_VALIDATION_FAILED"


class TestSerialization:
    def test_round_trips_through_canonical_parquet(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        payload = serialize_canonical_matrix(result)
        restored = pq.read_table(pa.BufferReader(payload.data))

        assert payload.kind == "omicsMatrix"
        assert payload.schema_id == "omics-wide-matrix/1.0"
        assert payload.media_type == "application/vnd.apache.parquet"
        assert payload.row_count == 3
        assert payload.sample_order_sha256 == result.sample_order_sha256
        assert restored.column_names == result.table.column_names
        assert restored.schema.metadata[b"uod_rg24.modality"] == b"mrna"
        assert restored.column(2).to_pylist() == result.table.column(2).to_pylist()

    def test_serialization_is_deterministic(self) -> None:
        plan = _plan(MRNA_HEADER)

        first = serialize_canonical_matrix(build_canonical_matrix(MRNA_CSV, plan))
        second = serialize_canonical_matrix(build_canonical_matrix(MRNA_CSV, plan))

        assert first.sha256 == second.sha256

    def test_qc_records_provenance_and_lineage(self) -> None:
        result = build_canonical_matrix(MRNA_CSV, _plan(MRNA_HEADER))

        payload = serialize_registration_qc(
            result,
            _plan(MRNA_HEADER),
            artifact_id="art_register_001",
            generated_at=GENERATED_AT,
            source_sha256=SOURCE_SHA,
        )
        document = payload.data.decode()

        assert payload.schema_id == "artifact-qc/1.0"
        assert '"artifactId":"art_register_001"' in document
        assert '"rowCount":3' in document
        assert '"featureCount":2' in document
        assert f'"sourceSha256":"{SOURCE_SHA}"' in document
        assert f'"sampleOrderSha256":"{result.sample_order_sha256}"' in document
        assert '"featureNamespace":"doubleUnderscore"' in document
