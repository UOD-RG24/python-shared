from __future__ import annotations

import pytest
from pydantic import ValidationError
from uod_rg24_preprocessing.ingestion import (
    AmbiguousModalityError,
    DuplicateFeatureError,
    FeatureNamespace,
    Modality,
    RegistrationParameters,
    SourceIntegrityError,
    SourceUpload,
    UnsupportedSourceLayoutError,
    ValueScale,
    plan_wide_matrix_registration,
    verify_source_integrity,
)

SHA = "a" * 64


def _upload(**overrides: object) -> SourceUpload:
    payload: dict[str, object] = {
        "ownerId": "6a8c0866226c0fff2658d31a",
        "datasetId": "68f0c1f4b1a24d5e8c7a9012",
        "containerName": "datasets",
        "blobPath": "6a8c0866226c0fff2658d31a/fd16dc774673462cb5976eac0dc9ba10.csv",
        "originalFileName": "mrna_matrix.csv",
        "storedFileName": "fd16dc774673462cb5976eac0dc9ba10.csv",
        "fileType": "csv",
        "contentType": "application/octet-stream",
        "sha256": SHA,
        "byteLength": 165660,
        "etag": '"0x8DD1"',
        "omicsType": "mrna",
    }
    payload.update(overrides)
    return SourceUpload.model_validate(payload)


def _parameters(**overrides: object) -> RegistrationParameters:
    payload: dict[str, object] = {"valueScale": "log2(FPKM+1)"}
    payload.update(overrides)
    return RegistrationParameters.model_validate(payload)


class TestSourceUpload:
    def test_accepts_a_live_upload_record(self) -> None:
        upload = _upload()
        assert upload.delimiter == ","
        assert upload.owner_id == "6a8c0866226c0fff2658d31a"

    def test_tsv_uploads_use_a_tab_delimiter(self) -> None:
        upload = _upload(
            fileType="tsv",
            blobPath="owner/abc.tsv",
            storedFileName="abc.tsv",
        )
        assert upload.delimiter == "\t"

    def test_rejects_a_blob_path_that_escapes_its_prefix(self) -> None:
        with pytest.raises(ValidationError):
            _upload(blobPath="owner/../secrets/fd16dc774673462cb5976eac0dc9ba10.csv")

    def test_rejects_a_blob_path_that_does_not_end_with_the_stored_name(self) -> None:
        with pytest.raises(ValidationError):
            _upload(blobPath="6a8c0866226c0fff2658d31a/somethingelse.csv")

    def test_rejects_an_unsupported_file_type(self) -> None:
        with pytest.raises(ValidationError):
            _upload(
                fileType="parquet",
                blobPath="owner/a.parquet",
                storedFileName="a.parquet",
            )

    def test_rejects_a_malformed_digest(self) -> None:
        with pytest.raises(ValidationError):
            _upload(sha256="not-a-digest")


class TestPlanWideMatrixRegistration:
    def test_plans_the_live_mrna_upload(self) -> None:
        header = ["patient_id", "mrna__1", "mrna__10", "mrna__100"]

        plan = plan_wide_matrix_registration(header, _parameters())

        assert plan.source_identifier_column == "patient_id"
        assert plan.modality is Modality.MRNA
        assert plan.value_scale is ValueScale.LOG2_FPKM_PLUS_1
        assert plan.feature_namespace is FeatureNamespace.DOUBLE_UNDERSCORE
        assert plan.feature_ids == ("mrna__1", "mrna__10", "mrna__100")
        assert plan.warnings == ()

    def test_plans_the_live_cna_upload(self) -> None:
        header = ["patient_id", "cna__1", "cna__10"]

        plan = plan_wide_matrix_registration(
            header, _parameters(valueScale="discreteCopyNumber")
        )

        assert plan.modality is Modality.CNA
        assert plan.value_scale_policy.allows_negative is True
        assert plan.value_scale_policy.integral is True

    def test_canonical_columns_lead_with_entity_then_patient(self) -> None:
        plan = plan_wide_matrix_registration(["patient_id", "mrna__1"], _parameters())

        assert plan.canonical_column_names == ("entityId", "patientId", "mrna__1")

    def test_preserves_source_feature_order(self) -> None:
        header = ["patient_id", "mrna__100", "mrna__1", "mrna__10"]

        plan = plan_wide_matrix_registration(header, _parameters())

        assert plan.feature_ids == ("mrna__100", "mrna__1", "mrna__10")

    def test_recognises_the_colon_namespace(self) -> None:
        header = ["entityId", "mrna::entrez::1", "mrna::entrez::2"]

        plan = plan_wide_matrix_registration(header, _parameters())

        assert plan.feature_namespace is FeatureNamespace.COLON
        assert plan.modality is Modality.MRNA

    def test_rejects_an_unrecognised_identifier_column(self) -> None:
        with pytest.raises(UnsupportedSourceLayoutError) as excinfo:
            plan_wide_matrix_registration(["Gene", "Sample_01"], _parameters())

        assert excinfo.value.code == "SCHEMA_VALIDATION_FAILED"

    def test_accepts_a_declared_identifier_column(self) -> None:
        plan = plan_wide_matrix_registration(
            ["Gene", "mrna__1"],
            _parameters(identifierColumn="Gene"),
        )

        assert plan.source_identifier_column == "Gene"

    def test_rejects_a_declared_identifier_column_that_does_not_match(self) -> None:
        with pytest.raises(UnsupportedSourceLayoutError):
            plan_wide_matrix_registration(
                ["patient_id", "mrna__1"],
                _parameters(identifierColumn="Gene"),
            )

    def test_rejects_a_header_with_no_features(self) -> None:
        with pytest.raises(UnsupportedSourceLayoutError):
            plan_wide_matrix_registration(["patient_id"], _parameters())

    def test_rejects_duplicate_features(self) -> None:
        header = ["patient_id", "mrna__1", "mrna__1"]

        with pytest.raises(DuplicateFeatureError) as excinfo:
            plan_wide_matrix_registration(header, _parameters())

        assert excinfo.value.code == "INCOMPATIBLE_FEATURE_SCHEMA"

    def test_rejects_a_feature_named_like_an_identifier_column(self) -> None:
        header = ["patient_id", "mrna__1", "entityId"]

        with pytest.raises(DuplicateFeatureError):
            plan_wide_matrix_registration(header, _parameters())

    def test_rejects_an_unnamed_feature_column(self) -> None:
        with pytest.raises(UnsupportedSourceLayoutError):
            plan_wide_matrix_registration(
                ["patient_id", "mrna__1", "  "], _parameters()
            )

    def test_rejects_an_uninferable_modality(self) -> None:
        with pytest.raises(AmbiguousModalityError) as excinfo:
            plan_wide_matrix_registration(
                ["patient_id", "gene1", "gene2"], _parameters()
            )

        assert excinfo.value.code == "INCOMPATIBLE_FEATURE_SCHEMA"

    def test_rejects_mixed_modalities_without_a_declaration(self) -> None:
        header = ["patient_id", "mrna__1", "cna__1"]

        with pytest.raises(AmbiguousModalityError):
            plan_wide_matrix_registration(header, _parameters())

    def test_allows_mixed_modalities_when_declared_and_records_a_warning(self) -> None:
        header = ["patient_id", "mrna__1", "cna__1"]

        plan = plan_wide_matrix_registration(
            header, _parameters(modality="mrna", featureNamespace="doubleUnderscore")
        )

        assert plan.modality is Modality.MRNA
        assert "mixedFeatureNamespaces:2" in plan.warnings

    def test_rejects_a_declared_modality_that_contradicts_the_prefix(self) -> None:
        header = ["patient_id", "cna__1", "cna__2"]

        with pytest.raises(AmbiguousModalityError):
            plan_wide_matrix_registration(header, _parameters(modality="mrna"))

    def test_warns_about_unprefixed_features(self) -> None:
        header = ["patient_id", "mrna__1", "age"]

        plan = plan_wide_matrix_registration(header, _parameters())

        assert plan.warnings == ("unprefixedFeatures:1",)

    def test_a_bare_prefix_is_not_a_feature_identifier(self) -> None:
        with pytest.raises(AmbiguousModalityError):
            plan_wide_matrix_registration(["patient_id", "mrna__"], _parameters())

    def test_strips_surrounding_whitespace_in_the_header(self) -> None:
        plan = plan_wide_matrix_registration(
            [" patient_id ", " mrna__1 "], _parameters()
        )

        assert plan.source_identifier_column == "patient_id"
        assert plan.feature_ids == ("mrna__1",)

    def test_a_declared_value_scale_is_never_inferred(self) -> None:
        with pytest.raises(ValidationError):
            RegistrationParameters.model_validate({})


class TestVerifySourceIntegrity:
    def test_accepts_a_matching_digest(self) -> None:
        verify_source_integrity(_upload(), observed_sha256=SHA)

    def test_rejects_a_changed_blob(self) -> None:
        with pytest.raises(SourceIntegrityError) as excinfo:
            verify_source_integrity(_upload(), observed_sha256="b" * 64)

        assert excinfo.value.code == "INPUT_HASH_MISMATCH"
