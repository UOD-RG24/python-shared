from __future__ import annotations

from copy import deepcopy
from random import Random

import pytest
from pydantic import ValidationError
from uod_rg24_preprocessing.sample_selection import (
    ClinicalMappingConflictError,
    CohortPolicy,
    DuplicateSampleCandidateError,
    HarmonizationError,
    HarmonizationRequest,
    InvalidTcgaBarcodeError,
    MatchLevel,
    NoCommonSamplesError,
    SampleCandidate,
    harmonize_samples,
    parse_tcga_barcode,
)


def candidate(
    role: str,
    sample_id: str | None = None,
    *,
    patient_id: str | None = None,
    artifact: str | None = None,
) -> SampleCandidate:
    return SampleCandidate(
        role=role,
        source_artifact_id=artifact or f"art_{role}",
        source_sample_id=sample_id,
        patient_id=patient_id,
    )


def synthetic_request(**overrides: object) -> HarmonizationRequest:
    data: dict[str, object] = {
        "candidates": [
            candidate("mrna", "TCGA-AA-0001-01B-01D-0001-01"),
            candidate("mrna", "TCGA-AA-0001-01A-01D-0001-01"),
            candidate("mrna", "TCGA-AA-0001-11A-01D-0001-01"),
            candidate("mrna", "TCGA-AA-0002-01A"),
            candidate("protein", "TCGA-AA-0001-01A"),
            candidate("protein", "TCGA-AA-0003-01A"),
            candidate("clinicalPatient", patient_id="TCGA-AA-0001"),
            candidate("clinicalPatient", patient_id="TCGA-AA-0002"),
            candidate("clinicalPatient", patient_id="TCGA-AA-0003"),
            candidate("proteinAnnotation", patient_id="TCGA-AA-9999"),
        ],
        "clinicalPatientIds": [
            "TCGA-AA-0001",
            "TCGA-AA-0002",
            "TCGA-AA-0003",
        ],
        "clinicalSampleToPatient": {
            "TCGA-AA-0001-01A": "TCGA-AA-0001",
            "TCGA-AA-0001-01B": "TCGA-AA-0001",
            "TCGA-AA-0001-11A": "TCGA-AA-0001",
            "TCGA-AA-0002-01A": "TCGA-AA-0002",
        },
        "cohortRoles": ["mrna", "protein", "clinicalPatient"],
        "cohortPolicy": "intersection",
        "matchLevel": "patient",
    }
    data.update(overrides)
    return HarmonizationRequest.model_validate(data)


def test_tcga_parser_preserves_full_barcode_and_derives_patient() -> None:
    barcode = parse_tcga_barcode("tcga-aa-0001-01b-01d-0001-01")
    assert barcode.original == "tcga-aa-0001-01b-01d-0001-01"
    assert barcode.patient_id == "TCGA-AA-0001"
    assert barcode.canonical_sample_id == "TCGA-AA-0001-01B"
    assert barcode.sample_type_code == 1
    assert barcode.vial == "B"

    patient = parse_tcga_barcode("TCGA-AA-0001")
    assert not patient.is_sample

    for invalid in (
        "not-a-tcga-id",
        "TCGA-AA-0001-01A-ABCD",
        "TCGA-AA-0001-01A-01D-01",
    ):
        with pytest.raises(InvalidTcgaBarcodeError):
            parse_tcga_barcode(invalid)


def test_intersection_selects_primary_lowest_vial_and_records_every_replicate() -> None:
    result = harmonize_samples(synthetic_request())

    assert result.retained_cohort == ["TCGA-AA-0001"]
    retained_availability = {
        row.patient_id: row.role_availability for row in result.availability
    }
    assert all(
        all(retained_availability[patient_id].values())
        for patient_id in result.retained_cohort
    )
    selected_mrna = [
        row
        for row in result.sample_map
        if row.role == "mrna" and row.selected and row.patient_id == "TCGA-AA-0001"
    ]
    assert [row.source_sample_id for row in selected_mrna] == [
        "TCGA-AA-0001-01A-01D-0001-01"
    ]
    assert selected_mrna[0].selection_reason == "primaryTumourPreferred"
    rejected = {
        row.source_sample_id: row.rejected_reason
        for row in result.sample_map
        if row.role == "mrna" and not row.selected and row.patient_id == "TCGA-AA-0001"
    }
    assert rejected == {
        "TCGA-AA-0001-01B-01D-0001-01": "higherVial",
        "TCGA-AA-0001-11A-01D-0001-01": "nonPrimaryTumour",
    }
    assert all(
        row.patient_id == "TCGA-AA-0001" for row in result.sample_map if row.selected
    )
    selected_pairs = [
        (row.patient_id, row.role) for row in result.sample_map if row.selected
    ]
    assert len(selected_pairs) == len(set(selected_pairs))
    dropped_mrna = next(
        row
        for row in result.sample_map
        if row.patient_id == "TCGA-AA-0002" and row.role == "mrna"
    )
    assert not dropped_mrna.selected
    assert dropped_mrna.rejected_reason == "notInRetainedCohort"
    assert result.qc.ignored_sidecar_roles == ["proteinAnnotation"]
    assert result.qc.patient_level_fallback_count == 1


def test_available_policy_keeps_union_and_aligns_missing_roles_as_null() -> None:
    result = harmonize_samples(
        synthetic_request(cohortPolicy=CohortPolicy.AVAILABLE.value)
    )

    assert result.retained_cohort == [
        "TCGA-AA-0001",
        "TCGA-AA-0002",
        "TCGA-AA-0003",
    ]
    assert result.aligned_sample_ids["protein"] == [
        "TCGA-AA-0001-01A",
        None,
        "TCGA-AA-0003-01A",
    ]
    assert result.aligned_sample_ids["clinicalPatient"] == [None, None, None]


def test_two_modalities_can_harmonize_without_clinical_artifacts() -> None:
    result = harmonize_samples(
        HarmonizationRequest(
            candidates=[
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
            ],
            clinical_patient_ids=[],
            clinical_sample_to_patient={},
            cohort_roles=["mrna", "protein"],
        )
    )

    assert result.retained_cohort == ["TCGA-AA-0001"]
    assert result.qc.barcode_derived_count == 2
    assert result.qc.warnings == ["clinicalValidationUnavailable:2"]


def test_result_is_invariant_to_candidate_and_mapping_order() -> None:
    request = synthetic_request(cohortPolicy="available")
    expected = harmonize_samples(request).model_dump(mode="json")
    random = Random(1729)

    for _ in range(20):
        shuffled_candidates = list(request.candidates)
        mapping_items = list(request.clinical_sample_to_patient.items())
        random.shuffle(shuffled_candidates)
        random.shuffle(mapping_items)
        shuffled_request = request.model_copy(
            update={
                "candidates": shuffled_candidates,
                "clinical_sample_to_patient": dict(mapping_items),
            }
        )
        assert harmonize_samples(shuffled_request).model_dump(mode="json") == expected


def test_exact_sample_policy_intersects_before_replicate_selection() -> None:
    request = synthetic_request(matchLevel=MatchLevel.EXACT_SAMPLE.value)
    shared_second_vial = [
        (
            candidate("protein", "TCGA-AA-0001-01B")
            if item.role == "protein" and item.source_sample_id == "TCGA-AA-0001-01A"
            else item
        )
        for item in request.candidates
    ]
    result = harmonize_samples(
        request.model_copy(update={"candidates": shared_second_vial})
    )

    assert result.retained_cohort == ["TCGA-AA-0001"]
    selected = {
        row.role: row.source_sample_id for row in result.sample_map if row.selected
    }
    assert selected["mrna"] == "TCGA-AA-0001-01B-01D-0001-01"
    assert selected["protein"] == "TCGA-AA-0001-01B"


def test_exact_sample_policy_rejects_a_true_empty_intersection() -> None:
    request = synthetic_request(matchLevel=MatchLevel.EXACT_SAMPLE.value)
    non_matching = [
        (
            candidate("protein", "TCGA-AA-0001-02A")
            if item.role == "protein" and item.source_sample_id == "TCGA-AA-0001-01A"
            else item
        )
        for item in request.candidates
    ]
    with pytest.raises(NoCommonSamplesError):
        harmonize_samples(request.model_copy(update={"candidates": non_matching}))


def test_exact_sample_qc_keeps_patients_without_a_common_sample() -> None:
    result = harmonize_samples(
        HarmonizationRequest(
            candidates=[
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
                candidate("mrna", "TCGA-AA-0002-01A"),
                candidate("protein", "TCGA-AA-0002-02A"),
            ],
            clinical_patient_ids=[],
            clinical_sample_to_patient={},
            cohort_roles=["mrna", "protein"],
            match_level=MatchLevel.EXACT_SAMPLE,
        )
    )

    assert result.retained_cohort == ["TCGA-AA-0001"]
    assert [row.patient_id for row in result.availability] == [
        "TCGA-AA-0001",
        "TCGA-AA-0002",
    ]
    assert result.qc.available_patient_count == 2
    assert result.qc.dropped_patient_count == 1


def test_conflicting_clinical_mapping_is_terminal() -> None:
    with pytest.raises(ClinicalMappingConflictError):
        harmonize_samples(
            synthetic_request(
                clinicalSampleToPatient={
                    "TCGA-AA-0001-01A": "TCGA-AA-0002",
                }
            )
        )


def test_duplicate_candidate_is_terminal() -> None:
    request = synthetic_request()
    duplicates = deepcopy(request.candidates)
    duplicates.append(deepcopy(request.candidates[0]))
    with pytest.raises(DuplicateSampleCandidateError):
        harmonize_samples(request.model_copy(update={"candidates": duplicates}))


def test_request_rejects_unknown_fields_and_overlapping_role_declarations() -> None:
    with pytest.raises(ValidationError):
        HarmonizationRequest.model_validate(
            {
                **synthetic_request().model_dump(by_alias=True),
                "storageAccount": "caller-controlled",
            }
        )

    with pytest.raises(ValidationError, match="both cohort-bearing and a sidecar"):
        synthetic_request(cohortRoles=["mrna", "proteinAnnotation"])


def two_role_request(candidates: list[SampleCandidate], **overrides: object):
    return HarmonizationRequest(
        candidates=candidates,
        clinical_patient_ids=[],
        clinical_sample_to_patient={},
        cohort_roles=["mrna", "protein"],
        **overrides,
    )


def test_candidates_outside_the_cohort_roles_are_reported_not_discarded() -> None:
    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
                candidate("methylation", "TCGA-AA-0001-01A"),
            ]
        )
    )

    qc = result.qc
    assert qc.candidate_count == (
        qc.selected_candidate_count + qc.rejected_candidate_count
    )
    assert [
        (row.role, row.reason)
        for row in result.drop_report
        if row.role == "methylation"
    ] == [("methylation", "roleNotInCohort")]
    assert "roleNotInCohort:1" in qc.warnings
    # The unused role must not invent a patient in the availability table.
    assert qc.available_patient_count == 1


def test_sample_type_code_zero_is_ranked_ahead_of_higher_codes() -> None:
    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-00A", artifact="art_a"),
                candidate("mrna", "TCGA-AA-0001-11A", artifact="art_b"),
                candidate("protein", "TCGA-AA-0001-01A"),
            ]
        )
    )

    selected = {
        row.role: row.source_sample_id for row in result.sample_map if row.selected
    }
    assert selected["mrna"] == "TCGA-AA-0001-00A"


def test_replicates_rejected_on_sample_type_are_not_labelled_as_vial() -> None:
    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-06A", artifact="art_a"),
                candidate("mrna", "TCGA-AA-0001-11B", artifact="art_b"),
                candidate("protein", "TCGA-AA-0001-01A"),
            ]
        )
    )

    mrna = {
        row.source_sample_id: (row.selection_reason, row.rejected_reason)
        for row in result.sample_map
        if row.role == "mrna"
    }
    assert mrna["TCGA-AA-0001-06A"] == ("lowestSampleTypeCodeSelected", None)
    assert mrna["TCGA-AA-0001-11B"] == (None, "higherSampleTypeCode")


def test_blood_derived_primary_codes_count_as_primary_tumour() -> None:
    assert parse_tcga_barcode("TCGA-AA-0001-03A").is_primary_tumour
    assert parse_tcga_barcode("TCGA-AA-0001-09A").is_primary_tumour
    assert not parse_tcga_barcode("TCGA-AA-0001-11A").is_primary_tumour

    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-03A", artifact="art_a"),
                candidate("mrna", "TCGA-AA-0001-11A", artifact="art_b"),
                candidate("protein", "TCGA-AA-0001-01A"),
            ]
        )
    )
    mrna = {
        row.source_sample_id: row.rejected_reason
        for row in result.sample_map
        if row.role == "mrna"
    }
    assert mrna["TCGA-AA-0001-03A"] is None
    assert mrna["TCGA-AA-0001-11A"] == "nonPrimaryTumour"


def test_cohort_roles_are_stripped_before_matching_candidates() -> None:
    result = harmonize_samples(
        HarmonizationRequest(
            candidates=[
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
            ],
            clinical_patient_ids=[],
            clinical_sample_to_patient={},
            cohort_roles=["mrna ", " protein"],
        )
    )

    assert result.qc.cohort_roles == ["mrna", "protein"]
    assert result.retained_cohort == ["TCGA-AA-0001"]


def test_availability_reports_retention_and_joins_to_aligned_sample_ids() -> None:
    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-01A-01D-0001-01"),
                candidate("protein", "TCGA-AA-0001-01A"),
                candidate("mrna", "TCGA-AA-0002-01A", artifact="art_other"),
            ]
        )
    )

    retention = {row.patient_id: row.retained for row in result.availability}
    assert retention == {"TCGA-AA-0001": True, "TCGA-AA-0002": False}
    assert [
        patient_id for patient_id, kept in retention.items() if kept
    ] == result.retained_cohort

    kept = next(row for row in result.availability if row.retained)
    for role in result.qc.cohort_roles:
        assert kept.selected_samples[role] == result.aligned_sample_ids[role][0]
    # Full and canonical barcodes still resolve to the same underlying sample.
    assert kept.exact_sample_overlap


def test_exact_sample_drops_do_not_claim_a_present_role_is_missing() -> None:
    result = harmonize_samples(
        two_role_request(
            [
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
                candidate("mrna", "TCGA-AA-0002-01A", artifact="art_other"),
            ],
            cohort_policy=CohortPolicy.AVAILABLE,
            match_level=MatchLevel.EXACT_SAMPLE,
        )
    )

    dropped = {
        (row.role, row.reason)
        for row in result.drop_report
        if row.patient_id == "TCGA-AA-0002"
    }
    assert ("mrna", "notInExactSampleIntersection") in dropped
    assert ("mrna", "missingRequiredRole") not in dropped
    assert ("protein", "missingRequiredRole") in dropped


def test_candidate_without_any_identity_raises_a_domain_error() -> None:
    identityless = candidate("mrna", "TCGA-AA-0001-01A").model_copy(
        update={"source_sample_id": None, "patient_id": None}
    )
    request = two_role_request(
        [
            candidate("mrna", "TCGA-AA-0001-01A"),
            candidate("protein", "TCGA-AA-0001-01A"),
        ]
    )

    with pytest.raises(HarmonizationError, match="neither"):
        harmonize_samples(
            request.model_copy(
                update={
                    "candidates": [
                        identityless,
                        candidate("protein", "TCGA-AA-0001-01A"),
                    ]
                }
            )
        )
