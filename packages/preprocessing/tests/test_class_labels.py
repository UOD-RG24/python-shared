from __future__ import annotations

import pytest
from uod_rg24_preprocessing.class_labels import (
    ClassLabelError,
    MappingStatus,
    RawClinicalLabel,
    derive_ajcc_stage_4class,
    derive_identity_labels,
    derive_tumour_normal,
)


def records(*values: str | None) -> list[RawClinicalLabel]:
    return [
        RawClinicalLabel(f"patient-{index}", f"patient-{index}", value)
        for index, value in enumerate(values, start=1)
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Stage I", "Stage I"),
        ("stage ia", "Stage I"),
        ("Pathologic Stage IB2", "Stage I"),
        ("Stage IIA", "Stage II"),
        ("AJCC Stage III C", "Stage III"),
        ("pathological stage IVB", "Stage IV"),
    ],
)
def test_ajcc_recipe_collapses_supported_substages(raw: str, expected: str) -> None:
    result = derive_ajcc_stage_4class(records(raw))

    assert result.labels[0].raw_value == raw
    assert result.labels[0].label_value == expected
    assert result.labels[0].mapping_status == MappingStatus.MAPPED
    assert result.labels[0].mapping_reason == "ajccStageGroupCollapsed"


def test_ajcc_recipe_retains_missing_and_unmapped_rows() -> None:
    result = derive_ajcc_stage_4class(
        records(None, "Not Reported", "Stage 0", "Stage V", "Stage IIIA")
    )

    assert [item.mapping_status for item in result.labels] == [
        MappingStatus.MISSING,
        MappingStatus.MISSING,
        MappingStatus.UNMAPPED,
        MappingStatus.UNMAPPED,
        MappingStatus.MAPPED,
    ]
    assert [item.mapping_reason for item in result.labels] == [
        "missingOrNotReported",
        "missingOrNotReported",
        "unsupportedStage0",
        "unrecognizedStage",
        "ajccStageGroupCollapsed",
    ]
    assert result.mapped_count == 1


def test_tumour_normal_uses_controlled_values_only() -> None:
    result = derive_tumour_normal(
        records("Primary Tumor", "Solid Tissue Normal", "Cell Line", "Unknown")
    )

    assert [item.label_value for item in result.labels] == [
        "Tumour",
        "Normal",
        None,
        None,
    ]
    assert result.labels[2].mapping_status == MappingStatus.UNMAPPED
    assert result.labels[3].mapping_status == MappingStatus.MISSING


def test_identity_recipe_preserves_approved_canonical_spelling() -> None:
    result = derive_identity_labels(
        records(" alive ", "DEAD", "lost"),
        source_field="vitalStatus",
        approved_values=("Alive", "Dead"),
    )

    assert [item.label_value for item in result.labels] == ["Alive", "Dead", None]
    assert result.labels[2].mapping_reason == "valueNotApproved"


def test_duplicate_or_empty_cohorts_fail() -> None:
    duplicate = [
        RawClinicalLabel("patient-1", "patient-1", "Stage I"),
        RawClinicalLabel("patient-1", "patient-1", "Stage II"),
    ]
    with pytest.raises(ClassLabelError) as caught:
        derive_ajcc_stage_4class(duplicate)
    assert caught.value.code == "DUPLICATE_SAMPLE_ID"

    with pytest.raises(ClassLabelError) as caught:
        derive_ajcc_stage_4class([])
    assert caught.value.code == "EMPTY_CLINICAL_COHORT"


def test_timeline_fields_are_not_approved_identity_labels_by_default() -> None:
    with pytest.raises(ClassLabelError) as caught:
        derive_identity_labels(
            records("120"), source_field="daysToDeath", approved_values=()
        )
    assert caught.value.code == "INVALID_PARAMETERS"
