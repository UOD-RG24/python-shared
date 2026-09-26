from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import ClassLabelError


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    MISSING = "missing"
    UNMAPPED = "unmapped"


@dataclass(frozen=True, slots=True)
class RawClinicalLabel:
    entity_id: str
    patient_id: str
    raw_value: str | None

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.patient_id.strip():
            raise ClassLabelError(
                "INVALID_CLINICAL_RECORD",
                "Clinical entity and patient identifiers must not be empty.",
            )


@dataclass(frozen=True, slots=True)
class LabelMapping:
    entity_id: str
    patient_id: str
    raw_value: str | None
    label_value: str | None
    mapping_status: MappingStatus
    mapping_reason: str

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.patient_id.strip():
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT", "Label identifiers must not be empty."
            )
        if not self.mapping_reason.strip():
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT", "Every label requires a mapping reason."
            )
        if (self.mapping_status == MappingStatus.MAPPED) != (
            self.label_value is not None
        ):
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT",
                "Only mapped rows may contain a label value.",
            )


@dataclass(frozen=True, slots=True)
class ClassLabelResult:
    schema_version: str
    recipe_id: str
    recipe_version: str
    source_field: str
    labels: tuple[LabelMapping, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "clinical-labels/1.0":
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT", "Unsupported label schema version."
            )
        if not self.recipe_id.strip() or not self.recipe_version.strip():
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT", "Label recipe provenance must not be empty."
            )
        if not self.source_field.strip() or not self.labels:
            raise ClassLabelError(
                "INVALID_LABEL_ARTIFACT",
                "Label output requires a source field and at least one row.",
            )
        ids = [item.entity_id for item in self.labels]
        if len(ids) != len(set(ids)):
            raise ClassLabelError(
                "DUPLICATE_SAMPLE_ID",
                "Clinical label entity identifiers must be unique.",
            )

    @property
    def mapped_count(self) -> int:
        return sum(item.mapping_status == MappingStatus.MAPPED for item in self.labels)
