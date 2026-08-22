from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic.alias_generators import to_camel


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


NonEmptyStr = Annotated[str, StringConstraints(min_length=1, pattern=r".*\S.*")]
# Role names are matched by equality throughout the pipeline, so they are
# stripped at the boundary; otherwise a stray space silently matches nothing.
RoleStr = Annotated[str, BeforeValidator(_strip), StringConstraints(min_length=1)]
SIDECAR_ROLES = frozenset(
    {
        "clinicalSample",
        "proteinAnnotation",
        "diagnosisTimeline",
        "treatmentTimeline",
    }
)


class DomainModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        validate_default=True,
    )


class CohortPolicy(StrEnum):
    INTERSECTION = "intersection"
    AVAILABLE = "available"


class MatchLevel(StrEnum):
    PATIENT = "patient"
    EXACT_SAMPLE = "exactSample"


class SampleCandidate(DomainModel):
    role: RoleStr
    source_artifact_id: NonEmptyStr
    source_sample_id: NonEmptyStr | None = None
    patient_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def require_an_identity(self) -> SampleCandidate:
        if self.source_sample_id is None and self.patient_id is None:
            raise ValueError("sourceSampleId or patientId is required")
        return self


class HarmonizationRequest(DomainModel):
    candidates: Annotated[list[SampleCandidate], Field(min_length=1)]
    clinical_patient_ids: list[NonEmptyStr]
    clinical_sample_to_patient: dict[NonEmptyStr, NonEmptyStr]
    cohort_roles: list[RoleStr] = Field(default_factory=list)
    cohort_policy: CohortPolicy = CohortPolicy.INTERSECTION
    match_level: MatchLevel = MatchLevel.PATIENT
    replicate_recipe: Annotated[str, Field(pattern=r"^tcga_primary_tumour_v1$")] = (
        "tcga_primary_tumour_v1"
    )

    @model_validator(mode="after")
    def validate_role_declarations(self) -> HarmonizationRequest:
        cohort_roles = set(self.cohort_roles)
        normalized_patients = {
            patient_id.strip().upper() for patient_id in self.clinical_patient_ids
        }
        if len(self.cohort_roles) != len(cohort_roles):
            raise ValueError("cohort roles must be unique")
        if len(self.clinical_patient_ids) != len(normalized_patients):
            raise ValueError("clinical patient IDs must be unique")
        if cohort_roles & SIDECAR_ROLES:
            raise ValueError("a role cannot be both cohort-bearing and a sidecar")
        return self


class SampleMapRow(DomainModel):
    entity_id: NonEmptyStr
    patient_id: NonEmptyStr
    modality: NonEmptyStr
    role: NonEmptyStr
    source_sample_id: NonEmptyStr | None
    selected: bool
    selection_reason: NonEmptyStr | None
    rejected_reason: NonEmptyStr | None
    replicate_group: NonEmptyStr
    source_artifact_id: NonEmptyStr
    mapping_status: NonEmptyStr


class AvailabilityRow(DomainModel):
    entity_id: NonEmptyStr
    patient_id: NonEmptyStr
    role_availability: dict[NonEmptyStr, bool]
    selected_samples: dict[NonEmptyStr, NonEmptyStr | None]
    exact_sample_overlap: bool
    retained: bool


class DropReportRow(DomainModel):
    patient_id: NonEmptyStr
    role: NonEmptyStr
    reason: NonEmptyStr
    source_sample_id: NonEmptyStr | None = None


class HarmonizationQC(DomainModel):
    candidate_count: Annotated[int, Field(ge=0)]
    valid_candidate_count: Annotated[int, Field(ge=0)]
    selected_candidate_count: Annotated[int, Field(ge=0)]
    rejected_candidate_count: Annotated[int, Field(ge=0)]
    available_patient_count: Annotated[int, Field(ge=0)]
    retained_patient_count: Annotated[int, Field(ge=0)]
    dropped_patient_count: Annotated[int, Field(ge=0)]
    patient_level_fallback_count: Annotated[int, Field(ge=0)]
    barcode_derived_count: Annotated[int, Field(ge=0)]
    cohort_roles: list[NonEmptyStr]
    ignored_sidecar_roles: list[NonEmptyStr]
    warnings: list[NonEmptyStr]


class HarmonizationResult(DomainModel):
    sample_map: list[SampleMapRow]
    availability: list[AvailabilityRow]
    retained_cohort: list[NonEmptyStr]
    drop_report: list[DropReportRow]
    aligned_sample_ids: dict[NonEmptyStr, list[NonEmptyStr | None]]
    sample_order_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    qc: HarmonizationQC
