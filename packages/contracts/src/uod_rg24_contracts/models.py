from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from .base import (
    ImageDigest,
    NonEmptyStr,
    Sha256,
    TraceId,
    TraceParent,
    UtcDatetime,
    WireModel,
)


class Layer(StrEnum):
    STANDARDIZATION = "standardization"
    NORMALIZATION = "normalization"
    SAMPLE_SELECTION = "sampleSelection"
    FEATURE_SELECTION = "featureSelection"
    CLASS_LABELS = "classLabels"
    MATRIX_EXTRACTION = "matrixExtraction"


class RunStatus(StrEnum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_SCHEDULED = "retryScheduled"
    CANCELLATION_REQUESTED = "cancellationRequested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactState(StrEnum):
    RESERVED = "reserved"
    WRITING = "writing"
    COMMITTED = "committed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class Modality(StrEnum):
    MRNA = "mrna"
    CNA = "cna"
    PROTEIN = "protein"
    CLINICAL = "clinical"
    MIXED = "mixed"


class LearningScope(StrEnum):
    NONE = "none"
    TRAIN_FOLD = "trainFold"
    COHORT_UNSUPERVISED = "cohortUnsupervised"


Extensions = dict[str, JsonValue]
COHORT_UNSUPERVISED_LEAKAGE_WARNING = "cohortUnsupervisedMayBiasSupervisedEvaluation"


class ArtifactRef(WireModel):
    artifact_id: NonEmptyStr
    dataset_id: NonEmptyStr
    kind: NonEmptyStr
    modality: Modality | None = None
    schema_id: NonEmptyStr
    state: ArtifactState
    sha256: Sha256
    created_at: UtcDatetime
    extensions: Extensions = Field(default_factory=dict)


class StepInput(WireModel):
    role: NonEmptyStr
    artifact_id: NonEmptyStr
    expected_sha256: Sha256
    expected_schema_id: NonEmptyStr


class StepOutputReservation(WireModel):
    name: NonEmptyStr
    artifact_id: NonEmptyStr
    kind: NonEmptyStr
    schema_id: NonEmptyStr


class ExecutionContext(WireModel):
    recipe_id: NonEmptyStr
    recipe_version: NonEmptyStr
    random_seed: StrictInt
    requested_at: UtcDatetime
    traceparent: TraceParent
    extensions: Extensions = Field(default_factory=dict)


class StepCommandV1(WireModel):
    schema_version: Literal["1.0"]
    message_id: NonEmptyStr
    run_id: NonEmptyStr
    step_run_id: NonEmptyStr
    attempt: Annotated[StrictInt, Field(ge=1)]
    experiment_id: NonEmptyStr
    dataset_id: NonEmptyStr
    owner_id: NonEmptyStr
    layer: Layer
    operation: NonEmptyStr
    inputs: Annotated[list[StepInput], Field(min_length=1)]
    outputs: Annotated[list[StepOutputReservation], Field(min_length=1)]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    execution_context: ExecutionContext
    extensions: Extensions = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_roles_and_outputs(self) -> StepCommandV1:
        roles = [item.role for item in self.inputs]
        output_names = [item.name for item in self.outputs]
        output_ids = [item.artifact_id for item in self.outputs]
        if len(roles) != len(set(roles)):
            raise ValueError("input roles must be unique")
        if len(output_names) != len(set(output_names)):
            raise ValueError("output names must be unique")
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output artifact IDs must be unique")
        input_ids = {item.artifact_id for item in self.inputs}
        if input_ids & set(output_ids):
            raise ValueError("output artifact IDs must not alias input artifacts")
        return self


class ProblemError(WireModel):
    path: NonEmptyStr | None = None
    reason: NonEmptyStr
    count: Annotated[StrictInt, Field(ge=0)] | None = None
    extensions: Extensions = Field(default_factory=dict)


class ProblemDetails(WireModel):
    type: NonEmptyStr
    title: NonEmptyStr
    status: Annotated[StrictInt, Field(ge=400, le=599)]
    code: NonEmptyStr
    detail: NonEmptyStr
    instance: NonEmptyStr | None = None
    trace_id: TraceId | None = None
    errors: list[ProblemError] = Field(default_factory=list[ProblemError])
    extensions: Extensions = Field(default_factory=dict)


class StepEventOutput(WireModel):
    name: NonEmptyStr
    artifact_id: NonEmptyStr
    sha256: Sha256


class StepMetrics(WireModel):
    duration_ms: Annotated[StrictInt, Field(ge=0)]
    input_rows: Annotated[StrictInt, Field(ge=0)] | None = None
    input_features: Annotated[StrictInt, Field(ge=0)] | None = None
    output_rows: Annotated[StrictInt, Field(ge=0)] | None = None
    output_features: Annotated[StrictInt, Field(ge=0)] | None = None
    bytes_read: Annotated[StrictInt, Field(ge=0)] = 0
    bytes_written: Annotated[StrictInt, Field(ge=0)] = 0


class StepEventBase(WireModel):
    schema_version: Literal["1.0"]
    event_id: NonEmptyStr
    occurred_at: UtcDatetime
    run_id: NonEmptyStr
    step_run_id: NonEmptyStr
    attempt: Annotated[StrictInt, Field(ge=1)]
    message_id: NonEmptyStr
    trace_id: TraceId
    layer: Layer
    operation: NonEmptyStr
    code_version: NonEmptyStr
    container_image_digest: ImageDigest
    extensions: Extensions = Field(default_factory=dict)


class StepSucceededEventV1(StepEventBase):
    event_type: Literal["preprocessing.step.succeeded"]
    outputs: Annotated[list[StepEventOutput], Field(min_length=1)]
    metrics: StepMetrics
    warnings: list[NonEmptyStr] = Field(default_factory=list)


class StepFailedEventV1(StepEventBase):
    event_type: Literal["preprocessing.step.failed"]
    problem: ProblemDetails
    retryable: StrictBool
    metrics: StepMetrics


class ManifestCreator(WireModel):
    service: NonEmptyStr
    code_version: NonEmptyStr
    container_image_digest: ImageDigest
    recipe_id: NonEmptyStr
    recipe_version: NonEmptyStr


class ManifestInput(WireModel):
    artifact_id: NonEmptyStr
    sha256: Sha256
    role: NonEmptyStr


class ManifestData(WireModel):
    media_type: NonEmptyStr
    sha256: Sha256
    byte_length: Annotated[StrictInt, Field(ge=0)]
    row_count: Annotated[StrictInt, Field(ge=0)] | None = None
    feature_count: Annotated[StrictInt, Field(ge=0)] | None = None
    schema_id: NonEmptyStr
    sample_order_sha256: Sha256 | None = None
    feature_order_sha256: Sha256 | None = None


class ManifestLearning(WireModel):
    learns_parameters: StrictBool
    learning_scope: LearningScope
    fit_population_artifact_id: NonEmptyStr | None = None
    fitted_artifact_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_learning_state(self) -> ManifestLearning:
        if not self.learns_parameters:
            if self.learning_scope != LearningScope.NONE:
                raise ValueError("non-learning operations must use learningScope=none")
            if self.fit_population_artifact_id or self.fitted_artifact_id:
                raise ValueError(
                    "non-learning operations cannot reference fitted state"
                )
        elif self.learning_scope == LearningScope.NONE:
            raise ValueError("learning operations must declare their learning scope")
        elif (
            self.learning_scope == LearningScope.TRAIN_FOLD
            and self.fit_population_artifact_id is None
        ):
            raise ValueError(
                "train-fold learning requires a fitting population artifact"
            )
        return self


class ArtifactManifestV1(WireModel):
    schema_version: Literal["artifact-manifest/1.0"]
    artifact_id: NonEmptyStr
    kind: NonEmptyStr
    modality: Modality | None = None
    dataset_id: NonEmptyStr
    experiment_id: NonEmptyStr
    run_id: NonEmptyStr
    step_run_id: NonEmptyStr
    created_at: UtcDatetime
    created_by: ManifestCreator
    inputs: Annotated[list[ManifestInput], Field(min_length=1)]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    data: ManifestData
    learning: ManifestLearning
    random_seed: StrictInt
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    extensions: Extensions = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cohort_learning_warning(self) -> ArtifactManifestV1:
        if (
            self.learning.learns_parameters
            and self.learning.learning_scope == LearningScope.COHORT_UNSUPERVISED
            and COHORT_UNSUPERVISED_LEAKAGE_WARNING not in self.warnings
        ):
            raise ValueError(
                "cohort-unsupervised learning requires the evaluation-bias warning"
            )
        return self


class QCSummaryV1(WireModel):
    schema_version: Literal["artifact-qc/1.0"]
    artifact_id: NonEmptyStr
    generated_at: UtcDatetime
    counts: dict[NonEmptyStr, Annotated[StrictInt, Field(ge=0)]] = Field(
        default_factory=dict
    )
    metrics: dict[NonEmptyStr, StrictInt | StrictFloat] = Field(default_factory=dict)
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    extensions: Extensions = Field(default_factory=dict)


class OperationInputSpec(WireModel):
    role: NonEmptyStr
    kind: NonEmptyStr
    accepted_schema_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    required: StrictBool = True


class OperationOutputSpec(WireModel):
    name: NonEmptyStr
    kind: NonEmptyStr
    schema_id: NonEmptyStr


class OperationDefinition(WireModel):
    layer: Layer
    operation: NonEmptyStr
    inputs: Annotated[list[OperationInputSpec], Field(min_length=1)]
    outputs: Annotated[list[OperationOutputSpec], Field(min_length=1)]
    parameter_schema: dict[str, JsonValue]
    learns_parameters: StrictBool
    allowed_learning_scopes: list[LearningScope] = Field(
        default_factory=list[LearningScope]
    )
    workload_class: NonEmptyStr


class OperationCatalogueV1(WireModel):
    schema_version: Literal["operation-catalogue/1.0"]
    operations: Annotated[list[OperationDefinition], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_operations(self) -> OperationCatalogueV1:
        keys = [(item.layer, item.operation) for item in self.operations]
        if len(keys) != len(set(keys)):
            raise ValueError("layer/operation pairs must be unique")
        return self
