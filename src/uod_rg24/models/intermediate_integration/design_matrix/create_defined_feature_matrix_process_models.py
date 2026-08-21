from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class CreateDefinedFeatureMatrixProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    requested_features: list[str] = Field(
        alias="requestedFeatures",
        description="Feature names requested for the output matrix.",
    )
    selected_features: list[str] = Field(
        alias="selectedFeatures",
        description="Feature names included in the output matrix.",
    )
    missing_features: list[str] = Field(
        alias="missingFeatures",
        description="Requested feature names not found in the input matrix.",
    )
    requested_feature_count: int = Field(
        alias="requestedFeatureCount",
        ge=0,
    )
    selected_feature_count: int = Field(
        alias="selectedFeatureCount",
        ge=0,
    )
    missing_feature_count: int = Field(
        alias="missingFeatureCount",
        ge=0,
    )
    samples_processed: int = Field(
        alias="samplesProcessed",
        ge=0,
        description="Number of sample rows written to the output matrix.",
    )


class CreateDefinedFeatureMatrixProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    defined_features: list[str] = Field(
        alias="definedFeatures",
        min_length=1,
        description="Features to include in the output matrix.",
    )


class CreateDefinedFeatureMatrixProcessResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    success: bool

    started_at: datetime = Field(
        alias="startedAt",
    )

    completed_at: datetime = Field(
        alias="completedAt",
    )

    total_duration_ms: float = Field(
        alias="totalDurationMs",
        ge=0,
    )

    input_file: FileProcessInfoModel = Field(
        alias="inputFile",
    )

    output_file: FileProcessInfoModel = Field(
        alias="outputFile",
    )

    temporary_files: TemporaryFileInfoModel = Field(
        alias="temporaryFiles",
    )

    create_defined_feature_matrix_process_info: (
        CreateDefinedFeatureMatrixProcessInfoModel
    ) = Field(
        alias="createDefinedFeatureMatrixProcessInfo",
    )

    steps: list[ProcessStepModel]
