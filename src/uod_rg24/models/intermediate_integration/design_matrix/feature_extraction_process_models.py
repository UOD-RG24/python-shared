from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class FeatureExtractionProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    feature_names: list[str] = Field(
        alias="featureNames",
    )

    feature_count: int = Field(
        alias="featureCount",
        ge=0,
    )

    total_column_count: int = Field(
        alias="totalColumnCount",
        ge=0,
    )

    excluded_columns: list[str] = Field(
        default_factory=list,
        alias="excludedColumns",
    )

    excluded_column_count: int = Field(
        alias="excludedColumnCount",
        ge=0,
    )
    missing_excluded_columns: list[str] = Field(
        default_factory=list,
        alias="missingExcludedColumns",
    )


class FeatureExtractionProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        alias="excludedColumns",
    )


class FeatureExtractionProcessResponseModel(BaseModel):
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

    output_file: FileProcessInfoModel | None = Field(
        default=None,
        alias="outputFile",
    )

    temporary_files: TemporaryFileInfoModel | None = Field(
        default=None,
        alias="temporaryFiles",
    )

    feature_extraction_process_info: FeatureExtractionProcessInfoModel = Field(
        alias="featureExtractionProcessInfo",
    )

    steps: list[ProcessStepModel]
