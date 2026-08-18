from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class RetainCommonSamplesAcrossLayersProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )


class RetainCommonSamplesAcrossLayersProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )


class RetainCommonSamplesAcrossLayersProcessResponseModel(BaseModel):
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

    retain_common_samples_across_layers_info: (
        RetainCommonSamplesAcrossLayersProcessInfoModel
    ) = Field(
        alias="retainCommonSamplesAcrossLayersInfo",
    )

    steps: list[ProcessStepModel]
