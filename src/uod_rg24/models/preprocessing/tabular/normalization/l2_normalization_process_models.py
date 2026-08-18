from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class L2NormalizationInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    norm: Literal["l2"] = "l2"

    numeric_columns: list[str] = Field(
        alias="numericColumns",
    )

    numeric_column_count: int = Field(
        alias="numericColumnCount",
        ge=0,
    )

    rows_processed: int = Field(
        alias="rowsProcessed",
        ge=0,
    )

    transform_chunks_processed: int = Field(
        alias="transformChunksProcessed",
        ge=0,
    )

    chunk_size: int = Field(
        alias="chunkSize",
        gt=0,
    )


class L2NormalizationProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    chunk_size: int = Field(
        alias="chunkSize",
        gt=0,
    )

    numeric_columns: list[str] | None = Field(
        default=None,
        alias="numericColumns",
    )

    copy_: bool = Field(
        default=True,
        alias="copy",
    )


class L2NormalizationProcessResponseModel(BaseModel):
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

    normalization_info: L2NormalizationInfoModel

    steps: list[ProcessStepModel]
