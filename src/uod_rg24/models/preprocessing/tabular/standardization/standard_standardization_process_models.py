from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class StandardStandardizationInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    numeric_columns: list[str] = Field(alias="numericColumns")
    numeric_column_count: int = Field(alias="numericColumnCount")
    rows_processed: int = Field(alias="rowsProcessed")
    fit_chunks_processed: int = Field(alias="fitChunksProcessed")
    transform_chunks_processed: int = Field(alias="transformChunksProcessed")
    chunk_size: int = Field(alias="chunkSize")
    mean: list[float] = Field(default_factory=list[float])
    variance: list[float] = Field(default_factory=list[float])
    scale: list[float] = Field(default_factory=list[float])
    samples_seen: int | list[int] | None = Field(
        default=None,
        alias="samplesSeen",
    )


class StandardStandardizationProcessResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    success: bool
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime = Field(alias="completedAt")
    total_duration_ms: float = Field(
        alias="totalDurationMs",
        ge=0,
    )
    input_file: FileProcessInfoModel = Field(alias="inputFile")
    output_file: FileProcessInfoModel = Field(alias="outputFile")
    temporary_files: TemporaryFileInfoModel = Field(alias="temporaryFiles")
    standardization_info: StandardStandardizationInfoModel = Field(
        alias="standardizationInfo",
    )
    steps: list[ProcessStepModel] = Field(
        default_factory=list[ProcessStepModel],
    )


class StandardStandardizationProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    with_mean: bool = Field(
        default=True,
        alias="withMean",
    )
    with_std: bool = Field(
        default=True,
        alias="withStd",
    )
    copy_: bool = Field(
        default=True,
        alias="copy",
    )
    chunk_size: int = Field(
        default=100_000,
        alias="chunkSize",
        ge=1,
    )
    numeric_columns: list[str] | None = Field(
        default=None,
        alias="numericColumns",
    )
