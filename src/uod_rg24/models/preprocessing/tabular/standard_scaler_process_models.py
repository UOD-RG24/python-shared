from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProcessStepModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    step: str
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime = Field(alias="completedAt")
    duration_ms: float = Field(
        alias="durationMs",
        ge=0,
    )
    message: str | None = None


class FileProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    azure_storage_account_name: str = Field(alias="azureStorageAccountName")
    azure_container_name: str = Field(alias="azureContainerName")
    directory_name: str = Field(alias="directoryName")
    blob_path: str = Field(alias="blobPath")
    file_name: str = Field(alias="fileName")
    extension: str = Field(alias="extension")
    size_bytes: int = Field(
        alias="sizeBytes",
        ge=0,
    )
    size_mb: float = Field(
        alias="sizeMb",
        ge=0,
    )


class TemporaryFileInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    temporary_directory: str = Field(alias="temporaryDirectory")
    input_file_path: str = Field(alias="inputFilePath")
    output_file_path: str = Field(alias="outputFilePath")


class StandardScalerInfoModel(BaseModel):
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


class StandardScalerStandardizationProcessResponseModel(BaseModel):
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
    scaler: StandardScalerInfoModel
    steps: list[ProcessStepModel] = Field(
        default_factory=list[ProcessStepModel],
    )


class StandardScalerStandardizationProcessRequestModel(BaseModel):
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
