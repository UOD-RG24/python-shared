from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class MinMaxScalerInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

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

    fit_chunks_processed: int = Field(
        alias="fitChunksProcessed",
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

    feature_range: tuple[float, float] = Field(
        alias="featureRange",
    )

    min_adjustment: list[float] = Field(
        alias="minAdjustment",
    )

    scale: list[float]

    data_min: list[float] = Field(
        alias="dataMin",
    )

    data_max: list[float] = Field(
        alias="dataMax",
    )

    data_range: list[float] = Field(
        alias="dataRange",
    )

    samples_seen: int = Field(
        alias="samplesSeen",
        ge=0,
    )


class MinMaxScalerStandardizationProcessRequestModel(BaseModel):
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

    feature_range: tuple[int, int] = Field(
        default=(0, 1),
        alias="featureRange",
    )

    copy_: bool = Field(
        default=True,
        alias="copy",
    )

    clip: bool = False

    @model_validator(mode="after")
    def validate_feature_range(
        self,
    ) -> MinMaxScalerStandardizationProcessRequestModel:
        minimum, maximum = self.feature_range

        if minimum >= maximum:
            raise ValueError("featureRange minimum must be " "less than maximum.")

        return self


class MinMaxScalerStandardizationProcessResponseModel(BaseModel):
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

    scaler: MinMaxScalerInfoModel

    steps: list[ProcessStepModel]
