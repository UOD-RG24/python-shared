from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)


class RobustScalerInfoModel(BaseModel):
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

    with_centering: bool = Field(
        alias="withCentering",
    )

    with_scaling: bool = Field(
        alias="withScaling",
    )

    quantile_range: tuple[float, float] = Field(
        alias="quantileRange",
    )

    unit_variance: bool = Field(
        alias="unitVariance",
    )

    center: list[float] | None

    scale: list[float] | None

    n_features_in: int = Field(
        alias="nFeaturesIn",
        ge=0,
    )


class RobustScalerStandardizationProcessRequestModel(BaseModel):
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

    with_centering: bool = Field(
        default=True,
        alias="withCentering",
    )

    with_scaling: bool = Field(
        default=True,
        alias="withScaling",
    )

    quantile_range: tuple[float, float] = Field(
        default=(25.0, 75.0),
        alias="quantileRange",
    )

    copy_: bool = Field(
        default=True,
        alias="copy",
    )

    unit_variance: bool = Field(
        default=False,
        alias="unitVariance",
    )

    @model_validator(mode="after")
    def validate_quantile_range(self) -> Self:
        q_min, q_max = self.quantile_range

        if not 0.0 < q_min < q_max < 100.0:
            raise ValueError(
                "quantileRange must satisfy " "0.0 < q_min < q_max < 100.0."
            )

        return self


class RobustScalerStandardizationProcessResponseModel(BaseModel):
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

    scaler: RobustScalerInfoModel

    steps: list[ProcessStepModel]
