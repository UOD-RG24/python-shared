from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import ProcessStepModel


class CreateFinalSelectedMatrixProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    block: str = Field(
        min_length=1,
    )
    identifier_column: str = Field(
        alias="identifierColumn",
        min_length=1,
    )


class CreateFinalSelectedMatrixProcessResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    message: str = Field(
        min_length=1,
    )
    output_file_path: Path = Field(
        alias="outputFilePath",
    )
    row_count: int = Field(
        alias="rowCount",
        ge=0,
    )
    column_count: int = Field(
        alias="columnCount",
        ge=0,
    )
    selected_feature_count: int = Field(
        alias="selectedFeatureCount",
        ge=0,
    )
    total_elapsed_seconds: float = Field(
        alias="totalElapsedSeconds",
    )
    steps: list[ProcessStepModel]
