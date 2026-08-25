from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import ProcessStepModel


class ApplySGCCAWeightsProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    component: int = Field(
            ge=1,
        )
    index_column: str = Field(
        alias="indexColumn",
        min_length=1,
    )


class ApplySGCCAWeightsBlockOutputModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    block: str = Field(
        min_length=1,
    )
    component_applied: int = Field(
        alias="componentApplied",
        ge=1,
    )
    samples_processed: int = Field(
        alias="samplesProcessed",
        ge=0,
    )
    features_weighted: int = Field(
        alias="featuresWeighted",
        ge=0,
    )
    zero_weighted_values: int = Field(
        alias="zeroWeightedValues",
        ge=0,
    )
    feature_columns: list[str] = Field(
        alias="featureColumns",
    )
    input_file_path: Path = Field(
        alias="inputFilePath",
    )
    output_file_path: Path = Field(
        alias="outputFilePath",
    )


class ApplySGCCAWeightsProcessResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    message: str = Field(
        min_length=1,
    )
    component_applied: int = Field(
        alias="componentApplied",
        ge=1,
    )
    weights_file_path: Path = Field(
        alias="weightsFilePath",
    )
    blocks_processed: int = Field(
        alias="blocksProcessed",
        ge=1,
    )
    outputs: list[ApplySGCCAWeightsBlockOutputModel] = Field(
        min_length=1,
    )
    total_elapsed_seconds: float = Field(
        alias="totalElapsedSeconds",
    )  
    steps: list[ProcessStepModel]
