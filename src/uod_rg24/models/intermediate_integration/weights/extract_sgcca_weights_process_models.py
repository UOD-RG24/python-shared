from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    ProcessStepModel,
)

SparsityValue = Annotated[
    float,
    Field(
        gt=0.0,
        le=1.0,
    ),
]


class SGCCABlockDetailsModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    patients: int = Field(ge=0)
    features: int = Field(ge=0)


class SGCCABlocksProcessedDetailsModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    mrna: SGCCABlockDetailsModel = Field(alias="mRNA")
    cna: SGCCABlockDetailsModel = Field(alias="CNA")


class ExtractSGCCAWeightsProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )


class ExtractSGCCAWeightsProcessRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    index_column: str = Field(
        alias="indexColumn",
        min_length=1,
    )
    n_components: int = Field(
        alias="nComponents",
        ge=1,
    )
    sparsity: list[SparsityValue] = Field(
        min_length=2,
        max_length=2,
    )
    scheme: Literal[
        "horst",
        "factorial",
        "centroid",
    ] = "factorial"
    scale: bool = True


class ExtractSGCCAWeightsProcessResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    message: str
    r_version: str = Field(alias="rVersion")
    rgcca_version: str = Field(alias="rgccaVersion")
    patients_processed: int = Field(
        alias="patientsProcessed",
        ge=0,
    )
    blocks_processed: int = Field(
        alias="blocksProcessed",
        ge=0,
    )
    block_details: SGCCABlocksProcessedDetailsModel = Field(
        alias="blockDetails",
    )

    components_requested: int = Field(
        alias="componentsRequested",
        ge=1,
    )
    sparsity: list[float] = Field(min_length=1)
    scheme: str = Field(min_length=1)
    scale: bool

    weights_extracted: int = Field(
        alias="weightsExtracted",
        ge=0,
    )
    selected_weights: int = Field(
        alias="selectedWeights",
        ge=0,
    )
    selected_features: int = Field(
        alias="selectedFeatures",
        ge=0,
    )
    output_file_path: str = Field(
        alias="outputFilePath",
        min_length=1,
    )
    total_elapsed_seconds: float = Field(
        alias="totalElapsedSeconds",
        ge=0.0,
    )
    steps: list[ProcessStepModel]
