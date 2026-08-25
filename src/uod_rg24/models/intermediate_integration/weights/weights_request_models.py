from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.intermediate_integration.weights.apply_sgcca_weights_process_models import (
    ApplySGCCAWeightsProcessRequestModel,
)
from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    MetadataModel,
)
from uod_rg24.models.intermediate_integration.weights.extract_sgcca_weights_process_models import (
    ExtractSGCCAWeightsProcessRequestModel,
)
from uod_rg24.models.intermediate_integration.weights.extract_sgcca_weights_process_models import (
    ExtractSGCCAWeightsProcessRequestModel,
)
from uod_rg24.models.intermediate_integration.weights.create_final_selected_matrix_process_models import (
    CreateFinalSelectedMatrixProcessRequestModel,
)
from uod_rg24.models.intermediate_integration.weights.weights_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TInput1 = TypeVar("TInput1")
TInput2 = TypeVar("TInput2")
TInput3 = TypeVar("TInput3")
TOutput1 = TypeVar("TOutput1")
TOutput2 = TypeVar("TOutput2")


class WeightsWithNoInputsOrOutputsRequestModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for RGCCA health assessment.",
    )
    requested_by: str | None = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: MetadataModel | None = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )

class WeightsWithOneInputAndOneOutputRequestModel(BaseModel, Generic[TProcessRequest, TInput1, TOutput1]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for RGCCA health assessment.",
    )
    requested_by: str | None = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: MetadataModel | None = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )
    weights_process_request: TProcessRequest = Field(
        alias="weightsProcessRequest",
        description="Optional additional details specific to the weights request.",
    )
    input_blob_1: TInput1 = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for the weights process.",
    )
    output_blob_1: TOutput1 = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the weights process.",
    )
class WeightsWithTwoInputsAndOneOutputRequestModel(BaseModel, Generic[TProcessRequest, TInput1, TInput2, TOutput1]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for RGCCA health assessment.",
    )
    requested_by: str | None = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: MetadataModel | None = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )
    weights_process_request: TProcessRequest = Field(
        alias="weightsProcessRequest",
        description="Optional additional details specific to the weights request.",
    )
    input_blob_1: TInput1 = Field(
        alias="inputBlob1",
        description="Configuration and Azure Blob information required for the input 1 of the weights process.",
    )
    input_blob_2: TInput2 = Field(
        alias="inputBlob2",
        description="Configuration and Azure Blob information required for the input 2 of the weights process.",
    )
    output_blob_1: TOutput1 = Field(
        alias="outputBlob1",
        description="Configuration and Azure Blob information for the output 1 of the weights process.",
    )

class WeightsWithTwoInputsAndTwoOutputsRequestModel(BaseModel, Generic[TProcessRequest, TInput1, TInput2, TOutput1, TOutput2]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for RGCCA health assessment.",
    )
    requested_by: str | None = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: MetadataModel | None = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )
    weights_process_request: TProcessRequest = Field(
        alias="weightsProcessRequest",
        description="Optional additional details specific to the weights request.",
    )
    input_blob_1: TInput1 = Field(
        alias="inputBlob1",
        description="Configuration and Azure Blob information required for the input 1 of the weights process.",
    )
    input_blob_2: TInput2 = Field(
        alias="inputBlob2",
        description="Configuration and Azure Blob information required for the input 2 of the weights process.",
    )
    output_blob_1: TOutput1 = Field(
        alias="outputBlob1",
        description="Configuration and Azure Blob information for the output 1 of the weights process.",
    )
    output_blob_2: TOutput2 = Field(
        alias="outputBlob2",
        description="Configuration and Azure Blob information for the output 2 of the weights process.",
    )

class WeightsWithThreeInputsAndTwoOutputsRequestModel(BaseModel, Generic[TProcessRequest, TInput1, TInput2, TInput3, TOutput1, TOutput2]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for RGCCA health assessment.",
    )
    requested_by: str | None = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: MetadataModel | None = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )
    weights_process_request: TProcessRequest = Field(
        alias="weightsProcessRequest",
        description="Optional additional details specific to the weights request.",
    )
    input_blob_1: TInput1 = Field(
        alias="inputBlob1",
        description="Configuration and Azure Blob information required for the input 1 of the weights process.",
    )
    input_blob_2: TInput2 = Field(
        alias="inputBlob2",
        description="Configuration and Azure Blob information required for the input 2 of the weights process.",
    )
    input_blob_3: TInput3 = Field(
        alias="inputBlob3",
        description="Configuration and Azure Blob information required for the input 3 of the weights process.",
    )
    output_blob_1: TOutput1 = Field(
        alias="outputBlob1",
        description="Configuration and Azure Blob information for the output 1 of the weights process.",
    )
    output_blob_2: TOutput2 = Field(
        alias="outputBlob2",
        description="Configuration and Azure Blob information for the output 2 of the weights process.",
    )

class RGCCAHealthRequestModel(
    WeightsWithNoInputsOrOutputsRequestModel
):
    pass


class ExtractSGCCAWeightsRequestModel(
    WeightsWithTwoInputsAndOneOutputRequestModel[
        ExtractSGCCAWeightsProcessRequestModel,
        InputModel,
        InputModel,
        OutputModel,
    ]
):
    pass

class ApplySGCCAWeightsRequestModel(
    WeightsWithThreeInputsAndTwoOutputsRequestModel[
        ApplySGCCAWeightsProcessRequestModel,
        InputModel,
        InputModel,
        InputModel,
        OutputModel,
        OutputModel,
    ]
):
    pass


class CreateFinalSelectedMatrixWeightsRequestModel(
    WeightsWithTwoInputsAndOneOutputRequestModel[
        CreateFinalSelectedMatrixProcessRequestModel,
        InputModel,
        InputModel,
        OutputModel,
    ]
):
    pass