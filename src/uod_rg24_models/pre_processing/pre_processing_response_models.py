from __future__ import annotations
from datetime import datetime
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from uod_rg24_tools import datetime_tools
from uod_rg24_models.pre_processing.pre_processing_shared_models import (
    ErrorModel,
    MetadataModel,
)
from uod_rg24_models.pre_processing.tabular_data.standardization_models import (
    DatasetModel,
    MaxAbsScalerModel,
    MinMaxScalerModel,
    StandardScalerModel,
)

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class StandardizationResponseModel(BaseModel, Generic[TInput, TOutput]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    experiment_id: str = Field(
        alias="experimentId",
        description="Identifier of the associated experiment.",
    )
    dataset_id: str = Field(
        alias="datasetId",
        description="Identifier of the associated dataset.",
    )
    trace_id: str = Field(
        alias="traceId",
        min_length=1,
        description="Identifier used for distributed tracing.",
    )
    success: bool = Field(
        alias="success",
        description="Indicates whether the request completed successfully.",
    )
    status_code: int = Field(
        alias="statusCode",
        ge=100,
        le=599,
        description="HTTP status code returned by the endpoint.",
    )
    message: str = Field(
        alias="message",
        description="Human-readable response message.",
    )
    requested_by: Optional[str] = Field(
        alias="requestedBy",
        default=None,
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        alias="requestedAt",
        description="UTC timestamp when request processing started.",
    )
    completed_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="completedAt",
        description="UTC timestamp when request processing completed.",
    )
    time_consumed_ms: float = Field(
        alias="timeConsumedMs",
        ge=0,
        description="Total request-processing duration in milliseconds.",
    )
    error: Optional[ErrorModel] = Field(
        alias="error",
        default=None,
        description="Error details when success is false.",
    )
    request_metadata: Optional[MetadataModel] = Field(
        alias="requestMetadata",
        default=None,
        description="Optional additional request metadata.",
    )
    reponse_metadata: Optional[MetadataModel] = Field(
        alias="responseMetadata",
        default=None,
        description="Optional additional response metadata.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for preprocessing.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information required for preprocessing.",
    )


class StandardizationSuccessResponseModel(
    StandardizationResponseModel[TInput, TOutput],
    Generic[TInput, TOutput],
):
    success: bool = True
    input_blob: TInput
    output_blob: TOutput
    error: None = None


class StandardizationErrorResponseModel(
    StandardizationResponseModel[None, None],
):
    success: bool = False
    input_blob: None = None
    output_blob: None = None
    error: ErrorModel


class TabularDataPreprocessingUsingStandardScalerStandardizationResponseModel(
    StandardizationSuccessResponseModel[
        DatasetModel,
        StandardScalerModel,
    ]
):
    pass


class TabularDataPreprocessingUsingMinMaxScalerStandardizationResponseModel(
    StandardizationSuccessResponseModel[
        DatasetModel,
        MinMaxScalerModel,
    ]
):
    pass


class TabularDataPreprocessingUsingMaxAbsScalerStandardizationResponseModel(
    StandardizationSuccessResponseModel[
        DatasetModel,
        MaxAbsScalerModel,
    ]
):
    pass
