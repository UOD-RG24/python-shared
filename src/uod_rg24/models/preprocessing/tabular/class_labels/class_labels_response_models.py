from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    ErrorModel,
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.assign_class_labels_process_models import (
    AssignClassLabelsProcessRequestModel,
    AssignClassLabelsProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.class_labels_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.detect_classification_type_process_models import (
    DetectClassificationTypeProcessRequestModel,
    DetectClassificationTypeProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.extract_subtype_labels_process_models import (
    ExtractSubtypeLabelsProcessRequestModel,
    ExtractSubtypeLabelsProcessResponseModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TProcessResponse = TypeVar("TProcessResponse")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TError = TypeVar("TError")


class ClassLabelsResponseModel(
    BaseModel,
    Generic[TProcessRequest, TProcessResponse, TInput, TOutput, TError],
):
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
    requested_by: str | None = Field(
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
    error: TError = Field(
        alias="error",
        description="Error details when success is false.",
    )
    request_metadata: MetadataModel | None = Field(
        alias="requestMetadata",
        default=None,
        description="Optional additional request metadata.",
    )
    response_metadata: MetadataModel | None = Field(
        alias="responseMetadata",
        default=None,
        description="Optional additional response metadata.",
    )
    class_labels_process_request: TProcessRequest = Field(
        alias="classLabelsProcessRequest",
        description="Optional additional details specific to the class labels request.",
    )
    class_labels_process_response: TProcessResponse = Field(
        alias="classLabelsProcessResponse",
        description="Optional detailed result of the class labels assignment operation.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for class labels assignment.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the class labels assignment process.",
    )


class ClassLabelsSuccessResponseModel(
    ClassLabelsResponseModel[
        TProcessRequest,
        TProcessResponse,
        TInput,
        TOutput,
        None,
    ],
    Generic[TProcessRequest, TProcessResponse, TInput, TOutput],
):
    success: bool = True
    class_labels_process_request: TProcessRequest
    class_labels_process_response: TProcessResponse
    input_blob: TInput
    output_blob: TOutput
    error: None = None


class ClassLabelsErrorResponseModel(
    ClassLabelsResponseModel[
        None,
        None,
        None,
        None,
        ErrorModel,
    ],
):
    success: bool = False
    class_labels_process_request: None = None
    class_labels_process_response: None = None
    input_blob: None = None
    output_blob: None = None


class AssignClassLabelsResponseModel(
    ClassLabelsSuccessResponseModel[
        AssignClassLabelsProcessRequestModel,
        AssignClassLabelsProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class DetectClassificationTypeResponseModel(
    ClassLabelsSuccessResponseModel[
        DetectClassificationTypeProcessRequestModel,
        DetectClassificationTypeProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class ExtractSubtypeLabelsResponseModel(
    ClassLabelsSuccessResponseModel[
        ExtractSubtypeLabelsProcessRequestModel,
        ExtractSubtypeLabelsProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass
