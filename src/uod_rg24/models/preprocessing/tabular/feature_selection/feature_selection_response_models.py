from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    ErrorModel,
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.annotate_process_models import (
    AnnotateProcessRequestModel,
    AnnotateProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.feature_selection_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.remove_low_variance_process_models import (
    RemoveLowVarianceProcessRequestModel,
    RemoveLowVarianceProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.remove_no_signal_process_models import (
    RemoveNoSignalProcessRequestModel,
    RemoveNoSignalProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.select_correlation_process_models import (
    SelectCorrelationProcessRequestModel,
    SelectCorrelationProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.select_top_features_process_models import (
    SelectTopFeaturesProcessRequestModel,
    SelectTopFeaturesProcessResponseModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TProcessResponse = TypeVar("TProcessResponse")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TError = TypeVar("TError")


class FeatureSelectionResponseModel(
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
    feature_selection_process_request: TProcessRequest = Field(
        alias="featureSelectionProcessRequest",
        description="Optional additional details specific to the feature selection request.",
    )
    feature_selection_process_response: TProcessResponse = Field(
        alias="featureSelectionProcessResponse",
        description="Optional detailed result of the feature selection operation.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for feature selection.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the feature selection process.",
    )


class FeatureSelectionSuccessResponseModel(
    FeatureSelectionResponseModel[
        TProcessRequest,
        TProcessResponse,
        TInput,
        TOutput,
        None,
    ],
    Generic[TProcessRequest, TProcessResponse, TInput, TOutput],
):
    success: bool = True
    feature_selection_process_request: TProcessRequest
    feature_selection_process_response: TProcessResponse
    input_blob: TInput
    output_blob: TOutput
    error: None = None


class FeatureSelectionErrorResponseModel(
    FeatureSelectionResponseModel[
        None,
        None,
        None,
        None,
        ErrorModel,
    ],
):
    success: bool = False
    feature_selection_process_request: None = None
    feature_selection_process_response: None = None
    input_blob: None = None
    output_blob: None = None


class AnnotateResponseModel(
    FeatureSelectionSuccessResponseModel[
        AnnotateProcessRequestModel,
        AnnotateProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class SelectTopFeaturesResponseModel(
    FeatureSelectionSuccessResponseModel[
        SelectTopFeaturesProcessRequestModel,
        SelectTopFeaturesProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class RemoveLowVarianceResponseModel(
    FeatureSelectionSuccessResponseModel[
        RemoveLowVarianceProcessRequestModel,
        RemoveLowVarianceProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class RemoveNoSignalResponseModel(
    FeatureSelectionSuccessResponseModel[
        RemoveNoSignalProcessRequestModel,
        RemoveNoSignalProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class SelectCorrelationResponseModel(
    FeatureSelectionSuccessResponseModel[
        SelectCorrelationProcessRequestModel,
        SelectCorrelationProcessResponseModel,
        InputModel,
        OutputModel,
    ]
):
    pass
