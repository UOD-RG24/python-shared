from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.annotate_process_models import (
    AnnotateProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.feature_selection_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.remove_low_variance_process_models import (
    RemoveLowVarianceProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.remove_no_signal_process_models import (
    RemoveNoSignalProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.select_correlation_process_models import (
    SelectCorrelationProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.feature_selection.select_top_features_process_models import (
    SelectTopFeaturesProcessRequestModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class FeatureSelectionRequestModel(
    BaseModel, Generic[TProcessRequest, TInput, TOutput]
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for feature selection.",
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
    feature_selection_process_request: TProcessRequest = Field(
        alias="featureSelectionProcessRequest",
        description="Optional additional details specific to the feature selection request.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for feature selection.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the feature selection process.",
    )


class AnnotateRequestModel(
    FeatureSelectionRequestModel[
        AnnotateProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class SelectTopFeaturesRequestModel(
    FeatureSelectionRequestModel[
        SelectTopFeaturesProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class RemoveLowVarianceRequestModel(
    FeatureSelectionRequestModel[
        RemoveLowVarianceProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class RemoveNoSignalRequestModel(
    FeatureSelectionRequestModel[
        RemoveNoSignalProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class SelectCorrelationRequestModel(
    FeatureSelectionRequestModel[
        SelectCorrelationProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass
