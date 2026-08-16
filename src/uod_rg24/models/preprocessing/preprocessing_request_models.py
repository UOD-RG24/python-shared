from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.min_max_scaler_process_models import (
    MinMaxScalerStandardizationProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.standard_scaler_process_models import (
    StandardScalerStandardizationProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.standardization_models import (
    DatasetModel,
    MaxAbsScalerModel,
    MinMaxScalerModel,
    StandardScalerModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class StandardizationRequestModel(BaseModel, Generic[TProcessRequest, TInput, TOutput]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be standardized.",
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
    standardization_process_request: TProcessRequest = Field(
        alias="standardizationProcessRequest",
        description="Optional additional details specific to the standardization request.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for preprocessing.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the standardized output.",
    )


class TabularDataPreprocessingUsingStandardScalerStandardizationRequestModel(
    StandardizationRequestModel[
        StandardScalerStandardizationProcessRequestModel,
        DatasetModel,
        StandardScalerModel,
    ]
):
    pass


class TabularDataPreprocessingUsingMinMaxScalerStandardizationRequestModel(
    StandardizationRequestModel[
        MinMaxScalerStandardizationProcessRequestModel, DatasetModel, MinMaxScalerModel
    ]
):
    pass


class TabularDataPreprocessingUsingMaxAbsScalerStandardizationRequestModel(
    StandardizationRequestModel[None, DatasetModel, MaxAbsScalerModel]
):
    pass
