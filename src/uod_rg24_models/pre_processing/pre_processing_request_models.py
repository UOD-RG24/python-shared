from __future__ import annotations
from datetime import datetime
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from uod_rg24_tools import datetime_tools
from uod_rg24_models.pre_processing.pre_processing_shared_models import (
    MetadataModel,
)
from uod_rg24_models.pre_processing.tabular_data.standardization_models import (
    DatasetModel,
)

TInput = TypeVar("TInput")


class StandardizationRequestModel(BaseModel, Generic[TInput]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    experiment_id: str = Field(
        alias="experimentId",
        description="Unique identifier used to correlate the experiment.",
    )
    dataset_id: str = Field(
        alias="datasetId",
        description="Unique identifier used to correlate the dataset.",
    )
    requested_by: Optional[str] = Field(
        default=None,
        alias="requestedBy",
        description="Optional identifier of the user or system that initiated the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    request_metadata: Optional[MetadataModel] = Field(
        default=None,
        alias="requestMetadata",
        description="Optional information about the request source.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for preprocessing.",
    )


class TabularDataPreprocessingUsingStandardScalerStandardizationRequestModel(
    StandardizationRequestModel[DatasetModel]
):
    pass


class TabularDataPreprocessingUsingMinMaxScalerStandardizationRequestModel(
    StandardizationRequestModel[DatasetModel]
):
    pass


class TabularDataPreprocessingUsingMaxAbsScalerStandardizationRequestModel(
    StandardizationRequestModel[DatasetModel]
):
    pass
