from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.matrix_extraction.data_extraction_process_models import (
    DataExtractionProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.matrix_extraction.matrix_extraction_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class MatrixExtractionRequestModel(
    BaseModel, Generic[TProcessRequest, TInput, TOutput]
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
        description="Unique identifier for the dataset to be processed for class labels assignment.",
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
    matrix_extraction_process_request: TProcessRequest = Field(
        alias="matrixExtractionProcessRequest",
        description="Optional additional details specific to the matrix extraction request.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for the matrix extraction process.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the matrix extraction process.",
    )


class DataExtractionRequestModel(
    MatrixExtractionRequestModel[
        DataExtractionProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass
