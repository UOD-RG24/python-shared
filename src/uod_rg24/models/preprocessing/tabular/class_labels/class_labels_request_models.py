from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    MetadataModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.assign_class_labels_process_models import (
    AssignClassLabelsProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.class_labels_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.detect_classification_type_process_models import (
    DetectClassificationTypeProcessRequestModel,
)
from uod_rg24.models.preprocessing.tabular.class_labels.extract_subtype_labels_process_models import (
    ExtractSubtypeLabelsProcessRequestModel,
)
from uod_rg24.tools import datetime_tools

TProcessRequest = TypeVar("TProcessRequest")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ClassLabelsRequestModel(BaseModel, Generic[TProcessRequest, TInput, TOutput]):
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
    class_labels_process_request: TProcessRequest = Field(
        alias="classLabelsProcessRequest",
        description="Optional additional details specific to the class labels request.",
    )
    input_blob: TInput = Field(
        alias="inputBlob",
        description="Configuration and Azure Blob information required for class labels assignment.",
    )
    output_blob: TOutput = Field(
        alias="outputBlob",
        description="Configuration and Azure Blob information for the output of the class labels assignment process.",
    )


class AssignClassLabelsRequestModel(
    ClassLabelsRequestModel[
        AssignClassLabelsProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class DetectClassificationTypeRequestModel(
    ClassLabelsRequestModel[
        DetectClassificationTypeProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass


class ExtractSubtypeLabelsRequestModel(
    ClassLabelsRequestModel[
        ExtractSubtypeLabelsProcessRequestModel,
        InputModel,
        OutputModel,
    ]
):
    pass
