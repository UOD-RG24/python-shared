from __future__ import annotations
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field
from uod_rg24_tools import datetime_tools
from typing import Literal

T = TypeVar("T")

class ApiRequestMetadataModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: Optional[str] = Field(
        default=None,
        description="Application or service that submitted the request.",
        examples=["web-dashboard"],
    )
    version: str = Field(
        default="1.0",
        description="Request model or API version.",
        examples=["1.0"],
    )
    additional_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional request metadata.",
    )

class ApiRequestModel(BaseModel, Generic[T]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    request_id: UUID = Field(
        default_factory=uuid4,
        alias="requestId",
        description="Unique identifier used to correlate the request.",
    )
    requested_at: datetime = Field(
        default_factory=datetime_tools.utc_now,
        alias="requestedAt",
        description="UTC timestamp when the request was created.",
    )
    operation: str = Field(
        min_length=1,
        description="Name of the operation to execute.",
        examples=["process-dataset"],
    )
    data: T = Field(
        description="Operation-specific request payload.",
    )
    metadata: Optional[ApiRequestMetadataModel] = Field(
        default=None,
        description="Optional information about the request source.",
    )

class ApiErrorModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    code: str = Field(
        min_length=1,
        description="Machine-readable error code.",
        examples=["INVALID_REQUEST"],
    )
    message: str = Field(
        min_length=1,
        description="Human-readable error description.",
    )
    details: Optional[Any] = Field(
        default=None,
        description="Optional structured error details.",
    )

class ApiResponseModel(BaseModel, Generic[T]):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    request_id: UUID = Field(
        alias="requestId",
        description="Identifier of the associated request.",
    )
    trace_id: str = Field(
        alias="traceId",
        min_length=1,
        description="Identifier used for distributed tracing.",
    )
    success: bool = Field(
        description="Indicates whether the request completed successfully.",
    )
    status_code: int = Field(
        alias="statusCode",
        ge=100,
        le=599,
        description="HTTP status code returned by the endpoint.",
    )
    message: str = Field(
        description="Human-readable response message.",
    )
    data: Optional[T] = Field(
        default=None,
        description="Operation-specific response payload.",
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
    error: Optional[ApiErrorModel] = Field(
        default=None,
        description="Error details when success is false.",
    )

class ApiSuccessResponseModel(ApiResponseModel[T], Generic[T]):
    success: bool = True
    error: None = None

class ApiErrorResponseModel(ApiResponseModel[None]):
    success: bool = False
    data: None = None
    error: ApiErrorModel

class ProcessDatasetRequestDataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
    )

class ProcessDatasetRequestModel(
    ApiRequestModel[ProcessDatasetRequestDataModel]
):
    operation: Literal["process-dataset"]