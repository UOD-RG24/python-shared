from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    application: str | None = Field(
        default=None,
        description="Application or service that submitted the request.",
        examples=["web-dashboard"],
    )
    version: str = Field(
        default="1.0",
        description="Request model or API version.",
        examples=["1.0"],
    )
    additional_data: dict[str, Any] | None = Field(
        default=None,
        description="Additional request metadata.",
    )


class ErrorModel(BaseModel):
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
    details: Any | None = Field(
        default=None,
        description="Optional structured error details.",
    )
