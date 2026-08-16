from __future__ import annotations

from datetime import datetime
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


class ProcessStepModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    step: str
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime = Field(alias="completedAt")
    duration_ms: float = Field(
        alias="durationMs",
        ge=0,
    )
    message: str | None = None


class FileProcessInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    azure_storage_account_name: str = Field(alias="azureStorageAccountName")
    azure_container_name: str = Field(alias="azureContainerName")
    directory_name: str = Field(alias="directoryName")
    blob_path: str = Field(alias="blobPath")
    file_name: str = Field(alias="fileName")
    extension: str = Field(alias="extension")
    size_bytes: int = Field(
        alias="sizeBytes",
        ge=0,
    )
    size_mb: float = Field(
        alias="sizeMb",
        ge=0,
    )


class TemporaryFileInfoModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    temporary_directory: str = Field(alias="temporaryDirectory")
    input_file_path: str = Field(alias="inputFilePath")
    output_file_path: str = Field(alias="outputFilePath")
