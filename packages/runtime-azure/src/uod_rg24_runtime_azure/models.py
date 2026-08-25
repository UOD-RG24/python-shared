from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from uod_rg24_contracts import ArtifactState

from .errors import RuntimeConfigurationError
from .hashing import sha256_bytes

PARQUET_MEDIA_TYPE: Final[str] = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE: Final[str] = "application/json"
ARTIFACT_BUNDLE_MEMBERS: Final[tuple[str, str, str]] = (
    "data.parquet",
    "manifest.json",
    "qc.json",
)
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_SAFE_LOG_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise RuntimeConfigurationError(f"{name} must not be empty")


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeConfigurationError(
            "sha256 must contain 64 lowercase hex characters"
        )


@dataclass(frozen=True, slots=True)
class Sha256Digest:
    value: str

    def __post_init__(self) -> None:
        _require_sha256(self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BlobETag:
    value: str

    def __post_init__(self) -> None:
        _require_text("etag", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class TraceContext:
    traceparent: str
    trace_id: str
    parent_id: str
    flags: str

    def __post_init__(self) -> None:
        match = _TRACEPARENT.fullmatch(self.traceparent)
        if match is None:
            raise RuntimeConfigurationError(
                "traceparent must be a W3C version-00 trace context"
            )
        trace_id, parent_id, flags = match.groups()
        if trace_id == "0" * 32 or parent_id == "0" * 16:
            raise RuntimeConfigurationError(
                "traceparent identifiers cannot be all zeroes"
            )
        if (self.trace_id, self.parent_id, self.flags) != (
            trace_id,
            parent_id,
            flags,
        ):
            raise RuntimeConfigurationError(
                "trace context fields must match traceparent"
            )

    @classmethod
    def parse(cls, traceparent: str) -> TraceContext:
        match = _TRACEPARENT.fullmatch(traceparent)
        if match is None:
            raise RuntimeConfigurationError(
                "traceparent must be a W3C version-00 trace context"
            )
        trace_id, parent_id, flags = match.groups()
        return cls(
            traceparent=traceparent,
            trace_id=trace_id,
            parent_id=parent_id,
            flags=flags,
        )


@dataclass(frozen=True, slots=True)
class SafeLogContext:
    """Allowlisted structured dimensions; secrets and physical paths have no field."""

    event: str
    run_id: str | None = None
    step_run_id: str | None = None
    attempt: int | None = None
    dataset_id: str | None = None
    artifact_id: str | None = None
    layer: str | None = None
    operation: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "event",
            "run_id",
            "step_run_id",
            "dataset_id",
            "artifact_id",
            "layer",
            "operation",
        ):
            value = getattr(self, name)
            if value is not None and _SAFE_LOG_IDENTIFIER.fullmatch(value) is None:
                raise RuntimeConfigurationError(f"{name} is not a safe log identifier")
        if self.attempt is not None and self.attempt < 1:
            raise RuntimeConfigurationError("attempt must be positive")

    def as_dict(self) -> dict[str, str | int]:
        values = {
            "event": self.event,
            "runId": self.run_id,
            "stepRunId": self.step_run_id,
            "attempt": self.attempt,
            "datasetId": self.dataset_id,
            "artifactId": self.artifact_id,
            "layer": self.layer,
            "operation": self.operation,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ResolvedInputArtifact:
    """Broker-resolved internal input; never deserialize this from a step command."""

    artifact_id: str
    dataset_id: str
    kind: str
    schema_id: str
    media_type: str
    state: ArtifactState
    sha256: str
    byte_length: int
    container_name: str
    blob_name: str
    etag: str
    version_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "dataset_id",
            "kind",
            "schema_id",
            "media_type",
            "container_name",
            "blob_name",
            "etag",
        ):
            _require_text(name, getattr(self, name))
        _require_sha256(self.sha256)
        if self.byte_length < 0:
            raise RuntimeConfigurationError("byte_length must not be negative")


@dataclass(frozen=True, slots=True)
class ResolvedOutputReservation:
    """Broker-reserved internal output location awaiting catalogue commit."""

    artifact_id: str
    output_name: str
    dataset_id: str
    kind: str
    schema_id: str
    media_type: str
    container_name: str
    blob_prefix: str

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "output_name",
            "dataset_id",
            "kind",
            "schema_id",
            "media_type",
            "container_name",
            "blob_prefix",
        ):
            _require_text(name, getattr(self, name))
        segments = self.blob_prefix.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise RuntimeConfigurationError("blob_prefix is not a canonical prefix")
        if segments[-1] in ARTIFACT_BUNDLE_MEMBERS:
            raise RuntimeConfigurationError("blob_prefix must not name a bundle member")


@dataclass(frozen=True, slots=True)
class ArtifactReadRequest:
    artifact: ResolvedInputArtifact
    accepted_schema_ids: tuple[str, ...]
    expected_schema_id: str
    expected_sha256: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        if not self.accepted_schema_ids:
            raise RuntimeConfigurationError("accepted_schema_ids must not be empty")
        if any(not schema_id.strip() for schema_id in self.accepted_schema_ids):
            raise RuntimeConfigurationError(
                "accepted_schema_ids cannot contain empty values"
            )
        if len(set(self.accepted_schema_ids)) != len(self.accepted_schema_ids):
            raise RuntimeConfigurationError("accepted_schema_ids must be unique")
        _require_text("expected_schema_id", self.expected_schema_id)
        _require_sha256(self.expected_sha256)
        if self.maximum_bytes <= 0:
            raise RuntimeConfigurationError("maximum_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    output_name: str
    kind: str
    schema_id: str
    media_type: str
    data: bytes
    sha256: str
    byte_length: int
    row_count: int | None = None
    sample_order_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("output_name", "kind", "schema_id", "media_type"):
            _require_text(name, getattr(self, name))
        _require_sha256(self.sha256)
        if self.sha256 != sha256_bytes(self.data):
            raise RuntimeConfigurationError("payload sha256 does not match its bytes")
        if self.byte_length != len(self.data):
            raise RuntimeConfigurationError(
                "payload byte_length does not match its bytes"
            )
        if self.row_count is not None and self.row_count < 0:
            raise RuntimeConfigurationError("row_count must not be negative")
        if self.sample_order_sha256 is not None:
            _require_sha256(self.sample_order_sha256)

    @classmethod
    def from_bytes(
        cls,
        *,
        output_name: str,
        kind: str,
        schema_id: str,
        media_type: str,
        data: bytes,
        row_count: int | None = None,
        sample_order_sha256: str | None = None,
    ) -> ArtifactPayload:
        return cls(
            output_name=output_name,
            kind=kind,
            schema_id=schema_id,
            media_type=media_type,
            data=data,
            sha256=sha256_bytes(data),
            byte_length=len(data),
            row_count=row_count,
            sample_order_sha256=sample_order_sha256,
        )


@dataclass(frozen=True, slots=True)
class ArtifactReadReceipt:
    artifact_id: str
    schema_id: str
    sha256: Sha256Digest
    byte_length: int
    etag: BlobETag
    version_id: str | None


@dataclass(frozen=True, slots=True)
class ArtifactReadResult:
    data: bytes
    receipt: ArtifactReadReceipt


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    path: Path
    receipt: ArtifactReadReceipt

    def __post_init__(self) -> None:
        if not self.path.is_file():
            raise RuntimeConfigurationError("materialized artifact path is not a file")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    member_name: str
    path: Path
    output_name: str
    kind: str
    schema_id: str
    media_type: str
    sha256: str
    byte_length: int
    sample_order_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.member_name not in ARTIFACT_BUNDLE_MEMBERS:
            raise RuntimeConfigurationError(
                "artifact bundle member name is not allowed"
            )
        if not self.path.is_file():
            raise RuntimeConfigurationError("artifact bundle member is not a file")
        for name in ("output_name", "kind", "schema_id", "media_type"):
            _require_text(name, getattr(self, name))
        _require_sha256(self.sha256)
        if self.byte_length != self.path.stat().st_size:
            raise RuntimeConfigurationError("artifact file byte_length is incorrect")
        if self.sample_order_sha256 is not None:
            _require_sha256(self.sample_order_sha256)


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    data: ArtifactFile
    manifest: ArtifactFile
    qc: ArtifactFile

    def __post_init__(self) -> None:
        actual = (self.data.member_name, self.manifest.member_name, self.qc.member_name)
        if actual != ARTIFACT_BUNDLE_MEMBERS:
            raise RuntimeConfigurationError(
                "artifact bundle must contain data.parquet, manifest.json, and qc.json"
            )


@dataclass(frozen=True, slots=True)
class StagedBlobReceipt:
    artifact_id: str
    schema_id: str
    sha256: Sha256Digest
    byte_length: int
    etag: BlobETag
    version_id: str | None
    already_existed: bool


@dataclass(frozen=True, slots=True)
class StagedArtifactBundleReceipt:
    artifact_id: str
    data: StagedBlobReceipt
    manifest: StagedBlobReceipt
    qc: StagedBlobReceipt

    def __post_init__(self) -> None:
        _require_text("artifact_id", self.artifact_id)
        if any(
            receipt.artifact_id != self.artifact_id
            for receipt in (self.data, self.manifest, self.qc)
        ):
            raise RuntimeConfigurationError(
                "artifact bundle receipts must share one artifact_id"
            )


# Compatibility aliases use the more explicit staged receipt types.
ArtifactWriteReceipt = StagedBlobReceipt
ArtifactBundleWriteReceipt = StagedArtifactBundleReceipt


@dataclass(frozen=True, slots=True)
class ArtifactCommitEvidence:
    bundle_receipt: ArtifactBundleWriteReceipt
    media_type: str

    def __post_init__(self) -> None:
        _require_text("media_type", self.media_type)


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    message_id: str
    destination: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class ManagedIdentitySettings:
    storage_account_url: str
    service_bus_fully_qualified_namespace: str
    managed_identity_client_id: str | None = None

    def __post_init__(self) -> None:
        if not self.storage_account_url.startswith("https://"):
            raise RuntimeConfigurationError("storage_account_url must use HTTPS")
        if not self.service_bus_fully_qualified_namespace.endswith(
            ".servicebus.windows.net"
        ):
            raise RuntimeConfigurationError(
                "service_bus_fully_qualified_namespace is invalid"
            )
