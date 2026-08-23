from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from uod_rg24_contracts import (
    StepCommandV1,
    StepFailedEventV1,
    StepSucceededEventV1,
)

from .models import (
    ArtifactBundle,
    ArtifactBundleWriteReceipt,
    ArtifactCommitEvidence,
    ArtifactReadRequest,
    MaterializedArtifact,
    PublishReceipt,
    ResolvedInputArtifact,
    ResolvedOutputReservation,
    SafeLogContext,
    TraceContext,
)
from .workspace import StepWorkspace


@runtime_checkable
class ArtifactResolver(Protocol):
    """Future broker port; no HTTP implementation is provided in v0.1.0."""

    def resolve_input(
        self,
        *,
        artifact_id: str,
        owner_id: str,
        dataset_id: str,
    ) -> ResolvedInputArtifact: ...


@runtime_checkable
class ArtifactReservationBroker(Protocol):
    """Future broker port for reservation lifecycle and catalogue commit."""

    def claim_output(
        self,
        *,
        artifact_id: str,
        owner_id: str,
        dataset_id: str,
    ) -> ResolvedOutputReservation: ...

    def commit(self, evidence: ArtifactCommitEvidence) -> None: ...

    def mark_failed(self, *, artifact_id: str, code: str) -> None: ...


@runtime_checkable
class ArtifactControl(ArtifactResolver, ArtifactReservationBroker, Protocol):
    """Broker lifecycle boundary: resolve, claim, commit, or mark failed."""


@runtime_checkable
class ArtifactReader(Protocol):
    def materialize(
        self,
        request: ArtifactReadRequest,
        *,
        workspace: StepWorkspace,
        logical_name: str,
    ) -> MaterializedArtifact: ...


@runtime_checkable
class ArtifactWriter(Protocol):
    def upload_bundle(
        self,
        reservation: ResolvedOutputReservation,
        bundle: ArtifactBundle,
    ) -> ArtifactBundleWriteReceipt: ...


@runtime_checkable
class CommandPublisher(Protocol):
    def publish_command(self, command: StepCommandV1) -> PublishReceipt: ...


@runtime_checkable
class EventPublisher(Protocol):
    def publish_event(
        self,
        event: StepSucceededEventV1 | StepFailedEventV1,
        *,
        trace_context: TraceContext,
    ) -> PublishReceipt: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class StructuredLogger(Protocol):
    def info(self, context: SafeLogContext) -> None: ...

    def warning(self, context: SafeLogContext) -> None: ...

    def error(self, context: SafeLogContext) -> None: ...
