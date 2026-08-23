from __future__ import annotations

from uod_rg24_contracts import ProblemDetails

from .errors import (
    ArtifactAuthorizationError,
    ArtifactHashMismatchError,
    ArtifactIoError,
    ArtifactNotCommittedError,
    ArtifactNotFoundError,
    ArtifactSchemaMismatchError,
    ArtifactSizeMismatchError,
    ArtifactTooLargeError,
    ArtifactWriteConflictError,
    MessagePublishError,
    MessageTooLargeError,
    RuntimeAzureError,
    SerializationError,
)
from .models import TraceContext

_ERROR_PRESENTATIONS: tuple[tuple[type[RuntimeAzureError], int, str, str], ...] = (
    (
        ArtifactNotFoundError,
        404,
        "Artifact not found",
        "The artifact data was not found.",
    ),
    (
        ArtifactAuthorizationError,
        403,
        "Artifact access denied",
        "Artifact access was denied.",
    ),
    (
        ArtifactWriteConflictError,
        409,
        "Output conflict",
        "The reserved output conflicts with existing data.",
    ),
    (
        ArtifactTooLargeError,
        413,
        "Artifact too large",
        "The artifact exceeds the configured size limit.",
    ),
    (
        MessageTooLargeError,
        413,
        "Message too large",
        "The message exceeds the configured size limit.",
    ),
    (
        ArtifactNotCommittedError,
        422,
        "Artifact not committed",
        "Only committed artifacts may be read.",
    ),
    (
        ArtifactSchemaMismatchError,
        422,
        "Artifact schema mismatch",
        "The artifact schema is not compatible.",
    ),
    (
        ArtifactHashMismatchError,
        422,
        "Artifact integrity failure",
        "The artifact hash or immutable identity does not match.",
    ),
    (
        ArtifactSizeMismatchError,
        422,
        "Artifact size mismatch",
        "The artifact size does not match its catalogue record.",
    ),
    (
        SerializationError,
        422,
        "Output validation failed",
        "The output could not be serialized safely.",
    ),
    (
        ArtifactIoError,
        503,
        "Artifact dependency unavailable",
        "Artifact storage is temporarily unavailable.",
    ),
    (
        MessagePublishError,
        503,
        "Messaging dependency unavailable",
        "Message publication is temporarily unavailable.",
    ),
)


def runtime_error_to_problem(
    error: RuntimeAzureError,
    *,
    instance: str | None = None,
    trace_context: TraceContext | None = None,
) -> ProblemDetails:
    status, title, detail = (
        500,
        "Runtime failure",
        "The runtime could not complete the operation.",
    )
    for (
        error_type,
        candidate_status,
        candidate_title,
        candidate_detail,
    ) in _ERROR_PRESENTATIONS:
        if isinstance(error, error_type):
            status, title, detail = candidate_status, candidate_title, candidate_detail
            break
    slug = error.code.lower().replace("_", "-")
    return ProblemDetails.model_validate(
        {
            "type": f"https://genome.example/problems/{slug}",
            "title": title,
            "status": status,
            "code": error.code,
            "detail": detail,
            "instance": instance,
            "traceId": None if trace_context is None else trace_context.trace_id,
            "errors": [],
            "extensions": {"retryable": error.retryable},
        }
    )
