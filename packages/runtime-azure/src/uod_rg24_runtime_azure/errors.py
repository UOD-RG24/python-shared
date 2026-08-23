from __future__ import annotations


class RuntimeAzureError(Exception):
    """Base error with a stable domain code and retry classification."""

    code = "RUNTIME_AZURE_ERROR"
    retryable = False

    def __init__(self, message: str, *, artifact_id: str | None = None) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id


class RuntimeConfigurationError(RuntimeAzureError):
    code = "RUNTIME_CONFIGURATION_INVALID"


class SerializationError(RuntimeAzureError):
    code = "OUTPUT_VALIDATION_FAILED"


class ArtifactNotCommittedError(RuntimeAzureError):
    code = "ARTIFACT_NOT_COMMITTED"


class ArtifactSchemaMismatchError(RuntimeAzureError):
    code = "SCHEMA_VALIDATION_FAILED"


class ArtifactHashMismatchError(RuntimeAzureError):
    code = "INPUT_HASH_MISMATCH"


class ArtifactSizeMismatchError(RuntimeAzureError):
    code = "ARTIFACT_SIZE_MISMATCH"


class ArtifactTooLargeError(RuntimeAzureError):
    code = "ARTIFACT_TOO_LARGE"


class ArtifactNotFoundError(RuntimeAzureError):
    code = "ARTIFACT_NOT_FOUND"


class ArtifactAuthorizationError(RuntimeAzureError):
    code = "ARTIFACT_ACCESS_DENIED"


class ArtifactWriteConflictError(RuntimeAzureError):
    code = "OUTPUT_RESERVATION_CONFLICT"


class ArtifactIoError(RuntimeAzureError):
    code = "ARTIFACT_IO_FAILED"
    retryable = True


class MessageTooLargeError(RuntimeAzureError):
    code = "MESSAGE_TOO_LARGE"


class MessagePublishError(RuntimeAzureError):
    code = "MESSAGE_PUBLISH_FAILED"
    retryable = True
