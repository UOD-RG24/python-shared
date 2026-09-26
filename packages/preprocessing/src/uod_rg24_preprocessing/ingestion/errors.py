from __future__ import annotations


class IngestionError(ValueError):
    """A stable, safe registration failure.

    Codes are the shared domain codes documented in
    ``docs/plan/api-and-data-contracts.md`` so an adapter can map them to a
    problem response without inspecting the message.
    """

    code = "REGISTRATION_FAILED"


class UnsupportedSourceLayoutError(IngestionError):
    code = "SCHEMA_VALIDATION_FAILED"


class DuplicateFeatureError(IngestionError):
    code = "INCOMPATIBLE_FEATURE_SCHEMA"


class AmbiguousModalityError(IngestionError):
    code = "INCOMPATIBLE_FEATURE_SCHEMA"


class UnknownValueScaleError(IngestionError):
    code = "INVALID_VALUE_DOMAIN"


class SourceIntegrityError(IngestionError):
    code = "INPUT_HASH_MISMATCH"
