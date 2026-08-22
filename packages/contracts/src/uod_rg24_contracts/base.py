from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StrictStr,
    StringConstraints,
    WithJsonSchema,
)
from pydantic.alias_generators import to_camel

_RFC3339_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|\+00:00)$"
)


def _require_rfc3339_utc_string(value: object) -> object:
    """Constrain the wire form without blocking native ``datetime`` values.

    JSON payloads must present an RFC 3339 UTC string. A ``datetime`` is passed
    through so a model can be built from a computed timestamp and can
    re-validate its own ``model_dump()`` output; ``_require_utc`` still rejects
    naive or non-UTC values.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or _RFC3339_UTC_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp must be an RFC 3339 UTC string")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must include a UTC offset")
    return value


def _require_nonzero_trace_id(value: str) -> str:
    if value == "0" * 32:
        raise ValueError("trace ID cannot be all zeroes")
    return value


def _require_valid_traceparent_ids(value: str) -> str:
    _, trace_id, parent_id, _ = value.split("-")
    if trace_id == "0" * 32:
        raise ValueError("traceparent trace ID cannot be all zeroes")
    if parent_id == "0" * 16:
        raise ValueError("traceparent parent ID cannot be all zeroes")
    return value


class WireModel(BaseModel):
    """Base for strict JSON contracts with camel-case wire aliases."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        allow_inf_nan=False,
        extra="forbid",
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=False,
        validate_default=True,
    )


NonEmptyStr = Annotated[
    StrictStr,
    StringConstraints(min_length=1, pattern=r".*\S.*"),
]
Sha256 = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
TraceId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9a-f]{32}$"),
    AfterValidator(_require_nonzero_trace_id),
    WithJsonSchema({"type": "string", "pattern": r"^(?!0{32}$)[0-9a-f]{32}$"}),
]
TraceParent = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"),
    AfterValidator(_require_valid_traceparent_ids),
    WithJsonSchema(
        {
            "type": "string",
            "pattern": (
                r"^00-(?!0{32}-)[0-9a-f]{32}-"
                r"(?!0{16}-)[0-9a-f]{16}-[0-9a-f]{2}$"
            ),
        }
    ),
]
ImageDigest = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
UtcDatetime = Annotated[
    datetime,
    BeforeValidator(_require_rfc3339_utc_string),
    AfterValidator(_require_utc),
    WithJsonSchema(
        {
            "type": "string",
            "format": "date-time",
            "pattern": _RFC3339_UTC_PATTERN.pattern,
        }
    ),
]
