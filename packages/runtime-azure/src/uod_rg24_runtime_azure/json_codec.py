from __future__ import annotations

import json
from typing import cast

from pydantic import BaseModel

from .errors import SerializationError


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    return value


def canonical_json_bytes(value: object, *, trailing_newline: bool = True) -> bytes:
    """Encode a JSON value with stable keys, separators, UTF-8, and no NaN."""

    try:
        encoded = json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SerializationError("Value cannot be encoded as canonical JSON.") from exc
    return encoded + (b"\n" if trailing_newline else b"")


def decode_json_object(data: bytes) -> dict[str, object]:
    try:
        value = cast(object, json.loads(data))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializationError("Payload is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise SerializationError("Payload must contain one JSON object.")
    untyped = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in untyped):
        raise SerializationError("Payload object keys must be strings.")
    return {cast(str, key): item for key, item in untyped.items()}
