import uuid
from collections.abc import Mapping
from time import perf_counter
from typing import Any, cast

import azure.functions as func


def elapsed_ms(start_time: float) -> float:
    return round(
        (perf_counter() - start_time) * 1000,
        3,
    )


def parse_experiment_id(
    req: func.HttpRequest,
) -> str:
    body = cast(dict[str, Any], req.get_json())
    value: Any = body.get("experimentId")

    if value is None:
        return str(uuid.uuid4())

    try:
        return str(uuid.UUID(str(value)))
    except (
        TypeError,
        ValueError,
        AttributeError,
    ):
        return str(uuid.uuid4())


def get_trace_id(
    req: func.HttpRequest,
) -> str:
    headers = cast(
        Mapping[str, str],
        req.headers,
    )

    trace_id = headers.get("X-Trace-ID")

    if trace_id:
        return trace_id

    traceparent = headers.get("traceparent")

    if traceparent:
        components: list[str] = traceparent.split("-")

        if len(components) == 4 and len(components[1]) == 32:
            return components[1]

    return uuid.uuid4().hex


def to_string_keyed_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}

    unknown_dict = cast(dict[object, object], value)
    typed_dict: dict[str, object] = {}

    for key, item in unknown_dict.items():
        if isinstance(key, str):
            typed_dict[key] = item

    return typed_dict
