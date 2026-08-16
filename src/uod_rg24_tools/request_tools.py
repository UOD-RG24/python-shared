import uuid
from time import perf_counter
from typing import Any

import azure.functions as func
from uod_rg24_models.shared.api_request_models import (
    ApiErrorModel,
    ApiErrorResponseModel,
    ApiResponseModel,
)


def create_error_response(
    *,
    start_time: float,
    requested_at,
    experiment_id: uuid.UUID,
    trace_id: str,
    status_code: int,
    message: str,
    error_code: str,
    error_message: str,
    error_details: Any = None,
) -> func.HttpResponse:
    response = ApiErrorResponseModel(
        requestId=experiment_id,
        traceId=trace_id,
        statusCode=status_code,
        message=message,
        requestedAt=requested_at,
        timeConsumedMs=elapsed_ms(start_time),
        error=ApiErrorModel(
            code=error_code,
            message=error_message,
            details=error_details,
        ),
    )

    return to_http_response(response)


def to_http_response(
    response: ApiResponseModel[Any],
) -> func.HttpResponse:
    return func.HttpResponse(
        body=response.model_dump_json(
            by_alias=True,
            exclude_none=False,
        ),
        status_code=response.status_code,
        mimetype="application/json",
        headers={
            "X-Request-ID": str(response.request_id),
            "X-Trace-ID": response.trace_id,
        },
    )


def elapsed_ms(start_time: float) -> float:
    return round(
        (perf_counter() - start_time) * 1000,
        3,
    )


def _parse_experiment_id(
    req: func.HttpRequest,
) -> str:
    value = req.get_json().get("experimentId")

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


def _get_trace_id(
    req: func.HttpRequest,
) -> str:
    trace_id = req.headers.get("X-Trace-ID")

    if trace_id:
        return trace_id

    traceparent = req.headers.get("traceparent")

    if traceparent:
        components = traceparent.split("-")

        if len(components) == 4 and len(components[1]) == 32:
            return components[1]

    return uuid.uuid4().hex