from datetime import datetime, timezone
from time import perf_counter
from typing import Any
import uuid
import azure.functions as func
from datetime import datetime, timezone as _timezone
from uod_rg24_models.shared.api_request_models import (
    ApiResponseModel,
    ApiErrorModel,
    ApiErrorResponseModel,
)
from uod_rg24_tools.datetime_tools import utc_now


def create_error_response(
    *,
    start_time: float,
    requested_at,
    request_id: uuid.UUID,
    trace_id: str,
    status_code: int,
    message: str,
    error_code: str,
    error_message: str,
    error_details: Any = None,
) -> func.HttpResponse:
    response = ApiErrorResponseModel(
        request_id=request_id,
        trace_id=trace_id,
        status_code=status_code,
        message=message,
        requested_at=requested_at,
        completed_at=utc_now(),
        time_consumed_ms=elapsed_ms(start_time),
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


def parse_request_id(value: Any) -> uuid.UUID:
    if value is None:
        return uuid.uuid4()
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return uuid.uuid4()


def get_trace_id(req: func.HttpRequest) -> str:
    trace_id = req.headers.get("X-Trace-ID")
    if trace_id:
        return trace_id
    traceparent = req.headers.get("traceparent")
    if traceparent:
        components = traceparent.split("-")
        if len(components) == 4 and len(components[1]) == 32:
            return components[1]

    return uuid.uuid4().hex
