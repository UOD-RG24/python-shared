from __future__ import annotations

from typing import Protocol, Self
from uuid import UUID

from azure.servicebus import ServiceBusMessage
from azure.servicebus.exceptions import MessageSizeExceededError, ServiceBusError
from uod_rg24_contracts import (
    StepCommandV1,
    StepFailedEventV1,
    StepSucceededEventV1,
)

from .errors import MessagePublishError, MessageTooLargeError
from .json_codec import canonical_json_bytes
from .models import PublishReceipt, TraceContext

DEFAULT_MAXIMUM_MESSAGE_BYTES = 192 * 1024
type ServiceBusApplicationProperties = dict[
    str | bytes,
    int | float | bytes | bool | str | UUID,
]


class _Sender(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def send_messages(self, message: ServiceBusMessage) -> None: ...


class ServiceBusClientPort(Protocol):
    def get_topic_sender(self, *, topic_name: str) -> _Sender: ...


class _AzureServiceBusJsonPublisher:
    def __init__(
        self,
        client: ServiceBusClientPort,
        *,
        topic_name: str,
        maximum_message_bytes: int = DEFAULT_MAXIMUM_MESSAGE_BYTES,
    ) -> None:
        if not topic_name.strip():
            raise ValueError("topic_name must not be empty")
        if maximum_message_bytes <= 0:
            raise ValueError("maximum_message_bytes must be positive")
        self._client = client
        self._topic_name = topic_name
        self._maximum_message_bytes = maximum_message_bytes

    def _publish(
        self,
        *,
        body_model: object,
        message_id: str,
        correlation_id: str,
        subject: str,
        properties: ServiceBusApplicationProperties,
    ) -> PublishReceipt:
        body = canonical_json_bytes(body_model)
        if len(body) > self._maximum_message_bytes:
            raise MessageTooLargeError("Service Bus message exceeds the runtime limit.")
        message = ServiceBusMessage(
            body,
            application_properties=properties,
            content_type="application/json",
            correlation_id=correlation_id,
            message_id=message_id,
            subject=subject,
        )
        try:
            with self._client.get_topic_sender(topic_name=self._topic_name) as sender:
                sender.send_messages(message)
        except MessageSizeExceededError as exc:
            raise MessageTooLargeError(
                "Service Bus rejected an oversized message."
            ) from exc
        except ServiceBusError as exc:
            raise MessagePublishError(
                "Service Bus message publication failed."
            ) from exc
        return PublishReceipt(
            message_id=message_id,
            destination=self._topic_name,
            byte_length=len(body),
        )


class AzureServiceBusCommandPublisher(_AzureServiceBusJsonPublisher):
    def publish_command(self, command: StepCommandV1) -> PublishReceipt:
        return self._publish(
            body_model=command,
            message_id=command.message_id,
            correlation_id=command.run_id,
            subject="preprocessing.step.command",
            properties={
                "schemaVersion": command.schema_version,
                "layer": command.layer.value,
                "operation": command.operation,
                "attempt": command.attempt,
                "traceparent": command.execution_context.traceparent,
            },
        )


class AzureServiceBusEventPublisher(_AzureServiceBusJsonPublisher):
    def publish_event(
        self,
        event: StepSucceededEventV1 | StepFailedEventV1,
        *,
        trace_context: TraceContext,
    ) -> PublishReceipt:
        if trace_context.trace_id != event.trace_id:
            raise ValueError("traceparent trace ID must match the event trace ID")
        return self._publish(
            body_model=event,
            message_id=event.event_id,
            correlation_id=event.run_id,
            subject=event.event_type,
            properties={
                "schemaVersion": event.schema_version,
                "layer": event.layer.value,
                "operation": event.operation,
                "attempt": event.attempt,
                "traceId": event.trace_id,
                "traceparent": trace_context.traceparent,
            },
        )
