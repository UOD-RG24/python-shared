from __future__ import annotations

import json
from typing import Self

import pytest
from azure.servicebus import ServiceBusMessage
from azure.servicebus.exceptions import MessageSizeExceededError, ServiceBusError
from conftest import HASH_B, IMAGE_DIGEST, TRACE_ID, TRACEPARENT
from uod_rg24_contracts import StepCommandV1, StepSucceededEventV1
from uod_rg24_runtime_azure import (
    AzureServiceBusCommandPublisher,
    AzureServiceBusEventPublisher,
    MessagePublishError,
    MessageTooLargeError,
    TraceContext,
)


class FakeSender:
    def __init__(
        self,
        sent: list[ServiceBusMessage],
        send_error: Exception | None,
    ) -> None:
        self.sent = sent
        self.send_error = send_error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def send_messages(self, message: ServiceBusMessage) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)


class FakeServiceBusClient:
    def __init__(self) -> None:
        self.sent: list[ServiceBusMessage] = []
        self.topics: list[str] = []
        self.send_error: Exception | None = None

    def get_topic_sender(self, *, topic_name: str) -> FakeSender:
        self.topics.append(topic_name)
        return FakeSender(self.sent, self.send_error)


def body_bytes(message: ServiceBusMessage) -> bytes:
    body = message.body
    if isinstance(body, bytes):
        return body
    return b"".join(bytes(part) for part in body)


def success_event() -> StepSucceededEventV1:
    return StepSucceededEventV1.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": "evt_01K",
            "eventType": "preprocessing.step.succeeded",
            "occurredAt": "2026-08-23T10:00:00Z",
            "runId": "run_01K",
            "stepRunId": "step_01K",
            "attempt": 1,
            "messageId": "msg_01K",
            "traceId": TRACE_ID,
            "layer": "sampleSelection",
            "operation": "testOperation",
            "codeVersion": "test-worker@0.0.0",
            "containerImageDigest": IMAGE_DIGEST,
            "outputs": [
                {"name": "sampleMap", "artifactId": "art_output", "sha256": HASH_B}
            ],
            "metrics": {
                "durationMs": 10,
                "inputRows": 3,
                "outputRows": 1,
                "bytesRead": 100,
                "bytesWritten": 50,
            },
            "warnings": [],
        }
    )


def test_command_publisher_sets_identity_trace_and_canonical_body(
    step_command: StepCommandV1,
) -> None:
    client = FakeServiceBusClient()
    receipt = AzureServiceBusCommandPublisher(
        client, topic_name="preprocessing-commands"
    ).publish_command(step_command)
    message = client.sent[0]
    assert receipt.message_id == "msg_01K"
    assert client.topics == ["preprocessing-commands"]
    assert message.message_id == "msg_01K"
    assert message.correlation_id == "run_01K"
    assert message.subject == "preprocessing.step.command"
    properties = message.application_properties
    assert properties is not None
    assert properties["traceparent"] == TRACEPARENT
    assert json.loads(body_bytes(message))["operation"] == "testOperation"


def test_event_publisher_requires_and_propagates_matching_traceparent() -> None:
    client = FakeServiceBusClient()
    event = success_event()
    receipt = AzureServiceBusEventPublisher(
        client, topic_name="preprocessing-events"
    ).publish_event(event, trace_context=TraceContext.parse(TRACEPARENT))
    message = client.sent[0]
    assert receipt.message_id == "evt_01K"
    assert message.subject == "preprocessing.step.succeeded"
    properties = message.application_properties
    assert properties is not None
    assert properties["traceparent"] == TRACEPARENT
    assert json.loads(body_bytes(message))["eventId"] == "evt_01K"

    with pytest.raises(ValueError, match="match"):
        AzureServiceBusEventPublisher(
            client, topic_name="preprocessing-events"
        ).publish_event(
            event,
            trace_context=TraceContext.parse(f"00-{'3' * 32}-{'2' * 16}-01"),
        )


def test_message_size_is_bounded_before_service_bus_io(
    step_command: StepCommandV1,
) -> None:
    client = FakeServiceBusClient()
    publisher = AzureServiceBusCommandPublisher(
        client,
        topic_name="preprocessing-commands",
        maximum_message_bytes=10,
    )
    with pytest.raises(MessageTooLargeError):
        publisher.publish_command(step_command)
    assert client.sent == []
    assert client.topics == []


@pytest.mark.parametrize(
    ("sdk_error", "mapped_error"),
    [
        (MessageSizeExceededError(), MessageTooLargeError),
        (ServiceBusError("unavailable"), MessagePublishError),
    ],
)
def test_service_bus_failures_have_stable_runtime_errors(
    step_command: StepCommandV1,
    sdk_error: Exception,
    mapped_error: type[Exception],
) -> None:
    client = FakeServiceBusClient()
    client.send_error = sdk_error
    publisher = AzureServiceBusCommandPublisher(
        client, topic_name="preprocessing-commands"
    )
    with pytest.raises(mapped_error):
        publisher.publish_command(step_command)
    assert client.sent == []
