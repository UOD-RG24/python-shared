from __future__ import annotations

from typing import Any

import pytest
from azure.core.credentials import AccessToken, TokenCredential
from azure.identity import DefaultAzureCredential
from uod_rg24_runtime_azure import (
    ManagedIdentitySettings,
    azure_clients,
    create_blob_service_client,
    create_default_credential,
    create_service_bus_client,
)


class StaticTokenCredential:
    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        del scopes, kwargs
        return AccessToken("test-token", 4_102_444_800)


def settings() -> ManagedIdentitySettings:
    return ManagedIdentitySettings(
        storage_account_url="https://genomeartifacts.blob.core.windows.net",
        service_bus_fully_qualified_namespace=(
            "genome-preprocessing.servicebus.windows.net"
        ),
        managed_identity_client_id="managed-client-id",
    )


def test_default_credential_is_the_managed_identity_compatible_chain() -> None:
    credential = create_default_credential(
        managed_identity_client_id="managed-client-id"
    )
    try:
        assert isinstance(credential, DefaultAzureCredential)
    finally:
        credential.close()


def test_client_factories_require_token_credentials_and_never_connection_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    def blob_factory(**kwargs: object) -> object:
        calls["blob"] = kwargs
        return object()

    def service_bus_factory(**kwargs: object) -> object:
        calls["service_bus"] = kwargs
        return object()

    monkeypatch.setattr(azure_clients, "BlobServiceClient", blob_factory)
    monkeypatch.setattr(azure_clients, "ServiceBusClient", service_bus_factory)
    credential = StaticTokenCredential()
    assert isinstance(credential, TokenCredential)

    create_blob_service_client(settings(), credential)
    create_service_bus_client(settings(), credential)

    assert calls == {
        "blob": {
            "account_url": "https://genomeartifacts.blob.core.windows.net",
            "credential": credential,
        },
        "service_bus": {
            "fully_qualified_namespace": (
                "genome-preprocessing.servicebus.windows.net"
            ),
            "credential": credential,
        },
    }
    assert all("connection_string" not in call for call in calls.values())
