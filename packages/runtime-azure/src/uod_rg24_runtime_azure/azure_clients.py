from __future__ import annotations

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.servicebus import ServiceBusClient
from azure.storage.blob import BlobServiceClient

from .models import ManagedIdentitySettings


def create_default_credential(
    *, managed_identity_client_id: str | None = None
) -> DefaultAzureCredential:
    """Create the shared credential chain; production resolves managed identity."""

    return DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id)


def create_blob_service_client(
    settings: ManagedIdentitySettings,
    credential: TokenCredential,
) -> BlobServiceClient:
    return BlobServiceClient(
        account_url=settings.storage_account_url,
        credential=credential,
    )


def create_service_bus_client(
    settings: ManagedIdentitySettings,
    credential: TokenCredential,
) -> ServiceBusClient:
    return ServiceBusClient(
        fully_qualified_namespace=settings.service_bus_fully_qualified_namespace,
        credential=credential,
    )
