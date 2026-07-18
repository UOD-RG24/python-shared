from dataclasses import dataclass, field
from typing import BinaryIO
from azure.core.exceptions import ResourceExistsError
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import (
    BlobClient,
    BlobServiceClient,
    ContainerClient,
)


@dataclass(slots=True)
class StorageAccountModel:
    storage_account_name: str

    storage_account_url: str = field(init=False)
    default_azure_credential: DefaultAzureCredential = field(
        init=False,
        repr=False,
    )
    blob_service_client: BlobServiceClient = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.storage_account_name = self.storage_account_name.strip()

        if not self.storage_account_name:
            raise ValueError("storage_account_name cannot be empty.")
        self.storage_account_url = (
            f"https://{self.storage_account_name}" ".blob.core.windows.net"
        )
        self.default_azure_credential = DefaultAzureCredential()

        self.blob_service_client = BlobServiceClient(
            account_url=self.storage_account_url,
            credential=self.default_azure_credential,
        )

    async def create_blob_container(
        self,
        container_name: str,
    ) -> bool:
        container_name = container_name.strip()
        if not container_name:
            raise ValueError("container_name cannot be empty.")
        container_client: ContainerClient = (
            self.blob_service_client.get_container_client(
                container=container_name,
            )
        )
        try:
            await container_client.create_container()
            return True
        except ResourceExistsError:
            return False

    async def upload_blob(
        self,
        container_name: str,
        blob_name: str,
        data: bytes | str | BinaryIO,
        *,
        overwrite: bool = True,
        content_type: str = "application/octet-stream",
    ) -> BlobClient:
        container_name = container_name.strip()
        blob_name = blob_name.strip()
        if not container_name:
            raise ValueError("container_name cannot be empty.")
        if not blob_name:
            raise ValueError("blob_name cannot be empty.")
        container_created = await self.create_blob_container(
            container_name=container_name,
        )
        if container_created:
            print(f"Container '{container_name}' created.")
        else:
            print(f"Container '{container_name}' already exists.")
        blob_client: BlobClient = self.blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name,
        )
        await blob_client.upload_blob(
            data=data,
            overwrite=overwrite,
            content_settings=ContentSettings(
                content_type=content_type,
            ),
        )
        return blob_client

    async def download_blob(
        self,
        container_name: str,
        blob_name: str,
    ) -> bytes:
        container_name = container_name.strip()
        blob_name = blob_name.strip()
        if not container_name:
            raise ValueError("container_name cannot be empty.")
        if not blob_name:
            raise ValueError("blob_name cannot be empty.")

        blob_client: BlobClient = self.blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name,
        )
        download_stream = await blob_client.download_blob()
        return await download_stream.readall()

    async def close(self) -> None:
        await self.blob_service_client.close()
        await self.default_azure_credential.close()

    async def __aenter__(
        self,
    ) -> "StorageAccountModel":
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        await self.close()
