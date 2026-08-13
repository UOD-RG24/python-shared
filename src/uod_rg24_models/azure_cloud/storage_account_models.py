from pydantic import BaseModel, Field


class BlobBaseModel(BaseModel):
    azure_storage_account_name: str = Field(
        alias="azureStorageAccountName",
        min_length=1,
    )
    azure_container_name: str = Field(
        alias="azureContainerName",
        min_length=1,
    )
    directory_name: str = Field(
        alias="directoryName",
        min_length=1,
    )
    file_name: str = Field(
        alias="fileName",
        min_length=1,
    )
