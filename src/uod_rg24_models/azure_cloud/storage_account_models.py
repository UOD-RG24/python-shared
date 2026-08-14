from pydantic import BaseModel, Field


class BlobBaseModel(BaseModel):
    azure_storage_account_name: str = Field(
        alias="azureStorageAccountName",
        min_length=1,
        description="Name of the Azure Storage Account where the file is located.",
    )
    azure_container_name: str = Field(
        alias="azureContainerName",
        min_length=1,
        description="Name of the Azure Blob Storage container where the file is located.",
    )
    directory_name: str = Field(
        alias="directoryName",
        min_length=1,
        description="Name of the directory in the Azure Blob Storage where the file is located.",
    )
    file_name: str = Field(
        alias="fileName",
        min_length=1,
        description="Name of the file in the Azure Blob Storage without extension.",
    )
    extension: str = Field(
        alias="extension",
        default=None,
        description="Optional file extension (e.g., .csv, .tsv, .txt)",
    )
