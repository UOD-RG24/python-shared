from pydantic import BaseModel, Field


class InputOutputBlobBaseModel(BaseModel):
    azure_storage_account_name: str = Field(
        alias="azureStorageAccountName",
        min_length=1,
    )
    azure_container_name: str = Field(
        alias="azureContainerName",
        min_length=1,
    )
    input_blob_path: str = Field(
        alias="inputBlobPath",
        min_length=1,
    )
    output_blob_path: str = Field(
        alias="outputBlobPath",
        min_length=1,
    )
