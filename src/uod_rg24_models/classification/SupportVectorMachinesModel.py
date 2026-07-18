from typing import Any
import asyncio
import io
import os
import joblib
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from uod_rg24_models.shared.api_request_models import ApiRequestModel
from uod_rg24_models.azure_cloud.StorageAccountModel import StorageAccountModel


class TrainModel(BaseModel):
    data: list[list[float]]
    feature: list[int]
    test_size: float = Field(alias="testSize", default=0.25, gt=0, lt=1)
    random_state: int = Field(alias="randomState", default=42)
    stratify: bool = True

    @model_validator(mode="after")
    def validate_training_data(self):
        if len(self.data) != len(self.feature):
            raise ValueError(
                "The number of data rows must match the number of feature labels."
            )
        if len(self.data) < 4:
            raise ValueError("At least four training samples are required.")
        if len(set(self.feature)) < 2:
            raise ValueError("At least two target classes are required.")
        feature_lengths = {len(row) for row in self.data}
        if len(feature_lengths) != 1:
            raise ValueError("Every data row must contain the same number of values.")
        return self


class SupportVectorMachinesRequestModel(ApiRequestModel):
    train: TrainModel


class SupportVectorMachinesResponseDataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_type: str = Field(
        alias="modelType",
    )
    kernel: str
    training_samples: int = Field(alias="trainingSamples")
    testing_samples: int = Field(
        alias="testingSamples",
    )
    number_of_features: int = Field(
        alias="numberOfFeatures",
    )
    classes: list[Any]
    accuracy: float
    actual_values: list[Any] = Field(alias="actualValues")
    predicted_values: list[Any] = Field(alias="predictedValues")
    classification_report: dict[str, Any] = Field(alias="classificationReport")


async def save_svm_model(
    request_id: str,
    model,
) -> str:
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    model_buffer = io.BytesIO()
    joblib.dump(
        model,
        model_buffer,
        compress=3,
    )
    model_buffer.seek(0)
    blob_name = f"{request_id}/model_svm.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=model_buffer,
        )
        return blob_client.url


async def read_svm_model(
    request_id: str,
) -> Any:
    if not request_id:
        raise ValueError("request_id cannot be empty.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{request_id}/model_svm.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name,
    ) as storage_account_model:
        model_bytes = await storage_account_model.download_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
        )
    model_buffer = io.BytesIO(model_bytes)
    model_buffer.seek(0)
    model = joblib.load(model_buffer)
    return model
