from typing import Any, Literal
import io
import os
import joblib
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from uod_rg24_models.shared.api_request_models import ApiRequestModel
from uod_rg24_models.azure_cloud.StorageAccountModel import StorageAccountModel
from sklearn.pipeline import Pipeline
from uod_rg24_tools.deployment_tools import (
    get_project_metadata,
)


class TrainModel(BaseModel):
    n_neighbors: int = Field(alias="nNeighbors", default=5, gt=0)
    weights: Literal["uniform", "distance"] = Field(alias="weights", default="uniform")
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = Field(
        alias="algorithm", default="auto"
    )
    metric: str = Field(alias="metric", default="minkowski")
    test_size: float = Field(alias="testSize", default=0.25, gt=0, le=1)
    random_state: int | None = Field(alias="randomState", default=None)
    p: int = Field(alias="p", default=2, gt=0)
    n_jobs: int = Field(alias="nJobs", default=-1)
    stratify: bool = True


class KNearestNeighborsRequestModel(ApiRequestModel):
    train: TrainModel


class ResponseMetadataModel(BaseModel):
    source: str
    version: str


def get_response_metadata() -> ResponseMetadataModel:
    project_metadata = get_project_metadata()

    return ResponseMetadataModel(
        source=str(
            project_metadata.get(
                "name",
                "unknown",
            )
        ),
        version=str(
            project_metadata.get(
                "version",
                "unknown",
            )
        ),
    )


class KNearestNeighborsResponseDataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_type: str = Field(
        alias="modelType",
    )
    n_neighbors: int = Field(
        alias="nNeighbors",
    )
    weights: str = Field(
        alias="weights",
    )
    algorithm: str = Field(
        alias="algorithm",
    )
    metric: str = Field(
        alias="metric",
    )
    p: int = Field(
        alias="p",
    )
    n_jobs: int = Field(
        alias="nJobs",
    )
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
    model_url: str = Field(alias="modelUrl")
    metadata: ResponseMetadataModel = Field(
        default_factory=get_response_metadata,
    )


async def save_k_nearest_neighbors_model(
    experiment_id: str,
    model: Pipeline,
    n_neighbors: int,
) -> str:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
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
    blob_name = f"{experiment_id}/model_k_nearest_neighbors_{n_neighbors}.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=model_buffer,
        )
        return blob_client.url


async def read_k_nearest_neighbors_model(
    experiment_id: str,
    n_neighbors: int,
) -> Any:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{experiment_id}/model_k_nearest_neighbors_{n_neighbors}.joblib"
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
