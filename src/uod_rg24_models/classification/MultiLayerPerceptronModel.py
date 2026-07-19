from typing import Any, Literal
import io
import os
import joblib
from pydantic import BaseModel, ConfigDict, Field
from uod_rg24_models.shared.api_request_models import ApiRequestModel
from uod_rg24_models.azure_cloud.StorageAccountModel import StorageAccountModel
from sklearn.pipeline import Pipeline
from uod_rg24_tools.deployment_tools import (
    get_project_metadata,
)


class TrainModel(BaseModel):
    hidden_layer_sizes: tuple[int, ...] = Field(
        alias="hiddenLayerSizes",
        default=(100, 50),
        min_length=1,
    )
    activation: Literal[
        "identity",
        "logistic",
        "tanh",
        "relu",
    ] = "relu"
    solver: Literal[
        "lbfgs",
        "sgd",
        "adam",
    ] = "adam"
    alpha: float = Field(
        default=0.0001,
        ge=0,
    )
    learning_rate_init: float = Field(
        alias="learningRateInit",
        default=0.001,
        gt=0,
    )
    max_iter: int = Field(
        alias="maxIter",
        default=500,
        gt=0,
    )
    early_stopping: bool = Field(
        alias="earlyStopping",
        default=True,
    )
    validation_fraction: float = Field(
        alias="validationFraction",
        default=0.1,
        gt=0,
        lt=1,
    )
    n_iter_no_change: int = Field(
        alias="nIterNoChange",
        default=20,
        gt=0,
    )
    test_size: float = Field(
        alias="testSize",
        default=0.25,
        gt=0.0,
        lt=1.0,
    )
    random_state: int | None = Field(
        alias="randomState",
        default=None,
    )
    stratify: bool = True


class MultiLayerPerceptronRequestModel(ApiRequestModel):
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


class MultiLayerPerceptronResponseDataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_type: str = Field(
        alias="modelType",
        default="Multi-Layer Perceptron",
    )
    hidden_layer_sizes: list[int] = Field(
        alias="hiddenLayerSizes",
    )
    activation: str
    solver: str
    iterations: int
    loss: float
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


async def save_multi_layer_perceptron_model(
    experiment_id: str,
    alpha: float,
    model: Pipeline,
) -> str:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if alpha is not None and alpha <= 0:
        raise ValueError("alpha must be greater than 0 if provided.")
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
    blob_name = f"{experiment_id}/model_multi_layer_perceptron_alpha{alpha}.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=model_buffer,
        )
        return blob_client.url


async def read_multi_layer_perceptron_model(
    experiment_id: str,
    alpha: float,
) -> Any:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if not alpha or alpha <= 0:
        raise ValueError("max_depth must be a positive integer.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{experiment_id}/model_multi_layer_perceptron_alpha{alpha}.joblib"
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
