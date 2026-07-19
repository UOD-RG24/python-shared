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
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    c: float = Field(
        default=1.0,
        gt=0,
        description="Inverse regularisation strength.",
    )
    penalty: Literal[
        "l1",
        "l2",
        "elasticnet",
        "none",
    ] = "l2"
    solver: Literal[
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ] = "lbfgs"
    max_iter: int = Field(
        alias="maxIter",
        default=1000,
        ge=1,
        le=100_000,
    )
    class_weight: Literal["balanced"] | None = Field(
        alias="classWeight",
        default=None,
    )
    test_size: float = Field(alias="testSize", default=0.25, gt=0, le=1)
    random_state: int | None = Field(alias="randomState", default=None)
    stratify: bool = True


class LogisticRegressionRequestModel(ApiRequestModel):
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


class LogisticRegressionResponseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_type: str = Field(
        alias="modelType",
    )
    var_smoothing: float = Field(
        alias="varSmoothing",
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


async def save_logistic_regression_model(
    experiment_id: str,
    c: float,
    model: Pipeline,
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
    blob_name = f"{experiment_id}/model_logistic_regression_c{c}.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=model_buffer,
        )
        return blob_client.url


async def read_logistic_regression_model(
    experiment_id: str,
    c: float,
) -> Any:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{experiment_id}/model_logistic_regression_c{c}.joblib"
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
