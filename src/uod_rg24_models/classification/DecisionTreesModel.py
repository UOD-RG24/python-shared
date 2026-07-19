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
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
    criterion: Literal[
        "gini",
        "entropy",
        "log_loss",
    ] = "gini"
    splitter: Literal[
        "best",
        "random",
    ] = "best"
    max_depth: int | None = Field(
        alias="maxDepth",
        default=None,
        gt=0,
    )
    min_samples_split: int | float = Field(
        alias="minSamplesSplit",
        default=2,
    )
    min_samples_leaf: int | float = Field(
        alias="minSamplesLeaf",
        default=1,
    )
    max_features: int | float | Literal["sqrt", "log2"] | None = Field(
        alias="maxFeatures",
        default=None,
    )
    class_weight: Literal["balanced"] | dict[Any, float] | None = Field(
        alias="classWeight",
        default=None,
    )
    ccp_alpha: float = Field(
        alias="ccpAlpha",
        default=0.0,
        ge=0.0,
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


class DecisionTreesRequestModel(ApiRequestModel):
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


class DecisionTreesResponseDataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_type: str = Field(
        alias="modelType",
    )
    max_depth: int | None = Field(
        alias="maxDepth",
        default=None,
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


async def save_decision_trees_model(
    experiment_id: str,
    max_depth: int,
    model: Pipeline,
) -> str:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be greater than 0 if provided.")
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
    blob_name = f"{experiment_id}/model_decision_trees_{max_depth}.joblib"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=model_buffer,
        )
        return blob_client.url


async def read_decision_trees_model(
    experiment_id: str,
    max_depth: int,
) -> Any:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if not max_depth or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{experiment_id}/model_decision_trees_{max_depth}.joblib"
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
