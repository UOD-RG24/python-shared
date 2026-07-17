from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from uod_rg24_models.shared.api_request_models import ApiRequestModel

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
            raise ValueError(
                "Every data row must contain the same number of values."
            )
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
    testing_samples: int = Field(alias="testingSamples",)
    number_of_features: int = Field(
        alias="numberOfFeatures",
    )
    classes: list[Any]
    accuracy: float
    actual_values: list[Any] = Field(alias="actualValues")
    predicted_values: list[Any] = Field(alias="predictedValues")
    classification_report: dict[str, Any] = Field(alias="classificationReport")