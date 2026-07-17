from pydantic import BaseModel, Field, ValidationError, model_validator

class TrainModel(BaseModel):
    data: list[list[float]]
    feature: list[int]
    test_size: float = Field(default=0.25, gt=0, lt=1)
    random_state: int = 42
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

class SupportVectorMachinesRequestModel(BaseModel):
    train: TrainModel
