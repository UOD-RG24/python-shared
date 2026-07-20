from typing import Annotated, Literal
from pydantic import BaseModel, Field
from sklearn.metrics import (
    multilabel_confusion_matrix,
)
import numpy as np
from uod_rg24_models.shared.api_request_models import ApiRequestModel


class SupportVectorMachinesVisualisationModel(BaseModel):
    model_type: Literal["support-vector-machines"]
    kernel: Literal["linear", "rbf", "poly"]


class RandomForestVisualisationModel(BaseModel):
    model_type: Literal["random-forest"]
    n_estimators: int = Field(
        alias="nEstimators",
        gt=0,
    )


class KNearestNeighborsVisualisationModel(BaseModel):
    model_type: Literal["k-nearest-neighbors"]
    n_neighbors: int = Field(alias="nNeighbors", default=5, gt=0)


class GaussianNaiveBayesVisualisationModel(BaseModel):
    model_type: Literal["gaussian-naive-bayes"]
    var_smoothing: float = Field(alias="varSmoothing", default=1e-9, gt=0)


class DecisionTreesVisualisationModel(BaseModel):
    model_type: Literal["decision-trees"]
    max_depth: int | None = Field(
        alias="maxDepth",
        default=None,
    )


class MultiLayerPerceptronVisualisationModel(BaseModel):
    model_type: Literal["multi-layer perceptron"]
    alpha: float = Field(
        default=0.0001,
        ge=0,
    )


class LogisticRegressionVisualisationModel(BaseModel):
    model_type: Literal["logistic-regression"]
    c: float = Field(
        default=1.0,
        gt=0,
        description="Inverse regularisation strength.",
    )


class XGBoostVisualisationModel(BaseModel):
    model_type: Literal["xgboost"]


def macro_specificity(
    y_true,
    y_pred,
) -> float:
    matrices = multilabel_confusion_matrix(
        y_true,
        y_pred,
    )
    true_negatives = matrices[:, 0, 0]
    false_positives = matrices[:, 0, 1]
    denominators = true_negatives + false_positives
    specificities = np.divide(
        true_negatives,
        denominators,
        out=np.zeros(
            true_negatives.shape,
            dtype=float,
        ),
        where=denominators != 0,
    )
    return float(np.mean(specificities))


VisualisationModel = Annotated[
    SupportVectorMachinesVisualisationModel
    | RandomForestVisualisationModel
    | KNearestNeighborsVisualisationModel
    | GaussianNaiveBayesVisualisationModel
    | DecisionTreesVisualisationModel
    | MultiLayerPerceptronVisualisationModel
    | LogisticRegressionVisualisationModel
    | XGBoostVisualisationModel,
    Field(discriminator="model_type"),
]


class FigureModel(BaseModel):
    height: int = Field(alias="height")
    width: int = Field(alias="width")


class CrossValdiationModel(BaseModel):
    class_size: int = Field(alias="classSize")
    n_repeats: int = Field(alias="nRepeats")
    random_state: int = Field(alias="randomState")


class BoxPlotForCrossValidationRequestModel(ApiRequestModel):
    visualisation: VisualisationModel
    cross_validation: CrossValdiationModel = Field(alias="crossValidation")
    figure: FigureModel
