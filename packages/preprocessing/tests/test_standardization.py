from __future__ import annotations

import math

import pytest
from uod_rg24_preprocessing.standardization import (
    StandardizationError,
    StandardizationStrategy,
    fit_transform,
    transform,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
FEATURES = ("mrna::entrez::1", "mrna::entrez::2")
TYPES = ("float64", "float64")


def fit(strategy: StandardizationStrategy, values: list[list[float]]):
    return fit_transform(
        values,
        feature_ids=FEATURES,
        feature_types=TYPES,
        strategy=strategy,
        recipe_id=f"{strategy.value}-scaler-v1",
        recipe_version="1.0.0",
        modality="mrna",
        source_value_scale="LOG2-VALUE",
        learning_scope="trainFold",
        fit_population_artifact_id="art_train_ids",
        fit_population_sha256=HASH_A,
        input_artifact_sha256=HASH_B,
    )


def test_standard_scaler_uses_population_variance_and_handles_constants() -> None:
    result = fit(StandardizationStrategy.STANDARD, [[1.0, 5.0], [3.0, 5.0]])

    assert result.values == ((-1.0, 0.0), (1.0, 0.0))
    first = result.transformer.feature_parameters[0].mapping
    second = result.transformer.feature_parameters[1].mapping
    assert first == {"center": 2.0, "scale": 1.0, "variance": 1.0}
    assert second == {"center": 5.0, "scale": 1.0, "variance": 0.0}


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (StandardizationStrategy.MIN_MAX, ((0.0, 0.0), (1.0, 0.0))),
        (StandardizationStrategy.MAX_ABS, ((1.0 / 3.0, 1.0), (1.0, 1.0))),
        (StandardizationStrategy.ROBUST, ((-1.0, 0.0), (1.0, 0.0))),
    ],
)
def test_strategy_golden_values_and_constant_features(
    strategy: StandardizationStrategy,
    expected: tuple[tuple[float, ...], ...],
) -> None:
    result = fit(strategy, [[1.0, 5.0], [3.0, 5.0]])

    for actual_row, expected_row in zip(result.values, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_transform_reuses_saved_state_for_held_out_values() -> None:
    fitted = fit(StandardizationStrategy.STANDARD, [[1.0, 10.0], [3.0, 14.0]])

    applied = transform(
        [[101.0, -86.0]],
        feature_ids=FEATURES,
        feature_types=TYPES,
        transformer=fitted.transformer,
    )

    assert applied.values == ((99.0, -49.0),)
    assert applied.transformer is fitted.transformer
    assert fitted.transformer.feature_parameters[0].mapping["center"] == 2.0


def test_changed_validation_values_cannot_change_fitted_parameters() -> None:
    fitted = fit(StandardizationStrategy.MIN_MAX, [[1.0, 2.0], [3.0, 6.0]])
    before = fitted.transformer

    transform(
        [[-1e12, 1e12]],
        feature_ids=FEATURES,
        feature_types=TYPES,
        transformer=before,
    )

    assert fitted.transformer == before


def test_robust_scaler_uses_linear_percentiles() -> None:
    result = fit_transform(
        [[0.0], [10.0], [20.0], [30.0]],
        feature_ids=("x",),
        feature_types=("float64",),
        strategy="robust",
        recipe_id="robust-scaler-v1",
        recipe_version="1.0.0",
        modality="mrna",
        source_value_scale="LOG2-VALUE",
        learning_scope="cohortUnsupervised",
        fit_population_artifact_id="art_cohort",
        fit_population_sha256=HASH_A,
        input_artifact_sha256=HASH_B,
    )

    parameters = result.transformer.feature_parameters[0].mapping
    assert parameters["quantileLow"] == 7.5
    assert parameters["quantileHigh"] == 22.5
    assert parameters["center"] == 15.0
    assert parameters["scale"] == 15.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(bad: float) -> None:
    with pytest.raises(StandardizationError) as caught:
        fit(StandardizationStrategy.STANDARD, [[bad, 1.0]])

    assert caught.value.code == "INVALID_VALUE_DOMAIN"


def test_transform_rejects_missing_reordered_or_retyped_features() -> None:
    fitted = fit(StandardizationStrategy.STANDARD, [[1.0, 2.0], [3.0, 4.0]])

    for feature_ids, feature_types in (
        ((FEATURES[0],), (TYPES[0],)),
        ((FEATURES[1], FEATURES[0]), TYPES),
        (FEATURES, ("float32", "float64")),
    ):
        with pytest.raises(StandardizationError) as caught:
            transform(
                [[1.0] * len(feature_ids)],
                feature_ids=feature_ids,
                feature_types=feature_types,
                transformer=fitted.transformer,
            )
        assert caught.value.code == "INCOMPATIBLE_FEATURE_SCHEMA"


def test_invalid_scope_and_robust_quantiles_are_rejected() -> None:
    with pytest.raises(StandardizationError) as caught:
        fit_transform(
            [[1.0]],
            feature_ids=("x",),
            feature_types=("float64",),
            strategy="standard",
            recipe_id="standard-scaler-v1",
            recipe_version="1.0.0",
            modality="mrna",
            source_value_scale="LOG2-VALUE",
            learning_scope="none",
            fit_population_artifact_id="art_train",
            fit_population_sha256=HASH_A,
            input_artifact_sha256=HASH_B,
        )
    assert caught.value.code == "INVALID_LEARNING_SCOPE"

    with pytest.raises(StandardizationError) as caught:
        fit_transform(
            [[1.0]],
            feature_ids=("x",),
            feature_types=("float64",),
            strategy="robust",
            recipe_id="robust-scaler-v1",
            recipe_version="1.0.0",
            modality="mrna",
            source_value_scale="LOG2-VALUE",
            learning_scope="trainFold",
            fit_population_artifact_id="art_train",
            fit_population_sha256=HASH_A,
            input_artifact_sha256=HASH_B,
            quantile_range=(75.0, 25.0),
        )
    assert caught.value.code == "INVALID_PARAMETERS"
