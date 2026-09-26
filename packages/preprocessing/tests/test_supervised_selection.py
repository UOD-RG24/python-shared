from __future__ import annotations

import math
import sys

import pytest
from uod_rg24_preprocessing.feature_selection import (
    FeatureSelectionError,
    apply_mask,
    fit_select,
)


def fit(values, **overrides):
    options = {
        "feature_ids": ("mrna::b", "mrna::a", "mrna::constant"),
        "feature_types": ("double",) * 3,
        "strategy": "topKByAnova",
        "recipe_id": "top-k-by-anova-v1",
        "recipe_version": "1.0.0",
        "modality": "mrna",
        "source_value_scale": "LOG2-VALUE",
        "learning_scope": "trainFold",
        "fit_population_artifact_id": "train",
        "fit_population_sha256": "a" * 64,
        "input_artifact_sha256": "b" * 64,
        "labels": ("A", "A", "B", "B"),
        "label_artifact_id": "labels",
        "label_artifact_sha256": "c" * 64,
        "top_k": 1,
    }
    options.update(overrides)
    return fit_select(values, **options)


def test_anova_known_score_ties_and_input_order():
    result = fit([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]])
    assert result.mask.selected_feature_ids == ("mrna::a",)
    assert result.mask.decisions[0].score == pytest.approx(50.0)
    assert result.mask.decisions[2].score == 0.0
    result = fit(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]], top_k=2
    )
    assert result.mask.selected_feature_ids == ("mrna::b", "mrna::a")


def test_perfect_separation_has_finite_persistable_score():
    result = fit([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert result.mask.decisions[0].score == sys.float_info.max
    assert all(math.isfinite(row.score) for row in result.mask.decisions)


@pytest.mark.parametrize(
    "override",
    [
        {"labels": None},
        {"labels": ("A",) * 4},
        {"labels": ("A", "A", "A", "B")},
        {"learning_scope": "cohortUnsupervised"},
        {"label_artifact_id": None},
        {"label_artifact_sha256": "bad"},
        {"top_k": True},
        {"top_k": 1.5},
    ],
)
def test_invalid_supervised_inputs_fail(override):
    with pytest.raises(FeatureSelectionError):
        fit(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]],
            **override,
        )


def test_transform_uses_fitted_state_without_labels():
    fitted = fit([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]])
    result = apply_mask(
        [[1e6, -1e6, 9e6]],
        feature_ids=fitted.mask.input_feature_ids,
        feature_types=fitted.mask.input_feature_types,
        mask=fitted.mask,
    )
    assert result.values == ((-1e6,),)
    assert result.mask is fitted.mask
