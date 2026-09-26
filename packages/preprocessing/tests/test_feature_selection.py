from __future__ import annotations

# pyright: reportArgumentType=false
import math

import pytest
from uod_rg24_preprocessing.feature_selection import (
    FeatureSelectionError,
    FeatureSelectionStrategy,
    apply_mask,
    fit_select,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
FEATURES = ("mrna::a", "mrna::b", "mrna::c", "mrna::d")
TYPES = ("double",) * 4


def select(
    strategy: FeatureSelectionStrategy | str,
    values: list[list[float | None]],
    **options: object,
):
    return fit_select(
        values,
        feature_ids=FEATURES,
        feature_types=TYPES,
        strategy=strategy,
        recipe_id=f"{strategy}-v1",
        recipe_version="1.0.0",
        modality="mrna",
        source_value_scale="standardized:standard",
        learning_scope="trainFold",
        fit_population_artifact_id="art_train_ids",
        fit_population_sha256=HASH_A,
        input_artifact_sha256=HASH_B,
        **options,
    )


def test_remove_all_missing_preserves_partial_features_and_reports_reason() -> None:
    result = select(
        "removeAllMissing",
        [[None, 1.0, 2.0, None], [None, None, 3.0, 4.0]],
    )

    assert result.mask.selected_feature_ids == ("mrna::b", "mrna::c", "mrna::d")
    assert result.values == ((1.0, 2.0, None), (None, 3.0, 4.0))
    assert result.mask.decisions[0].rejection_reason == "allMissing"
    assert result.mask.decisions[1].score == 1.0


def test_remove_constant_uses_observed_values_and_rejects_all_missing() -> None:
    result = select(
        "removeConstant",
        [[1.0, 1.0, None, 4.0], [1.0, 2.0, None, 4.0]],
    )

    assert result.mask.selected_feature_ids == ("mrna::b",)
    assert [item.rejection_reason for item in result.mask.decisions] == [
        "constant",
        None,
        "allMissing",
        "constant",
    ]


def test_low_variance_uses_population_variance_and_inclusive_threshold() -> None:
    result = select(
        "lowVariance",
        [[1.0, 0.0, 5.0, 2.0], [3.0, 0.0, 5.0, 6.0]],
        variance_threshold=1.0,
    )

    assert result.mask.selected_feature_ids == ("mrna::a", "mrna::d")
    assert [item.score for item in result.mask.decisions] == [1.0, 0.0, 0.0, 4.0]
    assert result.mask.option_mapping == {"varianceThreshold": 1.0}


def test_top_k_tie_breaks_by_feature_id_but_outputs_input_order() -> None:
    result = select(
        "topKByVariance",
        [[0.0, 0.0, 0.0, 10.0], [2.0, 2.0, 2.0, 14.0]],
        top_k=2,
    )

    # d has the highest variance; a/b/c tie, so lexical feature ID selects a.
    assert result.mask.selected_feature_ids == ("mrna::a", "mrna::d")
    assert result.values == ((0.0, 10.0), (2.0, 14.0))


def test_correlation_filter_keeps_highest_variance_then_lexical_tie() -> None:
    result = select(
        "correlationFilter",
        [
            [1.0, 2.0, 4.0, 1.0],
            [2.0, 4.0, 3.0, 0.0],
            [3.0, 6.0, 2.0, 1.0],
            [4.0, 8.0, 1.0, 0.0],
        ],
        maximum_correlation=0.9,
    )

    assert result.mask.selected_feature_ids == ("mrna::b", "mrna::d")
    assert result.mask.decisions[0].rejection_reason == "correlatedWith:mrna::b"
    assert result.mask.decisions[2].rejection_reason == "correlatedWith:mrna::b"


def test_apply_mask_reuses_state_without_recalculating_scores() -> None:
    fitted = select(
        "topKByVariance",
        [[0.0, 0.0, 0.0, 10.0], [2.0, 2.0, 2.0, 14.0]],
        top_k=2,
    )
    before = fitted.mask

    applied = apply_mask(
        [[1e12, -1e12, 9e12, -9e12]],
        feature_ids=FEATURES,
        feature_types=TYPES,
        mask=before,
    )

    assert applied.values == ((1e12, -9e12),)
    assert applied.mask is before


def test_schema_mismatch_and_missing_fit_values_fail() -> None:
    fitted = select(
        "topKByVariance",
        [[0.0, 0.0, 0.0, 10.0], [2.0, 2.0, 2.0, 14.0]],
        top_k=2,
    )
    with pytest.raises(FeatureSelectionError) as caught:
        apply_mask(
            [[1.0, 2.0, 3.0, 4.0]],
            feature_ids=("mrna::b", "mrna::a", "mrna::c", "mrna::d"),
            feature_types=TYPES,
            mask=fitted.mask,
        )
    assert caught.value.code == "INCOMPATIBLE_FEATURE_SCHEMA"

    with pytest.raises(FeatureSelectionError) as caught:
        select(
            "lowVariance",
            [[1.0, None, 3.0, 4.0]],
            variance_threshold=0.0,
        )
    assert caught.value.code == "INVALID_VALUE_DOMAIN"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_fail(bad: float) -> None:
    with pytest.raises(FeatureSelectionError) as caught:
        select("removeConstant", [[1.0, bad, 3.0, 4.0]])
    assert caught.value.code == "INVALID_VALUE_DOMAIN"


def test_no_features_and_invalid_options_fail() -> None:
    with pytest.raises(FeatureSelectionError) as caught:
        select("removeConstant", [[1.0, 1.0, 1.0, 1.0]])
    assert caught.value.code == "NO_FEATURES_SELECTED"

    for strategy, options in (
        ("lowVariance", {"variance_threshold": -1.0}),
        ("topKByVariance", {"top_k": 0}),
        ("correlationFilter", {"maximum_correlation": 1.1}),
    ):
        with pytest.raises(FeatureSelectionError) as caught:
            select(strategy, [[1.0, 2.0, 3.0, 4.0]], **options)
        assert caught.value.code == "INVALID_PARAMETERS"
