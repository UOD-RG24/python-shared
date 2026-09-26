from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Sequence

from .errors import FeatureSelectionError
from .models import (
    FeatureDecision,
    FeatureSelectionResult,
    FeatureSelectionStrategy,
    FittedFeatureMask,
    SelectionLearningScope,
)

NullableNumericRows = Sequence[Sequence[float | int | None]]


def fit_select(
    values: NullableNumericRows,
    *,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
    strategy: FeatureSelectionStrategy | str,
    recipe_id: str,
    recipe_version: str,
    modality: str,
    source_value_scale: str,
    learning_scope: SelectionLearningScope | str,
    fit_population_artifact_id: str,
    fit_population_sha256: str,
    input_artifact_sha256: str,
    variance_threshold: float = 0.0,
    top_k: int | None = None,
    maximum_correlation: float = 0.95,
    labels: Sequence[str] | None = None,
    label_artifact_id: str | None = None,
    label_artifact_sha256: str | None = None,
) -> FeatureSelectionResult:
    """Fit a deterministic mask and apply it to the fitting rows."""

    parsed_strategy = _strategy(strategy)
    parsed_scope = _scope(learning_scope)
    rows, ids, types = _validate(values, feature_ids, feature_types)
    columns = tuple(tuple(row[index] for row in rows) for index in range(len(ids)))
    if parsed_strategy == FeatureSelectionStrategy.TOP_K_BY_ANOVA:
        if parsed_scope != SelectionLearningScope.TRAIN_FOLD:
            raise FeatureSelectionError(
                "INVALID_LEARNING_SCOPE", "Supervised selection requires trainFold."
            )
        if not label_artifact_id or not label_artifact_sha256:
            raise FeatureSelectionError(
                "INVALID_LABEL_ARTIFACT", "Exact label artifact provenance is required."
            )
    selected, scores, reasons, options = _fit(
        columns,
        ids,
        parsed_strategy,
        variance_threshold=variance_threshold,
        top_k=top_k,
        maximum_correlation=maximum_correlation,
        labels=labels,
    )
    selected_positions = {
        position: selected_position
        for selected_position, position in enumerate(
            position for position in range(len(ids)) if position in selected
        )
    }
    decisions = tuple(
        FeatureDecision(
            input_position=position,
            feature_id=feature_id,
            feature_type=types[position],
            selected=position in selected,
            selected_position=selected_positions.get(position),
            score=scores[position],
            rejection_reason=reasons[position],
        )
        for position, feature_id in enumerate(ids)
    )
    mask = FittedFeatureMask(
        schema_version="feature-mask/1.0",
        strategy=parsed_strategy,
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        modality=modality,
        source_value_scale=source_value_scale,
        input_feature_ids=ids,
        input_feature_types=types,
        decisions=decisions,
        learning_scope=parsed_scope,
        fit_population_artifact_id=fit_population_artifact_id,
        fit_population_sha256=fit_population_sha256,
        input_artifact_sha256=input_artifact_sha256,
        fit_row_count=len(rows),
        options=options,
        label_artifact_id=label_artifact_id,
        label_artifact_sha256=label_artifact_sha256,
    )
    return FeatureSelectionResult(values=_apply(rows, mask), mask=mask)


def apply_mask(
    values: NullableNumericRows,
    *,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
    mask: FittedFeatureMask,
) -> FeatureSelectionResult:
    """Apply trusted fitted selection state without recalculating scores."""

    rows, ids, types = _validate(values, feature_ids, feature_types)
    if ids != mask.input_feature_ids or types != mask.input_feature_types:
        raise FeatureSelectionError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "The matrix feature identifiers, order, or types do not match the mask.",
        )
    if mask.strategy in {
        FeatureSelectionStrategy.LOW_VARIANCE,
        FeatureSelectionStrategy.TOP_K_BY_VARIANCE,
        FeatureSelectionStrategy.CORRELATION_FILTER,
        FeatureSelectionStrategy.TOP_K_BY_ANOVA,
    }:
        _require_complete(rows)
    return FeatureSelectionResult(values=_apply(rows, mask), mask=mask)


def _strategy(value: FeatureSelectionStrategy | str) -> FeatureSelectionStrategy:
    try:
        return FeatureSelectionStrategy(value)
    except ValueError as exc:
        raise FeatureSelectionError(
            "UNSUPPORTED_STRATEGY", "The feature-selection strategy is not supported."
        ) from exc


def _scope(value: SelectionLearningScope | str) -> SelectionLearningScope:
    try:
        return SelectionLearningScope(value)
    except ValueError as exc:
        raise FeatureSelectionError(
            "INVALID_LEARNING_SCOPE",
            "Feature-mask fitting requires trainFold or cohortUnsupervised.",
        ) from exc


def _validate(
    values: NullableNumericRows,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
) -> tuple[tuple[tuple[float | None, ...], ...], tuple[str, ...], tuple[str, ...]]:
    ids = tuple(feature_ids)
    types = tuple(feature_types)
    if not ids or len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
        raise FeatureSelectionError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "Feature identifiers must be non-empty and unique.",
        )
    if len(types) != len(ids) or any(not value.strip() for value in types):
        raise FeatureSelectionError(
            "INCOMPATIBLE_FEATURE_SCHEMA", "Every feature requires a canonical type."
        )
    rows = tuple(
        tuple(None if value is None else float(value) for value in row)
        for row in values
    )
    if not rows:
        raise FeatureSelectionError(
            "EMPTY_FITTING_POPULATION", "At least one fitting row is required."
        )
    if any(len(row) != len(ids) for row in rows):
        raise FeatureSelectionError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "Every row must contain exactly one value per feature.",
        )
    if any(
        value is not None and not math.isfinite(value) for row in rows for value in row
    ):
        raise FeatureSelectionError(
            "INVALID_VALUE_DOMAIN", "Feature values must be finite or null."
        )
    return rows, ids, types


def _fit(
    columns: tuple[tuple[float | None, ...], ...],
    feature_ids: tuple[str, ...],
    strategy: FeatureSelectionStrategy,
    *,
    variance_threshold: float,
    top_k: int | None,
    maximum_correlation: float,
    labels: Sequence[str] | None,
) -> tuple[
    set[int],
    tuple[float | None, ...],
    tuple[str | None, ...],
    tuple[tuple[str, float | int], ...],
]:
    count = len(columns)
    scores: list[float | None] = [None] * count
    reasons: list[str | None] = [None] * count
    selected: set[int] = set()

    if strategy == FeatureSelectionStrategy.REMOVE_ALL_MISSING:
        for position, column in enumerate(columns):
            present = sum(value is not None for value in column)
            scores[position] = float(present)
            if present:
                selected.add(position)
            else:
                reasons[position] = "allMissing"
        options: tuple[tuple[str, float | int], ...] = ()
    elif strategy == FeatureSelectionStrategy.REMOVE_CONSTANT:
        for position, column in enumerate(columns):
            present = tuple(float(value) for value in column if value is not None)
            if not present:
                reasons[position] = "allMissing"
                scores[position] = None
            else:
                scores[position] = _variance(present)
                if len(set(present)) > 1:
                    selected.add(position)
                else:
                    reasons[position] = "constant"
        options = ()
    elif strategy == FeatureSelectionStrategy.LOW_VARIANCE:
        _require_complete_columns(columns)
        if not math.isfinite(variance_threshold) or variance_threshold < 0.0:
            raise FeatureSelectionError(
                "INVALID_PARAMETERS",
                "varianceThreshold must be finite and non-negative.",
            )
        for position, column in enumerate(columns):
            score = _variance(
                tuple(float(value) for value in column if value is not None)
            )
            scores[position] = score
            if score >= variance_threshold:
                selected.add(position)
            else:
                reasons[position] = "belowVarianceThreshold"
        options = (("varianceThreshold", float(variance_threshold)),)
    elif strategy in {
        FeatureSelectionStrategy.TOP_K_BY_VARIANCE,
        FeatureSelectionStrategy.TOP_K_BY_ANOVA,
    }:
        _require_complete_columns(columns)
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= count
        ):
            raise FeatureSelectionError(
                "INVALID_PARAMETERS", "topK must be between one and the feature count."
            )
        if strategy == FeatureSelectionStrategy.TOP_K_BY_ANOVA:
            variances = _anova_scores(columns, labels)
        else:
            variances = tuple(
                _variance(tuple(float(value) for value in column if value is not None))
                for column in columns
            )
        scores[:] = variances
        ranking = sorted(
            range(count),
            key=lambda position: (-variances[position], feature_ids[position]),
        )
        selected.update(ranking[:top_k])
        for position in range(count):
            if position not in selected:
                reasons[position] = "outsideTopK"
        options = (("topK", top_k),)
    else:
        _require_complete_columns(columns)
        if (
            not math.isfinite(maximum_correlation)
            or not 0.0 <= maximum_correlation <= 1.0
        ):
            raise FeatureSelectionError(
                "INVALID_PARAMETERS", "maximumCorrelation must be between zero and one."
            )
        variances = tuple(
            _variance(tuple(float(value) for value in column if value is not None))
            for column in columns
        )
        scores[:] = variances
        ranking = sorted(
            range(count),
            key=lambda position: (-variances[position], feature_ids[position]),
        )
        ranked_selected: list[int] = []
        for position in ranking:
            conflict = next(
                (
                    accepted
                    for accepted in ranked_selected
                    if abs(_correlation(columns[position], columns[accepted]))
                    > maximum_correlation
                ),
                None,
            )
            if conflict is None:
                selected.add(position)
                ranked_selected.append(position)
            else:
                reasons[position] = f"correlatedWith:{feature_ids[conflict]}"
        options = (("maximumCorrelation", float(maximum_correlation)),)

    if not selected:
        raise FeatureSelectionError(
            "NO_FEATURES_SELECTED", "The configured strategy selected no features."
        )
    return selected, tuple(scores), tuple(reasons), options


def _anova_scores(
    columns: tuple[tuple[float | None, ...], ...], labels: Sequence[str] | None
) -> tuple[float, ...]:
    """One-way ANOVA F scores; no p-values or ordinal label encoding."""
    if (
        labels is None
        or len(labels) != len(columns[0])
        or any(not isinstance(label, str) or not label.strip() for label in labels)
    ):
        raise FeatureSelectionError(
            "INVALID_LABEL_ARTIFACT", "Every fitting row requires a mapped class label."
        )
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        raise FeatureSelectionError(
            "INSUFFICIENT_CLASS_SAMPLES",
            "ANOVA requires two classes with two rows each.",
        )
    groups = tuple(
        tuple(i for i, label in enumerate(labels) if label == cls)
        for cls in sorted(counts)
    )
    scores: list[float] = []
    for column in columns:
        values = tuple(float(value) for value in column if value is not None)
        # Rescaling avoids overflow and leaves the dimensionless F statistic unchanged.
        scale = max(abs(value) for value in values) or 1.0
        values = tuple(value / scale for value in values)
        mean = math.fsum(values) / len(values)
        means = tuple(
            math.fsum(values[i] for i in group) / len(group) for group in groups
        )
        between = math.fsum(
            len(group) * (center - mean) ** 2
            for group, center in zip(groups, means, strict=True)
        )
        within = math.fsum(
            (values[i] - center) ** 2
            for group, center in zip(groups, means, strict=True)
            for i in group
        )
        if within == 0.0:
            score = sys.float_info.max if between > 0.0 else 0.0
        else:
            score = min(
                (between / (len(groups) - 1)) / (within / (len(values) - len(groups))),
                sys.float_info.max,
            )
        scores.append(score)
    return tuple(scores)


def _require_complete(rows: tuple[tuple[float | None, ...], ...]) -> None:
    if any(value is None for row in rows for value in row):
        raise FeatureSelectionError(
            "INVALID_VALUE_DOMAIN",
            "This selection strategy does not accept null values.",
        )


def _require_complete_columns(columns: tuple[tuple[float | None, ...], ...]) -> None:
    if any(value is None for column in columns for value in column):
        raise FeatureSelectionError(
            "INVALID_VALUE_DOMAIN",
            "This selection strategy does not accept null values.",
        )


def _variance(values: tuple[float, ...]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _correlation(
    left: tuple[float | None, ...], right: tuple[float | None, ...]
) -> float:
    x = tuple(float(value) for value in left if value is not None)
    y = tuple(float(value) for value in right if value is not None)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = tuple(value - mean_x for value in x)
    centered_y = tuple(value - mean_y for value in y)
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / denominator


def _apply(
    rows: tuple[tuple[float | None, ...], ...], mask: FittedFeatureMask
) -> tuple[tuple[float | None, ...], ...]:
    positions = tuple(item.input_position for item in mask.decisions if item.selected)
    return tuple(tuple(row[position] for position in positions) for row in rows)
