from __future__ import annotations

import math
from collections.abc import Sequence

from .errors import StandardizationError
from .models import (
    FeatureParameters,
    FittedTransformer,
    LearningScope,
    StandardizationResult,
    StandardizationStrategy,
)

NumericRows = Sequence[Sequence[float]]


def fit_transform(
    values: NumericRows,
    *,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
    strategy: StandardizationStrategy | str,
    recipe_id: str,
    recipe_version: str,
    modality: str,
    source_value_scale: str,
    learning_scope: LearningScope | str,
    fit_population_artifact_id: str,
    fit_population_sha256: str,
    input_artifact_sha256: str,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
) -> StandardizationResult:
    """Fit on exactly ``values`` and transform those rows with saved parameters."""

    parsed_strategy = _parse_strategy(strategy)
    parsed_scope = _parse_scope(learning_scope)
    rows, ids, types = _validate_matrix(values, feature_ids, feature_types)
    columns = tuple(tuple(row[index] for row in rows) for index in range(len(ids)))
    options = _options(
        parsed_strategy,
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
    )
    parameters = tuple(
        FeatureParameters(
            feature_id=feature_id, values=_fit_column(column, parsed_strategy, options)
        )
        for feature_id, column in zip(ids, columns, strict=True)
    )
    transformer = FittedTransformer(
        schema_version="fitted-transformer/1.0",
        strategy=parsed_strategy,
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        modality=modality,
        source_value_scale=source_value_scale,
        feature_ids=ids,
        feature_types=types,
        feature_parameters=parameters,
        learning_scope=parsed_scope,
        fit_population_artifact_id=fit_population_artifact_id,
        fit_population_sha256=fit_population_sha256,
        input_artifact_sha256=input_artifact_sha256,
        fit_row_count=len(rows),
        options=options,
    )
    return StandardizationResult(
        values=_apply(rows, transformer), transformer=transformer
    )


def transform(
    values: NumericRows,
    *,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
    transformer: FittedTransformer,
) -> StandardizationResult:
    """Apply trusted fitted state without executing a fitting path."""

    rows, ids, types = _validate_matrix(values, feature_ids, feature_types)
    if ids != transformer.feature_ids or types != transformer.feature_types:
        raise StandardizationError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "The matrix feature identifiers, order, or types do not match the transformer.",
        )
    return StandardizationResult(
        values=_apply(rows, transformer), transformer=transformer
    )


def _parse_strategy(value: StandardizationStrategy | str) -> StandardizationStrategy:
    try:
        return StandardizationStrategy(value)
    except ValueError as exc:
        raise StandardizationError(
            "UNSUPPORTED_STRATEGY", "The standardization strategy is not supported."
        ) from exc


def _parse_scope(value: LearningScope | str) -> LearningScope:
    try:
        return LearningScope(value)
    except ValueError as exc:
        raise StandardizationError(
            "INVALID_LEARNING_SCOPE",
            "Fit-transform requires trainFold or cohortUnsupervised learning scope.",
        ) from exc


def _validate_matrix(
    values: NumericRows,
    feature_ids: Sequence[str],
    feature_types: Sequence[str],
) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...], tuple[str, ...]]:
    ids = tuple(feature_ids)
    types = tuple(feature_types)
    if not ids or len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
        raise StandardizationError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "Feature identifiers must be non-empty and unique.",
        )
    if len(types) != len(ids) or any(not value.strip() for value in types):
        raise StandardizationError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "Each feature must have one canonical non-empty type.",
        )
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows:
        raise StandardizationError(
            "EMPTY_FITTING_POPULATION", "At least one matrix row is required."
        )
    if any(len(row) != len(ids) for row in rows):
        raise StandardizationError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            "Every matrix row must contain exactly one value per feature.",
        )
    if any(not math.isfinite(value) for row in rows for value in row):
        raise StandardizationError(
            "INVALID_VALUE_DOMAIN", "Matrix feature values must be finite."
        )
    return rows, ids, types


def _options(
    strategy: StandardizationStrategy,
    *,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: tuple[float, float],
) -> tuple[tuple[str, float | bool], ...]:
    if strategy == StandardizationStrategy.STANDARD:
        return (("withMean", with_centering), ("withStd", with_scaling))
    if strategy == StandardizationStrategy.ROBUST:
        low, high = quantile_range
        if not (0.0 <= low < high <= 100.0):
            raise StandardizationError(
                "INVALID_PARAMETERS",
                "Robust quantileRange must satisfy 0 <= low < high <= 100.",
            )
        return (
            ("quantileLow", float(low)),
            ("quantileHigh", float(high)),
            ("withCentering", with_centering),
            ("withScaling", with_scaling),
        )
    if not with_centering or not with_scaling or quantile_range != (25.0, 75.0):
        raise StandardizationError(
            "INVALID_PARAMETERS",
            "This strategy does not accept centering, scaling, or quantile overrides.",
        )
    return ()


def _fit_column(
    column: tuple[float, ...],
    strategy: StandardizationStrategy,
    options: tuple[tuple[str, float | bool], ...],
) -> tuple[tuple[str, float], ...]:
    if strategy == StandardizationStrategy.STANDARD:
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        option_map = dict(options)
        center = mean if option_map["withMean"] else 0.0
        scale = math.sqrt(variance) if option_map["withStd"] and variance > 0.0 else 1.0
        return (("center", center), ("scale", scale), ("variance", variance))
    if strategy == StandardizationStrategy.MIN_MAX:
        minimum = min(column)
        maximum = max(column)
        data_range = maximum - minimum
        return (
            ("dataMin", minimum),
            ("dataMax", maximum),
            ("center", minimum),
            ("scale", data_range if data_range > 0.0 else 1.0),
        )
    if strategy == StandardizationStrategy.MAX_ABS:
        maximum = max(abs(value) for value in column)
        return (("center", 0.0), ("scale", maximum if maximum > 0.0 else 1.0))
    option_map = dict(options)
    low = _percentile(column, float(option_map["quantileLow"]))
    high = _percentile(column, float(option_map["quantileHigh"]))
    median = _percentile(column, 50.0)
    width = high - low
    center = median if option_map["withCentering"] else 0.0
    scale = width if option_map["withScaling"] and width > 0.0 else 1.0
    return (
        ("center", center),
        ("scale", scale),
        ("quantileLow", low),
        ("quantileHigh", high),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _apply(
    rows: tuple[tuple[float, ...], ...], transformer: FittedTransformer
) -> tuple[tuple[float, ...], ...]:
    output: list[tuple[float, ...]] = []
    for row in rows:
        transformed: list[float] = []
        for value, parameters in zip(row, transformer.feature_parameters, strict=True):
            mapping = parameters.mapping
            transformed.append((value - mapping["center"]) / mapping["scale"])
        output.append(tuple(transformed))
    return tuple(output)
