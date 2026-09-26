from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .errors import StandardizationError


class StandardizationStrategy(StrEnum):
    STANDARD = "standard"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"


class LearningScope(StrEnum):
    TRAIN_FOLD = "trainFold"
    COHORT_UNSUPERVISED = "cohortUnsupervised"


@dataclass(frozen=True, slots=True)
class FeatureParameters:
    feature_id: str
    values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.feature_id.strip():
            raise StandardizationError(
                "INCOMPATIBLE_FEATURE_SCHEMA", "Feature identifiers must not be empty."
            )
        names = [name for name, _ in self.values]
        if not names or len(names) != len(set(names)):
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER",
                "Each feature must contain a unique, non-empty parameter set.",
            )
        if any(not name.strip() for name in names):
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER",
                "Transformer parameter names must not be empty.",
            )

    @property
    def mapping(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.values))


@dataclass(frozen=True, slots=True)
class FittedTransformer:
    schema_version: str
    strategy: StandardizationStrategy
    recipe_id: str
    recipe_version: str
    modality: str
    source_value_scale: str
    feature_ids: tuple[str, ...]
    feature_types: tuple[str, ...]
    feature_parameters: tuple[FeatureParameters, ...]
    learning_scope: LearningScope
    fit_population_artifact_id: str
    fit_population_sha256: str
    input_artifact_sha256: str
    fit_row_count: int
    options: tuple[tuple[str, float | bool], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "fitted-transformer/1.0":
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER", "Unsupported transformer schema version."
            )
        for value in (
            self.recipe_id,
            self.recipe_version,
            self.modality,
            self.source_value_scale,
            self.fit_population_artifact_id,
        ):
            if not value.strip():
                raise StandardizationError(
                    "INVALID_FITTED_TRANSFORMER",
                    "Transformer provenance values must not be empty.",
                )
        if self.fit_row_count < 1:
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER",
                "The fitted population must contain at least one row.",
            )
        if not self.feature_ids or len(self.feature_ids) != len(set(self.feature_ids)):
            raise StandardizationError(
                "INCOMPATIBLE_FEATURE_SCHEMA",
                "Feature identifiers must be non-empty and unique.",
            )
        if any(not value.strip() for value in (*self.feature_ids, *self.feature_types)):
            raise StandardizationError(
                "INCOMPATIBLE_FEATURE_SCHEMA",
                "Feature identifiers and types must not be empty.",
            )
        if not (
            len(self.feature_ids)
            == len(self.feature_types)
            == len(self.feature_parameters)
        ):
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER",
                "Feature schema and fitted parameters must have equal lengths.",
            )
        if (
            tuple(item.feature_id for item in self.feature_parameters)
            != self.feature_ids
        ):
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER",
                "Fitted parameters must use the exact feature order.",
            )
        for digest in (self.fit_population_sha256, self.input_artifact_sha256):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise StandardizationError(
                    "INVALID_FITTED_TRANSFORMER",
                    "Transformer hashes must contain 64 lowercase hexadecimal characters.",
                )
        option_names = [name for name, _ in self.options]
        if len(option_names) != len(set(option_names)):
            raise StandardizationError(
                "INVALID_FITTED_TRANSFORMER", "Transformer options must be unique."
            )

    @property
    def option_mapping(self) -> Mapping[str, float | bool]:
        return MappingProxyType(dict(self.options))


@dataclass(frozen=True, slots=True)
class StandardizationResult:
    values: tuple[tuple[float, ...], ...]
    transformer: FittedTransformer
