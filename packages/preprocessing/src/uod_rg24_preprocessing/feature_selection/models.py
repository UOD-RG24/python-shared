from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .errors import FeatureSelectionError


class FeatureSelectionStrategy(StrEnum):
    REMOVE_ALL_MISSING = "removeAllMissing"
    REMOVE_CONSTANT = "removeConstant"
    LOW_VARIANCE = "lowVariance"
    TOP_K_BY_VARIANCE = "topKByVariance"
    CORRELATION_FILTER = "correlationFilter"
    TOP_K_BY_ANOVA = "topKByAnova"


class SelectionLearningScope(StrEnum):
    TRAIN_FOLD = "trainFold"
    COHORT_UNSUPERVISED = "cohortUnsupervised"


@dataclass(frozen=True, slots=True)
class FeatureDecision:
    input_position: int
    feature_id: str
    feature_type: str
    selected: bool
    selected_position: int | None
    score: float | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.input_position < 0
            or not self.feature_id.strip()
            or not self.feature_type
        ):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK",
                "Feature-mask rows must identify an input feature.",
            )
        if self.selected != (self.selected_position is not None):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK",
                "Selected features must have exactly one selected position.",
            )
        if self.selected_position is not None and self.selected_position < 0:
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "Selected positions must be non-negative."
            )
        if self.selected and self.rejection_reason is not None:
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK",
                "Selected features cannot have a rejection reason.",
            )
        if not self.selected and not self.rejection_reason:
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "Rejected features require a reason."
            )


@dataclass(frozen=True, slots=True)
class FittedFeatureMask:
    schema_version: str
    strategy: FeatureSelectionStrategy
    recipe_id: str
    recipe_version: str
    modality: str
    source_value_scale: str
    input_feature_ids: tuple[str, ...]
    input_feature_types: tuple[str, ...]
    decisions: tuple[FeatureDecision, ...]
    learning_scope: SelectionLearningScope
    fit_population_artifact_id: str
    fit_population_sha256: str
    input_artifact_sha256: str
    fit_row_count: int
    options: tuple[tuple[str, float | int], ...] = ()
    label_artifact_id: str | None = None
    label_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.strategy == FeatureSelectionStrategy.TOP_K_BY_ANOVA and (
            self.learning_scope != SelectionLearningScope.TRAIN_FOLD
            or not self.label_artifact_id
            or not self.label_artifact_sha256
            or len(self.label_artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.label_artifact_sha256)
        ):
            raise FeatureSelectionError(
                "INVALID_LABEL_ARTIFACT",
                "Supervised masks require training scope and exact label provenance.",
            )
        if self.schema_version != "feature-mask/1.0":
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "Unsupported feature-mask schema version."
            )
        if self.fit_row_count < 1:
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "A fitted mask requires at least one row."
            )
        text = (
            self.recipe_id,
            self.recipe_version,
            self.modality,
            self.source_value_scale,
            self.fit_population_artifact_id,
        )
        if any(not value.strip() for value in text):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "Feature-mask provenance must not be empty."
            )
        if (
            not self.input_feature_ids
            or len(self.input_feature_ids) != len(set(self.input_feature_ids))
            or any(not value.strip() for value in self.input_feature_ids)
            or len(self.input_feature_types) != len(self.input_feature_ids)
            or any(not value.strip() for value in self.input_feature_types)
        ):
            raise FeatureSelectionError(
                "INCOMPATIBLE_FEATURE_SCHEMA",
                "Input feature identifiers and types must be complete and unique.",
            )
        if len(self.decisions) != len(self.input_feature_ids):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "The mask must contain every input feature."
            )
        for position, decision in enumerate(self.decisions):
            if (
                decision.input_position != position
                or decision.feature_id != self.input_feature_ids[position]
                or decision.feature_type != self.input_feature_types[position]
            ):
                raise FeatureSelectionError(
                    "INVALID_FEATURE_MASK",
                    "Mask rows must preserve the exact input feature order.",
                )
        selected = tuple(item for item in self.decisions if item.selected)
        if not selected:
            raise FeatureSelectionError(
                "NO_FEATURES_SELECTED", "At least one feature must be selected."
            )
        if tuple(item.selected_position for item in selected) != tuple(
            range(len(selected))
        ):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK",
                "Selected positions must be contiguous in input feature order.",
            )
        for digest in (self.fit_population_sha256, self.input_artifact_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise FeatureSelectionError(
                    "INVALID_FEATURE_MASK",
                    "Feature-mask hashes must be lowercase SHA-256 values.",
                )
        names = [name for name, _ in self.options]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise FeatureSelectionError(
                "INVALID_FEATURE_MASK", "Feature-mask options must be unique."
            )

    @property
    def option_mapping(self) -> Mapping[str, float | int]:
        return MappingProxyType(dict(self.options))

    @property
    def selected_feature_ids(self) -> tuple[str, ...]:
        return tuple(item.feature_id for item in self.decisions if item.selected)


@dataclass(frozen=True, slots=True)
class FeatureSelectionResult:
    values: tuple[tuple[float | None, ...], ...]
    mask: FittedFeatureMask
