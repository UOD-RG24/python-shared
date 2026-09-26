from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256


class MatrixExtractionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class MatrixBlock:
    modality: str
    entity_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    feature_types: tuple[str, ...]
    values: tuple[tuple[float | int | None, ...], ...]
    matrix_artifact_id: str
    mask_artifact_id: str
    value_scale: str


@dataclass(frozen=True, slots=True)
class MatrixExtractionResult:
    blocks: tuple[MatrixBlock, ...]
    entity_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    combined_feature_ids: tuple[str, ...]
    combined_values: tuple[tuple[float | int | None, ...], ...]
    sample_order_sha256: str
    feature_order_sha256: str


def order_hash(ids: Sequence[str]) -> str:
    return sha256("".join(value + "\n" for value in ids).encode()).hexdigest()


def assemble_matrices(
    blocks: Sequence[MatrixBlock],
    *,
    entity_ids: Sequence[str],
    patient_ids: Sequence[str],
) -> MatrixExtractionResult:
    """Validate exact FA3 row order and concatenate in caller-declared block order."""
    entities, patients = tuple(entity_ids), tuple(patient_ids)
    if (
        not entities
        or len(entities) != len(set(entities))
        or len(patients) != len(entities)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in (*entities, *patients)
        )
    ):
        raise MatrixExtractionError(
            "INVALID_COHORT", "Cohort identifiers must be complete and unique."
        )
    if not blocks or len({block.modality for block in blocks}) != len(blocks):
        raise MatrixExtractionError(
            "INVALID_MODALITIES", "Supply one selected block per modality."
        )
    combined_ids: list[str] = []
    for block in blocks:
        if (
            block.modality not in {"mrna", "protein", "cna"}
            or not block.value_scale.strip()
        ):
            raise MatrixExtractionError(
                "INVALID_MODALITIES", "Declare a supported modality and value scale."
            )
        if block.entity_ids != entities or block.patient_ids != patients:
            raise MatrixExtractionError(
                "SAMPLE_ALIGNMENT_MISMATCH",
                "Matrix rows must match the exact FA3 cohort order.",
            )
        if (
            not block.feature_ids
            or len(block.feature_ids) != len(set(block.feature_ids))
            or len(block.feature_ids) != len(block.feature_types)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in block.feature_ids
            )
        ):
            raise MatrixExtractionError(
                "INVALID_FEATURE_SCHEMA",
                "Selected features must be complete and unique.",
            )
        if len(block.values) != len(entities) or any(
            len(row) != len(block.feature_ids) for row in block.values
        ):
            raise MatrixExtractionError(
                "INVALID_MATRIX_SHAPE", "Matrix shape does not match its identifiers."
            )
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            for row in block.values
            for value in row
        ):
            raise MatrixExtractionError(
                "INVALID_VALUE_DOMAIN",
                "Extracted matrices require finite values; impute explicitly upstream.",
            )
        for feature in block.feature_ids:
            prefix = feature.split("::", 1)[0]
            if (
                prefix in {"mrna", "protein", "cna", "clinical"}
                and prefix != block.modality
            ):
                raise MatrixExtractionError(
                    "INVALID_FEATURE_NAMESPACE",
                    "Feature namespace conflicts with its modality.",
                )
            combined_ids.append(
                feature
                if feature.startswith(block.modality + "::")
                else f"{block.modality}::{feature}"
            )
    if len(combined_ids) != len(set(combined_ids)):
        raise MatrixExtractionError(
            "DUPLICATE_FEATURE_ID", "Combined feature identifiers collide."
        )
    combined = tuple(
        tuple(value for block in blocks for value in block.values[position])
        for position in range(len(entities))
    )
    return MatrixExtractionResult(
        tuple(blocks),
        entities,
        patients,
        tuple(combined_ids),
        combined,
        order_hash(entities),
        order_hash(combined_ids),
    )
