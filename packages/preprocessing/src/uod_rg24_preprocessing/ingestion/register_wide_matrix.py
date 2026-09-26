from __future__ import annotations

from collections import Counter

from .errors import (
    AmbiguousModalityError,
    DuplicateFeatureError,
    SourceIntegrityError,
    UnsupportedSourceLayoutError,
)
from .models import (
    ACCEPTED_IDENTIFIER_COLUMNS,
    MODALITY_PREFIXES,
    RESERVED_COLUMNS,
    FeatureNamespace,
    Modality,
    RegistrationParameters,
    RegistrationPlan,
    SourceUpload,
)


def plan_wide_matrix_registration(
    header: list[str],
    parameters: RegistrationParameters,
) -> RegistrationPlan:
    """Decide how a wide, entity-per-row upload becomes a canonical matrix.

    The upload keeps its own identifier column name; the canonical matrix always
    leads with ``entityId`` then ``patientId``. In the patient-level V1 contract
    the two hold the same value, which is what FA2 verifies element-wise.

    This function reads the header only. It never sees a value, so it cannot
    infer a value scale by accident.
    """

    columns = [column.strip() for column in header]
    if len(columns) < 2:
        raise UnsupportedSourceLayoutError(
            "The upload must contain an identifier column and at least one feature."
        )

    identifier_column = columns[0]
    expected = parameters.identifier_column
    if expected is not None:
        if identifier_column != expected:
            raise UnsupportedSourceLayoutError(
                f"The first column is {identifier_column!r}, not the declared "
                f"identifier column {expected!r}."
            )
    elif identifier_column not in ACCEPTED_IDENTIFIER_COLUMNS:
        accepted = ", ".join(repr(name) for name in ACCEPTED_IDENTIFIER_COLUMNS)
        raise UnsupportedSourceLayoutError(
            f"The first column {identifier_column!r} is not a recognised entity "
            f"identifier. Expected one of {accepted}, or declare identifierColumn."
        )

    feature_ids = tuple(columns[1:])
    _reject_blank_features(feature_ids)
    _reject_duplicate_features(feature_ids)
    _reject_reserved_features(feature_ids)

    modality, namespace, warnings = _resolve_modality(feature_ids, parameters)

    return RegistrationPlan(
        source_identifier_column=identifier_column,
        modality=modality,
        value_scale=parameters.value_scale,
        feature_namespace=namespace,
        feature_ids=feature_ids,
        warnings=warnings,
    )


def verify_source_integrity(upload: SourceUpload, *, observed_sha256: str) -> None:
    """Fail before conversion when the blob is not the one the record describes."""

    if observed_sha256 != upload.sha256:
        raise SourceIntegrityError(
            "The stored blob does not match the SHA-256 recorded for the dataset."
        )


def _reject_blank_features(feature_ids: tuple[str, ...]) -> None:
    if any(not feature_id for feature_id in feature_ids):
        raise UnsupportedSourceLayoutError(
            "The upload contains an unnamed feature column."
        )


def _reject_duplicate_features(feature_ids: tuple[str, ...]) -> None:
    duplicates = sorted(
        name for name, count in Counter(feature_ids).items() if count > 1
    )
    if duplicates:
        shown = ", ".join(repr(name) for name in duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f" and {len(duplicates) - 5} more"
        raise DuplicateFeatureError(
            f"The upload contains duplicate feature identifiers: {shown}{suffix}."
        )


def _reject_reserved_features(feature_ids: tuple[str, ...]) -> None:
    collisions = sorted(RESERVED_COLUMNS.intersection(feature_ids))
    if collisions:
        shown = ", ".join(repr(name) for name in collisions)
        raise DuplicateFeatureError(
            f"Feature identifiers collide with canonical identifier columns: {shown}."
        )


def _resolve_modality(
    feature_ids: tuple[str, ...],
    parameters: RegistrationParameters,
) -> tuple[Modality, FeatureNamespace, tuple[str, ...]]:
    observed: set[tuple[Modality, FeatureNamespace]] = set()
    unprefixed = 0
    for feature_id in feature_ids:
        match = _match_prefix(feature_id)
        if match is None:
            unprefixed += 1
        else:
            observed.add(match)

    warnings: list[str] = []
    if unprefixed:
        warnings.append(f"unprefixedFeatures:{unprefixed}")

    declared_modality = parameters.modality
    declared_namespace = parameters.feature_namespace

    if len(observed) > 1:
        modalities = sorted({modality.value for modality, _ in observed})
        if declared_modality is None:
            raise AmbiguousModalityError(
                "The upload mixes feature namespaces for "
                f"{', '.join(modalities)}; declare modality explicitly."
            )
        warnings.append(f"mixedFeatureNamespaces:{len(observed)}")

    inferred = next(iter(observed)) if len(observed) == 1 else None

    if declared_modality is None:
        if inferred is None:
            raise AmbiguousModalityError(
                "The modality could not be inferred from the feature identifiers; "
                "declare modality explicitly."
            )
        modality = inferred[0]
    else:
        modality = declared_modality
        if inferred is not None and inferred[0] is not modality:
            raise AmbiguousModalityError(
                f"The declared modality {modality.value!r} does not match the "
                f"feature prefix {inferred[0].value!r}."
            )

    if declared_namespace is not None:
        namespace = declared_namespace
    elif inferred is not None:
        namespace = inferred[1]
    else:
        raise AmbiguousModalityError(
            "The feature namespace could not be inferred; declare "
            "featureNamespace explicitly."
        )

    return modality, namespace, tuple(warnings)


def _match_prefix(feature_id: str) -> tuple[Modality, FeatureNamespace] | None:
    for prefix, modality, namespace in MODALITY_PREFIXES:
        if feature_id.startswith(prefix) and len(feature_id) > len(prefix):
            return modality, namespace
    return None
