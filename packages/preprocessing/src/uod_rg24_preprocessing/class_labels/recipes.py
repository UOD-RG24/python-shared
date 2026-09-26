from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

from .errors import ClassLabelError
from .models import ClassLabelResult, LabelMapping, MappingStatus, RawClinicalLabel

_MISSING = frozenset(
    {
        "",
        "--",
        "[not available]",
        "not available",
        "not applicable",
        "not reported",
        "unknown",
        "missing",
        "na",
        "n/a",
    }
)
_STAGE_PATTERN = re.compile(r"^(IV|III|II|I)(?:[ABC])?(?:[123])?$")
_TUMOUR = frozenset(
    {
        "primary tumor",
        "primary tumour",
        "recurrent tumor",
        "recurrent tumour",
        "metastatic",
        "additional metastatic",
    }
)
_NORMAL = frozenset(
    {
        "solid tissue normal",
        "blood derived normal",
        "buccal cell normal",
    }
)


def derive_ajcc_stage_4class(
    records: Sequence[RawClinicalLabel],
    *,
    source_field: str = "ajccPathologicStage",
) -> ClassLabelResult:
    """Map AJCC stage groups and retain every raw or unmapped row."""

    return _derive(
        records,
        recipe_id="ajcc-stage-4class-v1",
        source_field=source_field,
        mapper=_map_ajcc,
    )


def derive_tumour_normal(
    records: Sequence[RawClinicalLabel],
    *,
    source_field: str = "sampleType",
) -> ClassLabelResult:
    """Map controlled TCGA-like sample types to Tumour or Normal."""

    return _derive(
        records,
        recipe_id="tumour-normal-v1",
        source_field=source_field,
        mapper=_map_tumour_normal,
    )


def derive_identity_labels(
    records: Sequence[RawClinicalLabel],
    *,
    source_field: str,
    approved_values: Sequence[str],
) -> ClassLabelResult:
    """Validate and retain a caller-selected approved categorical field."""

    allowed = {_normalize(value): value for value in approved_values if value.strip()}
    if not source_field.strip() or not allowed:
        raise ClassLabelError(
            "INVALID_PARAMETERS",
            "Identity labels require a source field and approved values.",
        )

    def mapper(raw: str | None) -> tuple[str | None, MappingStatus, str]:
        normalized = _normalize(raw)
        if normalized in _MISSING:
            return None, MappingStatus.MISSING, "missingSourceValue"
        if normalized not in allowed:
            return None, MappingStatus.UNMAPPED, "valueNotApproved"
        return allowed[normalized], MappingStatus.MAPPED, "approvedIdentityValue"

    return _derive(
        records,
        recipe_id="identity-label-v1",
        source_field=source_field,
        mapper=mapper,
    )


def _derive(
    records: Sequence[RawClinicalLabel],
    *,
    recipe_id: str,
    source_field: str,
    mapper: Callable[[str | None], tuple[str | None, MappingStatus, str]],
) -> ClassLabelResult:
    if not records:
        raise ClassLabelError(
            "EMPTY_CLINICAL_COHORT", "At least one clinical row is required."
        )
    ids = [item.entity_id for item in records]
    if len(ids) != len(set(ids)):
        raise ClassLabelError(
            "DUPLICATE_SAMPLE_ID", "Clinical entity identifiers must be unique."
        )
    labels: list[LabelMapping] = []
    for item in records:
        label, status, reason = mapper(item.raw_value)
        labels.append(
            LabelMapping(
                entity_id=item.entity_id,
                patient_id=item.patient_id,
                raw_value=item.raw_value,
                label_value=label,
                mapping_status=status,
                mapping_reason=reason,
            )
        )
    return ClassLabelResult(
        schema_version="clinical-labels/1.0",
        recipe_id=recipe_id,
        recipe_version="1.0.0",
        source_field=source_field,
        labels=tuple(labels),
    )


def _map_ajcc(raw: str | None) -> tuple[str | None, MappingStatus, str]:
    normalized = _normalize(raw)
    if normalized in _MISSING:
        return None, MappingStatus.MISSING, "missingOrNotReported"
    stage = normalized.upper()
    for prefix in ("PATHOLOGICAL STAGE ", "PATHOLOGIC STAGE ", "AJCC STAGE ", "STAGE "):
        if stage.startswith(prefix):
            stage = stage[len(prefix) :]
            break
    stage = stage.replace(" ", "").replace("-", "")
    match = _STAGE_PATTERN.fullmatch(stage)
    if match is None:
        if stage.startswith("0"):
            return None, MappingStatus.UNMAPPED, "unsupportedStage0"
        return None, MappingStatus.UNMAPPED, "unrecognizedStage"
    group = match.group(1)
    return f"Stage {group}", MappingStatus.MAPPED, "ajccStageGroupCollapsed"


def _map_tumour_normal(raw: str | None) -> tuple[str | None, MappingStatus, str]:
    normalized = _normalize(raw)
    if normalized in _MISSING:
        return None, MappingStatus.MISSING, "missingOrNotReported"
    if normalized in _TUMOUR:
        return "Tumour", MappingStatus.MAPPED, "controlledTumourSampleType"
    if normalized in _NORMAL:
        return "Normal", MappingStatus.MAPPED, "controlledNormalSampleType"
    return None, MappingStatus.UNMAPPED, "unrecognizedSampleType"


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).strip().lower().split())
