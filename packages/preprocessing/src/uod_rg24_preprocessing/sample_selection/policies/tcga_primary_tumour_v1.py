from __future__ import annotations

from dataclasses import dataclass

from ..models import SampleCandidate
from ..tcga import TcgaBarcode

# Ranks a barcode that carries no sample type (a patient-level record) below
# every real sample type, including sample type code 00.
_MISSING_SAMPLE_TYPE_RANK = 999


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    candidate: SampleCandidate
    barcode: TcgaBarcode
    mapping_status: str


def _selection_key(item: ValidatedCandidate) -> tuple[int, int, str, str, str]:
    barcode = item.barcode
    primary_rank = 0 if barcode.is_primary_tumour else 1
    sample_type_rank = (
        _MISSING_SAMPLE_TYPE_RANK
        if barcode.sample_type_code is None
        else barcode.sample_type_code
    )
    vial_rank = barcode.vial or "~"
    return (
        primary_rank,
        sample_type_rank,
        vial_rank,
        barcode.normalized,
        item.candidate.source_artifact_id,
    )


def select_replicate(
    candidates: list[ValidatedCandidate],
) -> tuple[ValidatedCandidate, str, dict[tuple[str, str], str]]:
    ordered = sorted(candidates, key=_selection_key)
    selected = ordered[0]
    if len(ordered) == 1:
        return selected, "onlyCandidate", {}

    rejected: dict[tuple[str, str], str] = {}
    for item in ordered[1:]:
        identity = (
            item.candidate.source_artifact_id,
            item.candidate.source_sample_id or item.barcode.patient_id,
        )
        if selected.barcode.is_primary_tumour and not item.barcode.is_primary_tumour:
            rejected[identity] = "nonPrimaryTumour"
        elif selected.barcode.sample_type_code != item.barcode.sample_type_code:
            rejected[identity] = "higherSampleTypeCode"
        elif selected.barcode.vial != item.barcode.vial:
            rejected[identity] = "higherVial"
        else:
            rejected[identity] = "barcodeTieBreak"

    reasons = set(rejected.values())
    if "nonPrimaryTumour" in reasons:
        reason = "primaryTumourPreferred"
    elif "higherSampleTypeCode" in reasons:
        reason = "lowestSampleTypeCodeSelected"
    elif "higherVial" in reasons:
        reason = "lowestVialSelected"
    else:
        reason = "deterministicBarcodeTieBreak"
    return selected, reason, rejected
