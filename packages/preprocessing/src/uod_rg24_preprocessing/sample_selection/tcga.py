from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidTcgaBarcodeError

_PATIENT_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
# TCGA sample type codes that denote a primary tumour: 01 primary solid tumour,
# 03 primary blood derived cancer (peripheral blood), 09 primary blood derived
# cancer (bone marrow).
_PRIMARY_TUMOUR_SAMPLE_TYPE_CODES = frozenset({1, 3, 9})
_SAMPLE_PATTERN = re.compile(
    r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})([A-Z0-9])"
    r"(?:-([0-9]{2})([A-Z0-9])"
    r"(?:-([A-Z0-9]{4})"
    r"(?:-([A-Z0-9]{2}))?"
    r")?"
    r")?$"
)


@dataclass(frozen=True, slots=True)
class TcgaBarcode:
    original: str
    normalized: str
    patient_id: str
    canonical_sample_id: str | None
    sample_type_code: int | None
    vial: str | None

    @property
    def is_sample(self) -> bool:
        return self.canonical_sample_id is not None

    @property
    def is_primary_tumour(self) -> bool:
        return self.sample_type_code in _PRIMARY_TUMOUR_SAMPLE_TYPE_CODES


def parse_tcga_barcode(value: str) -> TcgaBarcode:
    original = value
    normalized = value.strip().upper()
    if _PATIENT_PATTERN.fullmatch(normalized):
        return TcgaBarcode(
            original=original,
            normalized=normalized,
            patient_id=normalized,
            canonical_sample_id=None,
            sample_type_code=None,
            vial=None,
        )

    match = _SAMPLE_PATTERN.fullmatch(normalized)
    if match is None:
        raise InvalidTcgaBarcodeError(f"Invalid TCGA barcode: {value!r}")

    patient_id, sample_type, vial = match.group(1, 2, 3)
    return TcgaBarcode(
        original=original,
        normalized=normalized,
        patient_id=patient_id,
        canonical_sample_id=f"{patient_id}-{sample_type}{vial}",
        sample_type_code=int(sample_type),
        vial=vial,
    )
