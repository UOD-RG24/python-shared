from __future__ import annotations


class HarmonizationError(ValueError):
    code = "HARMONIZATION_FAILED"


class InvalidTcgaBarcodeError(HarmonizationError):
    code = "SCHEMA_VALIDATION_FAILED"


class ClinicalMappingConflictError(HarmonizationError):
    code = "SAMPLE_MAPPING_AMBIGUOUS"


class DuplicateSampleCandidateError(HarmonizationError):
    code = "DUPLICATE_SAMPLE_ID"


class NoCommonSamplesError(HarmonizationError):
    code = "NO_COMMON_SAMPLES"
