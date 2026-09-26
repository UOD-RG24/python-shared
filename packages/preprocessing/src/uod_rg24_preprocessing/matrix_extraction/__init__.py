"""Deterministic assembly of already processed, aligned modality blocks."""

from .assembly import (
    MatrixBlock,
    MatrixExtractionError,
    MatrixExtractionResult,
    assemble_matrices,
)

__all__ = [
    "MatrixBlock",
    "MatrixExtractionError",
    "MatrixExtractionResult",
    "assemble_matrices",
]
