from __future__ import annotations


class FeatureSelectionError(ValueError):
    """Stable scientific validation error raised by feature selectors."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
