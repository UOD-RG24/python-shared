from __future__ import annotations


class ClassLabelError(ValueError):
    """Stable scientific validation error raised by clinical label recipes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
