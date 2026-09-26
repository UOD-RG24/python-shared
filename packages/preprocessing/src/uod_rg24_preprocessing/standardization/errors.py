from __future__ import annotations


class StandardizationError(ValueError):
    """A stable validation or fitted-state failure from the scientific core."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
