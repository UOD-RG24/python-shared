from __future__ import annotations

import re
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self

from .errors import RuntimeConfigurationError
from .models import ArtifactBundle, ArtifactFile, ArtifactPayload

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StepWorkspace:
    """A private, automatically cleaned workspace for runtime adapters."""

    def __init__(self, *, prefix: str = "uod-rg24-") -> None:
        self._prefix = prefix
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    @property
    def root(self) -> Path:
        if self._temporary_directory is None:
            raise RuntimeError("TemporaryWorkspace has not been entered")
        return Path(self._temporary_directory.name)

    def __enter__(self) -> Self:
        if self._temporary_directory is not None:
            raise RuntimeError("TemporaryWorkspace cannot be entered twice")
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=self._prefix)
        self.root.chmod(0o700)
        return self

    def path(self, logical_name: str) -> Path:
        if _SAFE_NAME.fullmatch(logical_name) is None or logical_name in {".", ".."}:
            raise RuntimeConfigurationError("Workspace file name is not safe.")
        return self.root / logical_name

    def materialize_input_path(self, logical_name: str) -> Path:
        """Allocate one safe input file path without accepting relative paths."""

        return self.path(logical_name)

    def materialize_bundle(
        self,
        *,
        output_name: str,
        data: ArtifactPayload,
        manifest: ArtifactPayload,
        qc: ArtifactPayload,
    ) -> ArtifactBundle:
        """Write exactly the three immutable members of one artifact bundle."""

        if _SAFE_NAME.fullmatch(output_name) is None:
            raise RuntimeConfigurationError("output_name is not safe")
        directory = self.root / output_name
        directory.mkdir(mode=0o700, exist_ok=False)
        payloads = (
            ("data.parquet", data),
            ("manifest.json", manifest),
            ("qc.json", qc),
        )
        files: list[ArtifactFile] = []
        for member_name, payload in payloads:
            path = directory / member_name
            path.write_bytes(payload.data)
            files.append(
                ArtifactFile(
                    member_name=member_name,
                    path=path,
                    output_name=payload.output_name,
                    kind=payload.kind,
                    schema_id=payload.schema_id,
                    media_type=payload.media_type,
                    sha256=payload.sha256,
                    byte_length=payload.byte_length,
                    sample_order_sha256=payload.sample_order_sha256,
                )
            )
        return ArtifactBundle(data=files[0], manifest=files[1], qc=files[2])

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


# Backwards-friendly internal name; StepWorkspace is the public contract.
TemporaryWorkspace = StepWorkspace
