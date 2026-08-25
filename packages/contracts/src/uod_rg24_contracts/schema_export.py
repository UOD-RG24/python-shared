from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from .models import (
    ArtifactManifestV1,
    ArtifactRef,
    OperationCatalogueV1,
    ProblemDetails,
    QCSummaryV1,
    StepCommandV1,
    StepFailedEventV1,
    StepSucceededEventV1,
)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "artifact-manifest-v1.schema.json": ArtifactManifestV1,
    "artifact-ref.schema.json": ArtifactRef,
    "operation-catalogue-v1.schema.json": OperationCatalogueV1,
    "problem-details.schema.json": ProblemDetails,
    "qc-summary-v1.schema.json": QCSummaryV1,
    "step-command-v1.schema.json": StepCommandV1,
    "step-failed-event-v1.schema.json": StepFailedEventV1,
    "step-succeeded-event-v1.schema.json": StepSucceededEventV1,
}


def default_output_directory() -> Path:
    return Path(__file__).parent / "schemas" / "v1"


def export_schemas(output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for file_name, model in sorted(SCHEMA_MODELS.items()):
        destination = output_directory / file_name
        document = model.model_json_schema(by_alias=True, mode="serialization")
        destination.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the V1 JSON Schemas.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_output_directory(),
    )
    args = parser.parse_args()
    export_schemas(args.output_directory)


if __name__ == "__main__":
    main()
