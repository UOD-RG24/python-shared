from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Final

import pyarrow as pa

from ..errors import RuntimeConfigurationError


@dataclass(frozen=True, slots=True)
class LayoutField:
    name: str
    type: str
    nullable: bool


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    output_name: str
    kind: str
    schema_id: str
    media_type: str
    fields: tuple[LayoutField, ...]
    sort_keys: tuple[str, ...]

    @property
    def arrow_schema(self) -> pa.Schema:
        arrow_types = {
            "utf8": pa.string(),
            "bool": pa.bool_(),
            "int64": pa.int64(),
        }
        try:
            fields = [
                pa.field(field.name, arrow_types[field.type], nullable=field.nullable)
                for field in self.fields
            ]
        except KeyError as exc:
            raise RuntimeConfigurationError(
                "Unknown Arrow type in layout descriptor."
            ) from exc
        metadata = {
            b"uod.output_name": self.output_name.encode(),
            b"uod.kind": self.kind.encode(),
            b"uod.schema_id": self.schema_id.encode(),
        }
        return pa.schema(fields, metadata=metadata)


def _load_layout(file_name: str) -> ArtifactLayout:
    resource = files("uod_rg24_runtime_azure").joinpath("layouts", "v1", file_name)
    value = json.loads(resource.read_text(encoding="utf-8"))
    try:
        fields = tuple(
            LayoutField(
                name=field["name"],
                type=field["type"],
                nullable=field["nullable"],
            )
            for field in value["fields"]
        )
        layout = ArtifactLayout(
            output_name=value["outputName"],
            kind=value["kind"],
            schema_id=value["schemaId"],
            media_type=value["mediaType"],
            fields=fields,
            sort_keys=tuple(value["sortKeys"]),
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeConfigurationError("Invalid artifact layout descriptor.") from exc
    names = [field.name for field in fields]
    if len(names) != len(set(names)) or not fields:
        raise RuntimeConfigurationError("Layout fields must be non-empty and unique.")
    if any(key not in names for key in layout.sort_keys):
        raise RuntimeConfigurationError("Layout sort keys must name declared fields.")
    return layout


_LAYOUT_FILES: Final[tuple[str, ...]] = (
    "sample-map.json",
    "availability.json",
    "retained-cohort.json",
    "drop-report.json",
    "aligned-sample-ids.json",
)
_loaded = tuple(_load_layout(file_name) for file_name in _LAYOUT_FILES)
OUTPUT_LAYOUTS: Final[Mapping[str, ArtifactLayout]] = MappingProxyType(
    {layout.output_name: layout for layout in _loaded}
)
if len(OUTPUT_LAYOUTS) != 5:
    raise RuntimeConfigurationError("Exactly five sample-report layouts are required.")

OUTPUT_NAMES: Final[tuple[str, ...]] = tuple(OUTPUT_LAYOUTS)
OUTPUT_SCHEMA_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {name: layout.schema_id for name, layout in OUTPUT_LAYOUTS.items()}
)


def get_layout(output_name: str) -> ArtifactLayout:
    try:
        return OUTPUT_LAYOUTS[output_name]
    except KeyError as exc:
        raise RuntimeConfigurationError(
            f"Unsupported output name: {output_name}"
        ) from exc
