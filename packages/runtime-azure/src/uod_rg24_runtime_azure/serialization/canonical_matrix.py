from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from uod_rg24_contracts import QCSummaryV1
from uod_rg24_preprocessing.ingestion import (
    ENTITY_ID_COLUMN,
    MATRIX_KIND,
    MATRIX_SCHEMA_ID,
    PATIENT_ID_COLUMN,
    RegistrationPlan,
)

from ..errors import SerializationError
from ..hashing import sha256_bytes
from ..json_codec import canonical_json_bytes
from ..models import JSON_MEDIA_TYPE, PARQUET_MEDIA_TYPE, ArtifactPayload
from .sample_reports import PARQUET_WRITE_OPTIONS, QC_KIND, QC_SCHEMA_ID

REGISTERED_MATRIX_OUTPUT_NAME: Final[str] = "matrix"
_METADATA_PREFIX: Final[bytes] = b"uod_rg24."

#: One CSV row of a 38,000-column matrix is roughly 400 KB, so the reader block
#: must comfortably exceed it or Arrow cannot parse a single row.
_CSV_BLOCK_SIZE: Final[int] = 32 * 1024 * 1024


class CanonicalMatrixError(ValueError):
    """A stable, safe registration failure raised while reading values."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class CanonicalMatrixResult:
    table: pa.Table
    row_count: int
    feature_count: int
    value_count: int
    sample_order_sha256: str
    feature_order_sha256: str
    minimum: float
    maximum: float
    warnings: tuple[str, ...] = ()


def build_canonical_matrix(
    source: Path | bytes,
    plan: RegistrationPlan,
    *,
    delimiter: str = ",",
) -> CanonicalMatrixResult:
    """Convert a wide, entity-per-row upload into the canonical V1 matrix.

    The source bytes are never modified; this produces a new artifact beside
    them. Column order, entity order and feature order are taken from the source
    and preserved, then hashed, so a downstream step can verify it received the
    order this registration committed.
    """

    table = _read_source(source, plan, delimiter=delimiter)
    _require_expected_columns(table, plan)

    entity_ids = _identifier_values(table.column(0))
    if not entity_ids:
        raise CanonicalMatrixError(
            "SCHEMA_VALIDATION_FAILED", "The upload contains no entities."
        )
    if len(entity_ids) != len(set(entity_ids)):
        raise CanonicalMatrixError(
            "DUPLICATE_SAMPLE_ID", "The upload contains duplicate entity identifiers."
        )

    policy = plan.value_scale_policy
    columns: list[pa.Array | pa.ChunkedArray] = [
        pa.array(entity_ids, type=pa.string()),
        pa.array(entity_ids, type=pa.string()),
    ]
    minima: list[float] = []
    maxima: list[float] = []
    for position, feature_id in enumerate(plan.feature_ids, start=1):
        values, minimum, maximum = _numeric_feature(
            table.column(position),
            feature_id=feature_id,
            integral=policy.integral,
            allows_negative=policy.allows_negative,
            value_scale=plan.value_scale.value,
        )
        columns.append(values)
        minima.append(minimum)
        maxima.append(maximum)

    schema = pa.schema(
        [
            pa.field(ENTITY_ID_COLUMN, pa.string(), nullable=False),
            pa.field(PATIENT_ID_COLUMN, pa.string(), nullable=False),
            *(
                pa.field(feature_id, pa.float64(), nullable=False)
                for feature_id in plan.feature_ids
            ),
        ],
        metadata=_matrix_metadata(plan),
    )
    canonical = pa.Table.from_arrays(columns, schema=schema)

    feature_count = len(plan.feature_ids)
    row_count = len(entity_ids)
    return CanonicalMatrixResult(
        table=canonical,
        row_count=row_count,
        feature_count=feature_count,
        value_count=row_count * feature_count,
        sample_order_sha256=order_hash(entity_ids),
        feature_order_sha256=order_hash(list(plan.feature_ids)),
        minimum=min(minima),
        maximum=max(maxima),
        warnings=plan.warnings,
    )


def serialize_canonical_matrix(result: CanonicalMatrixResult) -> ArtifactPayload:
    try:
        sink = pa.BufferOutputStream()
        pq.write_table(result.table, sink, **PARQUET_WRITE_OPTIONS)
        data = sink.getvalue().to_pybytes()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise SerializationError(
            "Could not serialize the registered matrix as canonical Parquet."
        ) from exc
    return ArtifactPayload.from_bytes(
        output_name=REGISTERED_MATRIX_OUTPUT_NAME,
        kind=MATRIX_KIND,
        schema_id=MATRIX_SCHEMA_ID,
        media_type=PARQUET_MEDIA_TYPE,
        data=data,
        row_count=result.row_count,
        sample_order_sha256=result.sample_order_sha256,
    )


def serialize_registration_qc(
    result: CanonicalMatrixResult,
    plan: RegistrationPlan,
    *,
    artifact_id: str,
    generated_at: datetime,
    source_sha256: str,
) -> ArtifactPayload:
    qc = QCSummaryV1.model_validate(
        {
            "schemaVersion": QC_SCHEMA_ID,
            "artifactId": artifact_id,
            "generatedAt": generated_at,
            "counts": {
                "rowCount": result.row_count,
                "featureCount": result.feature_count,
                "registeredValueCount": result.value_count,
            },
            "metrics": {
                "minimum": result.minimum,
                "maximum": result.maximum,
            },
            "warnings": list(result.warnings),
            "extensions": _registration_extensions(
                result, plan, source_sha256=source_sha256
            ),
        }
    )
    return ArtifactPayload.from_bytes(
        output_name=f"{REGISTERED_MATRIX_OUTPUT_NAME}.qc",
        kind=QC_KIND,
        schema_id=QC_SCHEMA_ID,
        media_type=JSON_MEDIA_TYPE,
        data=canonical_json_bytes(qc),
    )


def order_hash(values: list[str]) -> str:
    """Hash an ordered identifier list exactly as FA2 and FA3 do."""

    return sha256_bytes("".join(f"{value}\n" for value in values).encode())


def _read_source(
    source: Path | bytes,
    plan: RegistrationPlan,
    *,
    delimiter: str,
) -> pa.Table:
    read_options = pacsv.ReadOptions(block_size=_CSV_BLOCK_SIZE)
    parse_options = pacsv.ParseOptions(delimiter=delimiter)
    convert_options = pacsv.ConvertOptions(
        column_types={plan.source_identifier_column: pa.string()}
    )
    # read_csv accepts a path or any Arrow NativeFile, so in-memory bytes and a
    # materialized blob take the same code path.
    handle = pa.BufferReader(source) if isinstance(source, bytes) else source
    try:
        return pacsv.read_csv(
            handle,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise CanonicalMatrixError(
            "SCHEMA_VALIDATION_FAILED",
            "The uploaded blob is not readable delimited text.",
        ) from exc


def _require_expected_columns(table: pa.Table, plan: RegistrationPlan) -> None:
    expected = [plan.source_identifier_column, *plan.feature_ids]
    actual = [name.strip() for name in table.column_names]
    if actual != expected:
        raise CanonicalMatrixError(
            "SCHEMA_VALIDATION_FAILED",
            "The uploaded columns do not match the registration plan.",
        )


def _identifier_values(column: pa.ChunkedArray) -> list[str]:
    if column.null_count or not (
        pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
    ):
        raise CanonicalMatrixError(
            "SCHEMA_VALIDATION_FAILED",
            "Entity identifiers must be non-null strings.",
        )
    values = [str(value).strip() for value in column.to_pylist()]
    if any(not value for value in values):
        raise CanonicalMatrixError(
            "SCHEMA_VALIDATION_FAILED", "Entity identifiers must not be empty."
        )
    return values


def _numeric_feature(
    column: pa.ChunkedArray,
    *,
    feature_id: str,
    integral: bool,
    allows_negative: bool,
    value_scale: str,
) -> tuple[pa.ChunkedArray, float, float]:
    # A wholly empty column infers as Arrow null rather than a numeric type, so
    # report it as the missing data it is instead of a type mismatch.
    if pa.types.is_null(column.type):
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Feature {feature_id!r} contains no values.",
        )
    if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
        raise CanonicalMatrixError(
            "INCOMPATIBLE_FEATURE_SCHEMA",
            f"Feature {feature_id!r} is not numeric.",
        )
    if column.null_count:
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Feature {feature_id!r} contains missing values.",
        )

    source_is_integer = pa.types.is_integer(column.type)
    try:
        values = pc.cast(column, pa.float64(), safe=True)
    except pa.ArrowException as exc:
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Feature {feature_id!r} contains values outside the supported domain.",
        ) from exc

    bounds = pc.min_max(values).as_py()
    minimum = bounds["min"]
    maximum = bounds["max"]
    if (
        minimum is None
        or maximum is None
        or not math.isfinite(minimum)
        or not math.isfinite(maximum)
    ):
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Feature {feature_id!r} contains non-finite values.",
        )
    if not allows_negative and minimum < 0.0:
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Value scale {value_scale!r} does not permit the negative values in "
            f"feature {feature_id!r}.",
        )
    # An integer source column is integral for free; only a float column has to
    # be proved, and that check is one vectorized pass.
    if (
        integral
        and not source_is_integer
        and not bool(pc.all(pc.equal(values, pc.round(values))).as_py())
    ):
        raise CanonicalMatrixError(
            "INVALID_VALUE_DOMAIN",
            f"Value scale {value_scale!r} requires whole numbers, but feature "
            f"{feature_id!r} contains fractional values.",
        )
    return values, float(minimum), float(maximum)


def _matrix_metadata(plan: RegistrationPlan) -> dict[bytes, bytes]:
    return {
        _METADATA_PREFIX + b"schemaId": MATRIX_SCHEMA_ID.encode(),
        _METADATA_PREFIX + b"modality": plan.modality.value.encode(),
        _METADATA_PREFIX + b"valueScale": plan.value_scale.value.encode(),
        _METADATA_PREFIX + b"entityIdColumn": ENTITY_ID_COLUMN.encode(),
        _METADATA_PREFIX + b"featureNamespace": plan.feature_namespace.value.encode(),
    }


def _registration_extensions(
    result: CanonicalMatrixResult,
    plan: RegistrationPlan,
    *,
    source_sha256: str,
) -> Mapping[str, object]:
    return {
        "outputName": REGISTERED_MATRIX_OUTPUT_NAME,
        "modality": plan.modality.value,
        "valueScale": plan.value_scale.value,
        "featureNamespace": plan.feature_namespace.value,
        "sourceIdentifierColumn": plan.source_identifier_column,
        "sourceSha256": source_sha256,
        "sampleOrderSha256": result.sample_order_sha256,
        "featureOrderSha256": result.feature_order_sha256,
    }
