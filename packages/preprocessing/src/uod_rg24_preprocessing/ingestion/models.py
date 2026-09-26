from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic.alias_generators import to_camel


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


NonEmptyStr = Annotated[str, StringConstraints(min_length=1, pattern=r".*\S.*")]
StrippedStr = Annotated[str, BeforeValidator(_strip), StringConstraints(min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

ENTITY_ID_COLUMN: Final[str] = "entityId"
PATIENT_ID_COLUMN: Final[str] = "patientId"
SAMPLE_ID_COLUMN: Final[str] = "sampleId"
RESERVED_COLUMNS: Final[frozenset[str]] = frozenset(
    {ENTITY_ID_COLUMN, PATIENT_ID_COLUMN, SAMPLE_ID_COLUMN}
)

MATRIX_SCHEMA_ID: Final[str] = "omics-wide-matrix/1.0"
MATRIX_KIND: Final[str] = "omicsMatrix"

#: Source identifier columns accepted as the entity key of a wide upload.
#: ``patient_id`` is what every live ``core-wa1-api`` upload uses.
ACCEPTED_IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "patient_id",
    "patientId",
    "entityId",
)


class Modality(StrEnum):
    MRNA = "mrna"
    CNA = "cna"
    PROTEIN = "protein"
    CLINICAL = "clinical"


class FeatureNamespace(StrEnum):
    """How a feature identifier encodes its modality.

    ``doubleUnderscore`` is ``<modality>__<sourceId>`` and is the namespace every
    live artifact and every ``dtmi:com:patient;1`` property uses. ``colon`` is
    the ``<modality>::<annotation>::<sourceId>`` form in the V1 contracts. The
    two are not interchangeable and the choice is recorded in every manifest.
    """

    DOUBLE_UNDERSCORE = "doubleUnderscore"
    COLON = "colon"


#: Modality prefix to namespace, longest prefix first so ``mrna__`` cannot be
#: mistaken for a bare ``mrna::`` match.
MODALITY_PREFIXES: Final[tuple[tuple[str, Modality, FeatureNamespace], ...]] = tuple(
    (f"{modality.value}{separator}", modality, namespace)
    for modality in Modality
    for separator, namespace in (
        ("__", FeatureNamespace.DOUBLE_UNDERSCORE),
        ("::", FeatureNamespace.COLON),
    )
)


class ValueScale(StrEnum):
    FPKM = "FPKM"
    LOG2_FPKM_PLUS_1 = "log2(FPKM+1)"
    DISCRETE_COPY_NUMBER = "discreteCopyNumber"
    Z_SCORE = "zScore"
    LOG2_VALUE = "LOG2-VALUE"


class ValueScalePolicy(BaseModel):
    """What a declared value scale permits in the numeric feature block.

    The declared scale governs validation. Values are never inspected to guess a
    scale, per ``docs/plan/api-and-data-contracts.md``.
    """

    model_config = ConfigDict(frozen=True)

    allows_negative: bool
    integral: bool


VALUE_SCALE_POLICIES: Final[dict[ValueScale, ValueScalePolicy]] = {
    ValueScale.FPKM: ValueScalePolicy(allows_negative=False, integral=False),
    ValueScale.LOG2_FPKM_PLUS_1: ValueScalePolicy(
        allows_negative=False, integral=False
    ),
    ValueScale.DISCRETE_COPY_NUMBER: ValueScalePolicy(
        allows_negative=True, integral=True
    ),
    ValueScale.Z_SCORE: ValueScalePolicy(allows_negative=True, integral=False),
    ValueScale.LOG2_VALUE: ValueScalePolicy(allows_negative=True, integral=False),
}


class IngestionModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        frozen=True,
    )


class SourceUpload(IngestionModel):
    """The ``core-wa1-api`` dataset record and the blob it points at.

    Every field is taken from server state. A dashboard cannot supply any of
    them, and none of them names a storage account, container, or path that the
    caller chose.
    """

    owner_id: NonEmptyStr
    dataset_id: NonEmptyStr
    container_name: NonEmptyStr
    blob_path: NonEmptyStr
    original_file_name: NonEmptyStr
    stored_file_name: NonEmptyStr
    file_type: StrippedStr
    content_type: str | None = None
    sha256: Sha256Hex
    byte_length: int = Field(ge=0)
    etag: NonEmptyStr
    version_id: str | None = None
    omics_type: str | None = None

    @model_validator(mode="after")
    def _check_blob_path(self) -> SourceUpload:
        segments = self.blob_path.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise ValueError("blobPath is not a canonical blob name")
        if not self.blob_path.endswith(self.stored_file_name):
            raise ValueError("blobPath must end with storedFileName")
        return self

    @model_validator(mode="after")
    def _check_file_type(self) -> SourceUpload:
        if self.file_type.lower().lstrip(".") not in {"csv", "tsv"}:
            raise ValueError("only csv and tsv uploads can be registered")
        return self

    @property
    def delimiter(self) -> str:
        return "\t" if self.file_type.lower().lstrip(".") == "tsv" else ","


class RegistrationParameters(IngestionModel):
    """Operator-declared facts that values alone must never be used to infer."""

    value_scale: ValueScale
    modality: Modality | None = None
    identifier_column: StrippedStr | None = None
    feature_namespace: FeatureNamespace | None = None


class RegistrationPlan(IngestionModel):
    """The validated decision about how one upload becomes a canonical matrix.

    It is produced without reading a single value, so it can be logged, tested
    and reviewed before any bytes are converted.
    """

    source_identifier_column: NonEmptyStr
    modality: Modality
    value_scale: ValueScale
    feature_namespace: FeatureNamespace
    feature_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    warnings: tuple[str, ...] = ()

    @property
    def identifier_columns(self) -> tuple[str, str]:
        return (ENTITY_ID_COLUMN, PATIENT_ID_COLUMN)

    @property
    def canonical_column_names(self) -> tuple[str, ...]:
        return (*self.identifier_columns, *self.feature_ids)

    @property
    def value_scale_policy(self) -> ValueScalePolicy:
        return VALUE_SCALE_POLICIES[self.value_scale]
