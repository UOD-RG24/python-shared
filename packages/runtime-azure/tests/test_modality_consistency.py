from __future__ import annotations

from uod_rg24_contracts import Modality as ContractModality
from uod_rg24_preprocessing.ingestion import Modality as IngestionModality


def test_every_ingestible_modality_can_be_written_to_a_manifest() -> None:
    """The ingestion enum is the narrower one and must stay a subset.

    ``uod_rg24_preprocessing`` deliberately does not depend on
    ``uod_rg24_contracts``, so the two enums are declared separately. Only this
    package sees both, which makes it the one place the drift can be caught. A
    modality that registration accepts but a manifest rejects would fail at the
    Blob write, after the conversion work is already done.
    """

    contract_values = {modality.value for modality in ContractModality}
    ingestion_values = {modality.value for modality in IngestionModality}

    assert ingestion_values <= contract_values, (
        "ingestion accepts modalities that ArtifactManifestV1 would reject: "
        f"{sorted(ingestion_values - contract_values)}"
    )


def test_the_live_cohort_modalities_are_representable() -> None:
    for value in ("mrna", "cna"):
        assert ContractModality(value).value == value
        assert IngestionModality(value).value == value
