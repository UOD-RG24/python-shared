from importlib.metadata import version

from uod_rg24_preprocessing import (
    HarmonizationRequest,
    SampleCandidate,
    harmonize_samples,
)
from uod_rg24_runtime_azure import OUTPUT_NAMES, serialize_harmonization_reports


def main() -> None:
    for distribution in (
        "uod-rg24-contracts",
        "uod-rg24-preprocessing",
        "uod-rg24-runtime-azure",
    ):
        assert version(distribution) == "0.1.0"

    result = harmonize_samples(
        HarmonizationRequest(
            candidates=[
                SampleCandidate(
                    role="mrna",
                    source_artifact_id="art_mrna",
                    source_sample_id="TCGA-AA-0001-01A",
                ),
                SampleCandidate(
                    role="protein",
                    source_artifact_id="art_protein",
                    source_sample_id="TCGA-AA-0001-01A",
                ),
            ],
            clinical_patient_ids=[],
            clinical_sample_to_patient={},
            cohort_roles=["mrna", "protein"],
        )
    )
    first = serialize_harmonization_reports(result)
    second = serialize_harmonization_reports(result)
    assert tuple(first) == OUTPUT_NAMES
    assert {name: payload.data for name, payload in first.items()} == {
        name: payload.data for name, payload in second.items()
    }


if __name__ == "__main__":
    main()
