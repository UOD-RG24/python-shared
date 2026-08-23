from __future__ import annotations

from datetime import UTC, datetime

import pytest
from uod_rg24_contracts import ArtifactManifestV1, StepCommandV1
from uod_rg24_preprocessing import (
    HarmonizationRequest,
    HarmonizationResult,
    SampleCandidate,
    harmonize_samples,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
TRACE_ID = "1" * 32
TRACEPARENT = f"00-{TRACE_ID}-{'2' * 16}-01"
IMAGE_DIGEST = f"sha256:{'c' * 64}"


@pytest.fixture
def harmonization_result() -> HarmonizationResult:
    def candidate(role: str, sample_id: str) -> SampleCandidate:
        return SampleCandidate(
            role=role,
            source_artifact_id=f"art_{role}",
            source_sample_id=sample_id,
        )

    return harmonize_samples(
        HarmonizationRequest(
            candidates=[
                candidate("mrna", "TCGA-AA-0001-01B"),
                candidate("mrna", "TCGA-AA-0001-01A"),
                candidate("mrna", "TCGA-AA-0002-01A"),
                candidate("protein", "TCGA-AA-0001-01A"),
                candidate("protein", "TCGA-AA-0003-01A"),
            ],
            clinical_patient_ids=[
                "TCGA-AA-0001",
                "TCGA-AA-0002",
                "TCGA-AA-0003",
            ],
            clinical_sample_to_patient={
                "TCGA-AA-0001-01A": "TCGA-AA-0001",
                "TCGA-AA-0001-01B": "TCGA-AA-0001",
                "TCGA-AA-0002-01A": "TCGA-AA-0002",
                "TCGA-AA-0003-01A": "TCGA-AA-0003",
            },
            cohort_roles=["mrna", "protein"],
        )
    )


@pytest.fixture
def step_command() -> StepCommandV1:
    return StepCommandV1.model_validate(
        {
            "schemaVersion": "1.0",
            "messageId": "msg_01K",
            "runId": "run_01K",
            "stepRunId": "step_01K",
            "attempt": 1,
            "experimentId": "exp_01K",
            "datasetId": "ds_01K",
            "ownerId": "usr_01K",
            "layer": "sampleSelection",
            "operation": "testOperation",
            "inputs": [
                {
                    "role": "matrix",
                    "artifactId": "art_input",
                    "expectedSha256": HASH_A,
                    "expectedSchemaId": "omics-wide-matrix/1.0",
                }
            ],
            "outputs": [
                {
                    "name": "sampleMap",
                    "artifactId": "art_output",
                    "kind": "sampleMap",
                    "schemaId": "sample-map/1.0",
                }
            ],
            "parameters": {"cohortPolicy": "intersection"},
            "executionContext": {
                "recipeId": "tcga-primary-tumour-v1",
                "recipeVersion": "1.0.0",
                "randomSeed": 1729,
                "requestedAt": "2026-08-22T12:00:00Z",
                "traceparent": TRACEPARENT,
            },
        }
    )


@pytest.fixture
def artifact_manifest() -> ArtifactManifestV1:
    return ArtifactManifestV1.model_validate(
        {
            "schemaVersion": "artifact-manifest/1.0",
            "artifactId": "art_output",
            "kind": "sampleMap",
            "modality": None,
            "datasetId": "ds_01K",
            "experimentId": "exp_01K",
            "runId": "run_01K",
            "stepRunId": "step_01K",
            "createdAt": "2026-08-23T10:00:00Z",
            "createdBy": {
                "service": "pp-fa3-sample-selection",
                "codeVersion": "2.0.0",
                "containerImageDigest": IMAGE_DIGEST,
                "recipeId": "tcga-primary-tumour-v1",
                "recipeVersion": "1.0.0",
            },
            "inputs": [{"artifactId": "art_input", "sha256": HASH_A, "role": "matrix"}],
            "parameters": {"cohortPolicy": "intersection"},
            "data": {
                "mediaType": "application/vnd.apache.parquet",
                "sha256": HASH_B,
                "byteLength": 123,
                "rowCount": 1,
                "featureCount": None,
                "schemaId": "sample-map/1.0",
                "sampleOrderSha256": HASH_A,
                "featureOrderSha256": None,
            },
            "learning": {
                "learnsParameters": False,
                "learningScope": "none",
                "fitPopulationArtifactId": None,
                "fittedArtifactId": None,
            },
            "randomSeed": 1729,
            "warnings": [],
        }
    )


@pytest.fixture
def utc_now() -> datetime:
    return datetime(2026, 8, 23, 10, 0, tzinfo=UTC)
