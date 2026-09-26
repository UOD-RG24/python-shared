from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from uod_rg24_contracts import (
    COHORT_UNSUPERVISED_LEAKAGE_WARNING,
    ArtifactManifestV1,
    ArtifactRef,
    ManifestLearning,
    OperationCatalogueV1,
    ProblemDetails,
    QCSummaryV1,
    StepCommandV1,
    StepFailedEventV1,
    StepSucceededEventV1,
)
from uod_rg24_contracts.schema_export import SCHEMA_MODELS, export_schemas

HASH_A = "a" * 64
HASH_B = "b" * 64
TRACE_ID = "1" * 32
IMAGE_DIGEST = f"sha256:{'c' * 64}"
TRACEPARENT = f"00-{TRACE_ID}-{'2' * 16}-01"


def command_payload() -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "messageId": "msg_01K",
        "runId": "run_01K",
        "stepRunId": "step_01K",
        "attempt": 1,
        "experimentId": "exp_01K",
        "datasetId": "ds_01K",
        "ownerId": "usr_01K",
        "layer": "standardization",
        "operation": "standard",
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
                "name": "matrix",
                "artifactId": "art_output",
                "kind": "omicsMatrix",
                "schemaId": "omics-wide-matrix/1.0",
            }
        ],
        "parameters": {"mode": "fitTransform", "withMean": True},
        "executionContext": {
            "recipeId": "standard-scaler-v1",
            "recipeVersion": "1.0.0",
            "randomSeed": 1729,
            "requestedAt": "2026-08-22T12:00:00Z",
            "traceparent": TRACEPARENT,
        },
    }


def test_command_example_round_trips_using_wire_aliases() -> None:
    command = StepCommandV1.model_validate(command_payload())

    dumped = command.model_dump(mode="json")

    assert dumped["schemaVersion"] == "1.0"
    assert dumped["executionContext"]["randomSeed"] == 1729
    assert dumped["outputs"][0]["schemaId"] == "omics-wide-matrix/1.0"
    assert "schema_version" not in dumped


def test_commands_reject_unknown_fields_and_duplicate_roles() -> None:
    unknown = command_payload()
    unknown["storageAccount"] = "caller-controlled"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StepCommandV1.model_validate(unknown)

    duplicate = command_payload()
    duplicate["inputs"] = [duplicate["inputs"][0], duplicate["inputs"][0]]  # type: ignore[index]
    with pytest.raises(ValidationError, match="input roles must be unique"):
        StepCommandV1.model_validate(duplicate)

    aliased_output = command_payload()
    aliased_output["outputs"][0]["artifactId"] = "art_input"  # type: ignore[index]
    with pytest.raises(ValidationError, match="must not alias input artifacts"):
        StepCommandV1.model_validate(aliased_output)

    missing_output_schema = command_payload()
    del missing_output_schema["outputs"][0]["schemaId"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="schemaId"):
        StepCommandV1.model_validate(missing_output_schema)


def test_wire_models_reject_python_field_names() -> None:
    payload = command_payload()
    payload["schema_version"] = payload.pop("schemaVersion")

    with pytest.raises(ValidationError):
        StepCommandV1.model_validate(payload)


def test_wire_models_do_not_coerce_scalar_types() -> None:
    payload = command_payload()
    payload["attempt"] = "1"

    with pytest.raises(ValidationError):
        StepCommandV1.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schemaVersion",), "2.0"),
        (("inputs", 0, "expectedSha256"), "ABC"),
        (("executionContext", "requestedAt"), "2026-08-22T13:00:00+01:00"),
        (("executionContext", "requestedAt"), 0),
    ],
)
def test_command_rejects_wrong_version_hash_or_non_utc_timestamp(
    path: tuple[str | int, ...], value: object
) -> None:
    payload = command_payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        StepCommandV1.model_validate(payload)


@pytest.mark.parametrize(
    "traceparent",
    [
        f"00-{'0' * 32}-{'2' * 16}-01",
        f"00-{TRACE_ID}-{'0' * 16}-01",
    ],
)
def test_command_rejects_all_zero_w3c_trace_identifiers(traceparent: str) -> None:
    payload = command_payload()
    payload["executionContext"]["traceparent"] = traceparent  # type: ignore[index]

    with pytest.raises(ValidationError):
        StepCommandV1.model_validate(payload)


def test_artifact_ref_accepts_only_declared_extensions() -> None:
    payload = {
        "artifactId": "art_01K",
        "datasetId": "ds_01K",
        "kind": "omicsMatrix",
        "modality": "mrna",
        "schemaId": "omics-wide-matrix/1.0",
        "state": "committed",
        "sha256": HASH_A,
        "createdAt": "2026-08-22T12:04:03Z",
        "extensions": {"source": "GDC"},
    }
    artifact = ArtifactRef.model_validate(payload)
    assert artifact.extensions == {"source": "GDC"}

    payload["blobPath"] = "must/not/leak"
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(payload)


def test_succeeded_event_matches_documented_identity_envelope() -> None:
    event = StepSucceededEventV1.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": "evt_01K",
            "eventType": "preprocessing.step.succeeded",
            "occurredAt": "2026-08-22T12:04:03Z",
            "runId": "run_01K",
            "stepRunId": "step_01K",
            "attempt": 1,
            "messageId": "msg_01K",
            "traceId": TRACE_ID,
            "layer": "standardization",
            "operation": "standard",
            "codeVersion": "pp-fa1-standardization@2.0.0",
            "containerImageDigest": IMAGE_DIGEST,
            "outputs": [
                {"name": "matrix", "artifactId": "art_output", "sha256": HASH_B}
            ],
            "metrics": {
                "durationMs": 243000,
                "inputRows": 280,
                "outputRows": 280,
                "bytesRead": 123,
                "bytesWritten": 456,
            },
            "warnings": [],
        }
    )
    assert event.trace_id == TRACE_ID


def problem_payload() -> dict[str, object]:
    return {
        "type": "https://genome.example/problems/schema-validation-failed",
        "title": "Schema validation failed",
        "status": 422,
        "code": "SCHEMA_VALIDATION_FAILED",
        "detail": "The input matrix did not match its declared schema.",
        "traceId": TRACE_ID,
        "errors": [
            {
                "path": "inputs[0]",
                "reason": "unexpectedSchemaId",
                "count": 1,
            }
        ],
    }


def failed_event_payload() -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "eventId": "evt_failed_01K",
        "eventType": "preprocessing.step.failed",
        "occurredAt": "2026-08-22T12:04:03Z",
        "runId": "run_01K",
        "stepRunId": "step_01K",
        "attempt": 1,
        "messageId": "msg_01K",
        "traceId": TRACE_ID,
        "layer": "sampleSelection",
        "operation": "harmonizeSamples",
        "codeVersion": "pp-fa3-sample-selection@2.0.0",
        "containerImageDigest": IMAGE_DIGEST,
        "problem": problem_payload(),
        "retryable": False,
        "metrics": {
            "durationMs": 120,
            "inputRows": 487,
            "bytesRead": 123,
            "bytesWritten": 0,
        },
    }


def test_failed_event_round_trips_problem_details_without_coercion() -> None:
    event = StepFailedEventV1.model_validate(failed_event_payload())

    dumped = event.model_dump(mode="json")

    assert dumped["eventType"] == "preprocessing.step.failed"
    assert dumped["problem"]["traceId"] == TRACE_ID
    assert dumped["problem"]["errors"][0] == {
        "path": "inputs[0]",
        "reason": "unexpectedSchemaId",
        "count": 1,
        "extensions": {},
    }
    assert dumped["retryable"] is False

    invalid = failed_event_payload()
    invalid["retryable"] = "false"
    with pytest.raises(ValidationError):
        StepFailedEventV1.model_validate(invalid)


def test_qc_summary_enforces_strict_nonnegative_finite_values() -> None:
    payload: dict[str, object] = {
        "schemaVersion": "artifact-qc/1.0",
        "artifactId": "art_sample_map",
        "generatedAt": "2026-08-22T12:04:03Z",
        "counts": {"retainedPatients": 361, "droppedPatients": 4},
        "metrics": {"retainedFraction": 0.989, "candidateCount": 365},
        "warnings": ["patientLevelFallback:1"],
    }
    summary = QCSummaryV1.model_validate(payload)
    assert summary.counts["retainedPatients"] == 361
    assert summary.metrics["retainedFraction"] == 0.989

    for invalid_counts in (
        {"retainedPatients": -1},
        {"retainedPatients": "361"},
    ):
        with pytest.raises(ValidationError):
            QCSummaryV1.model_validate({**payload, "counts": invalid_counts})

    with pytest.raises(ValidationError):
        QCSummaryV1.model_validate(
            {**payload, "metrics": {"retainedFraction": float("inf")}}
        )


def operation_catalogue_payload() -> dict[str, object]:
    return {
        "schemaVersion": "operation-catalogue/1.0",
        "operations": [
            {
                "layer": "sampleSelection",
                "operation": "testOperation",
                "inputs": [
                    {
                        "role": "mrna",
                        "kind": "omicsMatrix",
                        "acceptedSchemaIds": ["omics-wide-matrix/1.0"],
                        "required": True,
                    }
                ],
                "outputs": [
                    {
                        "name": "sampleMap",
                        "kind": "sampleMap",
                        "schemaId": "sample-map/1.0",
                    }
                ],
                "parameterSchema": {
                    "type": "object",
                    "additionalProperties": False,
                },
                "learnsParameters": False,
                "allowedLearningScopes": ["none"],
                "workloadClass": "standard",
            }
        ],
    }


def test_operation_catalogue_declares_input_and_output_schema_ids() -> None:
    catalogue = OperationCatalogueV1.model_validate(operation_catalogue_payload())
    operation = catalogue.operations[0]

    assert operation.inputs[0].accepted_schema_ids == ["omics-wide-matrix/1.0"]
    assert operation.outputs[0].schema_id == "sample-map/1.0"

    duplicate = operation_catalogue_payload()
    duplicate["operations"] = [
        duplicate["operations"][0],  # type: ignore[index]
        duplicate["operations"][0],  # type: ignore[index]
    ]
    with pytest.raises(ValidationError, match="layer/operation pairs must be unique"):
        OperationCatalogueV1.model_validate(duplicate)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("inputs", "acceptedSchemaIds"),
        ("outputs", "schemaId"),
    ],
)
def test_operation_catalogue_requires_schema_declarations(
    section: str, field: str
) -> None:
    payload = operation_catalogue_payload()
    del payload["operations"][0][section][0][field]  # type: ignore[index]

    with pytest.raises(ValidationError, match=field):
        OperationCatalogueV1.model_validate(payload)


def test_operation_catalogue_rejects_an_empty_accepted_schema_list() -> None:
    payload = operation_catalogue_payload()
    payload["operations"][0]["inputs"][0]["acceptedSchemaIds"] = []  # type: ignore[index]

    with pytest.raises(ValidationError):
        OperationCatalogueV1.model_validate(payload)


def test_manifest_enforces_learning_declarations() -> None:
    with pytest.raises(ValidationError, match="learningScope=none"):
        ManifestLearning.model_validate(
            {"learnsParameters": False, "learningScope": "trainFold"}
        )

    with pytest.raises(ValidationError, match="fitting population"):
        ManifestLearning.model_validate(
            {"learnsParameters": True, "learningScope": "trainFold"}
        )

    transform_learning = ManifestLearning.model_validate(
        {
            "learnsParameters": False,
            "learningScope": "none",
            "fittedArtifactId": "art_fitted_transformer",
        }
    )
    assert transform_learning.fitted_artifact_id == "art_fitted_transformer"

    with pytest.raises(ValidationError, match="cannot declare a fitting population"):
        ManifestLearning.model_validate(
            {
                "learnsParameters": False,
                "learningScope": "none",
                "fitPopulationArtifactId": "art_train_ids",
            }
        )

    manifest = ArtifactManifestV1.model_validate(
        {
            "schemaVersion": "artifact-manifest/1.0",
            "artifactId": "art_output",
            "kind": "omicsMatrix",
            "modality": "mrna",
            "datasetId": "ds_01K",
            "experimentId": "exp_01K",
            "runId": "run_01K",
            "stepRunId": "step_01K",
            "createdAt": "2026-08-22T12:04:03Z",
            "createdBy": {
                "service": "pp-fa2-normalization",
                "codeVersion": "2.0.0",
                "containerImageDigest": IMAGE_DIGEST,
                "recipeId": "mrna-fpkm-log2p1-v1",
                "recipeVersion": "1.0.0",
            },
            "inputs": [
                {"artifactId": "art_source", "sha256": HASH_A, "role": "matrix"}
            ],
            "parameters": {"formula": "log2(x + 1)"},
            "data": {
                "mediaType": "application/vnd.apache.parquet",
                "sha256": HASH_B,
                "byteLength": 987,
                "rowCount": 3,
                "featureCount": 2,
                "schemaId": "omics-wide-matrix/1.0",
                "sampleOrderSha256": HASH_A,
                "featureOrderSha256": HASH_B,
            },
            "learning": {"learnsParameters": False, "learningScope": "none"},
            "randomSeed": 1729,
            "warnings": [],
        }
    )
    assert manifest.data.row_count == 3

    cohort_payload = manifest.model_dump(mode="json")
    cohort_payload["learning"] = {
        "learnsParameters": True,
        "learningScope": "cohortUnsupervised",
        "fitPopulationArtifactId": None,
        "fittedArtifactId": None,
    }
    with pytest.raises(ValidationError, match="evaluation-bias warning"):
        ArtifactManifestV1.model_validate(cohort_payload)

    cohort_payload["warnings"] = [COHORT_UNSUPERVISED_LEAKAGE_WARNING]
    cohort_manifest = ArtifactManifestV1.model_validate(cohort_payload)
    assert cohort_manifest.learning.fit_population_artifact_id is None


def test_schema_export_is_deterministic_and_matches_checked_in_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_schemas(first)
    export_schemas(second)

    package_schemas = (
        Path(__file__).parents[1] / "src" / "uod_rg24_contracts" / "schemas" / "v1"
    )
    assert {path.name for path in first.iterdir()} == set(SCHEMA_MODELS)
    for file_name in SCHEMA_MODELS:
        first_text = (first / file_name).read_text(encoding="utf-8")
        assert first_text == (second / file_name).read_text(encoding="utf-8")
        assert json.loads(first_text)["type"] == "object"
        assert first_text == (package_schemas / file_name).read_text(encoding="utf-8")


def artifact_ref_payload() -> dict[str, object]:
    return {
        "artifactId": "art_01K",
        "datasetId": "ds_01K",
        "kind": "omicsMatrix",
        "schemaId": "omics-wide-matrix/1.0",
        "state": "committed",
        "sha256": HASH_A,
        "createdAt": "2026-08-22T12:04:03Z",
    }


def test_models_round_trip_through_a_python_mode_dump() -> None:
    reference = ArtifactRef.model_validate(artifact_ref_payload())

    restored = ArtifactRef.model_validate(reference.model_dump())

    assert restored.created_at == reference.created_at
    assert ArtifactRef.model_validate(reference.model_dump(mode="json")) == reference


def test_timestamps_accept_aware_utc_datetimes_but_nothing_looser() -> None:
    accepted = ArtifactRef.model_validate(
        {
            **artifact_ref_payload(),
            "createdAt": datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        }
    )
    assert accepted.created_at.utcoffset() == timedelta(0)

    # A naive datetime and a non-UTC offset are the cases under test.
    for rejected in (
        datetime(2026, 8, 22, 12, 0),  # noqa: DTZ001
        datetime(2026, 8, 22, 12, 0, tzinfo=timezone(timedelta(hours=1))),
        "2026-08-22T13:00:00+01:00",
        0,
    ):
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate(
                {**artifact_ref_payload(), "createdAt": rejected}
            )


def test_problem_details_trace_id_matches_the_event_trace_id_constraint() -> None:
    problem = {
        "type": "about:blank",
        "title": "Schema validation failed",
        "status": 422,
        "code": "SCHEMA_VALIDATION_FAILED",
        "detail": "The input matrix did not match its declared schema.",
    }
    assert ProblemDetails.model_validate({**problem, "traceId": TRACE_ID}).trace_id == (
        TRACE_ID
    )
    assert ProblemDetails.model_validate(problem).trace_id is None

    for rejected in ("not-a-trace-id", "0" * 32, "A" * 32, TRACE_ID[:-1]):
        with pytest.raises(ValidationError):
            ProblemDetails.model_validate({**problem, "traceId": rejected})
