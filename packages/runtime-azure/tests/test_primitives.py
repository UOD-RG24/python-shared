from __future__ import annotations

import io
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from uod_rg24_contracts import StepCommandV1
from uod_rg24_runtime_azure import (
    ArtifactBundleWriteReceipt,
    ArtifactCommitEvidence,
    ArtifactControl,
    ArtifactIoError,
    ArtifactPayload,
    ArtifactWriteReceipt,
    BlobETag,
    RuntimeConfigurationError,
    SafeLogContext,
    SerializationError,
    Sha256Digest,
    StepWorkspace,
    TraceContext,
    canonical_json_bytes,
    runtime_error_to_problem,
    sha256_bytes,
    sha256_stream,
    step_command_fingerprint,
)


def test_canonical_json_sorts_keys_and_rejects_nan() -> None:
    assert canonical_json_bytes({"z": 1, "a": "é"}) == b'{"a":"\xc3\xa9","z":1}\n'
    with pytest.raises(SerializationError):
        canonical_json_bytes({"bad": math.nan})


def test_stream_hash_restores_position() -> None:
    stream = io.BytesIO(b"prefix-payload")
    stream.seek(7)
    digest, length = sha256_stream(stream, chunk_size=2)
    assert digest == sha256_bytes(b"payload")
    assert length == 7
    assert stream.tell() == 7


def test_fingerprint_is_retry_stable_but_output_sensitive(
    step_command: StepCommandV1,
) -> None:
    expected = step_command_fingerprint(step_command, code_version="runtime@1")
    retry = step_command.model_copy(
        update={
            "attempt": 2,
            "message_id": "msg_retry",
            "execution_context": step_command.execution_context.model_copy(
                update={
                    "requested_at": datetime(2026, 8, 24, tzinfo=UTC),
                    "traceparent": f"00-{'3' * 32}-{'4' * 16}-01",
                }
            ),
        }
    )
    assert step_command_fingerprint(retry, code_version="runtime@1") == expected
    sensitive_cases = [
        (
            step_command.model_copy(
                update={
                    "inputs": [
                        step_command.inputs[0].model_copy(
                            update={"expected_sha256": "d" * 64}
                        )
                    ]
                }
            ),
            "runtime@1",
        ),
        (
            step_command.model_copy(update={"parameters": {"cohortPolicy": "union"}}),
            "runtime@1",
        ),
        (
            step_command.model_copy(
                update={
                    "execution_context": step_command.execution_context.model_copy(
                        update={"recipe_id": "changed-recipe"}
                    )
                }
            ),
            "runtime@1",
        ),
        *[
            (
                step_command.model_copy(
                    update={
                        "outputs": [step_command.outputs[0].model_copy(update=update)]
                    }
                ),
                "runtime@1",
            )
            for update in (
                {"name": "availability"},
                {"artifact_id": "art_changed"},
                {"kind": "sampleAvailability"},
                {"schema_id": "sample-availability/1.0"},
            )
        ],
        (step_command, "runtime@2"),
    ]
    for changed_command, code_version in sensitive_cases:
        assert (
            step_command_fingerprint(changed_command, code_version=code_version)
            != expected
        )


def test_trace_context_is_w3c_valid_even_when_constructed_directly() -> None:
    traceparent = f"00-{'1' * 32}-{'2' * 16}-01"
    context = TraceContext.parse(traceparent)
    assert context.trace_id == "1" * 32
    with pytest.raises(RuntimeConfigurationError, match="must match"):
        TraceContext(
            traceparent=traceparent,
            trace_id="3" * 32,
            parent_id="2" * 16,
            flags="01",
        )
    with pytest.raises(RuntimeConfigurationError, match="all zeroes"):
        TraceContext.parse(f"00-{'0' * 32}-{'2' * 16}-01")


def test_safe_log_context_and_problem_mapping_never_expose_paths_or_secrets() -> None:
    context = SafeLogContext(
        event="artifact.write",
        step_run_id="step_01K",
        artifact_id="art_01K",
        attempt=1,
    )
    assert context.as_dict() == {
        "event": "artifact.write",
        "stepRunId": "step_01K",
        "attempt": 1,
        "artifactId": "art_01K",
    }
    with pytest.raises(RuntimeConfigurationError, match="safe log"):
        SafeLogContext(event="artifact.write", artifact_id="/tmp/token?sig=secret")

    error = ArtifactIoError("failed at /tmp/private?sig=secret")
    problem = runtime_error_to_problem(
        error,
        trace_context=TraceContext.parse(f"00-{'1' * 32}-{'2' * 16}-01"),
    )
    rendered = problem.model_dump_json(by_alias=True)
    assert problem.status == 503
    assert problem.code == "ARTIFACT_IO_FAILED"
    assert problem.extensions == {"retryable": True}
    assert problem.trace_id == "1" * 32
    assert "/tmp" not in rendered
    assert "secret" not in rendered


def test_typed_hash_and_artifact_control_lifecycle_are_public() -> None:
    assert str(Sha256Digest("a" * 64)) == "a" * 64
    with pytest.raises(RuntimeConfigurationError):
        Sha256Digest("A" * 64)
    for method_name in ("resolve_input", "claim_output", "commit", "mark_failed"):
        assert hasattr(ArtifactControl, method_name)


def test_commit_evidence_retains_every_immutable_bundle_receipt() -> None:
    def receipt(
        *, artifact_id: str, schema_id: str, digest: str
    ) -> ArtifactWriteReceipt:
        return ArtifactWriteReceipt(
            artifact_id=artifact_id,
            schema_id=schema_id,
            sha256=Sha256Digest(digest),
            byte_length=42,
            etag=BlobETag(f'"{digest[:8]}"'),
            version_id="version-1",
            already_existed=False,
        )

    data = receipt(
        artifact_id="art_output", schema_id="sample-map/1.0", digest="a" * 64
    )
    manifest = receipt(
        artifact_id="art_output",
        schema_id="artifact-manifest/1.0",
        digest="b" * 64,
    )
    qc = receipt(artifact_id="art_output", schema_id="artifact-qc/1.0", digest="c" * 64)
    bundle_receipt = ArtifactBundleWriteReceipt(
        artifact_id="art_output", data=data, manifest=manifest, qc=qc
    )

    evidence = ArtifactCommitEvidence(
        bundle_receipt=bundle_receipt,
        media_type="application/vnd.apache.parquet",
    )

    assert evidence.bundle_receipt.data is data
    assert evidence.bundle_receipt.manifest is manifest
    assert evidence.bundle_receipt.qc is qc
    with pytest.raises(RuntimeConfigurationError, match="share one artifact_id"):
        ArtifactBundleWriteReceipt(
            artifact_id="art_output",
            data=data,
            manifest=manifest,
            qc=receipt(
                artifact_id="art_other",
                schema_id="artifact-qc/1.0",
                digest="d" * 64,
            ),
        )


def test_workspace_enforces_safe_names_and_fixed_bundle_members(tmp_path: Path) -> None:
    del tmp_path
    data = ArtifactPayload.from_bytes(
        output_name="sampleMap",
        kind="sampleMap",
        schema_id="sample-map/1.0",
        media_type="application/vnd.apache.parquet",
        data=b"PAR1",
    )
    manifest = ArtifactPayload.from_bytes(
        output_name="manifest",
        kind="artifactManifest",
        schema_id="artifact-manifest/1.0",
        media_type="application/json",
        data=b"{}\n",
    )
    qc = ArtifactPayload.from_bytes(
        output_name="sampleMap.qc",
        kind="qcSummary",
        schema_id="artifact-qc/1.0",
        media_type="application/json",
        data=b"{}\n",
    )
    with StepWorkspace() as workspace:
        root = workspace.root
        bundle = workspace.materialize_bundle(
            output_name="sampleMap", data=data, manifest=manifest, qc=qc
        )
        assert [
            bundle.data.path.name,
            bundle.manifest.path.name,
            bundle.qc.path.name,
        ] == ["data.parquet", "manifest.json", "qc.json"]
        with pytest.raises(RuntimeConfigurationError):
            workspace.path("../escape")
    assert not root.exists()
