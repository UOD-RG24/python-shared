from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from uod_rg24_contracts import ArtifactState
from uod_rg24_runtime_azure import (
    ArtifactAuthorizationError,
    ArtifactHashMismatchError,
    ArtifactIoError,
    ArtifactNotCommittedError,
    ArtifactNotFoundError,
    ArtifactPayload,
    ArtifactReadRequest,
    ArtifactSchemaMismatchError,
    ArtifactSizeMismatchError,
    ArtifactWriteConflictError,
    AzureBlobArtifactReader,
    AzureBlobArtifactWriter,
    BlobETag,
    ResolvedInputArtifact,
    ResolvedOutputReservation,
    Sha256Digest,
    StepWorkspace,
    canonical_json_bytes,
    sha256_bytes,
)

SAMPLE_ORDER_SHA256 = "a" * 64


def empty_metadata() -> dict[str, str]:
    return {}


@dataclass
class FakeBlobRecord:
    data: bytes
    etag: str = '"etag-1"'
    version_id: str | None = "version-1"
    metadata: dict[str, str] = field(default_factory=empty_metadata)
    reported_size: int | None = None
    download_data: bytes | None = None

    def properties(self) -> SimpleNamespace:
        return SimpleNamespace(
            etag=self.etag,
            metadata=self.metadata,
            size=len(self.data) if self.reported_size is None else self.reported_size,
            version_id=self.version_id,
        )


class FakeDownloader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def chunks(self) -> list[bytes]:
        midpoint = max(1, len(self._data) // 2)
        return [self._data[:midpoint], self._data[midpoint:]]


class FakeBlobClient:
    def __init__(self, service: FakeBlobService, key: tuple[str, str]) -> None:
        self._service = service
        self._key = key

    def get_blob_properties(self) -> SimpleNamespace:
        if self._service.properties_error is not None:
            raise self._service.properties_error
        return self._service.records[self._key].properties()

    def download_blob(self, **kwargs: object) -> FakeDownloader:
        self._service.download_kwargs.append(kwargs)
        record = self._service.records[self._key]
        return FakeDownloader(
            record.data if record.download_data is None else record.download_data
        )

    def upload_blob(self, data: object, **kwargs: object) -> dict[str, str]:
        self._service.upload_kwargs.append(kwargs)
        if self._key in self._service.records:
            raise ResourceExistsError("already exists")
        value = cast(BinaryIO, data).read()
        metadata = kwargs.get("metadata")
        assert isinstance(metadata, dict)
        raw_metadata = cast(dict[object, object], metadata)
        typed_metadata = {str(key): str(item) for key, item in raw_metadata.items()}
        record = FakeBlobRecord(data=value, metadata=typed_metadata)
        self._service.records[self._key] = record
        return {"etag": record.etag, "version_id": record.version_id or ""}


class FakeBlobService:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], FakeBlobRecord] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.download_kwargs: list[dict[str, object]] = []
        self.upload_kwargs: list[dict[str, object]] = []
        self.properties_error: Exception | None = None

    def get_blob_client(
        self,
        container: str,
        blob: str,
        *,
        version_id: str | None = None,
    ) -> FakeBlobClient:
        self.requests.append((container, blob, version_id))
        return FakeBlobClient(self, (container, blob))


def resolved_input(data: bytes, **overrides: object) -> ResolvedInputArtifact:
    values: dict[str, object] = {
        "artifact_id": "art_input",
        "dataset_id": "ds_01K",
        "kind": "sampleMap",
        "schema_id": "sample-map/1.0",
        "media_type": "application/vnd.apache.parquet",
        "state": ArtifactState.COMMITTED,
        "sha256": sha256_bytes(data),
        "byte_length": len(data),
        "container_name": "artifacts",
        "blob_name": "resolved/data.parquet",
        "etag": '"etag-1"',
        "version_id": "version-1",
    }
    values.update(overrides)
    return ResolvedInputArtifact(**values)  # type: ignore[arg-type]


def read_request(artifact: ResolvedInputArtifact) -> ArtifactReadRequest:
    return ArtifactReadRequest(
        artifact=artifact,
        accepted_schema_ids=("sample-map/1.0",),
        expected_schema_id="sample-map/1.0",
        expected_sha256=artifact.sha256,
        maximum_bytes=1024,
    )


def test_reader_materializes_verified_versioned_blob() -> None:
    data = b"deterministic-report"
    service = FakeBlobService()
    service.records[("artifacts", "resolved/data.parquet")] = FakeBlobRecord(data)
    reader = AzureBlobArtifactReader(service)
    artifact = resolved_input(data)

    with StepWorkspace() as workspace:
        materialized = reader.materialize(
            read_request(artifact), workspace=workspace, logical_name="input.parquet"
        )
        assert materialized.path.read_bytes() == data
        assert isinstance(materialized.receipt.sha256, Sha256Digest)
        assert isinstance(materialized.receipt.etag, BlobETag)
        assert str(materialized.receipt.sha256) == sha256_bytes(data)
    assert service.requests == [("artifacts", "resolved/data.parquet", "version-1")]
    assert service.download_kwargs[0]["etag"] == '"etag-1"'


def test_reader_rejects_command_mismatch_before_blob_io() -> None:
    data = b"report"
    service = FakeBlobService()
    reader = AzureBlobArtifactReader(service)
    artifact = resolved_input(data)
    request = ArtifactReadRequest(
        artifact=artifact,
        accepted_schema_ids=("sample-map/1.0",),
        expected_schema_id="sample-map/1.0",
        expected_sha256="f" * 64,
        maximum_bytes=1024,
    )
    with (
        StepWorkspace() as workspace,
        pytest.raises(ArtifactHashMismatchError, match="step command"),
    ):
        reader.materialize(request, workspace=workspace, logical_name="input.bin")
    assert service.requests == []


def test_reader_rejects_state_schema_hash_and_version_mismatches() -> None:
    data = b"report"
    service = FakeBlobService()
    reader = AzureBlobArtifactReader(service)
    with StepWorkspace() as workspace:
        uncommitted = resolved_input(data, state=ArtifactState.RESERVED)
        with pytest.raises(ArtifactNotCommittedError):
            reader.materialize(
                read_request(uncommitted), workspace=workspace, logical_name="one.bin"
            )
        wrong_schema = resolved_input(data, schema_id="other/1.0")
        with pytest.raises(ArtifactSchemaMismatchError):
            reader.materialize(
                read_request(wrong_schema), workspace=workspace, logical_name="two.bin"
            )

    service.records[("artifacts", "resolved/data.parquet")] = FakeBlobRecord(
        data, version_id="different-version"
    )
    with (
        StepWorkspace() as workspace,
        pytest.raises(ArtifactHashMismatchError, match="version"),
    ):
        reader.materialize(
            read_request(resolved_input(data)),
            workspace=workspace,
            logical_name="three.bin",
        )


@pytest.mark.parametrize(
    ("record", "error_type", "message"),
    [
        (FakeBlobRecord(b"report", etag='"other"'), ArtifactHashMismatchError, "ETag"),
        (
            FakeBlobRecord(b"report", reported_size=99),
            ArtifactSizeMismatchError,
            "size",
        ),
        (
            FakeBlobRecord(b"report", download_data=b"rep0rt"),
            ArtifactHashMismatchError,
            "hash",
        ),
        (
            FakeBlobRecord(b"report", download_data=b"short"),
            ArtifactSizeMismatchError,
            "size",
        ),
    ],
)
def test_reader_rejects_catalogue_or_download_integrity_mismatch_and_cleans_up(
    record: FakeBlobRecord,
    error_type: type[Exception],
    message: str,
) -> None:
    data = b"report"
    service = FakeBlobService()
    service.records[("artifacts", "resolved/data.parquet")] = record
    reader = AzureBlobArtifactReader(service)
    with StepWorkspace() as workspace:
        target = workspace.path("failed.bin")
        with pytest.raises(error_type, match=message):
            reader.materialize(
                read_request(resolved_input(data)),
                workspace=workspace,
                logical_name="failed.bin",
            )
        assert not target.exists()


@pytest.mark.parametrize(
    ("sdk_error", "mapped_error"),
    [
        (ResourceNotFoundError("missing"), ArtifactNotFoundError),
        (ClientAuthenticationError("denied"), ArtifactAuthorizationError),
        (HttpResponseError("unavailable"), ArtifactIoError),
    ],
)
def test_reader_maps_sdk_failures_without_leaving_partial_files(
    sdk_error: Exception,
    mapped_error: type[Exception],
) -> None:
    data = b"report"
    service = FakeBlobService()
    service.records[("artifacts", "resolved/data.parquet")] = FakeBlobRecord(data)
    service.properties_error = sdk_error
    reader = AzureBlobArtifactReader(service)
    with StepWorkspace() as workspace:
        target = workspace.path("failed.bin")
        with pytest.raises(mapped_error):
            reader.materialize(
                read_request(resolved_input(data)),
                workspace=workspace,
                logical_name="failed.bin",
            )
        assert not target.exists()


def payload(
    *,
    output_name: str,
    kind: str,
    schema_id: str,
    media_type: str,
    data: bytes,
    sample_order_sha256: str | None = None,
) -> ArtifactPayload:
    return ArtifactPayload.from_bytes(
        output_name=output_name,
        kind=kind,
        schema_id=schema_id,
        media_type=media_type,
        data=data,
        sample_order_sha256=sample_order_sha256,
    )


def reservation(**overrides: object) -> ResolvedOutputReservation:
    values: dict[str, object] = {
        "artifact_id": "art_output",
        "output_name": "sampleMap",
        "dataset_id": "ds_01K",
        "kind": "sampleMap",
        "schema_id": "sample-map/1.0",
        "media_type": "application/vnd.apache.parquet",
        "container_name": "artifacts",
        "blob_prefix": "owner/run/art_output",
    }
    values.update(overrides)
    return ResolvedOutputReservation(**values)  # type: ignore[arg-type]


def materialize_bundle(
    workspace: StepWorkspace,
    *,
    data: bytes = b"PAR1",
    manifest_sha256: str | None = None,
):
    data_sha256 = sha256_bytes(data)
    manifest_data = canonical_json_bytes(
        {
            "schemaVersion": "artifact-manifest/1.0",
            "artifactId": "art_output",
            "kind": "sampleMap",
            "data": {
                "schemaId": "sample-map/1.0",
                "mediaType": "application/vnd.apache.parquet",
                "sha256": manifest_sha256 or data_sha256,
                "byteLength": len(data),
                "sampleOrderSha256": SAMPLE_ORDER_SHA256,
            },
        }
    )
    qc_data = canonical_json_bytes(
        {
            "schemaVersion": "artifact-qc/1.0",
            "artifactId": "art_output",
        }
    )
    return workspace.materialize_bundle(
        output_name="sampleMap",
        data=payload(
            output_name="sampleMap",
            kind="sampleMap",
            schema_id="sample-map/1.0",
            media_type="application/vnd.apache.parquet",
            data=data,
            sample_order_sha256=SAMPLE_ORDER_SHA256,
        ),
        manifest=payload(
            output_name="manifest",
            kind="artifactManifest",
            schema_id="artifact-manifest/1.0",
            media_type="application/json",
            data=manifest_data,
        ),
        qc=payload(
            output_name="sampleMap.qc",
            kind="qcSummary",
            schema_id="artifact-qc/1.0",
            media_type="application/json",
            data=qc_data,
        ),
    )


def test_writer_uploads_only_fixed_members_and_is_idempotent() -> None:
    service = FakeBlobService()
    writer = AzureBlobArtifactWriter(service)
    with StepWorkspace() as workspace:
        bundle = materialize_bundle(workspace)
        first = writer.upload_bundle(reservation(), bundle)
        second = writer.upload_bundle(reservation(), bundle)

    assert sorted(blob for _, blob in service.records) == [
        "owner/run/art_output/data.parquet",
        "owner/run/art_output/manifest.json",
        "owner/run/art_output/qc.json",
    ]
    assert not first.data.already_existed
    assert isinstance(first.data.sha256, Sha256Digest)
    assert isinstance(first.data.etag, BlobETag)
    assert second.data.already_existed
    assert all(kwargs["overwrite"] is False for kwargs in service.upload_kwargs)


def test_writer_detects_existing_different_member() -> None:
    service = FakeBlobService()
    writer = AzureBlobArtifactWriter(service)
    with StepWorkspace() as first_workspace:
        writer.upload_bundle(reservation(), materialize_bundle(first_workspace))
    with (
        StepWorkspace() as second_workspace,
        pytest.raises(ArtifactWriteConflictError),
    ):
        writer.upload_bundle(
            reservation(), materialize_bundle(second_workspace, data=b"DIFFERENT")
        )


def test_writer_rejects_name_or_manifest_mismatch_before_blob_io() -> None:
    service = FakeBlobService()
    writer = AzureBlobArtifactWriter(service)
    with StepWorkspace() as workspace:
        bundle = materialize_bundle(workspace)
        with pytest.raises(ArtifactSchemaMismatchError, match="reservation"):
            writer.upload_bundle(
                reservation(output_name="availability"),
                bundle,
            )
    assert service.requests == []

    with StepWorkspace() as workspace:
        bundle = materialize_bundle(workspace, manifest_sha256="f" * 64)
        with pytest.raises(ArtifactSchemaMismatchError, match="metadata"):
            writer.upload_bundle(reservation(), bundle)
    assert service.requests == []


def test_writer_rehashes_every_materialized_member_before_blob_io() -> None:
    service = FakeBlobService()
    writer = AzureBlobArtifactWriter(service)
    with StepWorkspace() as workspace:
        bundle = materialize_bundle(workspace)
        bundle.qc.path.write_bytes(b"tampered\n")
        with pytest.raises(ArtifactHashMismatchError, match="member hash"):
            writer.upload_bundle(reservation(), bundle)
    assert service.requests == []
