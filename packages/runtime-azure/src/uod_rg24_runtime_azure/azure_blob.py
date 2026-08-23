from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, cast

from azure.core import MatchConditions
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.storage.blob import ContentSettings
from uod_rg24_contracts import ArtifactState

from .errors import (
    ArtifactAuthorizationError,
    ArtifactHashMismatchError,
    ArtifactIoError,
    ArtifactNotCommittedError,
    ArtifactNotFoundError,
    ArtifactSchemaMismatchError,
    ArtifactSizeMismatchError,
    ArtifactTooLargeError,
    ArtifactWriteConflictError,
    RuntimeAzureError,
    SerializationError,
)
from .hashing import sha256_stream
from .json_codec import decode_json_object
from .models import (
    ArtifactBundle,
    ArtifactBundleWriteReceipt,
    ArtifactFile,
    ArtifactReadReceipt,
    ArtifactReadRequest,
    ArtifactReadResult,
    ArtifactWriteReceipt,
    BlobETag,
    MaterializedArtifact,
    ResolvedOutputReservation,
    Sha256Digest,
)
from .serialization import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_ID,
    OUTPUT_LAYOUTS,
    QC_KIND,
    QC_SCHEMA_ID,
)
from .workspace import StepWorkspace


class _Downloader(Protocol):
    def chunks(self) -> Iterable[bytes]: ...


class _BlobClient(Protocol):
    def get_blob_properties(self) -> object: ...

    def download_blob(self, **kwargs: object) -> _Downloader: ...

    def upload_blob(self, data: object, **kwargs: object) -> object: ...


class BlobServiceClientPort(Protocol):
    def get_blob_client(
        self,
        container: str,
        blob: str,
        *,
        version_id: str | None = None,
    ) -> _BlobClient: ...


def _property(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value).get(name, default)
    return getattr(value, name, default)


def _etag(value: object) -> str:
    result = _property(value, "etag")
    if not isinstance(result, str) or not result:
        raise ArtifactIoError("Azure Blob response did not include an ETag.")
    return result


def _version_id(value: object) -> str | None:
    result = _property(value, "version_id")
    return result if isinstance(result, str) and result else None


def _size(value: object) -> int:
    result = _property(value, "size")
    if not isinstance(result, int) or result < 0:
        raise ArtifactIoError("Azure Blob properties did not include a valid size.")
    return result


def _metadata(value: object) -> dict[str, str]:
    result = _property(value, "metadata", {})
    if not isinstance(result, Mapping):
        return {}
    metadata = cast(Mapping[object, object], result)
    return {str(key): str(item) for key, item in metadata.items()}


def _status_code(exc: HttpResponseError) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


class AzureBlobArtifactReader:
    """Materialize broker-resolved immutable blobs into a StepWorkspace."""

    def __init__(
        self,
        client: BlobServiceClientPort,
        *,
        maximum_concurrency: int = 1,
    ) -> None:
        if maximum_concurrency <= 0:
            raise ValueError("maximum_concurrency must be positive")
        self._client = client
        self._maximum_concurrency = maximum_concurrency

    def materialize(
        self,
        request: ArtifactReadRequest,
        *,
        workspace: StepWorkspace,
        logical_name: str,
    ) -> MaterializedArtifact:
        source = request.artifact
        if source.state != ArtifactState.COMMITTED:
            raise ArtifactNotCommittedError(
                "Input artifact is not committed.", artifact_id=source.artifact_id
            )
        if source.schema_id != request.expected_schema_id:
            raise ArtifactSchemaMismatchError(
                "Resolved input schema differs from the step command.",
                artifact_id=source.artifact_id,
            )
        if source.schema_id not in request.accepted_schema_ids:
            raise ArtifactSchemaMismatchError(
                "Input artifact schema is not accepted.", artifact_id=source.artifact_id
            )
        if source.sha256 != request.expected_sha256:
            raise ArtifactHashMismatchError(
                "Resolved input hash differs from the step command.",
                artifact_id=source.artifact_id,
            )
        if source.byte_length > request.maximum_bytes:
            raise ArtifactTooLargeError(
                "Input artifact exceeds the configured byte limit.",
                artifact_id=source.artifact_id,
            )

        blob = self._client.get_blob_client(
            source.container_name,
            source.blob_name,
            version_id=source.version_id,
        )
        target = workspace.materialize_input_path(logical_name)
        try:
            properties = blob.get_blob_properties()
            if _etag(properties) != source.etag:
                raise ArtifactHashMismatchError(
                    "Input artifact ETag differs from the catalogue.",
                    artifact_id=source.artifact_id,
                )
            if _size(properties) != source.byte_length:
                raise ArtifactSizeMismatchError(
                    "Input artifact size differs from the catalogue.",
                    artifact_id=source.artifact_id,
                )
            if (
                source.version_id is not None
                and _version_id(properties) != source.version_id
            ):
                raise ArtifactHashMismatchError(
                    "Input artifact version differs from the catalogue.",
                    artifact_id=source.artifact_id,
                )
            downloader = blob.download_blob(
                etag=source.etag,
                match_condition=MatchConditions.IfNotModified,
                max_concurrency=self._maximum_concurrency,
            )
            byte_length = 0
            with target.open("xb") as destination:
                for chunk in downloader.chunks():
                    byte_length += len(chunk)
                    if byte_length > request.maximum_bytes:
                        raise ArtifactTooLargeError(
                            "Downloaded artifact exceeds the configured byte limit.",
                            artifact_id=source.artifact_id,
                        )
                    destination.write(chunk)
            with target.open("rb") as stream:
                digest, hashed_length = sha256_stream(stream)
            if byte_length != source.byte_length or hashed_length != source.byte_length:
                raise ArtifactSizeMismatchError(
                    "Downloaded artifact size differs from the catalogue.",
                    artifact_id=source.artifact_id,
                )
            if digest != source.sha256:
                raise ArtifactHashMismatchError(
                    "Downloaded artifact hash differs from the catalogue.",
                    artifact_id=source.artifact_id,
                )
        except RuntimeAzureError:
            target.unlink(missing_ok=True)
            raise
        except ResourceNotFoundError as exc:
            target.unlink(missing_ok=True)
            raise ArtifactNotFoundError(
                "Input artifact data was not found.", artifact_id=source.artifact_id
            ) from exc
        except ClientAuthenticationError as exc:
            target.unlink(missing_ok=True)
            raise ArtifactAuthorizationError(
                "Input artifact access was denied.", artifact_id=source.artifact_id
            ) from exc
        except ResourceModifiedError as exc:
            target.unlink(missing_ok=True)
            raise ArtifactHashMismatchError(
                "Input artifact changed during download.",
                artifact_id=source.artifact_id,
            ) from exc
        except HttpResponseError as exc:
            target.unlink(missing_ok=True)
            if _status_code(exc) in {401, 403}:
                raise ArtifactAuthorizationError(
                    "Input artifact access was denied.", artifact_id=source.artifact_id
                ) from exc
            raise ArtifactIoError(
                "Input artifact download failed.", artifact_id=source.artifact_id
            ) from exc
        except OSError as exc:
            target.unlink(missing_ok=True)
            raise ArtifactIoError(
                "Input artifact could not be materialized.",
                artifact_id=source.artifact_id,
            ) from exc

        receipt = ArtifactReadReceipt(
            artifact_id=source.artifact_id,
            schema_id=source.schema_id,
            sha256=Sha256Digest(source.sha256),
            byte_length=source.byte_length,
            etag=BlobETag(source.etag),
            version_id=source.version_id,
        )
        return MaterializedArtifact(path=target, receipt=receipt)

    def read_bytes(self, request: ArtifactReadRequest) -> ArtifactReadResult:
        """Bounded convenience helper for small report artifacts."""

        with StepWorkspace() as workspace:
            materialized = self.materialize(
                request,
                workspace=workspace,
                logical_name="input.bin",
            )
            return ArtifactReadResult(
                data=materialized.path.read_bytes(), receipt=materialized.receipt
            )


class AzureBlobArtifactWriter:
    """Conditionally upload fixed artifact bundle members to a reservation."""

    def __init__(
        self,
        client: BlobServiceClientPort,
        *,
        maximum_concurrency: int = 1,
    ) -> None:
        if maximum_concurrency <= 0:
            raise ValueError("maximum_concurrency must be positive")
        self._client = client
        self._maximum_concurrency = maximum_concurrency

    def _validate_bundle(
        self,
        reservation: ResolvedOutputReservation,
        bundle: ArtifactBundle,
    ) -> None:
        try:
            layout = OUTPUT_LAYOUTS[reservation.output_name]
        except KeyError as exc:
            raise ArtifactSchemaMismatchError(
                "Output reservation name is not supported.",
                artifact_id=reservation.artifact_id,
            ) from exc
        if (
            bundle.data.output_name != reservation.output_name
            or reservation.kind != layout.kind
            or reservation.schema_id != layout.schema_id
            or reservation.media_type != layout.media_type
            or bundle.data.kind != reservation.kind
            or bundle.data.schema_id != reservation.schema_id
            or bundle.data.media_type != reservation.media_type
        ):
            raise ArtifactSchemaMismatchError(
                "Data bundle does not match its output reservation.",
                artifact_id=reservation.artifact_id,
            )
        if (
            bundle.manifest.output_name != "manifest"
            or bundle.manifest.kind != MANIFEST_KIND
            or bundle.manifest.schema_id != MANIFEST_SCHEMA_ID
            or bundle.manifest.media_type != "application/json"
        ):
            raise ArtifactSchemaMismatchError(
                "Bundle manifest has the wrong schema.",
                artifact_id=reservation.artifact_id,
            )
        if (
            bundle.qc.output_name != f"{reservation.output_name}.qc"
            or bundle.qc.kind != QC_KIND
            or bundle.qc.schema_id != QC_SCHEMA_ID
            or bundle.qc.media_type != "application/json"
        ):
            raise ArtifactSchemaMismatchError(
                "Bundle QC has the wrong schema.", artifact_id=reservation.artifact_id
            )
        if bundle.data.sample_order_sha256 is None:
            raise ArtifactSchemaMismatchError(
                "Data bundle is missing the retained-cohort hash.",
                artifact_id=reservation.artifact_id,
            )
        for artifact_file in (bundle.data, bundle.manifest, bundle.qc):
            try:
                with artifact_file.path.open("rb") as source:
                    digest, byte_length = sha256_stream(source)
            except OSError as exc:
                raise ArtifactIoError(
                    "Output artifact file could not be inspected.",
                    artifact_id=reservation.artifact_id,
                ) from exc
            if digest != artifact_file.sha256:
                raise ArtifactHashMismatchError(
                    "Output bundle member hash is invalid.",
                    artifact_id=reservation.artifact_id,
                )
            if byte_length != artifact_file.byte_length:
                raise ArtifactSizeMismatchError(
                    "Output bundle member size is invalid.",
                    artifact_id=reservation.artifact_id,
                )
        try:
            manifest = decode_json_object(bundle.manifest.path.read_bytes())
            qc = decode_json_object(bundle.qc.path.read_bytes())
            manifest_data = manifest["data"]
            if not isinstance(manifest_data, dict):
                raise KeyError("data")
            typed_manifest_data = cast(dict[str, object], manifest_data)
            matches = (
                manifest.get("schemaVersion") == MANIFEST_SCHEMA_ID
                and manifest.get("artifactId") == reservation.artifact_id
                and manifest.get("kind") == reservation.kind
                and typed_manifest_data.get("schemaId") == reservation.schema_id
                and typed_manifest_data.get("mediaType") == reservation.media_type
                and typed_manifest_data.get("sha256") == str(bundle.data.sha256)
                and typed_manifest_data.get("byteLength") == bundle.data.byte_length
                and typed_manifest_data.get("sampleOrderSha256")
                == bundle.data.sample_order_sha256
                and qc.get("schemaVersion") == QC_SCHEMA_ID
                and qc.get("artifactId") == reservation.artifact_id
            )
        except (KeyError, OSError, SerializationError) as exc:
            raise ArtifactSchemaMismatchError(
                "Bundle metadata is incomplete.", artifact_id=reservation.artifact_id
            ) from exc
        if not matches:
            raise ArtifactSchemaMismatchError(
                "Bundle metadata does not describe the reserved data artifact.",
                artifact_id=reservation.artifact_id,
            )

    def _upload_file(
        self,
        reservation: ResolvedOutputReservation,
        artifact_file: ArtifactFile,
    ) -> ArtifactWriteReceipt:
        blob_name = f"{reservation.blob_prefix}/{artifact_file.member_name}"
        blob = self._client.get_blob_client(
            reservation.container_name,
            blob_name,
            version_id=None,
        )
        metadata = {
            "artifact_id": reservation.artifact_id,
            "member_name": artifact_file.member_name,
            "schema_id": artifact_file.schema_id,
            "sha256": artifact_file.sha256,
        }
        try:
            with artifact_file.path.open("rb") as source:
                response = blob.upload_blob(
                    source,
                    blob_type="BlockBlob",
                    content_settings=ContentSettings(
                        content_type=artifact_file.media_type
                    ),
                    length=artifact_file.byte_length,
                    max_concurrency=self._maximum_concurrency,
                    metadata=metadata,
                    overwrite=False,
                )
            return ArtifactWriteReceipt(
                artifact_id=reservation.artifact_id,
                schema_id=artifact_file.schema_id,
                sha256=Sha256Digest(artifact_file.sha256),
                byte_length=artifact_file.byte_length,
                etag=BlobETag(_etag(response)),
                version_id=_version_id(response),
                already_existed=False,
            )
        except ResourceExistsError as exc:
            try:
                properties = blob.get_blob_properties()
            except HttpResponseError as properties_exc:
                raise ArtifactIoError(
                    "Existing output artifact could not be inspected.",
                    artifact_id=reservation.artifact_id,
                ) from properties_exc
            existing_metadata = _metadata(properties)
            matches = (
                _size(properties) == artifact_file.byte_length
                and existing_metadata.get("artifact_id") == reservation.artifact_id
                and existing_metadata.get("member_name") == artifact_file.member_name
                and existing_metadata.get("schema_id") == artifact_file.schema_id
                and existing_metadata.get("sha256") == artifact_file.sha256
            )
            if not matches:
                raise ArtifactWriteConflictError(
                    "Reserved output already contains different data.",
                    artifact_id=reservation.artifact_id,
                ) from exc
            return ArtifactWriteReceipt(
                artifact_id=reservation.artifact_id,
                schema_id=artifact_file.schema_id,
                sha256=Sha256Digest(artifact_file.sha256),
                byte_length=artifact_file.byte_length,
                etag=BlobETag(_etag(properties)),
                version_id=_version_id(properties),
                already_existed=True,
            )
        except ClientAuthenticationError as exc:
            raise ArtifactAuthorizationError(
                "Output artifact access was denied.",
                artifact_id=reservation.artifact_id,
            ) from exc
        except HttpResponseError as exc:
            if _status_code(exc) in {401, 403}:
                raise ArtifactAuthorizationError(
                    "Output artifact access was denied.",
                    artifact_id=reservation.artifact_id,
                ) from exc
            raise ArtifactIoError(
                "Output artifact upload failed.", artifact_id=reservation.artifact_id
            ) from exc
        except OSError as exc:
            raise ArtifactIoError(
                "Output artifact file could not be read.",
                artifact_id=reservation.artifact_id,
            ) from exc

    def upload_bundle(
        self,
        reservation: ResolvedOutputReservation,
        bundle: ArtifactBundle,
    ) -> ArtifactBundleWriteReceipt:
        self._validate_bundle(reservation, bundle)
        return ArtifactBundleWriteReceipt(
            artifact_id=reservation.artifact_id,
            data=self._upload_file(reservation, bundle.data),
            manifest=self._upload_file(reservation, bundle.manifest),
            qc=self._upload_file(reservation, bundle.qc),
        )
