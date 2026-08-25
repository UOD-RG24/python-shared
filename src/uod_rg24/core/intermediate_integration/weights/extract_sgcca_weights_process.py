import tempfile
import threading
from pathlib import Path, PurePosixPath
from time import perf_counter

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient, BlobServiceClient

from uod_rg24.models.intermediate_integration.weights.extract_sgcca_weights_process_models import (
    ExtractSGCCAWeightsProcessRequestModel,
    ExtractSGCCAWeightsProcessResponseModel,
    SGCCABlockDetailsModel,
    SGCCABlocksProcessedDetailsModel,
)
from uod_rg24.models.intermediate_integration.weights.weights_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    ProcessStepModel
)
from uod_rg24.tools.r.rgcca import (
    align_matrices,
    get_r_version,
    get_rgcca_version,
    load_matrix,
    parse_boolean,
    parse_sparsity,
    run_sgcca,
)


def create_blob_path(
    *,
    directory_name: str,
    file_name: str,
    extension: str,
) -> str:
    normalized_directory = directory_name.strip("/\\").replace(
        "\\",
        "/",
    )
    normalized_extension = extension.lstrip(".")

    expected_suffix = f".{normalized_extension}"

    if file_name.lower().endswith(expected_suffix.lower()):
        complete_file_name = file_name
    else:
        complete_file_name = f"{file_name}{expected_suffix}"

    if not normalized_directory:
        return complete_file_name

    return str(
        PurePosixPath(normalized_directory)
        / complete_file_name
    )


def extract_sgcca_weights_process(
    blob_service_client: BlobServiceClient,
    mrna_matrix: InputModel,
    cna_matrix: InputModel,
    sgcca_feature_weights_output: OutputModel,
    extract_sgcca_weights_process_request: (
        ExtractSGCCAWeightsProcessRequestModel
    ),
    r_lock: threading.RLock,
) -> ExtractSGCCAWeightsProcessResponseModel:
    from contextlib import contextmanager
    from datetime import UTC, datetime
    from typing import Iterator

    total_started = perf_counter()
    process_steps: list[ProcessStepModel] = []

    @contextmanager
    def record_step(
        step: str,
        message: str | None = None,
    ) -> Iterator[None]:
        step_started_at = datetime.now(UTC)
        step_started = perf_counter()

        try:
            yield
        finally:
            step_completed_at = datetime.now(UTC)
            process_steps.append(
                ProcessStepModel(
                    step=step,
                    startedAt=step_started_at,
                    completedAt=step_completed_at,
                    durationMs=(
                        perf_counter() - step_started
                    )
                    * 1000,
                    message=message,
                )
            )

    with record_step(
        step="validate-inputs",
        message=(
            "Validated blob extensions, storage accounts, "
            "and sGCCA process parameters."
        ),
    ):
        mrna_extension = mrna_matrix.extension
        if mrna_extension is None:
            raise ValueError(
                "The mRNA matrix file extension is required."
            )

        cna_extension = cna_matrix.extension
        if cna_extension is None:
            raise ValueError(
                "The CNA matrix file extension is required."
            )

        output_extension = (
            sgcca_feature_weights_output.extension
        )
        if output_extension is None:
            raise ValueError(
                "The sGCCA feature weights output extension "
                "is required."
            )

        mrna_extension = mrna_extension.lstrip(".")
        cna_extension = cna_extension.lstrip(".")
        output_extension = output_extension.lstrip(".")

        blob_service_account_name = (
            blob_service_client.account_name
        )

        blob_models = (
            (mrna_matrix, "mRNA matrix"),
            (cna_matrix, "CNA matrix"),
            (
                sgcca_feature_weights_output,
                "sGCCA feature weights output",
            ),
        )

        for blob_model, blob_description in blob_models:
            if (
                blob_model.azure_storage_account_name
                != blob_service_account_name
            ):
                raise ValueError(
                    f"The {blob_description} specifies storage "
                    f"account "
                    f"{blob_model.azure_storage_account_name!r}, "
                    "but the supplied BlobServiceClient belongs to "
                    f"{blob_service_account_name!r}."
                )

        n_components = (
            extract_sgcca_weights_process_request.n_components
        )
        scheme = (
            extract_sgcca_weights_process_request.scheme.lower()
        )
        scale = parse_boolean(
            value=(
                extract_sgcca_weights_process_request.scale
            ),
            field_name="scale",
        )
        sparsity = parse_sparsity(
            value=(
                extract_sgcca_weights_process_request.sparsity
            ),
            block_count=2,
        )

    with record_step(
        step="create-blob-clients",
        message=(
            "Created blob paths and Azure Blob Storage clients."
        ),
    ):
        mrna_blob_path = create_blob_path(
            directory_name=mrna_matrix.directory_name,
            file_name=mrna_matrix.file_name,
            extension=mrna_extension,
        )
        cna_blob_path = create_blob_path(
            directory_name=cna_matrix.directory_name,
            file_name=cna_matrix.file_name,
            extension=cna_extension,
        )
        output_blob_path = create_blob_path(
            directory_name=(
                sgcca_feature_weights_output.directory_name
            ),
            file_name=(
                sgcca_feature_weights_output.file_name
            ),
            extension=output_extension,
        )

        mrna_blob_client: BlobClient = (
            blob_service_client.get_blob_client(
                container=(
                    mrna_matrix.azure_container_name
                ),
                blob=mrna_blob_path,
            )
        )
        cna_blob_client: BlobClient = (
            blob_service_client.get_blob_client(
                container=cna_matrix.azure_container_name,
                blob=cna_blob_path,
            )
        )
        output_blob_client: BlobClient = (
            blob_service_client.get_blob_client(
                container=(
                    sgcca_feature_weights_output
                    .azure_container_name
                ),
                blob=output_blob_path,
            )
        )

    with record_step(
        step="verify-input-blobs",
        message=(
            "Verified that the mRNA and CNA input blobs exist."
        ),
    ):
        try:
            mrna_blob_client.get_blob_properties()
        except ResourceNotFoundError as exception:
            raise FileNotFoundError(
                "The mRNA matrix blob does not exist. "
                f"Account: "
                f"{mrna_matrix.azure_storage_account_name!r}; "
                f"container: "
                f"{mrna_matrix.azure_container_name!r}; "
                f"blob: {mrna_blob_path!r}."
            ) from exception

        try:
            cna_blob_client.get_blob_properties()
        except ResourceNotFoundError as exception:
            raise FileNotFoundError(
                "The CNA matrix blob does not exist. "
                f"Account: "
                f"{cna_matrix.azure_storage_account_name!r}; "
                f"container: "
                f"{cna_matrix.azure_container_name!r}; "
                f"blob: {cna_blob_path!r}."
            ) from exception

    with tempfile.TemporaryDirectory(
        prefix="extract-sgcca-weights-",
    ) as temporary_directory:
        temporary_directory_path = Path(
            temporary_directory
        )
        local_mrna_path = (
            temporary_directory_path
            / f"{mrna_matrix.file_name}.{mrna_extension}"
        )
        local_cna_path = (
            temporary_directory_path
            / f"{cna_matrix.file_name}.{cna_extension}"
        )
        local_output_path = (
            temporary_directory_path
            / (
                f"{sgcca_feature_weights_output.file_name}."
                f"{output_extension}"
            )
        )

        with record_step(
            step="download-input-blobs",
            message=(
                "Downloaded the mRNA and CNA matrices to "
                "temporary local files."
            ),
        ):
            with local_mrna_path.open(
                "wb"
            ) as local_mrna_file:
                mrna_blob_client.download_blob().readinto(
                    local_mrna_file,
                )

            with local_cna_path.open(
                "wb"
            ) as local_cna_file:
                cna_blob_client.download_blob().readinto(
                    local_cna_file,
                )

        with record_step(
            step="load-matrices",
            message=(
                "Loaded the mRNA and CNA matrices."
            ),
        ):
            mrna = load_matrix(
                file_path=local_mrna_path,
                index_column=(
                    extract_sgcca_weights_process_request
                    .index_column
                ),
            )
            cna = load_matrix(
                file_path=local_cna_path,
                index_column=(
                    extract_sgcca_weights_process_request
                    .index_column
                ),
            )

        with record_step(
            step="align-matrices",
            message=(
                "Aligned the mRNA and CNA matrices by patient."
            ),
        ):
            mrna, cna = align_matrices(
                mrna=mrna,
                cna=cna,
            )
            blocks = {
                "mRNA": mrna,
                "CNA": cna,
            }

        with record_step(
            step="run-sgcca",
            message=(
                "Ran sGCCA and extracted feature weights."
            ),
        ):
            feature_weights = run_sgcca(
                blocks=blocks,
                n_components=n_components,
                sparsity=sparsity,
                scheme=scheme,
                scale=scale,
                r_lock=r_lock,
            )

        with record_step(
            step="write-output-file",
            message=(
                "Wrote the extracted feature weights to a "
                "temporary CSV file."
            ),
        ):
            feature_weights.to_csv(
                local_output_path,
                index=False,
            )

        with record_step(
            step="upload-output-blob",
            message=(
                "Uploaded the sGCCA feature weights output."
            ),
        ):
            with local_output_path.open(
                "rb"
            ) as local_output_file:
                output_blob_client.upload_blob(
                    data=local_output_file,
                    overwrite=True,
                )

    with record_step(
        step="calculate-output-statistics",
        message=(
            "Calculated selected-weight and selected-feature "
            "statistics."
        ),
    ):
        selected_weights = int(
            feature_weights["selected"].sum()
        )
        selected_features = int(
            feature_weights.loc[
                feature_weights["selected"],
                ["block", "feature"],
            ]
            .drop_duplicates()
            .shape[0]
        )

    with record_step(
        step="get-r-package-versions",
        message=(
            "Retrieved the R and RGCCA package versions."
        ),
    ):
        r_version = get_r_version(
            r_lock=r_lock,
        )
        rgcca_version = get_rgcca_version(
            r_lock=r_lock,
        )

    total_elapsed_seconds = (
        perf_counter() - total_started
    )

    return ExtractSGCCAWeightsProcessResponseModel(
        message=(
            "sGCCA feature weights extracted successfully."
        ),
        rVersion=r_version,
        rgccaVersion=rgcca_version,
        patientsProcessed=len(mrna),
        blocksProcessed=len(blocks),
        blockDetails=SGCCABlocksProcessedDetailsModel(
            mRNA=SGCCABlockDetailsModel(
                patients=mrna.shape[0],
                features=mrna.shape[1],
            ),
            CNA=SGCCABlockDetailsModel(
                patients=cna.shape[0],
                features=cna.shape[1],
            ),
        ),
        componentsRequested=n_components,
        sparsity=sparsity,
        scheme=scheme,
        scale=scale,
        weightsExtracted=len(feature_weights),
        selectedWeights=selected_weights,
        selectedFeatures=selected_features,
        outputFilePath=output_blob_path,
        totalElapsedSeconds=total_elapsed_seconds,
        steps=process_steps
    )