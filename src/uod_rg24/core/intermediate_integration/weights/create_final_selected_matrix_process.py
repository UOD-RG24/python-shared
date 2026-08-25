from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient

from uod_rg24.models.intermediate_integration.weights.create_final_selected_matrix_process_models import (
    CreateFinalSelectedMatrixProcessRequestModel,
    CreateFinalSelectedMatrixProcessResponseModel,
)
from uod_rg24.models.intermediate_integration.weights.weights_models import (
    InputModel,
    OutputModel,
)


def create_final_selected_matrix_process(
    blob_service_client: BlobServiceClient,
    weights_input_blob: InputModel,
    weighted_matrix_input_blob: InputModel,
    weighted_output_blob: OutputModel,
    final_matrix_request: CreateFinalSelectedMatrixProcessRequestModel,
) -> CreateFinalSelectedMatrixProcessResponseModel:
    from contextlib import contextmanager
    from datetime import UTC, datetime
    from time import perf_counter
    from typing import Iterator

    from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
        ProcessStepModel,
    )

    total_started = perf_counter()
    process_steps: list[ProcessStepModel] = []

    @contextmanager
    def record_step(
        step: str,
        message: str | None = None,
    ) -> Iterator[None]:
        started_at = datetime.now(UTC)
        step_started = perf_counter()

        try:
            yield
        finally:
            completed_at = datetime.now(UTC)
            process_steps.append(
                ProcessStepModel(
                    step=step,
                    startedAt=started_at,
                    completedAt=completed_at,
                    durationMs=(
                        perf_counter() - step_started
                    )
                    * 1000,
                    message=message,
                )
            )

    with TemporaryDirectory(
        prefix="create_final_selected_matrix_",
    ) as temporary_directory:
        with record_step(
            step="prepare-temporary-files",
            message=(
                "Created temporary input and output file paths."
            ),
        ):
            temporary_directory_path = Path(
                temporary_directory
            )

            weights_extension = (
                weights_input_blob.extension or "csv"
            ).lstrip(".")
            weighted_matrix_extension = (
                weighted_matrix_input_blob.extension or "csv"
            ).lstrip(".")
            output_extension = (
                weighted_output_blob.extension or "csv"
            ).lstrip(".")

            weights_file_path = (
                temporary_directory_path
                / (
                    f"{weights_input_blob.file_name}."
                    f"{weights_extension}"
                )
            )
            weighted_matrix_file_path = (
                temporary_directory_path
                / (
                    f"{weighted_matrix_input_blob.file_name}."
                    f"{weighted_matrix_extension}"
                )
            )
            output_file_path = (
                temporary_directory_path
                / (
                    f"{weighted_output_blob.file_name}."
                    f"{output_extension}"
                )
            )

            weights_blob_name = (
                f"{weights_input_blob.directory_name}/"
                f"{weights_input_blob.file_name}."
                f"{weights_extension}"
            )
            weighted_matrix_blob_name = (
                f"{weighted_matrix_input_blob.directory_name}/"
                f"{weighted_matrix_input_blob.file_name}."
                f"{weighted_matrix_extension}"
            )
            output_blob_name = (
                f"{weighted_output_blob.directory_name}/"
                f"{weighted_output_blob.file_name}."
                f"{output_extension}"
            )

        with record_step(
            step="create-blob-clients",
            message=(
                "Created Blob Storage clients for the weights, "
                "weighted matrix, and final output."
            ),
        ):
            weights_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        weights_input_blob.azure_container_name
                    ),
                    blob=weights_blob_name,
                )
            )
            weighted_matrix_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        weighted_matrix_input_blob
                        .azure_container_name
                    ),
                    blob=weighted_matrix_blob_name,
                )
            )
            output_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        weighted_output_blob.azure_container_name
                    ),
                    blob=output_blob_name,
                )
            )

        with record_step(
            step="download-weights",
            message=(
                "Downloaded the sGCCA feature weights file."
            ),
        ):
            with weights_file_path.open(
                "wb"
            ) as weights_file:
                weights_blob_client.download_blob().readinto(
                    weights_file
                )

        with record_step(
            step="download-weighted-matrix",
            message="Downloaded the weighted feature matrix.",
        ):
            with weighted_matrix_file_path.open(
                "wb"
            ) as weighted_matrix_file:
                weighted_matrix_blob_client.download_blob().readinto(
                    weighted_matrix_file
                )

        with record_step(
            step="load-input-files",
            message=(
                "Loaded the weights and weighted matrix CSV files."
            ),
        ):
            weights = pd.read_csv(weights_file_path)
            weighted_matrix = pd.read_csv(
                weighted_matrix_file_path
            )

        with record_step(
            step="validate-input-columns",
            message=(
                "Validated the required weights columns and "
                "identifier column."
            ),
        ):
            required_weight_columns = {
                "block",
                "feature",
                "selected",
            }
            missing_weight_columns = (
                required_weight_columns - set(weights.columns)
            )

            if missing_weight_columns:
                raise ValueError(
                    "The weights file is missing required "
                    f"columns: {sorted(missing_weight_columns)}"
                )

            identifier_column = (
                final_matrix_request.identifier_column
            )

            if identifier_column not in weighted_matrix.columns:
                raise ValueError(
                    "The identifier column is missing from the "
                    f"weighted matrix: {identifier_column}"
                )

        with record_step(
            step="select-features",
            message=(
                f"Selected features for the "
                f"{final_matrix_request.block} block."
            ),
        ):
            selected_column = weights["selected"]

            if pd.api.types.is_bool_dtype(selected_column):
                selected_mask = selected_column.fillna(False)
            else:
                selected_mask = (
                    selected_column.astype(str)
                    .str.strip()
                    .str.casefold()
                    .eq("true")
                )

            block_mask = (
                weights["block"]
                .astype(str)
                .str.strip()
                .str.casefold()
                .eq(
                    final_matrix_request.block
                    .strip()
                    .casefold()
                )
            )

            selected_features = (
                weights.loc[
                    block_mask & selected_mask,
                    "feature",
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            if not selected_features:
                raise ValueError(
                    "No selected features were found for block: "
                    f"{final_matrix_request.block}"
                )

        with record_step(
            step="validate-selected-features",
            message=(
                "Validated that all selected features exist in "
                "the weighted matrix."
            ),
        ):
            missing_features = [
                feature
                for feature in selected_features
                if feature not in weighted_matrix.columns
            ]

            if missing_features:
                raise ValueError(
                    "Selected features are missing from the "
                    f"weighted matrix: {missing_features}"
                )

        with record_step(
            step="create-final-matrix",
            message=(
                "Created the final matrix containing the "
                "identifier and selected features."
            ),
        ):
            final_matrix = weighted_matrix.loc[
                :,
                [
                    identifier_column,
                    *selected_features,
                ],
            ].copy()

        with record_step(
            step="write-final-matrix",
            message=(
                "Wrote the final selected matrix to a "
                "temporary CSV file."
            ),
        ):
            final_matrix.to_csv(
                output_file_path,
                index=False,
            )

        with record_step(
            step="upload-final-matrix",
            message=(
                "Uploaded the final selected matrix to Azure "
                "Blob Storage."
            ),
        ):
            with output_file_path.open(
                "rb"
            ) as output_file:
                output_blob_client.upload_blob(
                    data=output_file,
                    overwrite=True,
                )

    total_elapsed_seconds = (
        perf_counter() - total_started
    )

    return CreateFinalSelectedMatrixProcessResponseModel(
        message=(
            "The final selected matrix was created and "
            "uploaded successfully."
        ),
        outputFilePath=Path(output_blob_name),
        rowCount=len(final_matrix),
        columnCount=len(final_matrix.columns),
        selectedFeatureCount=len(selected_features),
        steps=process_steps,
        totalElapsedSeconds=total_elapsed_seconds,
    )