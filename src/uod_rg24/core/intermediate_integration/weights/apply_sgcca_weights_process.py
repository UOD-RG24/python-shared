from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient
from tempfile import TemporaryDirectory
from uod_rg24.models.intermediate_integration.weights.weights_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.intermediate_integration.weights.apply_sgcca_weights_process_models import (
    ApplySGCCAWeightsBlockOutputModel,
    ApplySGCCAWeightsProcessRequestModel,
    ApplySGCCAWeightsProcessResponseModel,
)
from uod_rg24.tools.r.rgcca import (
    apply_sgcca_weights_to_block,
    load_matrix,
    load_sgcca_weights,
)
from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    ProcessStepModel
)

def process_sgcca_weighted_block(
    *,
    block_name: str,
    component: int,
    input_file_path: Path,
    output_file_path: Path,
    weights: pd.DataFrame,
    index_column: str,
) -> dict[str, Any]:
    matrix = load_matrix(
        file_path=input_file_path,
        index_column=index_column,
    )
    weighted_matrix = apply_sgcca_weights_to_block(
        matrix=matrix,
        weights=weights,
        block_name=block_name,
        component=component,
    )
    output_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    weighted_matrix.to_csv(
        output_file_path,
        index=False,
    )
    feature_columns = [
        column for column in weighted_matrix.columns if column != index_column
    ]
    zero_value_count = int(
        np.isclose(
            weighted_matrix[feature_columns].to_numpy(dtype=np.float64),
            0.0,
        ).sum()
    )
    return {
        "block": block_name,
        "componentApplied": component,
        "samplesProcessed": len(weighted_matrix),
        "featuresWeighted": len(feature_columns),
        "zeroWeightedValues": (zero_value_count),
        "featureColumns": feature_columns,
        "inputFilePath": str(input_file_path),
        "outputFilePath": str(output_file_path),
    }


def apply_sgcca_weights_process(
    blob_service_client: BlobServiceClient,
    weights_input_blob: InputModel,
    mrna_input_blob: InputModel,
    cna_input_blob: InputModel,
    mrna_output_blob: OutputModel,
    cna_output_blob: OutputModel,
    weights_process_request: ApplySGCCAWeightsProcessRequestModel,
) -> ApplySGCCAWeightsProcessResponseModel:
    from contextlib import contextmanager
    from datetime import UTC, datetime
    from time import perf_counter
    from typing import Iterator

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
        prefix="apply-sgcca-weights-",
    ) as temporary_directory:
        with record_step(
            step="prepare-temporary-files",
            message=(
                "Created temporary input and output file paths."
            ),
        ):
            temporary_directory_path = Path(
                temporary_directory,
            )

            weights_file_path = (
                temporary_directory_path
                / weights_input_blob.file_name
            )
            mrna_file_path = (
                temporary_directory_path
                / mrna_input_blob.file_name
            )
            cna_file_path = (
                temporary_directory_path
                / cna_input_blob.file_name
            )
            mrna_output_file_path = (
                temporary_directory_path
                / mrna_output_blob.file_name
            )
            cna_output_file_path = (
                temporary_directory_path
                / cna_output_blob.file_name
            )

        with record_step(
            step="create-input-blob-clients",
            message=(
                "Created Blob Storage clients for the weights, "
                "mRNA, and CNA input files."
            ),
        ):
            weights_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        weights_input_blob.azure_container_name
                    ),
                    blob=(
                        f"{weights_input_blob.directory_name}/"
                        f"{weights_input_blob.file_name}."
                        f"{weights_input_blob.extension}"
                    ),
                )
            )
            mrna_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        mrna_input_blob.azure_container_name
                    ),
                    blob=(
                        f"{mrna_input_blob.directory_name}/"
                        f"{mrna_input_blob.file_name}."
                        f"{mrna_input_blob.extension}"
                    ),
                )
            )
            cna_blob_client: BlobClient = (
                blob_service_client.get_blob_client(
                    container=(
                        cna_input_blob.azure_container_name
                    ),
                    blob=(
                        f"{cna_input_blob.directory_name}/"
                        f"{cna_input_blob.file_name}."
                        f"{cna_input_blob.extension}"
                    ),
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
                weights_file.write(
                    weights_blob_client.download_blob().readall()
                )

        with record_step(
            step="download-mrna-matrix",
            message="Downloaded the mRNA input matrix.",
        ):
            with mrna_file_path.open("wb") as mrna_file:
                mrna_file.write(
                    mrna_blob_client.download_blob().readall()
                )

        with record_step(
            step="download-cna-matrix",
            message="Downloaded the CNA input matrix.",
        ):
            with cna_file_path.open("wb") as cna_file:
                cna_file.write(
                    cna_blob_client.download_blob().readall()
                )

        with record_step(
            step="load-sgcca-weights",
            message=(
                "Loaded and validated the sGCCA feature weights."
            ),
        ):
            weights = load_sgcca_weights(
                weights_file_path=weights_file_path,
            )

        outputs: list[
            ApplySGCCAWeightsBlockOutputModel
        ] = []

        block_configurations = [
            (
                "mRNA",
                mrna_file_path,
                mrna_output_file_path,
                mrna_input_blob,
                mrna_output_blob,
            ),
            (
                "CNA",
                cna_file_path,
                cna_output_file_path,
                cna_input_blob,
                cna_output_blob,
            ),
        ]

        for (
            block_name,
            input_file_path,
            output_file_path,
            input_blob,
            output_blob,
        ) in block_configurations:
            with record_step(
                step=f"apply-{block_name.lower()}-weights",
                message=(
                    f"Applied component "
                    f"{weights_process_request.component} sGCCA "
                    f"weights to the {block_name} matrix."
                ),
            ):
                result = process_sgcca_weighted_block(
                    block_name=block_name,
                    component=(
                        weights_process_request.component
                    ),
                    input_file_path=input_file_path,
                    output_file_path=output_file_path,
                    weights=weights,
                    index_column=(
                        weights_process_request.index_column
                    ),
                )

            with record_step(
                step=f"upload-{block_name.lower()}-matrix",
                message=(
                    f"Uploaded the weighted {block_name} matrix "
                    "to Azure Blob Storage."
                ),
            ):
                output_blob_path = (
                    f"{output_blob.directory_name}/"
                    f"{output_blob.file_name}."
                    f"{output_blob.extension}"
                )

                output_blob_client: BlobClient = (
                    blob_service_client.get_blob_client(
                        container=(
                            output_blob.azure_container_name
                        ),
                        blob=output_blob_path,
                    )
                )

                with output_file_path.open(
                    "rb"
                ) as output_file:
                    output_blob_client.upload_blob(
                        data=output_file,
                        overwrite=True,
                    )

            result["inputFilePath"] = (
                f"{input_blob.directory_name}/"
                f"{input_blob.file_name}."
                f"{input_blob.extension}"
            )
            result["outputFilePath"] = output_blob_path

            outputs.append(
                ApplySGCCAWeightsBlockOutputModel.model_validate(
                    result,
                )
            )

    total_elapsed_seconds = (
        perf_counter() - total_started
    )

    return ApplySGCCAWeightsProcessResponseModel(
        message=(
            "sGCCA weights were multiplied with individual "
            "feature values successfully."
        ),
        componentApplied=weights_process_request.component,
        weightsFilePath=Path(
            f"{weights_input_blob.directory_name}/"
            f"{weights_input_blob.file_name}."
            f"{weights_input_blob.extension}"
        ),
        blocksProcessed=len(outputs),
        outputs=outputs,
        steps=process_steps,
        totalElapsedSeconds=total_elapsed_seconds,
    )