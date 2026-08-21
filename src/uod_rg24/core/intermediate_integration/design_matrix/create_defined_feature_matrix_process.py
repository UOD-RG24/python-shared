import logging
import os
import tempfile
from time import perf_counter

import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient

from uod_rg24.models.intermediate_integration.design_matrix.create_defined_feature_matrix_process_models import (
    CreateDefinedFeatureMatrixProcessInfoModel,
    CreateDefinedFeatureMatrixProcessRequestModel,
    CreateDefinedFeatureMatrixProcessResponseModel,
)
from uod_rg24.models.intermediate_integration.design_matrix.design_matrix_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def create_defined_feature_matrix_process(
    blob_service_client: BlobServiceClient,
    input_blob: InputModel,
    output_blob: OutputModel,
    create_defined_feature_matrix_process_request: (
        CreateDefinedFeatureMatrixProcessRequestModel
    ),
) -> CreateDefinedFeatureMatrixProcessResponseModel:
    total_started: float = perf_counter()
    processing_started_at = utc_now()

    steps: list[ProcessStepModel] = []

    input_size_bytes: int = 0
    output_size_bytes: int = 0
    samples_processed: int = 0

    requested_features: list[str] = list(
        dict.fromkeys(create_defined_feature_matrix_process_request.defined_features)
    )

    input_blob_extension: str | None = input_blob.extension

    if input_blob_extension is None:
        raise ValueError("Input blob file extension is required.")

    input_extension: str = f".{input_blob_extension.lstrip('.').lower()}"

    if input_extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported input dataset type: {input_extension}. "
            "Only .csv and .tsv files are supported."
        )

    output_blob_extension: str = output_blob.extension or input_blob_extension
    output_extension: str = f".{output_blob_extension.lstrip('.').lower()}"

    if output_extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported output dataset type: {output_extension}. "
            "Only .csv and .tsv files are supported."
        )

    input_blob_path: str = (
        f"{input_blob.directory_name}/" f"{input_blob.file_name}" f"{input_extension}"
    )

    output_blob_path: str = (
        f"{output_blob.directory_name}/"
        f"{output_blob.file_name}"
        f"{output_extension}"
    )

    input_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=input_blob.azure_container_name,
        blob=input_blob_path,
    )

    output_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=output_blob.azure_container_name,
        blob=output_blob_path,
    )

    input_separator: str = "\t" if input_extension == ".tsv" else ","
    output_separator: str = "\t" if output_extension == ".tsv" else ","

    temporary_file_info: TemporaryFileInfoModel
    selected_features: list[str] = []
    missing_features: list[str] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        input_path: str = os.path.join(
            temporary_directory,
            f"input{input_extension}",
        )

        output_path: str = os.path.join(
            temporary_directory,
            f"output{output_extension}",
        )

        step_started_at = utc_now()
        step_timer: float = perf_counter()

        logger.info(
            "Downloading input blob. source_blob=%s",
            input_blob_path,
        )

        with open(input_path, "wb") as input_file:
            download_stream = input_blob_client.download_blob()
            download_stream.readinto(input_file)

        if not os.path.isfile(input_path):
            raise RuntimeError(
                f"Input file was not downloaded successfully: {input_path}"
            )

        input_size_bytes = os.path.getsize(input_path)

        steps.append(
            ProcessStepModel(
                step="download",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=f"Downloaded {input_size_bytes} bytes.",
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Creating defined-feature matrix. "
            "input_file=%s requested_feature_count=%s",
            input_path,
            len(requested_features),
        )

        dataframe: pd.DataFrame = pd.read_csv(
            input_path,
            sep=input_separator,
        )

        samples_processed = len(dataframe)
        available_columns: set[str] = set(dataframe.columns)

        selected_features = [
            feature for feature in requested_features if feature in available_columns
        ]

        missing_features = [
            feature
            for feature in requested_features
            if feature not in available_columns
        ]

        if missing_features:
            raise ValueError(
                "The following requested features were not found in the "
                f"input matrix: {', '.join(missing_features)}"
            )

        defined_feature_matrix: pd.DataFrame = dataframe.loc[
            :,
            selected_features,
        ].copy()

        defined_feature_matrix.to_csv(
            output_path,
            sep=output_separator,
            index=False,
        )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                "Defined-feature matrix output file was not created: " f"{output_path}"
            )

        output_size_bytes = os.path.getsize(output_path)

        logger.info(
            "Defined-feature matrix created. "
            "samples_processed=%s selected_feature_count=%s",
            samples_processed,
            len(selected_features),
        )

        steps.append(
            ProcessStepModel(
                step="select_defined_features",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Selected {len(selected_features)} features for "
                    f"{samples_processed} samples."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Uploading output blob. destination_blob=%s",
            output_blob_path,
        )

        with open(output_path, "rb") as output_file:
            output_blob_client.upload_blob(
                data=output_file,
                overwrite=True,
            )

        steps.append(
            ProcessStepModel(
                step="upload",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=f"Uploaded {output_size_bytes} bytes.",
            )
        )

        temporary_file_info = TemporaryFileInfoModel(
            temporaryDirectory=temporary_directory,
            inputFilePath=input_path,
            outputFilePath=output_path,
        )

    create_defined_feature_matrix_process_info = (
        CreateDefinedFeatureMatrixProcessInfoModel(
            requestedFeatures=requested_features,
            selectedFeatures=selected_features,
            missingFeatures=missing_features,
            requestedFeatureCount=len(requested_features),
            selectedFeatureCount=len(selected_features),
            missingFeatureCount=len(missing_features),
            samplesProcessed=samples_processed,
        )
    )

    processing_completed_at = utc_now()

    return CreateDefinedFeatureMatrixProcessResponseModel(
        success=True,
        startedAt=processing_started_at,
        completedAt=processing_completed_at,
        totalDurationMs=(perf_counter() - total_started) * 1000,
        inputFile=FileProcessInfoModel(
            azureStorageAccountName=input_blob.azure_storage_account_name,
            azureContainerName=input_blob.azure_container_name,
            directoryName=input_blob.directory_name,
            blobPath=input_blob_path,
            fileName=input_blob.file_name,
            extension=input_extension.lstrip("."),
            sizeBytes=input_size_bytes,
            sizeMb=input_size_bytes / 1024 / 1024,
        ),
        outputFile=FileProcessInfoModel(
            azureStorageAccountName=output_blob.azure_storage_account_name,
            azureContainerName=output_blob.azure_container_name,
            directoryName=output_blob.directory_name,
            blobPath=output_blob_path,
            fileName=output_blob.file_name,
            extension=output_extension.lstrip("."),
            sizeBytes=output_size_bytes,
            sizeMb=output_size_bytes / 1024 / 1024,
        ),
        temporaryFiles=temporary_file_info,
        createDefinedFeatureMatrixProcessInfo=(
            create_defined_feature_matrix_process_info
        ),
        steps=steps,
    )
