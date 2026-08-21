import logging
import os
import tempfile
from time import perf_counter

import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient

from uod_rg24.models.intermediate_integration.design_matrix.design_matrix_models import (
    InputModel,
)
from uod_rg24.models.intermediate_integration.design_matrix.feature_extraction_process_models import (
    FeatureExtractionProcessInfoModel,
    FeatureExtractionProcessRequestModel,
    FeatureExtractionProcessResponseModel,
)
from uod_rg24.models.intermediate_integration.intermediate_integration_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def feature_extraction_process(
    blob_service_client: BlobServiceClient,
    input_blob: InputModel,
    feature_extraction_process_request: FeatureExtractionProcessRequestModel,
) -> FeatureExtractionProcessResponseModel:
    total_started: float = perf_counter()
    processing_started_at = utc_now()
    steps: list[ProcessStepModel] = []
    input_size_bytes: int = 0
    feature_names: list[str] = []
    matrix_columns: list[str] = []
    excluded_columns_found: list[str] = []
    excluded_columns_missing: list[str] = []
    input_blob_extension: str | None = input_blob.extension
    if input_blob_extension is None:
        raise ValueError("Input blob file extension is required.")
    input_extension: str = f".{input_blob_extension.lstrip('.').lower()}"
    if input_extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported input dataset type: {input_extension}. "
            "Only .csv and .tsv files are supported."
        )
    input_separator: str = "\t" if input_extension == ".tsv" else ","
    input_blob_path: str = (
        f"{input_blob.directory_name}/" f"{input_blob.file_name}" f"{input_extension}"
    )
    input_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=input_blob.azure_container_name,
        blob=input_blob_path,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        input_path: str = os.path.join(
            temporary_directory,
            f"input{input_extension}",
        )
        step_started_at = utc_now()
        step_timer: float = perf_counter()
        logger.info(
            "Downloading feature matrix. source_blob=%s",
            input_blob_path,
        )
        with open(input_path, "wb") as input_file:
            download_stream = input_blob_client.download_blob()
            download_stream.readinto(input_file)
        if not os.path.isfile(input_path):
            raise RuntimeError(
                "Input feature matrix was not downloaded successfully: " f"{input_path}"
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
            "Extracting feature names. input_file=%s",
            input_path,
        )
        matrix_columns = (
            pd.read_csv(
                input_path,
                sep=input_separator,
                nrows=0,
            )
            .columns.astype(str)
            .tolist()
        )
        excluded_columns: set[str] = set(
            feature_extraction_process_request.excluded_columns
        )
        feature_names = [
            column for column in matrix_columns if column not in excluded_columns
        ]
        excluded_columns_found = [
            column for column in matrix_columns if column in excluded_columns
        ]
        excluded_columns_missing = sorted(excluded_columns.difference(matrix_columns))
        logger.info(
            "Feature extraction completed. "
            "total_columns=%s feature_count=%s "
            "excluded_columns_found=%s "
            "excluded_columns_missing=%s",
            len(matrix_columns),
            len(feature_names),
            len(excluded_columns_found),
            len(excluded_columns_missing),
        )
        steps.append(
            ProcessStepModel(
                step="extract_feature_names",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Extracted {len(feature_names)} feature names "
                    f"from {len(matrix_columns)} matrix columns."
                ),
            )
        )
    feature_extraction_process_info = FeatureExtractionProcessInfoModel(
        featureNames=feature_names,
        featureCount=len(feature_names),
        totalColumnCount=len(matrix_columns),
        excludedColumns=excluded_columns_found,
        excludedColumnCount=len(excluded_columns_found),
        missingExcludedColumns=excluded_columns_missing,
    )
    processing_completed_at = utc_now()
    return FeatureExtractionProcessResponseModel(
        success=True,
        startedAt=processing_started_at,
        completedAt=processing_completed_at,
        totalDurationMs=(perf_counter() - total_started) * 1000,
        inputFile=FileProcessInfoModel(
            azureStorageAccountName=(input_blob.azure_storage_account_name),
            azureContainerName=input_blob.azure_container_name,
            directoryName=input_blob.directory_name,
            blobPath=input_blob_path,
            fileName=input_blob.file_name,
            extension=input_extension.lstrip("."),
            sizeBytes=input_size_bytes,
            sizeMb=input_size_bytes / 1024 / 1024,
        ),
        outputFile=None,
        featureExtractionProcessInfo=(feature_extraction_process_info),
        steps=steps,
    )
