import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter

import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient
from sklearn.preprocessing import MinMaxScaler

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)
from uod_rg24.models.preprocessing.tabular.min_max_scaler_process_models import (
    MinMaxScalerInfoModel,
    MinMaxScalerStandardizationProcessRequestModel,
    MinMaxScalerStandardizationProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.standardization_models import (
    DatasetModel,
    MinMaxScalerModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def min_max_scaler_process(
    blob_service_client: BlobServiceClient,
    dataset_blob: DatasetModel,
    output_blob: MinMaxScalerModel,
    standardization_process_request: MinMaxScalerStandardizationProcessRequestModel,
) -> MinMaxScalerStandardizationProcessResponseModel:
    total_started = perf_counter()
    processing_started_at = utc_now()
    steps: list[ProcessStepModel] = []

    dataset_extension = dataset_blob.extension

    if dataset_extension is None:
        raise ValueError("Dataset file extension is required.")

    dataset_blob_path = (
        f"{dataset_blob.directory_name}/"
        f"{dataset_blob.file_name}"
        f".{dataset_extension.lstrip('.')}"
    )

    dataset_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=dataset_blob.azure_container_name,
        blob=dataset_blob_path,
    )

    extension = Path(dataset_blob_path).suffix.lower()

    if extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported dataset type: {extension}. "
            "Only .csv and .tsv files are supported."
        )

    separator = "\t" if extension == ".tsv" else ","

    output_blob_path = (
        f"{output_blob.directory_name}/" f"{output_blob.file_name}" f"{extension}"
    )

    output_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=output_blob.azure_container_name,
        blob=output_blob_path,
    )

    chunk_size = standardization_process_request.chunk_size

    numeric_columns: list[str] | None = None

    rows_processed = 0
    fit_chunks_processed = 0
    transform_chunks_processed = 0

    input_size_bytes = 0
    output_size_bytes = 0

    scaler = MinMaxScaler(
        feature_range=standardization_process_request.feature_range,
        copy=standardization_process_request.copy_,
        clip=standardization_process_request.clip,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(
            temp_dir,
            f"input{extension}",
        )

        output_path = os.path.join(
            temp_dir,
            f"output{extension}",
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Downloading dataset. source_blob=%s",
            dataset_blob_path,
        )

        with open(input_path, "wb") as file:
            download_stream = dataset_blob_client.download_blob()
            download_stream.readinto(file)

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

        for chunk in pd.read_csv(
            input_path,
            sep=separator,
            chunksize=chunk_size,
        ):
            if numeric_columns is None:
                if standardization_process_request.numeric_columns:
                    missing_columns = [
                        column
                        for column in standardization_process_request.numeric_columns
                        if column not in chunk.columns
                    ]

                    if missing_columns:
                        raise ValueError(
                            f"Columns not found in dataset: {missing_columns}"
                        )

                    non_numeric_columns = [
                        column
                        for column in standardization_process_request.numeric_columns
                        if not pd.api.types.is_numeric_dtype(chunk[column])
                    ]

                    if non_numeric_columns:
                        raise ValueError(
                            f"Columns are not numeric: {non_numeric_columns}"
                        )

                    numeric_columns = standardization_process_request.numeric_columns

                else:
                    numeric_columns = chunk.select_dtypes(
                        include="number",
                    ).columns.tolist()

                logger.info(
                    "MinMaxScaler columns=%s",
                    numeric_columns,
                )

                if not numeric_columns:
                    raise ValueError("Dataset contains no numeric columns.")

            scaler.partial_fit(chunk[numeric_columns])

            fit_chunks_processed += 1
            rows_processed += len(chunk)

        if numeric_columns is None:
            raise ValueError("Dataset contains no rows.")

        steps.append(
            ProcessStepModel(
                step="min_max_scaler_fit",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Fitted MinMaxScaler using "
                    f"{rows_processed} rows across "
                    f"{fit_chunks_processed} chunks."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        first_chunk = True

        for chunk in pd.read_csv(
            input_path,
            sep=separator,
            chunksize=chunk_size,
        ):
            chunk[numeric_columns] = scaler.transform(chunk[numeric_columns])

            chunk.to_csv(
                output_path,
                sep=separator,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )

            first_chunk = False
            transform_chunks_processed += 1

        output_size_bytes = os.path.getsize(output_path)

        steps.append(
            ProcessStepModel(
                step="min_max_scaler_transform",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Transformed {rows_processed} rows "
                    f"across {transform_chunks_processed} chunks."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Uploading Min-Max scaled dataset. " "destination_blob=%s",
            output_blob_path,
        )

        with open(output_path, "rb") as file:
            output_blob_client.upload_blob(
                file,
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
            temporaryDirectory=temp_dir,
            inputFilePath=input_path,
            outputFilePath=output_path,
        )

    scaler_info = MinMaxScalerInfoModel(
        numericColumns=numeric_columns,
        numericColumnCount=len(numeric_columns),
        rowsProcessed=rows_processed,
        fitChunksProcessed=fit_chunks_processed,
        transformChunksProcessed=transform_chunks_processed,
        chunkSize=chunk_size,
        featureRange=standardization_process_request.feature_range,
        minAdjustment=scaler.min_.tolist(),
        scale=scaler.scale_.tolist(),
        dataMin=scaler.data_min_.tolist(),
        dataMax=scaler.data_max_.tolist(),
        dataRange=scaler.data_range_.tolist(),
        samplesSeen=int(scaler.n_samples_seen_),
    )

    processing_completed_at = utc_now()

    return MinMaxScalerStandardizationProcessResponseModel(
        success=True,
        startedAt=processing_started_at,
        completedAt=processing_completed_at,
        totalDurationMs=(perf_counter() - total_started) * 1000,
        inputFile=FileProcessInfoModel(
            azureStorageAccountName=(dataset_blob.azure_storage_account_name),
            azureContainerName=(dataset_blob.azure_container_name),
            directoryName=dataset_blob.directory_name,
            blobPath=dataset_blob_path,
            fileName=dataset_blob.file_name,
            extension=dataset_extension,
            sizeBytes=input_size_bytes,
            sizeMb=(input_size_bytes / 1024 / 1024),
        ),
        outputFile=FileProcessInfoModel(
            azureStorageAccountName=(output_blob.azure_storage_account_name),
            azureContainerName=(output_blob.azure_container_name),
            directoryName=output_blob.directory_name,
            blobPath=output_blob_path,
            fileName=output_blob.file_name,
            extension=dataset_extension,
            sizeBytes=output_size_bytes,
            sizeMb=(output_size_bytes / 1024 / 1024),
        ),
        temporaryFiles=temporary_file_info,
        scaler=scaler_info,
        steps=steps,
    )
