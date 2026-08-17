import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient
from sklearn.preprocessing import StandardScaler

from uod_rg24.models.preprocessing.tabular.standardization.standard_standardization_process_models import (
    StandardStandardizationInfoModel,
    StandardStandardizationProcessRequestModel,
    StandardStandardizationProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.standardization.standardization_models import (
    DatasetModel,
    StandardStandardizationModel,
)
from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def standard_standardization_process(
    blob_service_client: BlobServiceClient,
    dataset_blob: DatasetModel,
    output_blob: StandardStandardizationModel,
    standardization_process_request: StandardStandardizationProcessRequestModel,
) -> StandardStandardizationProcessResponseModel:
    total_started = perf_counter()
    processing_started_at = utc_now()
    steps: list[ProcessStepModel] = []
    if dataset_blob.extension is None:
        raise ValueError("Dataset file extension is required.")
    dataset_blob_path: str = (
        f"{dataset_blob.directory_name}/"
        f"{dataset_blob.file_name}"
        f".{dataset_blob.extension.lstrip('.')}"
    )
    dataset_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=dataset_blob.azure_container_name,
        blob=dataset_blob_path,
    )
    extension: str = Path(dataset_blob_path).suffix.lower()
    if extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported dataset type: {extension}. "
            "Only .csv and .tsv files are supported."
        )
    separator: str = "\t" if extension == ".tsv" else ","
    output_blob_path: str = (
        f"{output_blob.directory_name}/" f"{output_blob.file_name}" f"{extension}"
    )
    output_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=output_blob.azure_container_name,
        blob=output_blob_path,
    )
    chunk_size: int = standardization_process_request.chunk_size
    numeric_columns: list[str] | None = None
    rows_processed = 0
    fit_chunks_processed = 0
    transform_chunks_processed = 0
    input_size_bytes = 0
    output_size_bytes = 0
    scaler = StandardScaler(
        copy=standardization_process_request.copy_,
        with_mean=standardization_process_request.with_mean,
        with_std=standardization_process_request.with_std,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path: str = os.path.join(
            temp_dir,
            f"input{extension}",
        )
        output_path: str = os.path.join(
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
                message=(f"Downloaded " f"{input_size_bytes} bytes."),
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
                            "Columns not found in dataset: " f"{missing_columns}"
                        )
                    non_numeric_columns = [
                        column
                        for column in standardization_process_request.numeric_columns
                        if not pd.api.types.is_numeric_dtype(chunk[column])
                    ]
                    if non_numeric_columns:
                        raise ValueError(
                            "Columns are not numeric: " f"{non_numeric_columns}"
                        )
                    numeric_columns = standardization_process_request.numeric_columns
                else:
                    numeric_columns = chunk.select_dtypes(
                        include="number"
                    ).columns.tolist()
                logger.info(
                    "StandardScaler columns=%s",
                    numeric_columns,
                )
                if not numeric_columns:
                    raise ValueError("Dataset contains no " "numeric columns.")
            scaler.partial_fit(chunk[numeric_columns])
            fit_chunks_processed += 1
            rows_processed += len(chunk)
        steps.append(
            ProcessStepModel(
                step="standard_scaler_fit",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Fitted StandardScaler "
                    f"using {rows_processed} rows "
                    f"across "
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
                mode=("w" if first_chunk else "a"),
                header=first_chunk,
            )
            first_chunk = False
            transform_chunks_processed += 1
        output_size_bytes = os.path.getsize(output_path)
        steps.append(
            ProcessStepModel(
                step="standard_scaler_transform",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Transformed "
                    f"{rows_processed} rows "
                    f"across "
                    f"{transform_chunks_processed} "
                    f"chunks."
                ),
            )
        )
        step_started_at = utc_now()
        step_timer = perf_counter()
        logger.info(
            "Uploading standardised dataset. " "destination_blob=%s",
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
                message=(f"Uploaded " f"{output_size_bytes} bytes."),
            )
        )
        temporary_file_info = TemporaryFileInfoModel(
            temporaryDirectory=temp_dir,
            inputFilePath=input_path,
            outputFilePath=output_path,
        )
    samples_seen: Any = getattr(
        scaler,
        "n_samples_seen_",
        None,
    )
    if hasattr(
        samples_seen,
        "tolist",
    ):
        samples_seen = samples_seen.tolist()
    mean = getattr(
        scaler,
        "mean_",
        None,
    )
    variance = getattr(
        scaler,
        "var_",
        None,
    )
    scale = getattr(
        scaler,
        "scale_",
        None,
    )
    scaler_info = StandardStandardizationInfoModel(
        numericColumns=numeric_columns or [],
        numericColumnCount=len(numeric_columns or []),
        rowsProcessed=rows_processed,
        fitChunksProcessed=fit_chunks_processed,
        transformChunksProcessed=(transform_chunks_processed),
        chunkSize=chunk_size,
        mean=mean.tolist() if mean is not None else [],
        variance=variance.tolist() if variance is not None else [],
        scale=scale.tolist() if scale is not None else [],
        samplesSeen=samples_seen,
    )
    processing_completed_at = utc_now()
    return StandardStandardizationProcessResponseModel(
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
            extension=dataset_blob.extension,
            sizeBytes=input_size_bytes,
            sizeMb=(input_size_bytes / 1024 / 1024),
        ),
        outputFile=FileProcessInfoModel(
            azureStorageAccountName=(output_blob.azure_storage_account_name),
            azureContainerName=(output_blob.azure_container_name),
            directoryName=output_blob.directory_name,
            blobPath=output_blob_path,
            fileName=output_blob.file_name,
            extension=dataset_blob.extension,
            sizeBytes=output_size_bytes,
            sizeMb=(output_size_bytes / 1024 / 1024),
        ),
        temporaryFiles=temporary_file_info,
        standardizationInfo=scaler_info,
        steps=steps,
    )
