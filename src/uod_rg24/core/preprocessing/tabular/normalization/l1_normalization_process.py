import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient
from numpy.typing import NDArray
from sklearn.preprocessing import Normalizer

from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)
from uod_rg24.models.preprocessing.tabular.normalization.l1_normalization_process_models import (
    L1NormalizationInfoModel,
    L1NormalizationProcessRequestModel,
    L1NormalizationProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.normalization.normalization_models import (
    InputModel,
    OutputModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def l1_normalization_process(
    blob_service_client: BlobServiceClient,
    input_blob: InputModel,
    output_blob: OutputModel,
    normalization_process_request: L1NormalizationProcessRequestModel,
) -> L1NormalizationProcessResponseModel:
    total_started = perf_counter()
    processing_started_at = utc_now()

    steps: list[ProcessStepModel] = []

    input_blob_extension = input_blob.extension

    if input_blob_extension is None:
        raise ValueError("Input file extension is required.")

    input_blob_extension = input_blob_extension.lstrip(".")

    input_blob_path = (
        f"{input_blob.directory_name}/"
        f"{input_blob.file_name}."
        f"{input_blob_extension}"
    )

    input_blob_client: BlobClient = blob_service_client.get_blob_client(
        container=input_blob.azure_container_name,
        blob=input_blob_path,
    )

    extension = Path(input_blob_path).suffix.lower()

    if extension not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported input type: {extension}. "
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

    chunk_size = normalization_process_request.chunk_size

    numeric_columns: list[str] | None = None

    rows_processed = 0
    transform_chunks_processed = 0

    input_size_bytes = 0
    output_size_bytes = 0

    normalizer = Normalizer(
        norm="l1",
        copy=normalization_process_request.copy_,
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
            input_blob_path,
        )

        with open(input_path, "wb") as file:
            download_stream = input_blob_client.download_blob()
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

        first_chunk = True

        for chunk in pd.read_csv(
            input_path,
            sep=separator,
            chunksize=chunk_size,
        ):
            if numeric_columns is None:
                if normalization_process_request.numeric_columns:
                    requested_numeric_columns = (
                        normalization_process_request.numeric_columns
                    )

                    missing_columns = [
                        column
                        for column in requested_numeric_columns
                        if column not in chunk.columns
                    ]

                    if missing_columns:
                        raise ValueError(
                            f"Columns not found in dataset: {missing_columns}"
                        )

                    non_numeric_columns = [
                        column
                        for column in requested_numeric_columns
                        if not pd.api.types.is_numeric_dtype(chunk[column])
                    ]

                    if non_numeric_columns:
                        raise ValueError(
                            f"Columns are not numeric: {non_numeric_columns}"
                        )

                    numeric_columns = requested_numeric_columns

                else:
                    numeric_columns = chunk.select_dtypes(
                        include="number",
                    ).columns.tolist()

                if not numeric_columns:
                    raise ValueError("Dataset contains no numeric columns.")

                logger.info(
                    "L1 normalization columns=%s",
                    numeric_columns,
                )

            normalized_values: NDArray[np.float64] = cast(
                NDArray[np.float64],
                normalizer.transform(
                    chunk[numeric_columns],
                ),
            )

            chunk.loc[:, numeric_columns] = normalized_values

            chunk.to_csv(
                output_path,
                sep=separator,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )

            rows_processed += len(chunk)
            transform_chunks_processed += 1
            first_chunk = False

        if numeric_columns is None:
            raise ValueError("Dataset contains no rows.")

        output_size_bytes = os.path.getsize(output_path)

        steps.append(
            ProcessStepModel(
                step="l1_normalization",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"L1-normalized {rows_processed} rows across "
                    f"{transform_chunks_processed} chunks."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Uploading L1-normalized dataset. destination_blob=%s",
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

    normalization_info = L1NormalizationInfoModel(
        norm="l1",
        numericColumns=numeric_columns,
        numericColumnCount=len(numeric_columns),
        rowsProcessed=rows_processed,
        transformChunksProcessed=transform_chunks_processed,
        chunkSize=chunk_size,
    )

    processing_completed_at = utc_now()

    return L1NormalizationProcessResponseModel(
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
            extension=input_blob_extension,
            sizeBytes=input_size_bytes,
            sizeMb=(input_size_bytes / 1024 / 1024),
        ),
        outputFile=FileProcessInfoModel(
            azureStorageAccountName=output_blob.azure_storage_account_name,
            azureContainerName=output_blob.azure_container_name,
            directoryName=output_blob.directory_name,
            blobPath=output_blob_path,
            fileName=output_blob.file_name,
            extension=input_blob_extension,
            sizeBytes=output_size_bytes,
            sizeMb=(output_size_bytes / 1024 / 1024),
        ),
        temporaryFiles=temporary_file_info,
        normalization_info=normalization_info,
        steps=steps,
    )
