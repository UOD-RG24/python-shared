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
from sklearn.preprocessing import RobustScaler

from uod_rg24.models.preprocessing.tabular.standardization.robust_standardization_process_models import (
    RobustStandardizationInfoModel,
    RobustStandardizationProcessRequestModel,
    RobustStandardizationProcessResponseModel,
)
from uod_rg24.models.preprocessing.tabular.standardization.standardization_models import (
    DatasetModel,
    RobustStandardizationModel,
)
from uod_rg24.models.preprocessing.preprocessing_shared_models import (
    FileProcessInfoModel,
    ProcessStepModel,
    TemporaryFileInfoModel,
)
from uod_rg24.tools.datetime_tools import utc_now

logger = logging.getLogger(__name__)


def robust_standardization_process(
    blob_service_client: BlobServiceClient,
    dataset_blob: DatasetModel,
    output_blob: RobustStandardizationModel,
    standardization_process_request: RobustStandardizationProcessRequestModel,
) -> RobustStandardizationProcessResponseModel:
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

    q_min, q_max = standardization_process_request.quantile_range

    scaler = RobustScaler(
        with_centering=standardization_process_request.with_centering,
        with_scaling=standardization_process_request.with_scaling,
        quantile_range=(q_min, q_max),
        copy=standardization_process_request.copy_,
        unit_variance=standardization_process_request.unit_variance,
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

        fit_data_path = os.path.join(
            temp_dir,
            "robust_scaler_fit_data.dat",
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

                    numeric_columns = list(
                        standardization_process_request.numeric_columns
                    )
                else:
                    numeric_columns = chunk.select_dtypes(
                        include="number",
                    ).columns.tolist()

                logger.info(
                    "RobustScaler columns=%s",
                    numeric_columns,
                )

                if not numeric_columns:
                    raise ValueError("Dataset contains no numeric columns.")

            missing_columns = [
                column for column in numeric_columns if column not in chunk.columns
            ]

            if missing_columns:
                raise ValueError(f"Columns not found in dataset: {missing_columns}")

            non_numeric_columns = [
                column
                for column in numeric_columns
                if not pd.api.types.is_numeric_dtype(chunk[column])
            ]

            if non_numeric_columns:
                raise ValueError(f"Columns are not numeric: {non_numeric_columns}")

            rows_processed += len(chunk)
            fit_chunks_processed += 1

        if numeric_columns is None or rows_processed == 0:
            raise ValueError("Dataset contains no rows.")

        steps.append(
            ProcessStepModel(
                step="robust_scaler_inspection",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Inspected {rows_processed} rows across "
                    f"{fit_chunks_processed} chunks."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        numeric_column_count = len(numeric_columns)

        fit_data = np.memmap(
            fit_data_path,
            dtype=np.float64,
            mode="w+",
            shape=(
                rows_processed,
                numeric_column_count,
            ),
        )

        row_offset = 0

        for chunk in pd.read_csv(
            input_path,
            sep=separator,
            chunksize=chunk_size,
        ):
            chunk_row_count = len(chunk)
            next_row_offset = row_offset + chunk_row_count

            numeric_values: NDArray[np.float64] = chunk[numeric_columns].to_numpy(
                dtype=np.float64,
                copy=False,
            )

            fit_data[
                row_offset:next_row_offset,
                :,
            ] = numeric_values

            row_offset = next_row_offset

        fit_data.flush()

        steps.append(
            ProcessStepModel(
                step="robust_scaler_fit_data",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Prepared {rows_processed} rows and "
                    f"{numeric_column_count} numeric columns "
                    "for RobustScaler fitting."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Fitting RobustScaler. rows=%s columns=%s " "quantile_range=(%s, %s)",
            rows_processed,
            numeric_column_count,
            q_min,
            q_max,
        )

        scaler.fit(fit_data)

        steps.append(
            ProcessStepModel(
                step="robust_scaler_fit",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(f"Fitted RobustScaler using " f"{rows_processed} rows."),
            )
        )

        del fit_data

        step_started_at = utc_now()
        step_timer = perf_counter()

        first_chunk = True

        for chunk in pd.read_csv(
            input_path,
            sep=separator,
            chunksize=chunk_size,
        ):
            numeric_values = cast(
                NDArray[np.float64],
                chunk[numeric_columns].to_numpy(
                    dtype=np.float64,
                    copy=False,
                ),
            )

            transformed_values = cast(
                NDArray[np.float64],
                scaler.transform(numeric_values),
            )

            chunk[numeric_columns] = transformed_values

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
                step="robust_scaler_transform",
                startedAt=step_started_at,
                completedAt=utc_now(),
                durationMs=(perf_counter() - step_timer) * 1000,
                message=(
                    f"Transformed {rows_processed} rows across "
                    f"{transform_chunks_processed} chunks."
                ),
            )
        )

        step_started_at = utc_now()
        step_timer = perf_counter()

        logger.info(
            "Uploading RobustScaler standardized dataset. " "destination_blob=%s",
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

    center_values = cast(
        NDArray[np.float64] | None,
        scaler.center_,
    )

    scale_values = cast(
        NDArray[np.float64] | None,
        scaler.scale_,
    )

    center: list[float] | None = (
        center_values.tolist() if center_values is not None else None
    )

    scale: list[float] | None = (
        scale_values.tolist() if scale_values is not None else None
    )

    scaler_info = RobustStandardizationInfoModel(
        numericColumns=numeric_columns,
        numericColumnCount=len(numeric_columns),
        rowsProcessed=rows_processed,
        fitChunksProcessed=fit_chunks_processed,
        transformChunksProcessed=transform_chunks_processed,
        chunkSize=chunk_size,
        withCentering=standardization_process_request.with_centering,
        withScaling=standardization_process_request.with_scaling,
        quantileRange=standardization_process_request.quantile_range,
        unitVariance=standardization_process_request.unit_variance,
        center=center,
        scale=scale,
        nFeaturesIn=int(scaler.n_features_in_),
    )

    processing_completed_at = utc_now()

    return RobustStandardizationProcessResponseModel(
        success=True,
        startedAt=processing_started_at,
        completedAt=processing_completed_at,
        totalDurationMs=(perf_counter() - total_started) * 1000,
        inputFile=FileProcessInfoModel(
            azureStorageAccountName=dataset_blob.azure_storage_account_name,
            azureContainerName=dataset_blob.azure_container_name,
            directoryName=dataset_blob.directory_name,
            blobPath=dataset_blob_path,
            fileName=dataset_blob.file_name,
            extension=dataset_extension,
            sizeBytes=input_size_bytes,
            sizeMb=(input_size_bytes / 1024 / 1024),
        ),
        outputFile=FileProcessInfoModel(
            azureStorageAccountName=output_blob.azure_storage_account_name,
            azureContainerName=output_blob.azure_container_name,
            directoryName=output_blob.directory_name,
            blobPath=output_blob_path,
            fileName=output_blob.file_name,
            extension=dataset_extension,
            sizeBytes=output_size_bytes,
            sizeMb=(output_size_bytes / 1024 / 1024),
        ),
        temporaryFiles=temporary_file_info,
        standardizationInfo=scaler_info,
        steps=steps,
    )
