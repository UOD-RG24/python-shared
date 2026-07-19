import pandas as pd
import io
import os
from uod_rg24_models.azure_cloud.StorageAccountModel import StorageAccountModel


async def save_digital_twins_sample(experiment_id: str, dataframe: pd.DataFrame) -> str:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if dataframe.empty:
        raise ValueError("dataframe cannot be empty.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    csv_buffer = io.BytesIO()
    dataframe.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8",
    )
    csv_buffer.seek(0)
    blob_name = f"{experiment_id}/dt_sample.csv"
    async with StorageAccountModel(
        storage_account_name=storage_account_name
    ) as storage_account_model:
        blob_client = await storage_account_model.upload_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
            data=csv_buffer,
            overwrite=True,
            content_type="text/csv",
        )
        return blob_client.url


async def read_digital_twins_sample(
    experiment_id: str,
) -> pd.DataFrame:
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    experiments_container_name = os.getenv("EXPERIMENTS_CONTAINER_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured.")
    if not experiments_container_name:
        raise ValueError("EXPERIMENTS_CONTAINER_NAME is not configured.")
    blob_name = f"{experiment_id}/dt_sample.csv"
    async with StorageAccountModel(
        storage_account_name=storage_account_name,
    ) as storage_account_model:
        csv_bytes = await storage_account_model.download_blob(
            container_name=experiments_container_name,
            blob_name=blob_name,
        )
    return pd.read_csv(io.BytesIO(csv_bytes))
