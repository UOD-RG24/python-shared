import io
import os

import joblib
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings
from sklearn.pipeline import Pipeline


def save_model_to_blob_storage(
    request_id: str,
    model: Pipeline,
) -> str:
    connection_string = os.environ["MODEL_STORAGE_CONNECTION_STRING"]

    container_name = "experiments"
    blob_name = f"{request_id}_svm_model.joblib"

    # Serialize the complete pipeline into memory.
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    blob_client = container_client.get_blob_client(blob=blob_name)

    blob_client.upload_blob(
        data=model_buffer,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/octet-stream"),
        metadata={
            "request_id": request_id,
            "model_type": "support-vector-machine",
        },
    )

    blob_path = f"{container_name}/{blob_name}"

    return blob_path
