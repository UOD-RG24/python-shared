from __future__ import annotations

from hashlib import sha256

from uod_rg24_contracts import StepCommandV1

from .json_codec import canonical_json_bytes


def step_command_fingerprint(
    command: StepCommandV1,
    *,
    code_version: str,
) -> str:
    """Return a retry-stable execution fingerprint for one logical step.

    Attempt/message IDs, request time, and trace context are deliberately
    excluded so a retry of the same logical step has the same fingerprint.
    Output reservations are included because changing an output target changes
    the execution identity.
    """

    if not code_version.strip():
        raise ValueError("code_version must not be empty")
    material = {
        "schemaVersion": command.schema_version,
        "runId": command.run_id,
        "stepRunId": command.step_run_id,
        "ownerId": command.owner_id,
        "experimentId": command.experiment_id,
        "datasetId": command.dataset_id,
        "layer": command.layer.value,
        "operation": command.operation,
        "inputs": [
            {
                "role": item.role,
                "artifactId": item.artifact_id,
                "sha256": item.expected_sha256,
                "schemaId": item.expected_schema_id,
            }
            for item in sorted(command.inputs, key=lambda item: item.role)
        ],
        "outputs": [
            {
                "name": item.name,
                "artifactId": item.artifact_id,
                "kind": item.kind,
                "schemaId": item.schema_id,
            }
            for item in sorted(command.outputs, key=lambda item: item.name)
        ],
        "parameters": command.parameters,
        "recipeId": command.execution_context.recipe_id,
        "recipeVersion": command.execution_context.recipe_version,
        "randomSeed": command.execution_context.random_seed,
        "executionExtensions": command.execution_context.extensions,
        "extensions": command.extensions,
        "codeVersion": code_version.strip(),
    }
    return sha256(canonical_json_bytes(material, trailing_newline=False)).hexdigest()
