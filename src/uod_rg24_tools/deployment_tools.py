from functools import lru_cache
import os
from pathlib import Path
from typing import Any
import tomllib


@lru_cache(maxsize=1)
def get_project_metadata() -> dict[str, Any]:
    project_root = Path(
        os.getenv(
            "AzureWebJobsScriptRoot",
            Path.cwd(),
        )
    ).resolve()

    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return {
            "name": "unknown",
            "version": "unknown",
            "description": None,
        }

    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    project = pyproject.get("project", {})

    return {
        "name": project.get(
            "name",
            "unknown",
        ),
        "version": project.get(
            "version",
            "unknown",
        ),
        "description": project.get(
            "description",
        ),
    }


def get_project_version() -> str:
    return str(
        get_project_metadata().get(
            "version",
            "unknown",
        )
    )
