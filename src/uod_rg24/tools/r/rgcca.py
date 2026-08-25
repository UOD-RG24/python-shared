from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rpy2.robjects import (  # pyright: ignore[reportMissingTypeStubs]
    FloatVector,
    ListVector,
    conversion,
    default_converter,
    numpy2ri,
    r,
)
from rpy2.robjects.conversion import (  # pyright: ignore[reportMissingTypeStubs]
    localconverter,
)
from rpy2.robjects.packages import (  # pyright: ignore[reportMissingTypeStubs]
    importr,  # pyright: ignore[reportUnknownVariableType]
)
from rpy2.robjects.vectors import StrVector  # pyright: ignore[reportMissingTypeStubs]


@lru_cache(maxsize=1)
def get_rgcca_package(r_lock: threading.RLock) -> Any:
    user_library = os.environ["R_LIBS_USER"].replace("\\", "/")
    with r_lock, localconverter(default_converter):
        r(f"""
                .libPaths(
                    unique(
                        c(
                            "{user_library}",
                            .libPaths()
                        )
                    )
                )
                """)
        return importr(
            "RGCCA",
            on_conflict="warn",
        )


def get_r_version(r_lock: threading.RLock) -> str:
    with r_lock, localconverter(default_converter):
        version = cast(StrVector, r("R.version.string"))
        if len(version) == 0:
            return "Unknown"
        return cast(str, version[0])


def get_rgcca_version(r_lock: threading.RLock) -> str:
    with r_lock:
        get_rgcca_package(r_lock)
        with localconverter(default_converter):
            version = cast(
                StrVector,
                r('as.character(packageVersion("RGCCA"))'),
            )
            if len(version) == 0:
                return "Unknown"
            return cast(str, version[0])


def parse_boolean(
    value: Any,
    field_name: str,
) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be true or false.")


def parse_sparsity(
    value: object,
    block_count: int,
) -> list[float]:
    if not isinstance(value, list):
        raise TypeError("sparsity must be a JSON array.")
    items = cast(list[object], value)
    sparsity: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(
            item,
            (int, float),
        ):
            raise TypeError("Every sparsity value must be numeric.")

        sparsity.append(float(item))
    if len(sparsity) != block_count:
        raise ValueError(
            "The number of sparsity values must match "
            f"the number of blocks ({block_count})."
        )
    if any(not 0.0 < item <= 1.0 for item in sparsity):
        raise ValueError(
            "Every sparsity value must be greater than "
            "zero and less than or equal to one."
        )
    return sparsity


def align_matrices(
    mrna: pd.DataFrame,
    cna: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_patients = mrna.index.intersection(
        cna.index,
        sort=False,
    )
    if len(common_patients) < 2:
        raise ValueError(
            "At least two patients must exist in both " "mRNA and CNA matrices."
        )
    aligned_mrna = mrna.loc[common_patients].copy()
    aligned_cna = cna.loc[common_patients].copy()
    if not aligned_mrna.index.equals(aligned_cna.index):
        raise ValueError("Patient order does not match after alignment.")
    return aligned_mrna, aligned_cna


def load_matrix(
    file_path: Path,
    index_column: str,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Matrix file was not found: {file_path}")
    try:
        matrix = pd.read_csv(
            file_path,
            index_col=index_column,
        )
    except ValueError as exception:
        raise ValueError(
            f"Index column '{index_column}' was not found " f"in {file_path}."
        ) from exception
    if matrix.empty:
        raise ValueError(f"Matrix contains no data: {file_path}")
    if matrix.index.has_duplicates:
        duplicates = (
            matrix.index[matrix.index.duplicated()].astype(str).unique().tolist()
        )
        raise ValueError("Matrix contains duplicate patient IDs: " f"{duplicates[:10]}")
    if matrix.columns.has_duplicates:
        duplicates = (
            matrix.columns[matrix.columns.duplicated()].astype(str).unique().tolist()
        )
        raise ValueError("Matrix contains duplicate features: " f"{duplicates[:10]}")
    non_numeric_columns = (
        matrix.select_dtypes(exclude=[np.number]).columns.astype(str).tolist()
    )
    if non_numeric_columns:
        raise ValueError(
            "All feature columns must be numeric. "
            "Non-numeric columns: "
            f"{non_numeric_columns[:10]}"
        )
    matrix = matrix.astype(np.float64)
    values = matrix.to_numpy(
        dtype=np.float64,
        copy=False,
    )
    if not np.isfinite(values).all():
        raise ValueError("Matrix contains missing or infinite values: " f"{file_path}")
    if matrix.shape[1] == 0:
        raise ValueError(f"Matrix contains no feature columns: {file_path}")
    return matrix


def run_sgcca(
    blocks: dict[str, pd.DataFrame],
    n_components: int,
    sparsity: list[float],
    scheme: str,
    scale: bool,
    r_lock: threading.RLock,
) -> pd.DataFrame:
    if n_components < 1:
        raise ValueError("nComponents must be at least 1.")
    valid_schemes = {
        "horst",
        "factorial",
        "centroid",
    }
    if scheme not in valid_schemes:
        raise ValueError("scheme must be one of: " "horst, factorial or centroid.")
    numpy_blocks = {
        block_name: np.ascontiguousarray(
            block.to_numpy(
                dtype=np.float64,
                copy=False,
            ),
            dtype=np.float64,
        )
        for block_name, block in blocks.items()
    }
    converter = default_converter + numpy2ri.converter
    with r_lock, localconverter(converter):
        rgcca = get_rgcca_package(r_lock)
        r_blocks = {
            block_name: conversion.py2rpy(values)
            for block_name, values in (numpy_blocks.items())
        }
        blocks_r = ListVector(r_blocks)
        r("gc()")
        model = rgcca.rgcca(
            blocks=blocks_r,
            ncomp=n_components,
            sparsity=FloatVector(sparsity),
            scale=scale,
            scheme=scheme,
            verbose=False,
        )
        model_names = list(model.names())
        if "a" not in model_names:
            raise RuntimeError(
                "RGCCA output does not contain the "
                "'a' block-weight element. "
                f"Available elements: {model_names}"
            )
        block_weights_r = model[model_names.index("a")]
        weight_matrices = [
            np.array(
                block_weights_r[index],
                dtype=np.float64,
                copy=True,
            )
            for index in range(len(block_weights_r))
        ]
        del block_weights_r
        del model
        del blocks_r
        del r_blocks
        r("gc()")
    results: list[pd.DataFrame] = []
    for block_index, (
        block_name,
        block,
    ) in enumerate(blocks.items()):
        weight_matrix = weight_matrices[block_index]
        if weight_matrix.ndim == 1:
            weight_matrix = weight_matrix.reshape(
                -1,
                1,
            )
        expected_feature_count = len(block.columns)
        if weight_matrix.shape[0] != expected_feature_count:
            raise RuntimeError(
                f"RGCCA returned "
                f"{weight_matrix.shape[0]} weights for "
                f"{block_name}, but the block contains "
                f"{expected_feature_count} features."
            )
        for component_index in range(weight_matrix.shape[1]):
            component_weights = weight_matrix[
                :,
                component_index,
            ]
            results.append(
                pd.DataFrame(
                    {
                        "block": block_name,
                        "feature": (block.columns.astype(str)),
                        "component": (component_index + 1),
                        "weight": component_weights,
                        "absolute_weight": np.abs(component_weights),
                        "selected": ~np.isclose(
                            component_weights,
                            0.0,
                        ),
                    }
                )
            )
    if not results:
        raise RuntimeError("RGCCA returned no feature weights.")
    feature_weights = pd.concat(
        results,
        ignore_index=True,
    )
    feature_weights = feature_weights.sort_values(
        by=[
            "block",
            "component",
            "absolute_weight",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)
    return feature_weights


def load_sgcca_weights(
    weights_file_path: Path,
) -> pd.DataFrame:
    """Load and validate learned sGCCA weights."""
    if not weights_file_path.exists():
        raise FileNotFoundError(
            f"sGCCA weights file was not found: " f"{weights_file_path}"
        )
    weights = pd.read_csv(weights_file_path)
    required_columns = {
        "block",
        "feature",
        "component",
        "weight",
    }
    missing_columns = required_columns - set(weights.columns)
    if missing_columns:
        raise ValueError(
            "The weights file is missing columns: " f"{sorted(missing_columns)}"
        )
    if weights.empty:
        raise ValueError("The sGCCA weights file is empty.")
    weights = weights[
        [
            "block",
            "feature",
            "component",
            "weight",
        ]
    ].copy()
    weights["block"] = weights["block"].astype(str).str.strip().str.lower()
    weights["feature"] = weights["feature"].astype(str).str.strip()
    weights["component"] = pd.to_numeric(
        weights["component"],
        errors="raise",
    ).astype(int)
    weights["weight"] = pd.to_numeric(
        weights["weight"],
        errors="raise",
    ).astype(np.float64)
    if not np.isfinite(weights["weight"].to_numpy()).all():
        raise ValueError(
            "The weights file contains missing or " "infinite weight values."
        )
    if (weights["component"] < 1).any():
        raise ValueError("Component numbers must start from 1.")
    duplicate_rows = weights.duplicated(
        subset=[
            "block",
            "feature",
            "component",
        ]
    )
    if duplicate_rows.any():
        duplicates = weights.loc[
            duplicate_rows,
            [
                "block",
                "feature",
                "component",
            ],
        ].head(10)
        raise ValueError(
            "The weights file contains duplicate "
            "block-feature-component rows: "
            f"{duplicates.to_dict(orient='records')}"
        )
    return weights


def apply_sgcca_weights_to_block(
    matrix: pd.DataFrame,
    weights: pd.DataFrame,
    block_name: str,
    component: int,
) -> pd.DataFrame:
    if component < 1:
        raise ValueError("component must be at least 1.")
    normalized_block_name = block_name.strip().lower()
    component_weights = weights.loc[
        (weights["block"] == normalized_block_name)
        & (weights["component"] == component),
        [
            "feature",
            "weight",
        ],
    ].copy()
    if component_weights.empty:
        available_components = sorted(
            weights.loc[
                weights["block"] == normalized_block_name,
                "component",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"No weights were found for block "
            f"'{block_name}' and component "
            f"{component}. Available components: "
            f"{available_components}"
        )
    weight_series = component_weights.set_index("feature")["weight"]
    required_features = weight_series.index.tolist()
    missing_features = [
        feature for feature in required_features if feature not in matrix.columns
    ]
    if missing_features:
        raise ValueError(
            f"{block_name} is missing "
            f"{len(missing_features)} features "
            f"required by component {component}. "
            "First missing features: "
            f"{missing_features[:10]}"
        )
    ordered_features = [
        column for column in matrix.columns if column in weight_series.index
    ]
    feature_matrix = matrix.loc[
        :,
        ordered_features,
    ].astype(np.float64)
    feature_values = feature_matrix.to_numpy(
        dtype=np.float64,
        copy=False,
    )
    if not np.isfinite(feature_values).all():
        raise ValueError(
            f"{block_name} contains missing or " "infinite feature values."
        )
    ordered_weights = weight_series.reindex(ordered_features)
    if ordered_weights.isna().any():
        features_without_weights = ordered_weights[
            ordered_weights.isna()
        ].index.tolist()
        raise ValueError(
            "Weights could not be matched for: " f"{features_without_weights[:10]}"
        )
    weighted_features = feature_matrix.mul(
        ordered_weights,
        axis="columns",
    )
    weighted_features.index = matrix.index
    weighted_features.index.name = matrix.index.name
    return weighted_features.reset_index()
