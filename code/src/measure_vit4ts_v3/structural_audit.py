"""CPU-only structural audit for v3 incidence and temporal operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    build_linear_nctp,
    build_nearest_full_column_operator,
    literal_incidence,
    patch_count,
    released_incidence,
    validate_patch_grid,
    validate_temporal_operator,
)
from .dynamic_cache import stride1_pool_mask


STRUCTURAL_AUDIT_COLUMNS = (
    "audit_type",
    "operator",
    "scale",
    "availability",
    "na_reason",
    "patch_grid",
    "shape_rows",
    "shape_columns",
    "effective_patch_count",
    "row_crossings",
    "terminal_holes",
    "minimum",
    "nonnegative",
    "row_sum_min",
    "row_sum_max",
    "zero_rows",
)


def _pooling_masks(patch_grid: tuple[int, int]) -> Mapping[str, np.ndarray]:
    cells = patch_count(patch_grid)
    return {
        "P": np.arange(cells, dtype=np.int64).reshape(1, cells),
        "M": stride1_pool_mask(patch_grid, 2),
        "L": stride1_pool_mask(patch_grid, 3),
    }


def _row_crossings(patch_grid: tuple[int, int], *, released: bool) -> int:
    height, width = validate_patch_grid(patch_grid)
    cells = height * width
    shift = 1 if released else 0
    source = np.arange(cells, dtype=np.int64) + shift
    valid = source < cells
    return int(
        np.sum(
            (np.arange(cells, dtype=np.int64)[valid] // width)
            != (source[valid] // width)
        )
    )


def _terminal_holes(valid: np.ndarray) -> int:
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return int(valid.size)
    return int(valid.size - 1 - positions[-1])


def _incidence_rows(patch_grid: tuple[int, int]) -> list[dict[str, Any]]:
    grid = validate_patch_grid(patch_grid)
    rows: list[dict[str, Any]] = []
    for scale, mask in _pooling_masks(grid).items():
        for operator, builder, shifted in (
            ("literal", literal_incidence, False),
            ("released", released_incidence, True),
        ):
            incidence = builder(mask, grid).cpu().numpy().astype(bool, copy=False)
            valid = incidence.any(axis=1)
            row_sum = incidence.sum(axis=1, dtype=np.int64)
            rows.append(
                {
                    "audit_type": "incidence",
                    "operator": operator,
                    "scale": scale,
                    "availability": "static_core",
                    "na_reason": "",
                    "patch_grid": f"{grid[0]}x{grid[1]}",
                    "shape_rows": int(incidence.shape[0]),
                    "shape_columns": int(incidence.shape[1]),
                    "effective_patch_count": int(valid.sum()),
                    "row_crossings": _row_crossings(grid, released=shifted),
                    "terminal_holes": _terminal_holes(valid),
                    "minimum": float(incidence.min()),
                    "nonnegative": True,
                    "row_sum_min": float(row_sum.min()),
                    "row_sum_max": float(row_sum.max()),
                    "zero_rows": int(np.sum(row_sum == 0)),
                }
            )
    return rows


def _temporal_row(
    name: str,
    operator: np.ndarray,
    patch_grid: tuple[int, int],
    window_length: int,
    *,
    availability: str,
) -> dict[str, Any]:
    values = validate_temporal_operator(
        operator, window_length=window_length, patch_grid=patch_grid
    )
    row_sum = values.sum(axis=1, dtype=np.float64)
    return {
        "audit_type": "temporal_operator",
        "operator": name,
        "scale": "P",
        "availability": availability,
        "na_reason": "",
        "patch_grid": f"{patch_grid[0]}x{patch_grid[1]}",
        "shape_rows": int(values.shape[0]),
        "shape_columns": int(values.shape[1]),
        "effective_patch_count": int(np.sum(np.any(values > 0.0, axis=0))),
        "row_crossings": np.nan,
        "terminal_holes": np.nan,
        "minimum": float(values.min()),
        "nonnegative": bool(np.all(values >= 0.0)),
        "row_sum_min": float(row_sum.min()),
        "row_sum_max": float(row_sum.max()),
        "zero_rows": int(np.sum(row_sum == 0.0)),
    }


def _deferred_temporal_row(
    name: str,
    patch_grid: tuple[int, int],
    window_length: int,
) -> dict[str, Any]:
    trace = name in {"trace_soft", "trace_hard"}
    return {
        "audit_type": "temporal_operator",
        "operator": name,
        "scale": "P",
        "availability": "requires_frozen_trace_npz" if trace else "not_applicable",
        "na_reason": (
            "PER_SERIES_RENDERER_OPERATOR_NOT_STATIC"
            if trace
            else "NONLINEAR_REDUCER_AND_STITCH_HAS_NO_FIXED_OPERATOR_MATRIX"
        ),
        "patch_grid": f"{patch_grid[0]}x{patch_grid[1]}",
        "shape_rows": int(window_length) if trace else np.nan,
        "shape_columns": patch_count(patch_grid) if trace else np.nan,
        "effective_patch_count": np.nan,
        "row_crossings": np.nan,
        "terminal_holes": np.nan,
        "minimum": np.nan,
        "nonnegative": np.nan,
        "row_sum_min": np.nan,
        "row_sum_max": np.nan,
        "zero_rows": np.nan,
    }


def structural_audit_frame(
    *,
    patch_grid: tuple[int, int] = (14, 14),
    window_length: int = 240,
    image_size: tuple[int, int] = (224, 224),
    extra_temporal_operators: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return the mandatory static incidence/operator audit table.

    ``extra_temporal_operators`` permits later read-only per-series renderer
    ownership operators to replace the explicit trace NA rows using the same
    schema. Legacy reduction remains explicitly NA because quantile/top-tail
    reduction plus overlap stitching is nonlinear and has no fixed ``Q``.
    """

    grid = validate_patch_grid(patch_grid)
    length = int(window_length)
    if length <= 0:
        raise ValueError("window_length must be positive")
    operators: dict[str, np.ndarray] = {
        "nctp_linear": build_linear_nctp(length, grid, image_size=image_size),
        "nctp_nearest": build_nearest_full_column_operator(
            length, grid, image_size=image_size
        ),
    }
    if extra_temporal_operators:
        for name, operator in extra_temporal_operators.items():
            key = str(name)
            if not key or key in operators:
                raise ValueError("extra temporal-operator names must be unique and nonempty")
            operators[key] = np.asarray(operator)
    rows = _incidence_rows(grid)
    rows.extend(
        _temporal_row(
            name,
            operator,
            grid,
            length,
            availability=(
                "static_core"
                if name in {"nctp_linear", "nctp_nearest"}
                else "provided_read_only"
            ),
        )
        for name, operator in operators.items()
    )
    rows.extend(
        _deferred_temporal_row(name, grid, length)
        for name in ("trace_soft", "trace_hard", "legacy")
        if name not in operators
    )
    frame = pd.DataFrame(rows, columns=STRUCTURAL_AUDIT_COLUMNS)
    if tuple(frame.columns) != STRUCTURAL_AUDIT_COLUMNS or frame.empty:
        raise RuntimeError("structural audit schema construction failed")
    return frame


__all__ = ["STRUCTURAL_AUDIT_COLUMNS", "structural_audit_frame"]
