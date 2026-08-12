"""Horizon projection, time conversion, and RGT calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from sage_avo.data.interpretation import PreparedWell


@dataclass(frozen=True)
class HorizonCalibration:
    name: str
    rgt_reference: float
    input_time_ms: np.ndarray
    picked_time_ms: np.ndarray
    qc: dict[str, float | int]


def project_horizon_depth(
    horizon: pd.DataFrame,
    line_xy: np.ndarray,
    *,
    max_distance: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a Petrel XYZ horizon onto every seismic-line position."""
    points = horizon[["X", "Y"]].to_numpy(float)
    values = horizon["Z"].to_numpy(float)
    line = np.asarray(line_xy, dtype=float)
    distances, indices = cKDTree(line).query(points)
    valid = np.isfinite(values) & (distances <= max_distance)
    depth = np.full(line.shape[0], np.nan, dtype=float)
    nearest_distance = np.full(line.shape[0], np.nan, dtype=float)
    for index in np.unique(indices[valid]):
        selected = valid & (indices == index)
        depth[index] = np.median(values[selected])
        nearest_distance[index] = np.min(distances[selected])
    finite = np.isfinite(depth)
    if finite.sum() < 2:
        raise ValueError("Fewer than two horizon points project within max_distance")
    x = np.arange(depth.size)
    depth[~finite] = np.interp(x[~finite], x[finite], depth[finite])
    return depth, nearest_distance


def _well_time_at_depth(well: PreparedWell, depth: float) -> float:
    logs = well.logs
    valid = np.isfinite(logs["DEPTH"]) & np.isfinite(logs["TWT_MS"])
    if valid.sum() < 2:
        return float("nan")
    z = logs.loc[valid, "DEPTH"].to_numpy(float)
    t = logs.loc[valid, "TWT_MS"].to_numpy(float)
    order = np.argsort(z)
    if depth < z[order][0] or depth > z[order][-1]:
        return float("nan")
    return float(np.interp(depth, z[order], t[order]))


def depth_surface_to_time(
    depth: np.ndarray,
    line_xy: np.ndarray,
    wells: list[PreparedWell],
    *,
    distance_scale: float = 1_500.0,
) -> np.ndarray:
    """Convert a depth horizon using spatially weighted per-well time curves."""
    depth = np.asarray(depth, dtype=float)
    line = np.asarray(line_xy, dtype=float)
    output = np.full(depth.shape, np.nan, dtype=float)
    well_xy = np.array([[well.x, well.y] for well in wells], dtype=float)
    for index, (z_value, xy) in enumerate(zip(depth, line)):
        estimates = np.array([_well_time_at_depth(well, z_value) for well in wells])
        distances = np.linalg.norm(well_xy - xy, axis=1)
        weights = np.exp(-((distances / distance_scale) ** 2))
        valid = np.isfinite(estimates) & (weights > 1e-8)
        if valid.any():
            output[index] = np.sum(estimates[valid] * weights[valid]) / np.sum(weights[valid])
    finite = np.isfinite(output)
    if finite.sum() < 2:
        raise ValueError("Well depth-time curves do not cover the projected horizon")
    x = np.arange(output.size)
    output[~finite] = np.interp(x[~finite], x[finite], output[finite])
    return output


def build_well_horizon_table(
    wells: list[PreparedWell],
    t6_horizon: pd.DataFrame,
    t7_horizon: pd.DataFrame,
    line_xy: np.ndarray,
    cdps: np.ndarray,
) -> pd.DataFrame:
    """Sample depth horizons at wells, convert them to TWT, and match the line."""
    line = np.asarray(line_xy, dtype=float)
    tree = cKDTree(line)
    horizon_data = {
        "T6": (t6_horizon[["X", "Y"]].to_numpy(float), t6_horizon["Z"].to_numpy(float)),
        "T7": (t7_horizon[["X", "Y"]].to_numpy(float), t7_horizon["Z"].to_numpy(float)),
    }
    rows = []
    for well in wells:
        distance, line_index = tree.query([well.x, well.y])
        row: dict[str, float | int | str] = {
            "WELL": well.name,
            "X": well.x,
            "Y": well.y,
            "LINE_INDEX": int(line_index),
            "MATCHED_CDP": int(cdps[line_index]),
            "DISTANCE_TO_LINE_M": float(distance),
        }
        for name, (points, values) in horizon_data.items():
            depth = griddata(points, values, (well.x, well.y), method="linear")
            if not np.isfinite(depth):
                depth = griddata(points, values, (well.x, well.y), method="nearest")
            row[f"{name}_DEPTH_M"] = float(depth)
            row[f"{name}_TWT_MS"] = _well_time_at_depth(well, float(depth))
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_rgt(rgt: np.ndarray, time_ms: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.array(
        [np.interp(value, time_ms, rgt[:, index]) for index, value in enumerate(times)],
        dtype=float,
    )


def calibrate_horizon_rgt(
    name: str,
    rgt: np.ndarray,
    time_ms: np.ndarray,
    initial_time_ms: np.ndarray,
    well_table: pd.DataFrame,
) -> HorizonCalibration:
    """Fit one constant RGT level and invert it on all monotonic traces."""
    rgt = np.asarray(rgt, dtype=float)
    times = np.asarray(time_ms, dtype=float)
    initial = np.asarray(initial_time_ms, dtype=float)
    line_refs = _sample_rgt(rgt, times, initial)
    well_refs = []
    for _, row in well_table.iterrows():
        t_value = row[f"{name}_TWT_MS"]
        index = int(row["LINE_INDEX"])
        if np.isfinite(t_value):
            well_refs.append(np.interp(float(t_value), times, rgt[:, index]))
    references = np.concatenate([line_refs[np.isfinite(line_refs)], np.asarray(well_refs)])
    reference = float(np.median(references))
    picked = np.array(
        [np.interp(reference, rgt[:, index], times) for index in range(rgt.shape[1])],
        dtype=float,
    )
    residual = picked - initial
    well_residuals = []
    for _, row in well_table.iterrows():
        t_value = row[f"{name}_TWT_MS"]
        if np.isfinite(t_value):
            well_residuals.append(picked[int(row["LINE_INDEX"])] - float(t_value))
    qc = {
        "input_rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "input_median_absolute_ms": float(np.median(np.abs(residual))),
        "well_rmse_ms": float(np.sqrt(np.mean(np.square(well_residuals)))) if well_residuals else float("nan"),
        "well_count": int(len(well_residuals)),
    }
    return HorizonCalibration(name, reference, initial, picked, qc)
