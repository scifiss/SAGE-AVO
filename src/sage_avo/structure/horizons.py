"""Horizon projection and RGT inversion helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def project_horizon_to_line(
    horizon_xyv: np.ndarray,
    line_xy: np.ndarray,
    max_distance: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project horizon samples ``[x, y, value]`` to the nearest line location."""
    horizon = np.asarray(horizon_xyv, dtype=float)
    line = np.asarray(line_xy, dtype=float)
    if horizon.ndim != 2 or horizon.shape[1] != 3:
        raise ValueError("horizon_xyv must have shape [sample, 3]")
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("line_xy must have shape [trace, 2]")
    distances, indices = cKDTree(line).query(horizon[:, :2])
    valid = np.isfinite(horizon[:, 2]) & (distances <= max_distance)
    values = np.full(line.shape[0], np.nan)
    for index in np.unique(indices[valid]):
        values[index] = np.median(horizon[valid & (indices == index), 2])
    return values, distances[valid]


def time_at_rgt(
    rgt_trace: np.ndarray,
    time_axis: np.ndarray,
    rgt_value: float,
    guess: float | None = None,
    window: float | None = None,
) -> float:
    """Invert a monotonic RGT trace, optionally inside a time window."""
    rgt_trace = np.asarray(rgt_trace, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    valid = np.isfinite(rgt_trace) & np.isfinite(time_axis)
    if guess is not None and window is not None:
        valid &= np.abs(time_axis - guess) <= window
    if not valid.any():
        return float("nan")
    indices = np.flatnonzero(valid)
    return float(time_axis[indices[np.argmin(np.abs(rgt_trace[valid] - rgt_value))]])
