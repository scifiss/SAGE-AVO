"""Conservative field-deployment consistency checks against processed wells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import elastic_metrics


PROPERTY_COLUMNS = (("vp", "VP"), ("vs", "VS"), ("density", "RHOB"))


def field_well_consistency(
    prediction: np.ndarray,
    *,
    time_ms: np.ndarray,
    line_xy: np.ndarray,
    wells_directory: str | Path,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray | str | int | float]]]:
    """Sample a field prediction at local wells and return non-blind QC metrics.

    The wells may have contributed to upstream structure and elastic modeling;
    these rows are field consistency checks, not independent validation.
    """
    elastic = np.asarray(prediction, dtype=float)
    times = np.asarray(time_ms, dtype=float)
    coordinates = np.asarray(line_xy, dtype=float)
    if elastic.ndim != 3 or elastic.shape[0] != 3:
        raise ValueError("prediction must have shape [3, time, trace]")
    if elastic.shape[1:] != (times.size, coordinates.shape[0]) or coordinates.shape[1] != 2:
        raise ValueError("time/line coordinates do not match the prediction")
    rows: list[dict[str, float | str | int]] = []
    overlays: list[dict[str, np.ndarray | str | int | float]] = []
    for path in sorted(Path(wells_directory).glob("*.csv")):
        frame = pd.read_csv(path)
        required = {"WELL", "X", "Y", "TWT_MS", "VP", "VS", "RHOB"}
        if not required.issubset(frame.columns):
            continue
        well = str(frame["WELL"].dropna().iloc[0])
        xy = frame[["X", "Y"]].dropna().median().to_numpy(float)
        distances = np.linalg.norm(coordinates - xy[None], axis=1)
        trace = int(np.argmin(distances))
        well_time = frame["TWT_MS"].to_numpy(float)
        for channel, (property_name, column) in enumerate(PROPERTY_COLUMNS):
            observed = frame[column].to_numpy(float)
            predicted = np.interp(well_time, times, elastic[channel, :, trace], left=np.nan, right=np.nan)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            if valid.any():
                metrics = elastic_metrics(predicted[valid], observed[valid])
                rows.append(
                    {
                        "well": well,
                        "property": property_name,
                        "trace_index": trace,
                        "distance_to_line_m": float(distances[trace]),
                        "samples": int(valid.sum()),
                        **metrics,
                    }
                )
                overlays.append(
                    {
                        "well": well,
                        "property": property_name,
                        "channel": channel,
                        "trace_index": trace,
                        "distance_to_line_m": float(distances[trace]),
                        "time_ms": well_time[valid],
                        "observed": observed[valid],
                        "predicted": predicted[valid],
                    }
                )
    if not rows:
        raise ValueError("No compatible processed well samples intersect the field model")
    return pd.DataFrame(rows), overlays
