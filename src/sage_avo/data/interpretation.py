"""Readers and QC for Petrel horizons and S01 LAS wells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedWell:
    """One well with coordinates, calibrated time, and standardized curves."""

    name: str
    x: float
    y: float
    logs: pd.DataFrame
    source_path: str
    qc: dict[str, float | int | str]


def read_petrel_points(path: str | Path) -> pd.DataFrame:
    """Read numeric points after a Petrel ``END HEADER`` marker."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Petrel point file not found: {source}")
    rows: list[list[float]] = []
    in_data = False
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not in_data:
                if stripped.upper() == "END HEADER":
                    in_data = True
                continue
            if not stripped or stripped.startswith("#"):
                continue
            try:
                values = [float(value) for value in stripped.split()]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric points found after END HEADER in {source}")
    width = max(len(row) for row in rows)
    padded = [row + [np.nan] * (width - len(row)) for row in rows]
    columns = ["X", "Y", "Z"] + [f"ATTRIBUTE_{index}" for index in range(1, width - 2)]
    frame = pd.DataFrame(padded, columns=columns)
    frame.attrs.update({"source_path": str(source), "z_interpretation": "depth_m"})
    return frame


def _require_lasio():
    try:
        import lasio
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "LAS processing requires lasio. Install `python -m pip install -e \".[field]\"`."
        ) from error
    return lasio


def _coordinates(location: str) -> tuple[float, float]:
    pattern = r"{name}\s*=\s*([+-]?\d+(?:\.\d+)?)"
    values = []
    for name in ("X", "Y"):
        match = re.search(pattern.format(name=name), location, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Could not parse {name} coordinate from LAS LOC={location!r}")
        values.append(float(match.group(1)))
    return values[0], values[1]


def _monotonic_depth_time(depth: np.ndarray, twt_ms: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Well time calibration requires scikit-learn") from error
    valid = np.isfinite(depth) & np.isfinite(twt_ms)
    if valid.sum() < 2:
        raise ValueError("A well requires at least two finite depth-time samples")
    order = np.argsort(depth[valid])
    z_fit = depth[valid][order]
    t_fit = twt_ms[valid][order]
    unique_z, inverse = np.unique(z_fit, return_inverse=True)
    median_t = np.array([np.median(t_fit[inverse == index]) for index in range(unique_z.size)])
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(unique_z, median_t)
    repaired_fit = model.predict(unique_z)
    repaired = np.full(depth.shape, np.nan, dtype=float)
    inside = np.isfinite(depth) & (depth >= unique_z[0]) & (depth <= unique_z[-1])
    repaired[inside] = model.predict(depth[inside])
    return repaired, {
        "depth_time_samples": int(valid.sum()),
        "raw_downward_steps": int(np.sum(np.diff(t_fit) < 0.0)),
        "isotonic_rmse_ms": float(np.sqrt(np.mean((repaired_fit - median_t) ** 2))),
        "isotonic_max_adjustment_ms": float(np.max(np.abs(repaired_fit - median_t))),
    }


def read_las_well(path: str | Path) -> PreparedWell:
    """Read and standardize one S01 well without changing the source file."""
    lasio = _require_lasio()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"LAS file not found: {source}")
    las = lasio.read(source)
    logs = las.df().reset_index().replace(-999.25, np.nan)
    logs.columns = [str(column).upper() for column in logs.columns]
    if "DEPTH" not in logs:
        logs = logs.rename(columns={logs.columns[0]: "DEPTH"})
    required = {"DEPTH", "DELTA", "PORO", "RHOB", "DPTM", "DT", "SDT"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"{source.name} is missing required curves: {sorted(missing)}")
    for column in required:
        logs[column] = pd.to_numeric(logs[column], errors="coerce")

    name = str(las.well["WELL"].value).strip() if "WELL" in las.well else source.stem
    location = str(las.well["LOC"].value) if "LOC" in las.well else ""
    x, y = _coordinates(location)
    porosity_scale = 0.01 if float(logs["PORO"].quantile(0.99)) > 1.0 else 1.0
    logs["PORO"] = np.clip(logs["PORO"] * porosity_scale, 0.0, 0.6)
    logs["DELTA"] = np.clip(logs["DELTA"], 0.0, 1.0)
    logs["SAND_PROBABILITY"] = 1.0 - logs["DELTA"]
    logs["VP"] = 304_800.0 / logs["DT"]
    logs["VS"] = 304_800.0 / logs["SDT"]
    logs["TWT_MS"], time_qc = _monotonic_depth_time(
        logs["DEPTH"].to_numpy(float), logs["DPTM"].to_numpy(float)
    )
    logs["WELL"] = name
    logs["X"] = x
    logs["Y"] = y
    qc: dict[str, float | int | str] = {
        **time_qc,
        "rows": int(len(logs)),
        "porosity_scale_applied": float(porosity_scale),
        "delta_interpretation": "shaliness; sand_probability=1-delta",
    }
    return PreparedWell(name, x, y, logs, str(source), qc)


def read_las_wells(paths: list[str | Path] | tuple[str | Path, ...]) -> list[PreparedWell]:
    wells = [read_las_well(path) for path in paths]
    if len({well.name for well in wells}) != len(wells):
        raise ValueError("Well names must be unique")
    return wells
