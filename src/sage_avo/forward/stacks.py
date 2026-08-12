"""Explicit, non-overlapping angle-band definitions and front mute."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngleBand:
    name: str
    minimum_degrees: float
    maximum_degrees: float


DEFAULT_BANDS = (
    AngleBand("near", 3.0, 17.0),
    AngleBand("mid", 18.0, 31.0),
    AngleBand("far", 32.0, 45.0),
)


def validate_bands(bands: tuple[AngleBand, ...]) -> None:
    """Reject reversed or overlapping inclusive bands."""
    ordered = sorted(bands, key=lambda item: item.minimum_degrees)
    if any(item.maximum_degrees < item.minimum_degrees for item in ordered):
        raise ValueError("Angle-band maximum must not be below its minimum")
    if any(left.maximum_degrees >= right.minimum_degrees for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Inclusive angle bands must not overlap")


def stack_bands(
    gather: np.ndarray,
    angles_degrees: np.ndarray,
    bands: tuple[AngleBand, ...] = DEFAULT_BANDS,
) -> np.ndarray:
    """Mean-stack ``[angle, time, trace]`` data into named angle bands."""
    data = np.asarray(gather, dtype=float)
    angles = np.asarray(angles_degrees, dtype=float)
    if data.ndim != 3 or data.shape[0] != angles.size:
        raise ValueError("gather must have shape [angle, time, trace]")
    validate_bands(bands)
    outputs = []
    for band in bands:
        selection = (angles >= band.minimum_degrees) & (angles <= band.maximum_degrees)
        if not selection.any():
            raise ValueError(f"No samples fall in the {band.name!r} band")
        outputs.append(np.nanmean(data[selection], axis=0))
    return np.stack(outputs, axis=0)


def apply_front_mute(
    gather: np.ndarray,
    angles_degrees: np.ndarray,
    dt_seconds: float,
    start: tuple[float, float] = (30.0, 0.0),
    end: tuple[float, float] = (45.0, 0.1),
    taper_samples: int = 5,
) -> np.ndarray:
    """Apply the source workflow's angle-dependent early-time mute."""
    output = np.asarray(gather, dtype=float).copy()
    angles = np.asarray(angles_degrees, dtype=float)
    if output.ndim != 3 or output.shape[0] != angles.size:
        raise ValueError("gather must have shape [angle, time, trace]")
    mute_times = np.interp(angles, [start[0], end[0]], [start[1], end[1]])
    for index, mute_time in enumerate(mute_times):
        sample = int(np.clip(np.floor(mute_time / dt_seconds + 1e-6), 0, output.shape[1]))
        output[index, :sample] = 0.0
        stop = min(sample + taper_samples, output.shape[1])
        if stop > sample:
            output[index, sample:stop] *= np.linspace(0.0, 1.0, stop - sample)[:, None]
    return output
