"""Explicit production angle-band definitions and front mute."""

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
    AngleBand("mid", 17.0, 31.0),
    AngleBand("far", 31.0, 45.0),
)


def validate_bands(bands: tuple[AngleBand, ...]) -> None:
    """Reject reversed bands and overlap beyond intentional shared endpoints."""
    ordered = sorted(bands, key=lambda item: item.minimum_degrees)
    if any(item.maximum_degrees < item.minimum_degrees for item in ordered):
        raise ValueError("Angle-band maximum must not be below its minimum")
    if any(left.maximum_degrees > right.minimum_degrees for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Angle bands may share endpoints but must not overlap beyond them")


def representative_band_angles(
    bands: tuple[AngleBand, ...] = DEFAULT_BANDS,
) -> tuple[float, ...]:
    """Return arithmetic band midpoints used by compact P/G summaries."""
    validate_bands(bands)
    return tuple((band.minimum_degrees + band.maximum_degrees) / 2.0 for band in bands)


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
    sample_origin: int = 0,
    mute_time_origin_seconds: float = 0.0,
) -> np.ndarray:
    """Apply the angle-dependent mute in the full-realization sample frame."""
    output = np.asarray(gather, dtype=float).copy()
    angles = np.asarray(angles_degrees, dtype=float)
    if output.ndim != 3 or output.shape[0] != angles.size:
        raise ValueError("gather must have shape [angle, time, trace]")
    mute_times = np.interp(angles, [start[0], end[0]], [start[1], end[1]])
    for index, mute_time in enumerate(mute_times):
        global_sample = int(
            np.floor((mute_time - mute_time_origin_seconds) / dt_seconds + 1e-6)
        )
        sample = global_sample - int(sample_origin)
        local_axis = np.arange(output.shape[1])
        if taper_samples == 0:
            taper = (local_axis >= sample).astype(float)
        else:
            denominator = max(taper_samples - 1, 1)
            taper = np.clip((local_axis - sample) / denominator, 0.0, 1.0)
        output[index] *= taper[:, None]
    return output
