"""Dependency-light exact-Zoeppritz three-band forward workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stacks import AngleBand, DEFAULT_BANDS, apply_front_mute, stack_bands
from .wavelet import convolve_time, ricker
from .zoeppritz import reflectivity_gather


@dataclass(frozen=True)
class ForwardConfig:
    angles_degrees: tuple[float, ...] = tuple(float(value) for value in range(3, 46))
    bands: tuple[AngleBand, ...] = DEFAULT_BANDS
    wavelet_hz: float = 14.0
    dt_seconds: float = 0.004
    wavelet_samples: int = 81
    apply_mute: bool = True


def forward_avo_three_band(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    config: ForwardConfig = ForwardConfig(),
) -> np.ndarray:
    """Generate raw near/mid/far stacks with exact P-P Zoeppritz reflectivity."""
    angles = np.asarray(config.angles_degrees, dtype=float)
    reflectivity = reflectivity_gather(vp, vs, density, angles)
    seismic = convolve_time(
        reflectivity,
        ricker(config.wavelet_hz, config.dt_seconds, config.wavelet_samples),
        time_axis=1,
    )
    if config.apply_mute:
        seismic = apply_front_mute(seismic, angles, config.dt_seconds)
    return stack_bands(seismic, angles, config.bands).astype(np.float32)
