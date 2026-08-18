"""Dependency-light exact-Zoeppritz three-band forward workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .stacks import AngleBand, DEFAULT_BANDS, apply_front_mute, stack_bands
from .wavelet import convolve_time, ricker
from .zoeppritz import reflectivity_gather

if TYPE_CHECKING:
    from .specification import ForwardModelSpecification


@dataclass(frozen=True)
class ForwardConfig:
    angles_degrees: tuple[float, ...] = tuple(float(value) for value in range(3, 46))
    bands: tuple[AngleBand, ...] = DEFAULT_BANDS
    wavelet_hz: float = 14.0
    dt_seconds: float = 0.004
    wavelet_samples: int = 81
    apply_mute: bool = True
    mute_start: tuple[float, float] = (30.0, 0.0)
    mute_end: tuple[float, float] = (45.0, 0.1)
    taper_samples: int = 5


@dataclass(frozen=True)
class ForwardResult:
    """Dense-angle and band-limited outputs from one exact forward run."""

    reflectivity: np.ndarray
    seismic: np.ndarray
    stacks: np.ndarray
    angles_degrees: np.ndarray
    band_names: tuple[str, ...]


def forward_avo_dense(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    config: ForwardConfig = ForwardConfig(),
) -> ForwardResult:
    """Generate exact P-P reflectivity, convolved gathers, and three stacks."""
    angles = np.asarray(config.angles_degrees, dtype=float)
    reflectivity = reflectivity_gather(vp, vs, density, angles)
    seismic = convolve_time(
        reflectivity,
        ricker(config.wavelet_hz, config.dt_seconds, config.wavelet_samples),
        time_axis=1,
    )
    if config.apply_mute:
        seismic = apply_front_mute(
            seismic,
            angles,
            config.dt_seconds,
            start=config.mute_start,
            end=config.mute_end,
            taper_samples=config.taper_samples,
        )
    stacks = stack_bands(seismic, angles, config.bands)
    return ForwardResult(
        reflectivity=reflectivity.astype(np.float32),
        seismic=seismic.astype(np.float32),
        stacks=stacks.astype(np.float32),
        angles_degrees=angles.astype(np.float32),
        band_names=tuple(item.name for item in config.bands),
    )


def forward_avo_three_band(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    config: ForwardConfig = ForwardConfig(),
) -> np.ndarray:
    """Generate raw near/mid/far stacks with exact P-P Zoeppritz reflectivity."""
    return forward_avo_dense(vp, vs, density, config).stacks


def forward_avo_dense_spec(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    specification: "ForwardModelSpecification",
    *,
    sample_origin: int = 0,
) -> ForwardResult:
    """Run the exact NumPy operator from the shared v003 specification."""
    from .specification import ForwardModelSpecification

    if not isinstance(specification, ForwardModelSpecification):
        raise TypeError("specification must be a ForwardModelSpecification")
    angles = np.asarray(specification.angles_degrees, dtype=float)
    reflectivity = reflectivity_gather(vp, vs, density, angles)
    convolved = []
    for index, angle in enumerate(angles):
        wavelet = specification.wavelet_for_angle(float(angle)).samples_array(
            specification.dt_seconds
        )
        convolved.append(convolve_time(reflectivity[index], wavelet, time_axis=0))
    seismic = np.stack(convolved, axis=0)
    if specification.apply_mute:
        seismic = apply_front_mute(
            seismic,
            angles,
            specification.dt_seconds,
            start=specification.mute_start,
            end=specification.mute_end,
            taper_samples=specification.taper_samples,
            sample_origin=sample_origin,
            mute_time_origin_seconds=specification.mute_time_origin_seconds,
        )
    stacks = stack_bands(seismic, angles, specification.bands)
    return ForwardResult(
        reflectivity=reflectivity.astype(np.float32),
        seismic=seismic.astype(np.float32),
        stacks=stacks.astype(np.float32),
        angles_degrees=angles.astype(np.float32),
        band_names=tuple(item.name for item in specification.bands),
    )
