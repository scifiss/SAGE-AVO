"""Wavelet construction and trace-axis convolution."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d


def ricker(frequency_hz: float = 14.0, dt_seconds: float = 0.004, samples: int = 81) -> np.ndarray:
    """Create an odd-length, L1-normalized zero-phase Ricker wavelet."""
    if frequency_hz <= 0 or dt_seconds <= 0 or samples < 3 or samples % 2 == 0:
        raise ValueError("frequency/dt must be positive and samples must be odd and >= 3")
    time = (np.arange(samples) - samples // 2) * dt_seconds
    argument = (np.pi * frequency_hz * time) ** 2
    wavelet = (1.0 - 2.0 * argument) * np.exp(-argument)
    return wavelet / np.sum(np.abs(wavelet))


def convolve_time(data: np.ndarray, wavelet: np.ndarray, time_axis: int = -2) -> np.ndarray:
    """Convolve an array along its seismic-time axis using reflective boundaries."""
    return convolve1d(np.asarray(data), np.asarray(wavelet), axis=time_axis, mode="constant")
