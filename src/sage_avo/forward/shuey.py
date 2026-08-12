"""Shuey-derived compact diagnostics; not the synthetic-data forward solver."""

from __future__ import annotations

import numpy as np


def shuey_intercept_gradient(
    avo_stacks: np.ndarray,
    representative_angles_degrees: tuple[float, float, float] = (10.0, 24.0, 38.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``amplitude = intercept + gradient * sin(angle)^2`` per pixel."""
    amplitudes = np.asarray(avo_stacks, dtype=float)
    if amplitudes.ndim < 2 or amplitudes.shape[0] != 3:
        raise ValueError("avo_stacks must have three channels on axis zero")
    angles = np.asarray(representative_angles_degrees, dtype=float)
    if angles.shape != (3,):
        raise ValueError("Exactly three representative angles are required")
    predictor = np.sin(np.deg2rad(angles)) ** 2
    centered_x = predictor - predictor.mean()
    centered_y = amplitudes - amplitudes.mean(axis=0, keepdims=True)
    gradient = np.sum(centered_x.reshape((3,) + (1,) * (amplitudes.ndim - 1)) * centered_y, axis=0)
    gradient /= np.sum(centered_x**2)
    intercept = amplitudes.mean(axis=0) - gradient * predictor.mean()
    return intercept, gradient


def angular_features(
    avo_stacks: np.ndarray,
    representative_angles_degrees: tuple[float, float, float] = (10.0, 24.0, 38.0),
) -> np.ndarray:
    """Return near/mid/far, Shuey P/G, and curvature channels."""
    amplitudes = np.asarray(avo_stacks, dtype=float)
    intercept, gradient = shuey_intercept_gradient(amplitudes, representative_angles_degrees)
    curvature = amplitudes[0] - 2.0 * amplitudes[1] + amplitudes[2]
    return np.concatenate((amplitudes, intercept[None], gradient[None], curvature[None]), axis=0)
