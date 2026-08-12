"""Canonical facies conventions used throughout SAGE-AVO."""

from __future__ import annotations

import numpy as np


def delta_from_sand_probability(sand_probability: np.ndarray) -> np.ndarray:
    """Convert sand probability to shale-like DELTA.

    The source well workflow interprets lower DELTA as cleaner sand. Therefore
    this public implementation defines ``DELTA = 1 - P(sand)`` everywhere.
    """
    probability = np.asarray(sand_probability, dtype=float)
    if np.nanmin(probability) < 0 or np.nanmax(probability) > 1:
        raise ValueError("sand_probability must lie in [0, 1]")
    return 1.0 - probability


def sand_probability_from_delta(delta: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_from_sand_probability` for normalized DELTA."""
    normalized = np.asarray(delta, dtype=float)
    if np.nanmin(normalized) < 0 or np.nanmax(normalized) > 1:
        raise ValueError("delta must lie in [0, 1]")
    return 1.0 - normalized
