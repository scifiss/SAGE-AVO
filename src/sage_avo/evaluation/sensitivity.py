"""Model/prior sensitivity summaries; not posterior uncertainty."""

from __future__ import annotations

import numpy as np


def ensemble_sensitivity(members: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize checkpoint/prior-cutoff ensemble spread along axis zero."""
    values = np.asarray(members, dtype=float)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("At least two ensemble members are required")
    mean = np.nanmean(values, axis=0)
    standard_deviation = np.nanstd(values, axis=0, ddof=1)
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "relative_standard_deviation": standard_deviation / np.maximum(np.abs(mean), 1e-8),
    }
