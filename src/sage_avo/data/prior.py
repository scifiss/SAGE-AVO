"""Explicit truth-derived low-frequency elastic-prior construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class PriorDefinition:
    """Complete smoothing definition for a synthetic elastic prior."""

    source: str = "gaussian_filter_of_synthetic_truth"
    truth_derived: bool = True
    cutoff_hz: float = 2.0
    dt_seconds: float = 0.004
    sigma_constant: float = 0.133
    lateral_sigma_ratio: float = 2.0
    boundary_mode: str = "reflect"

    @property
    def sigma_time_samples(self) -> float:
        """Approximate vertical Gaussian sigma used in the source workflow."""
        if self.cutoff_hz <= 0 or self.dt_seconds <= 0:
            raise ValueError("cutoff_hz and dt_seconds must be positive")
        return self.sigma_constant / (self.cutoff_hz * self.dt_seconds)

    @property
    def sigma_lateral_samples(self) -> float:
        return self.lateral_sigma_ratio * self.sigma_time_samples

    def to_dict(self) -> dict[str, float | bool | str]:
        values = asdict(self)
        values["sigma_time_samples"] = self.sigma_time_samples
        values["sigma_lateral_samples"] = self.sigma_lateral_samples
        return values


def make_truth_derived_prior(
    elastic_truth: np.ndarray,
    definition: PriorDefinition = PriorDefinition(),
) -> np.ndarray:
    """Filter Vp, Vs, and density truth with identical spatial bandwidths.

    Parameters
    ----------
    elastic_truth:
        Array with shape ``[3, time, trace]`` ordered as Vp, Vs, density.

    Notes
    -----
    This intentionally uses synthetic truth. It creates an oracle background
    model for a controlled prior-refinement experiment, not an AVO-only input.
    """
    truth = np.asarray(elastic_truth, dtype=np.float64)
    if truth.ndim != 3 or truth.shape[0] != 3:
        raise ValueError("elastic_truth must have shape [3, time, trace]")
    sigma = (0.0, definition.sigma_time_samples, definition.sigma_lateral_samples)
    return gaussian_filter(truth, sigma=sigma, mode=definition.boundary_mode).astype(np.float32)
