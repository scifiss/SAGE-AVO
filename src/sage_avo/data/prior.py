"""One explicit low-frequency elastic-prior operator for synthetic and field data."""

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


def make_low_frequency_prior(
    elastic_model: np.ndarray,
    definition: PriorDefinition = PriorDefinition(),
) -> np.ndarray:
    """Filter Vp, Vs, and density with the configured 2-D Gaussian operator.

    Parameters
    ----------
    elastic_model:
        Array with shape ``[3, time, trace]`` ordered as Vp, Vs, density.
    """
    model = np.asarray(elastic_model, dtype=np.float64)
    if model.ndim != 3 or model.shape[0] != 3:
        raise ValueError("elastic_model must have shape [3, time, trace]")
    sigma = (0.0, definition.sigma_time_samples, definition.sigma_lateral_samples)
    return gaussian_filter(model, sigma=sigma, mode=definition.boundary_mode).astype(np.float32)


def make_truth_derived_prior(
    elastic_truth: np.ndarray,
    definition: PriorDefinition = PriorDefinition(),
) -> np.ndarray:
    """Build the disclosed oracle prior used by the synthetic experiment.

    This intentionally smooths synthetic truth. It defines a controlled
    AVO-guided prior-refinement experiment, not unconstrained AVO-only absolute
    inversion.
    """
    if not definition.truth_derived:
        raise ValueError("Synthetic truth-derived priors require truth_derived=True")
    return make_low_frequency_prior(elastic_truth, definition)
