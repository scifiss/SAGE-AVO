"""Exact/approximate AVO, wavelets, angle stacks, and forward QC."""

from importlib import import_module

from .pipeline import (
    ForwardConfig,
    ForwardResult,
    forward_avo_dense,
    forward_avo_dense_spec,
    forward_avo_three_band,
)
from .specification import (
    ForwardModelSpecification,
    WaveletSpecification,
    forward_specification_from_mapping,
)
from .madagascar import (
    forward_avo_madagascar,
    madagascar_availability,
    reflectivity_gather_madagascar,
)
from .perturbations import (
    ObservationPerturbationConfig,
    PerturbedObservation,
    apply_observation_perturbations,
    observation_config_from_mapping,
)
from .shuey import shuey_intercept_gradient
from .stacks import AngleBand, DEFAULT_BANDS
from .zoeppritz import zoeppritz_pp

__all__ = [
    "AngleBand",
    "DEFAULT_BANDS",
    "ForwardConfig",
    "ForwardModelSpecification",
    "ForwardResult",
    "ObservationPerturbationConfig",
    "PerturbedObservation",
    "WaveletSpecification",
    "forward_avo_dense",
    "forward_avo_dense_spec",
    "forward_avo_madagascar",
    "forward_avo_three_band",
    "forward_avo_three_band_spec_torch",
    "forward_specification_from_mapping",
    "exact_zoeppritz_pp_closed_form",
    "exact_zoeppritz_pp_matrix",
    "apply_observation_perturbations",
    "madagascar_availability",
    "observation_config_from_mapping",
    "reflectivity_gather_madagascar",
    "shuey_intercept_gradient",
    "zoeppritz_pp",
]


def __getattr__(name: str):
    """Load the optional differentiable operator only when it is requested."""
    if name in {
        "exact_zoeppritz_pp_closed_form",
        "exact_zoeppritz_pp_matrix",
        "forward_avo_three_band_spec_torch",
    }:
        return getattr(import_module(".torch_forward", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
