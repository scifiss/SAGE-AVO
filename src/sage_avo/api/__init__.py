"""Versioned, application-independent scientific NumPy facade.

Discovery imports metadata only. Numerical implementations are loaded lazily;
importing this namespace never imports Torch/PyG or initializes a GPU.
"""

from __future__ import annotations

from importlib import import_module

from .catalog import API_VERSION, get_operation, list_operations, provenance

_OPERATORS = {
    "make_coherent_deformation",
    "warp_geological_arrays",
    "elastic_moduli",
    "elastic_from_moduli",
    "brine_fluid_state",
    "co2_fluid_state",
    "exact_pp_reflectivity",
    "simulate_three_band_avo",
    "compare_avo_outputs",
    "report_rgt_monotonicity",
    "plot_elastic_comparison",
    "substitute_calibrated_fluid",
    "CalibratedFluidApplication",
}
_CONTRACTS = {
    "AngleBand": ("sage_avo.forward.stacks", "AngleBand"),
    "WaveletSpecification": ("sage_avo.forward.specification", "WaveletSpecification"),
    "ForwardModelSpecification": ("sage_avo.forward.specification", "ForwardModelSpecification"),
    "forward_specification_from_mapping": (
        "sage_avo.forward.specification",
        "forward_specification_from_mapping",
    ),
    "CalibratedDryFrameModel": ("sage_avo.geology.fluid_calibration", "CalibratedDryFrameModel"),
    "FluidRockPhysics": ("sage_avo.geology.fluid_calibration", "FluidRockPhysics"),
    "SupportAcceptanceContract": ("sage_avo.geology.support", "SupportAcceptanceContract"),
    "support_contract_from_mapping": ("sage_avo.geology.support", "support_contract_from_mapping"),
}

__all__ = [
    "API_VERSION",
    "get_operation",
    "list_operations",
    "provenance",
    *sorted(_OPERATORS | _CONTRACTS.keys()),
]


def __getattr__(name: str):
    if name in _OPERATORS:
        return getattr(import_module("._operators", __name__), name)
    if name in _CONTRACTS:
        module, attribute = _CONTRACTS[name]
        return getattr(import_module(module), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
