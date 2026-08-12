"""Exact/approximate AVO, wavelets, angle stacks, and forward QC."""

from .pipeline import ForwardConfig, ForwardResult, forward_avo_dense, forward_avo_three_band
from .madagascar import forward_avo_madagascar, madagascar_availability
from .shuey import shuey_intercept_gradient
from .stacks import AngleBand, DEFAULT_BANDS
from .zoeppritz import zoeppritz_pp

__all__ = [
    "AngleBand",
    "DEFAULT_BANDS",
    "ForwardConfig",
    "ForwardResult",
    "forward_avo_dense",
    "forward_avo_madagascar",
    "forward_avo_three_band",
    "madagascar_availability",
    "shuey_intercept_gradient",
    "zoeppritz_pp",
]
