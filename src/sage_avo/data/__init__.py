"""Dataset splitting, normalization, and multiscale patch metadata."""

from .field import (
    FieldLineStacks,
    SegySummary,
    inspect_segy,
    load_field_line_stacks,
    save_field_line_stacks,
    stack_segy_line,
)
from .layout import DataLayout
from .interpretation import PreparedWell, read_las_well, read_las_wells, read_petrel_points
from .normalization import NormalizationStats, compute_normalization_stats
from .prior import PriorDefinition, make_truth_derived_prior
from .splits import RealizationSplit, split_realizations


def __getattr__(name: str):
    """Load ML-only public symbols without requiring Torch for core data utilities."""
    if name == "IndexedRealizationPatches":
        from .indexed_dataset import IndexedRealizationPatches as indexed_realization_patches

        return indexed_realization_patches
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataLayout",
    "FieldLineStacks",
    "IndexedRealizationPatches",
    "NormalizationStats",
    "PreparedWell",
    "PriorDefinition",
    "RealizationSplit",
    "SegySummary",
    "compute_normalization_stats",
    "inspect_segy",
    "load_field_line_stacks",
    "make_truth_derived_prior",
    "read_las_well",
    "read_las_wells",
    "read_petrel_points",
    "save_field_line_stacks",
    "split_realizations",
    "stack_segy_line",
]
