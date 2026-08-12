"""Reusable publication-oriented SAGE-AVO figures."""

from .figures import (
    plot_ablation_metrics,
    plot_graph_mechanism,
    plot_inversion_comparison,
    plot_training_diversity,
)

__all__ = [
    "plot_ablation_metrics",
    "plot_graph_mechanism",
    "plot_inversion_comparison",
    "plot_training_diversity",
]

try:
    from .publication import generate_all_publication_figures

    __all__.append("generate_all_publication_figures")
except ImportError:
    pass
