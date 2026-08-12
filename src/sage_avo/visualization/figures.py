"""Small, reusable figure set for notebooks, papers, and presentations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.structure.graph import GraphEdges


def plot_training_diversity(
    examples: Sequence[Mapping[str, np.ndarray]],
    keys: tuple[str, ...] = ("near", "mid", "far", "vp", "vs", "density", "facies", "rgt"),
) -> plt.Figure:
    """Show representative synthetic-domain variation with consistent columns."""
    figure, axes = plt.subplots(len(examples), len(keys), figsize=(2.4 * len(keys), 2.2 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, example in enumerate(examples):
        for column, key in enumerate(keys):
            cmap = "gray" if key in {"near", "mid", "far"} else "viridis"
            axes[row, column].imshow(example[key], aspect="auto", cmap=cmap)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(key.replace("_", " ").title())
    figure.tight_layout()
    return figure


def plot_inversion_comparison(
    truth: np.ndarray,
    low_prior: np.ndarray,
    prediction: np.ndarray,
    property_names: tuple[str, str, str] = ("Vp", "Vs", "Density"),
) -> plt.Figure:
    """Plot truth, low prior, full prediction, and absolute error per property."""
    arrays = [np.asarray(value) for value in (truth, low_prior, prediction)]
    if any(value.shape != arrays[0].shape for value in arrays) or arrays[0].shape[0] != 3:
        raise ValueError("Inputs must have matching [3, height, width] shapes")
    figure, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, name in enumerate(property_names):
        low = min(np.nanpercentile(value[row], 2) for value in arrays)
        high = max(np.nanpercentile(value[row], 98) for value in arrays)
        panels = (truth[row], low_prior[row], prediction[row], np.abs(prediction[row] - truth[row]))
        titles = ("Truth", "Low-frequency prior", "SAGE-AVO", "Absolute error")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            kwargs = {"vmin": low, "vmax": high, "cmap": "viridis"} if column < 3 else {"cmap": "magma"}
            image = axes[row, column].imshow(panel, aspect="auto", **kwargs)
            axes[row, column].set_title(f"{name}: {title}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.tight_layout()
    return figure


def plot_ablation_metrics(table: pd.DataFrame) -> plt.Figure:
    """Plot required low/full/no-GNN/no-RGT/no-physics metrics."""
    required = {"model", "rmse_vp", "rmse_vs", "rmse_density", "miou"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(table))
    axes[0].bar(x - 0.18, table["rmse_vp"], 0.36, label="Vp")
    axes[0].bar(x + 0.18, table["rmse_vs"], 0.36, label="Vs")
    axes[0].set_ylabel("RMSE (m/s)")
    axes[0].legend(frameon=False)
    axes[1].bar(x, table["rmse_density"], color="tab:purple")
    axes[1].set_ylabel("Density RMSE (g/cc)")
    axes[2].bar(x, table["miou"], color="tab:green")
    axes[2].set_ylabel("Segmentation mIoU")
    for axis in axes:
        axis.set_xticks(x, table["model"], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _draw_edges(axis: plt.Axes, edges: GraphEdges, width: int) -> None:
    for source, destination, weight in zip(edges.source, edges.destination, edges.weight):
        source_row, source_column = divmod(int(source), width)
        target_row, target_column = divmod(int(destination), width)
        axis.plot(
            (source_column, target_column),
            (source_row, target_row),
            color="cyan",
            linewidth=0.4 + 1.0 * float(weight),
            alpha=0.55,
        )


def plot_graph_mechanism(
    avo: np.ndarray,
    rgt: np.ndarray,
    avo_gradient: np.ndarray,
    edges: GraphEdges,
    vp_full: np.ndarray,
    vp_no_gnn: np.ndarray,
    vp_true: np.ndarray | None = None,
) -> plt.Figure:
    """Create the prescribed 2×4 graph mechanism and benefit figure.

    Only the strongest 15% of edges ranked by actual message-passing edge
    weight are displayed. All edges remain in model message passing.
    """
    if avo.shape[0] != 3 or rgt.shape != avo.shape[1:]:
        raise ValueError("avo must be [3, H, W] and rgt must match its spatial shape")
    strongest = edges.strongest(0.15)
    figure, axes = plt.subplots(2, 4, figsize=(15, 7))
    top = ((avo[0], "(a) Low-angle AVO"), (avo[1], "(b) Mid-angle AVO"), (avo[2], "(c) High-angle AVO"), (rgt, "(d) RGT"))
    for axis, (data, title) in zip(axes[0], top):
        axis.imshow(data, aspect="auto", cmap="gray" if "AVO" in title else "viridis")
        axis.set_title(title)
    axes[1, 0].imshow(avo_gradient, aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title("(e) AVO gradient G")
    axes[1, 1].imshow(rgt, aspect="auto", cmap="gray")
    _draw_edges(axes[1, 1], strongest, rgt.shape[1])
    axes[1, 1].set_title("(f) Strongest 15% of graph edges")
    axes[1, 2].imshow(vp_full, aspect="auto", cmap="viridis")
    axes[1, 2].set_title("(g) SAGE-AVO predicted Vp")
    if vp_true is None:
        benefit = vp_full - vp_no_gnn
        title = "(h) Graph sensitivity: Vp full − no-GNN"
    else:
        benefit = np.abs(vp_no_gnn - vp_true) - np.abs(vp_full - vp_true)
        title = "(h) Graph error reduction (>0 helps)"
    limit = np.nanpercentile(np.abs(benefit), 98)
    benefit_image = axes[1, 3].imshow(
        benefit, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axes[1, 3].set_title(title)
    label = "Vp graph sensitivity (m/s)" if vp_true is None else "Vp absolute-error reduction (m/s)"
    figure.colorbar(benefit_image, ax=axes[1, 3], shrink=0.75, label=label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    return figure
