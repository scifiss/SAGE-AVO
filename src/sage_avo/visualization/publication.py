"""Publication figures generated only from completed controlled-run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sage_avo.forward.shuey import shuey_intercept_gradient
from sage_avo.models.variants import build_sage_avo_variant
from sage_avo.structure.graph import GraphEdges

from .figures import plot_graph_mechanism, plot_inversion_comparison, plot_training_diversity


VARIANTS = ("low_prior", "full", "no_gnn", "no_rgt", "no_physics")
DISPLAY_NAMES = {
    "low_prior": "Low-frequency prior",
    "full": "Full SAGE-AVO",
    "no_gnn": "No GNN",
    "no_rgt": "No RGT steering",
    "no_physics": "No physics",
}


def _prediction(experiment: Path, variant: str, realization_id: int) -> np.ndarray:
    path = experiment / "predictions" / variant / f"realization_{realization_id:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Controlled prediction is missing: {path}")
    with np.load(path) as archive:
        return archive["elastic"]


def _save(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main_synthetic_inversion(
    experiment: Path,
    dataset: Path,
    realization_id: int,
    figures: Path,
) -> None:
    with np.load(dataset / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
        truth, low = archive["elastic"], archive["low"]
    full = _prediction(experiment, "full", realization_id)
    figure = plot_inversion_comparison(truth, low, full)
    figure.suptitle(
        f"Controlled test realization {realization_id}: median full-model Vp RMSE rule",
        y=1.01,
    )
    _save(figure, figures / "main_synthetic_inversion")


def controlled_ablation_figure(
    experiment: Path,
    dataset: Path,
    realization_id: int,
    figures: Path,
) -> None:
    with np.load(dataset / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
        truth = archive["elastic"]
    predictions = {variant: _prediction(experiment, variant, realization_id) for variant in VARIANTS}
    values = [truth[0]] + [prediction[0] for prediction in predictions.values()]
    lower = min(np.nanpercentile(value, 2) for value in values)
    upper = max(np.nanpercentile(value, 98) for value in values)
    figure = plt.figure(figsize=(20, 9))
    grid = figure.add_gridspec(2, 5, height_ratios=(3.0, 1.25), hspace=0.22)
    for column, variant in enumerate(VARIANTS):
        axis = figure.add_subplot(grid[0, column])
        image = axis.imshow(
            predictions[variant][0], aspect="auto", cmap="viridis", vmin=lower, vmax=upper
        )
        axis.set_title(DISPLAY_NAMES[variant])
        axis.set(xlabel="Trace", ylabel="Time sample" if column == 0 else None)
    figure.colorbar(image, ax=figure.axes[:5], shrink=0.75, label="Vp (m/s)")
    metrics = pd.read_csv(experiment / "per_realization_metrics.csv")
    selected = metrics[
        (metrics["realization_id"] == realization_id)
        & (metrics["domain"].isin(("vp", "vs", "density")))
        & (metrics["metric"].isin(("rmse", "mae", "r2", "ssim")))
    ]
    pivot = selected.pivot(index="variant", columns=["domain", "metric"], values="value").reindex(VARIANTS)
    columns = [(domain, metric) for domain in ("vp", "vs", "density") for metric in ("rmse", "mae", "r2", "ssim")]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(columns))
    formatted = []
    for row in pivot.to_numpy():
        formatted.append(
            [f"{value:.4f}" if metric in {"r2", "ssim"} else f"{value:.3f}" for value, (_, metric) in zip(row, columns)]
        )
    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    labels = [f"{domain.title()} {metric.upper()}" for domain, metric in columns]
    table = table_axis.table(
        cellText=formatted,
        rowLabels=[DISPLAY_NAMES[name] for name in VARIANTS],
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    figure.suptitle(f"Controlled ablation — test realization {realization_id}")
    _save(figure, figures / "controlled_ablation")


def _load_full_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_config = config["model"]
    model = build_sage_avo_variant(
        "full",
        hidden_channels=int(model_config["hidden_channels"]),
        graph_layers=int(model_config["graph_layers"]),
        graph_heads=int(model_config["graph_heads"]),
        max_rgt_shift=int(model_config["max_rgt_shift_samples"]),
        classes=int(model_config["classes"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def graph_mechanism_figure(
    config: dict[str, Any],
    experiment: Path,
    dataset: Path,
    realization_id: int,
    figures: Path,
    device: torch.device,
) -> None:
    with np.load(dataset / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
        avo, low, rgt, truth = archive["avo"], archive["low"], archive["rgt"], archive["elastic"]
    full = _prediction(experiment, "full", realization_id)
    no_gnn = _prediction(experiment, "no_gnn", realization_id)
    normalization = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    patch_height, patch_width = (int(value) for value in config["patches"]["shape"])
    top = (rgt.shape[0] - patch_height) // 2
    left = (rgt.shape[1] - patch_width) // 2
    spatial = np.s_[top : top + patch_height, left : left + patch_width]
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    avo_normalized = (avo[(slice(None),) + spatial] - x_mean) / x_std
    low_normalized = (low[(slice(None),) + spatial] - y_mean) / y_std
    full_normalized = (full[(slice(None),) + spatial] - y_mean) / y_std
    model = _load_full_model(
        config, experiment / "runs" / "full" / "best_sampling.pt", device
    )
    model_output = model(
        torch.from_numpy(full_normalized[None]).to(device),
        torch.ones(1, device=device),
        torch.from_numpy(avo_normalized[None]).to(device),
        torch.from_numpy(low_normalized[None]).to(device),
        torch.from_numpy(rgt[spatial][None].astype(np.float32)).to(device),
    )
    edge_index = model_output.edge_indices[0].cpu().numpy()
    edge_weights = model_output.edge_weights[0].cpu().numpy()
    edges = GraphEdges(edge_index[0], edge_index[1], edge_weights)
    _, gradient = shuey_intercept_gradient(avo_normalized)
    figure = plot_graph_mechanism(
        avo_normalized,
        rgt[spatial],
        gradient,
        edges,
        full[0][spatial],
        no_gnn[0][spatial],
        truth[0][spatial],
    )
    figure.suptitle(
        f"Graph mechanism — central training-scale patch from test realization {realization_id}",
        y=1.01,
    )
    _save(figure, figures / "graph_mechanism_and_benefit")


def synthetic_diversity_figure(dataset: Path, figures: Path) -> None:
    split_ids = json.loads((dataset / "split_ids.json").read_text(encoding="utf-8"))["train"]
    positions = np.linspace(0, len(split_ids) - 1, 3, dtype=int)
    selected_ids = [int(split_ids[index]) for index in positions]
    examples = []
    for realization_id in selected_ids:
        with np.load(dataset / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
            examples.append(
                {
                    "near": archive["avo"][0],
                    "mid": archive["avo"][1],
                    "far": archive["avo"][2],
                    "vp": archive["elastic"][0],
                    "vs": archive["elastic"][1],
                    "density": archive["elastic"][2],
                    "facies": archive["segmentation"],
                    "rgt": archive["rgt"],
                }
            )
    figure = plot_training_diversity(examples)
    figure.suptitle(
        f"Systematic training-set diversity examples: realization IDs {selected_ids}", y=1.01
    )
    _save(figure, figures / "synthetic_training_diversity")


def generate_all_publication_figures(
    *,
    config: dict[str, Any],
    experiment_directory: str | Path,
    dataset_directory: str | Path,
    figures_directory: str | Path,
    representative_id: int,
    device_name: str | None = None,
) -> None:
    experiment = Path(experiment_directory)
    dataset = Path(dataset_directory)
    figures = Path(figures_directory)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    main_synthetic_inversion(experiment, dataset, representative_id, figures)
    controlled_ablation_figure(experiment, dataset, representative_id, figures)
    graph_mechanism_figure(
        config, experiment, dataset, representative_id, figures, device
    )
    synthetic_diversity_figure(dataset, figures)
