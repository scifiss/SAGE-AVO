#!/usr/bin/env python3
"""Generate CPU-only live diagnostics for an active Revision-2 training run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sage_avo.config import load_config
from sage_avo.data import IndexedRealizationPatches
from sage_avo.models import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.models.graph import build_rgt_edges
from sage_avo.models.sage_avo import angular_features


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
DATASET = (
    PRIVATE_ROOT
    / "stage_artifacts"
    / "stage03"
    / "ds_v002_production100_multiscale"
    / "dataset"
)
RUN = (
    PRIVATE_ROOT
    / "stage_artifacts"
    / "stage04"
    / "sage_avo_s01_v002_production"
    / "runs"
    / "full"
)
OUTPUT = PRIVATE_ROOT / "figures" / "revision2" / "training_live"
CONFIG = REPOSITORY / "configs" / "sage_avo_s01.yaml"

FACIES_CMAP = ListedColormap(["#4f5963", "#f2c66d", "#d73027"])
FACIES_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FACIES_CMAP.N)
COLORS = {
    "train": "#277da1",
    "validation": "#f3722c",
    "vp": "#264653",
    "vs": "#2a9d8f",
    "density": "#e76f51",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "figure.titlesize": 14,
            "savefig.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, stem: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{stem}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _clean_log(checkpoint_epoch: int) -> pd.DataFrame:
    table = pd.read_csv(RUN / "training_log.csv")
    table = table[pd.to_numeric(table["epoch"], errors="coerce").notna()].copy()
    table["epoch"] = table["epoch"].astype(int)
    return table[table["epoch"] <= checkpoint_epoch].sort_values("epoch")


def _plot_pair(axis: plt.Axes, table: pd.DataFrame, name: str, title: str) -> None:
    axis.plot(
        table["epoch"],
        table[f"train_{name}"],
        color=COLORS["train"],
        lw=1.8,
        marker="o",
        ms=2.8,
        label="train",
    )
    axis.plot(
        table["epoch"],
        table[f"validation_{name}"],
        color=COLORS["validation"],
        lw=1.8,
        marker="o",
        ms=2.8,
        label="validation",
    )
    axis.set(title=title, xlabel="epoch", ylabel="loss")
    axis.legend(frameon=False)


def loss_dashboard(table: pd.DataFrame, epoch: int) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    _plot_pair(axes[0, 0], table, "total", "(a) Total weighted objective")
    _plot_pair(axes[0, 1], table, "inversion", "(b) Composite inversion loss")

    for name, label, color in (
        ("vp", "Vp", COLORS["vp"]),
        ("vs", "Vs", COLORS["vs"]),
        ("density", "density", COLORS["density"]),
    ):
        axes[0, 2].plot(
            table["epoch"],
            table[f"train_flow_{name}"],
            color=color,
            lw=1.5,
            label=f"train {label}",
        )
        axes[0, 2].plot(
            table["epoch"],
            table[f"validation_flow_{name}"],
            color=color,
            lw=1.5,
            ls="--",
            label=f"validation {label}",
        )
    axes[0, 2].set(title="(c) Conditional flow velocity by property", xlabel="epoch", ylabel="MSE")
    axes[0, 2].legend(frameon=False, ncol=2, fontsize=8)

    _plot_pair(axes[1, 0], table, "physics", "(d) Exact-PP AVO consistency")
    _plot_pair(axes[1, 1], table, "structure", "(e) Graph-edge structural loss")
    _plot_pair(axes[1, 2], table, "segmentation", "(f) Facies/plume segmentation")

    first = table.iloc[0]
    last = table.iloc[-1]
    reduction = 100.0 * (first.validation_total - last.validation_total) / first.validation_total
    figure.suptitle(
        f"Revision-2 live training through epoch {epoch} — validation total reduced {reduction:.1f}%"
    )
    return _save(figure, "live01_training_loss_dashboard")


def sampling_dashboard(table: pd.DataFrame, epoch: int) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    best_index = table["sample_criterion"].idxmin()
    best = table.loc[best_index]

    axes[0, 0].plot(table.epoch, table.sample_criterion, "o-", color="#7b2cbf", lw=1.7)
    axes[0, 0].scatter(best.epoch, best.sample_criterion, color="black", zorder=3)
    axes[0, 0].annotate(
        f"best e{int(best.epoch)}: {best.sample_criterion:.3f}",
        (best.epoch, best.sample_criterion),
        xytext=(8, 10),
        textcoords="offset points",
    )
    axes[0, 0].set(title="(a) Checkpoint sampling criterion (lower is better)", xlabel="epoch")

    for name, label, color in (
        ("vp", "Vp", COLORS["vp"]),
        ("vs", "Vs", COLORS["vs"]),
        ("density", "density", COLORS["density"]),
    ):
        axes[0, 1].plot(
            table.epoch,
            table[f"sample_rmse_{name}_normalized"],
            "o-",
            ms=3,
            color=color,
            label=label,
        )
    axes[0, 1].set(title="(b) Sampled elastic RMSE", xlabel="epoch", ylabel="normalized RMSE")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(table.epoch, table.sample_miou, "o-", color="#2a9d8f", lw=1.7)
    axes[1, 0].set(title="(c) Sampled segmentation mIoU", xlabel="epoch", ylabel="mIoU")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].plot(table.epoch, table.learning_rate, color="#264653", lw=1.8, label="learning rate")
    weight_axis = axes[1, 1].twinx()
    for column, label, color in (
        ("physics_weight", "physics weight", "#e76f51"),
        ("structure_weight", "structure weight", "#f4a261"),
        ("ssim_weight", "SSIM weight", "#457b9d"),
    ):
        weight_axis.plot(table.epoch, table[column], lw=1.3, label=label, color=color)
    axes[1, 1].set(title="(d) Scheduler and objective curriculum", xlabel="epoch", ylabel="learning rate")
    weight_axis.set_ylabel("effective loss weight")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = weight_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)

    figure.suptitle(
        f"Inference-time validation through epoch {epoch} — best sampling checkpoint remains epoch {int(best.epoch)}"
    )
    return _save(figure, "live02_sampling_validation_dashboard")


def objective_composition(table: pd.DataFrame, epoch: int) -> Path:
    contributions = pd.DataFrame(
        {
            "inversion": table.validation_inversion,
            "segmentation": 0.30 * table.validation_segmentation,
            "exact-PP physics": table.physics_weight * table.validation_physics,
            "graph structure": table.structure_weight * table.validation_structure,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].stackplot(
        table.epoch,
        *[contributions[column] for column in contributions],
        labels=list(contributions),
        alpha=0.85,
    )
    axes[0].plot(table.epoch, table.validation_total, color="black", lw=1.5, label="logged total")
    axes[0].set(title="(a) Weighted validation-objective composition", xlabel="epoch", ylabel="contribution")
    axes[0].legend(frameon=False, fontsize=8)

    train_last = table.iloc[-1]
    validation_values = contributions.iloc[-1]
    train_values = np.array(
        [
            train_last.train_inversion,
            0.30 * train_last.train_segmentation,
            train_last.physics_weight * train_last.train_physics,
            train_last.structure_weight * train_last.train_structure,
        ]
    )
    x = np.arange(len(contributions.columns))
    axes[1].bar(x - 0.18, train_values, 0.36, label="train", color=COLORS["train"])
    axes[1].bar(x + 0.18, validation_values, 0.36, label="validation", color=COLORS["validation"])
    axes[1].set_xticks(x, ["inversion", "segmentation", "physics", "structure"], rotation=18)
    axes[1].set(title=f"(b) Epoch {epoch} weighted contributions", ylabel="contribution")
    axes[1].legend(frameon=False)
    figure.suptitle("What the reported total loss contains")
    return _save(figure, "live03_weighted_objective_composition")


def _tensor_group_statistics(
    earlier: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    groups: dict[str, str],
) -> pd.DataFrame:
    records = []
    for label, prefix in groups.items():
        keys = [
            key
            for key, value in earlier.items()
            if key.startswith(prefix) and torch.is_floating_point(value)
        ]
        earlier_sq = current_sq = difference_sq = 0.0
        changed = count = 0
        for key in keys:
            old = earlier[key].float()
            new = current[key].float()
            difference = new - old
            earlier_sq += float(old.square().sum())
            current_sq += float(new.square().sum())
            difference_sq += float(difference.square().sum())
            changed += int(torch.count_nonzero(difference))
            count += difference.numel()
        records.append(
            {
                "module": label,
                "parameters": count,
                "changed_percent": 100.0 * changed / max(count, 1),
                "norm_earlier": earlier_sq**0.5,
                "norm_current": current_sq**0.5,
                "delta_norm": difference_sq**0.5,
                "relative_delta": difference_sq**0.5 / (earlier_sq**0.5 + 1e-12),
            }
        )
    return pd.DataFrame(records)


def parameter_activity(
    earlier_checkpoint: dict[str, Any],
    current_checkpoint: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    groups = {
        "time": "time_embedding.",
        "AVO/prior condition": "condition_embedding.",
        "CNN encoder": "encoder.",
        "graph (all)": "graph.",
        "elastic decoder": "decoder.",
    }
    table = _tensor_group_statistics(
        earlier_checkpoint["model_state"], current_checkpoint["model_state"], groups
    )
    graph_groups = {
        "node projection": "graph.node_projection.",
        "TransformerConv 1": "graph.layers.0.",
        "TransformerConv 2": "graph.layers.1.",
        "graph normalization": "graph.normalizations.",
        "segmentation decoder": "graph.segmentation.",
    }
    graph_table = _tensor_group_statistics(
        earlier_checkpoint["model_state"], current_checkpoint["model_state"], graph_groups
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(table.module, table.relative_delta, color="#277da1")
    earlier_epoch = int(earlier_checkpoint["epoch"])
    current_epoch = int(current_checkpoint["epoch"])
    axes[0].set(
        title="(a) Relative parameter change by model branch",
        ylabel=f"||w{current_epoch}-w{earlier_epoch}|| / ||w{earlier_epoch}||",
    )
    axes[0].tick_params(axis="x", rotation=25)
    for index, value in enumerate(table.relative_delta):
        axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    axes[1].bar(graph_table.module, graph_table.relative_delta, color="#f3722c")
    axes[1].set(title="(b) Activity inside the conditional graph branch", ylabel="relative tensor change")
    axes[1].tick_params(axis="x", rotation=25)
    for index, value in enumerate(graph_table.relative_delta):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom")
    figure.suptitle(
        f"Checkpoint parameter activity: epoch {earlier_checkpoint['epoch']} → {current_checkpoint['epoch']}"
    )
    return _save(figure, "live04_checkpoint_parameter_activity"), table


def _select_validation_patch() -> tuple[IndexedRealizationPatches, int]:
    dataset = IndexedRealizationPatches(DATASET, "validation")
    realization_id = int(dataset.index.realization_id.min())
    candidates = np.flatnonzero(
        (dataset.index.realization_id.to_numpy() == realization_id)
        & (dataset.index.scale_index.to_numpy() == 1)
    )
    if not len(candidates):
        raise RuntimeError("No native 50x100 validation patches found")
    support = [
        int(torch.count_nonzero(dataset.sampling_fields(int(index))["segmentation"]))
        for index in candidates
    ]
    selected = int(candidates[int(np.argmax(support))])
    return dataset, selected


def _model_and_patch(
    current_checkpoint: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    config = load_config(CONFIG)
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).cpu()
    model.load_state_dict(current_checkpoint["model_state"], strict=True)
    model.eval()
    dataset, patch_index = _select_validation_patch()
    item = dataset[patch_index]
    batch = {
        key: value.unsqueeze(0) if value.ndim > 0 else value
        for key, value in item.items()
        if key in {"avo", "target", "low", "rgt", "mask", "segmentation"}
    }
    metadata = {
        "patch_index": patch_index,
        "realization_id": int(item["realization_id"]),
        "top": int(item["top"]),
        "left": int(item["left"]),
        "normalization": dataset.normalization,
    }
    return model, batch, metadata


def _functional_outputs(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    target = batch["target"]
    low = batch["low"]
    time = torch.full((1,), 0.5)
    state = 0.5 * low + 0.5 * target
    with torch.inference_mode():
        output = model(state, time, batch["avo"], low, batch["rgt"])
        height, width = batch["rgt"].shape[-2:]
        time_features = model.time_embedding(time[:, None]).unsqueeze(-1).unsqueeze(-1)
        time_features = time_features.expand(-1, -1, height, width)
        condition = model.condition_embedding(torch.cat((batch["avo"], low), dim=1))
        cnn = model.encoder(torch.cat((state, time_features, condition), dim=1))
        bypass_velocity = model.decoder(cnn)
        cartesian_rgt = torch.arange(height, dtype=batch["rgt"].dtype)[None, :, None]
        cartesian_rgt = cartesian_rgt.expand(1, height, width)
        cartesian_output = model(state, time, batch["avo"], low, cartesian_rgt)
        sampled = model.sample(
            batch["avo"], low, batch["rgt"], steps=20, guidance_scale=0.0
        )
        final_output = model(
            sampled,
            torch.ones(1),
            batch["avo"],
            low,
            batch["rgt"],
        )
    return {
        "velocity": output.velocity,
        "bypass_velocity": bypass_velocity,
        "cartesian_velocity": cartesian_output.velocity,
        "edge_index": output.edge_indices[0],
        "edge_weight": output.edge_weights[0],
        "sampled": sampled,
        "segmentation": final_output.segmentation_logits.argmax(dim=1),
    }


def graph_mechanism(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, float]]:
    with torch.inference_mode():
        _, gradient = angular_features(batch["avo"], model.representative_angles)
    avo = batch["avo"][0].numpy()
    rgt = batch["rgt"][0].numpy()
    gradient_np = gradient[0, 0].numpy()
    edge_index = outputs["edge_index"].numpy()
    edge_weight = outputs["edge_weight"].numpy()
    height, width = rgt.shape
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for index, title in enumerate(("near 3–17°", "mid 17–31°", "far 31–45°")):
        limit = np.percentile(np.abs(avo[index]), 99)
        image = axes[0, index].imshow(avo[index], cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        axes[0, index].set_title(f"({chr(97 + index)}) normalized {title} AVO")
        plt.colorbar(image, ax=axes[0, index], shrink=0.75)
    image = axes[0, 3].imshow(rgt, cmap="turbo", aspect="auto")
    axes[0, 3].set_title("(d) warped RGT conditioning coordinate")
    plt.colorbar(image, ax=axes[0, 3], shrink=0.75)

    limit = np.percentile(np.abs(gradient_np), 99)
    image = axes[1, 0].imshow(gradient_np, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
    axes[1, 0].set_title("(e) compact AVO gradient G")
    plt.colorbar(image, ax=axes[1, 0], shrink=0.75)

    crop = (8, 38, 25, 75)
    top, bottom, left, right = crop
    source, destination = edge_index
    source_row, source_col = source // width, source % width
    destination_row, destination_col = destination // width, destination % width
    inside = (
        (source_row >= top)
        & (source_row < bottom)
        & (source_col >= left)
        & (source_col < right)
        & (destination_row >= top)
        & (destination_row < bottom)
        & (destination_col >= left)
        & (destination_col < right)
    )
    crop_weights = edge_weight[inside]
    threshold = np.quantile(crop_weights, 0.85)
    show = inside & (edge_weight >= threshold)
    segments = np.stack(
        (
            np.column_stack((source_col[show] - left, source_row[show] - top)),
            np.column_stack((destination_col[show] - left, destination_row[show] - top)),
        ),
        axis=1,
    )
    axes[1, 1].imshow(
        rgt[top:bottom, left:right],
        cmap="gray",
        aspect="auto",
        extent=(-0.5, right - left - 0.5, bottom - top - 0.5, -0.5),
    )
    collection = LineCollection(segments, cmap="inferno", lw=0.55, alpha=0.75)
    collection.set_array(edge_weight[show])
    axes[1, 1].add_collection(collection)
    axes[1, 1].set_xlim(-0.5, right - left - 0.5)
    axes[1, 1].set_ylim(bottom - top - 0.5, -0.5)
    axes[1, 1].set_title("(f) strongest 15% edge attributes in crop")

    y_std = np.asarray(metadata["normalization"]["y_std"], dtype=np.float32)
    reinjection = (
        outputs["velocity"] - outputs["bypass_velocity"]
    )[0, 0].numpy() * y_std[0]
    steering = (
        outputs["velocity"] - outputs["cartesian_velocity"]
    )[0, 0].numpy() * y_std[0]
    for axis, values, title in (
        (axes[1, 2], reinjection, "(g) graph-reinjection effect on Vp flow"),
        (axes[1, 3], steering, "(h) real-RGT vs Cartesian graph effect"),
    ):
        effect_limit = np.percentile(np.abs(values), 99)
        image = axis.imshow(values, cmap="seismic", aspect="auto", vmin=-effect_limit, vmax=effect_limit)
        axis.set_title(title)
        plt.colorbar(image, ax=axis, shrink=0.75, label="m/s")

    for axis in axes.ravel():
        axis.set(xlabel="tensor trace", ylabel="tensor time")
    diagnostics = {
        "edge_count": int(edge_weight.size),
        "edge_weight_mean": float(edge_weight.mean()),
        "graph_reinjection_vp_rms_mps": float(np.sqrt(np.mean(reinjection**2))),
        "rgt_steering_vp_rms_mps": float(np.sqrt(np.mean(steering**2))),
    }
    figure.suptitle(
        "Conditional graph functional diagnostic — "
        f"validation ID {metadata['realization_id']}, patch ({metadata['top']}, {metadata['left']})"
    )
    return _save(figure, "live05_conditional_graph_functional_diagnostic"), diagnostics


def _attention_coefficients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    target = batch["target"]
    low = batch["low"]
    time = torch.full((1,), 0.5)
    state = 0.5 * low + 0.5 * target
    height, width = batch["rgt"].shape[-2:]
    with torch.inference_mode():
        time_features = model.time_embedding(time[:, None]).unsqueeze(-1).unsqueeze(-1)
        time_features = time_features.expand(-1, -1, height, width)
        condition = model.condition_embedding(torch.cat((batch["avo"], low), dim=1))
        cnn = model.encoder(torch.cat((state, time_features, condition), dim=1))
        tokens = cnn.flatten(2).transpose(1, 2)
        features, gradient = angular_features(batch["avo"], model.representative_angles)
        angle_tokens = features.flatten(2).transpose(1, 2)
        node_features = model.graph.node_projection(
            torch.cat((tokens, angle_tokens), dim=-1)
        )[0]
        edge_index = build_rgt_edges(
            batch["rgt"], model.graph.max_shift, steered=True
        )[0]
        flattened_gradient = gradient[0].reshape(height * width)
        contrast = torch.abs(
            flattened_gradient[edge_index[0]] - flattened_gradient[edge_index[1]]
        )
        edge_attribute = torch.exp(
            -contrast / (contrast.std(unbiased=False) + 1e-6)
        )
        attentions = []
        for layer, normalization in zip(
            model.graph.layers, model.graph.normalizations
        ):
            result, (_, alpha) = layer(
                node_features,
                edge_index,
                edge_attr=edge_attribute.unsqueeze(-1),
                return_attention_weights=True,
            )
            attentions.append(alpha.mean(dim=1))
            node_features = F.gelu(normalization(result))
    return edge_index, edge_attribute, attentions


def _draw_strongest_edges(
    axis: plt.Axes,
    background: np.ndarray,
    edge_index: np.ndarray,
    values: np.ndarray,
    crop: tuple[int, int, int, int],
    title: str,
) -> None:
    height, width = background.shape
    top, bottom, left, right = crop
    source, destination = edge_index
    source_row, source_col = source // width, source % width
    destination_row, destination_col = destination // width, destination % width
    inside = (
        (source_row >= top)
        & (source_row < bottom)
        & (source_col >= left)
        & (source_col < right)
        & (destination_row >= top)
        & (destination_row < bottom)
        & (destination_col >= left)
        & (destination_col < right)
    )
    threshold = np.quantile(values[inside], 0.85)
    show = inside & (values >= threshold)
    segments = np.stack(
        (
            np.column_stack((source_col[show] - left, source_row[show] - top)),
            np.column_stack(
                (destination_col[show] - left, destination_row[show] - top)
            ),
        ),
        axis=1,
    )
    axis.imshow(
        background[top:bottom, left:right],
        cmap="gray",
        aspect="auto",
        extent=(-0.5, right - left - 0.5, bottom - top - 0.5, -0.5),
    )
    collection = LineCollection(segments, cmap="inferno", lw=0.55, alpha=0.78)
    collection.set_array(values[show])
    axis.add_collection(collection)
    axis.set_xlim(-0.5, right - left - 0.5)
    axis.set_ylim(bottom - top - 0.5, -0.5)
    axis.set_title(title)
    axis.set(xlabel="crop trace", ylabel="crop time")


def learned_attention(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, float]]:
    edge_index, edge_attribute, attentions = _attention_coefficients(model, batch)
    edge_numpy = edge_index.numpy()
    attribute_numpy = edge_attribute.numpy()
    attention_numpy = [attention.numpy() for attention in attentions]
    rgt = batch["rgt"][0].numpy()
    height, width = rgt.shape
    crop = (8, 38, 25, 75)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    image = axes[0, 0].imshow(rgt, cmap="turbo", aspect="auto")
    axes[0, 0].set_title("(a) warped RGT")
    plt.colorbar(image, ax=axes[0, 0], shrink=0.75)
    _draw_strongest_edges(
        axes[0, 1],
        rgt,
        edge_numpy,
        attribute_numpy,
        crop,
        "(b) strongest 15% AVO edge attributes",
    )
    _draw_strongest_edges(
        axes[0, 2],
        rgt,
        edge_numpy,
        attention_numpy[0],
        crop,
        "(c) strongest 15% learned attention — layer 1",
    )
    _draw_strongest_edges(
        axes[1, 0],
        rgt,
        edge_numpy,
        attention_numpy[1],
        crop,
        "(d) strongest 15% learned attention — layer 2",
    )
    for layer_index, values in enumerate(attention_numpy):
        entropy = np.zeros(height * width, dtype=np.float32)
        degree = np.zeros(height * width, dtype=np.int32)
        np.add.at(entropy, edge_numpy[1], -values * np.log(values + 1e-12))
        np.add.at(degree, edge_numpy[1], 1)
        concentration = np.zeros_like(entropy)
        multiple = degree > 1
        concentration[multiple] = 1.0 - entropy[multiple] / np.log(degree[multiple])
        image = axes[1, layer_index + 1].imshow(
            concentration.reshape(height, width),
            cmap="magma",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, layer_index + 1].set_title(
            f"({chr(101 + layer_index)}) attention concentration — layer {layer_index + 1}"
        )
        plt.colorbar(
            image,
            ax=axes[1, layer_index + 1],
            shrink=0.75,
            label="1 − normalized entropy",
        )
    for axis in (axes[0, 0], axes[1, 1], axes[1, 2]):
        axis.set(xlabel="tensor trace", ylabel="tensor time")
    figure.suptitle(
        "True TransformerConv attention diagnostic — edge threshold is visualization only; "
        f"all edges remain active (ID {metadata['realization_id']})"
    )
    diagnostics = {}
    for index, values in enumerate(attention_numpy, start=1):
        diagnostics[f"attention_layer{index}_mean"] = float(values.mean())
        diagnostics[f"attention_layer{index}_std"] = float(values.std())
        diagnostics[f"attention_layer{index}_coefficient_of_variation"] = float(
            values.std() / (values.mean() + 1e-12)
        )
    return _save(figure, "live08_learned_transformerconv_attention"), diagnostics


def inversion_sample(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, float]]:
    normalization = metadata["normalization"]
    mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    prior = batch["low"][0].numpy() * std + mean
    truth = batch["target"][0].numpy() * std + mean
    prediction = outputs["sampled"][0].numpy() * std + mean
    mask = batch["mask"][0, 0].numpy() > 0.5
    names = ("Vp", "Vs", "density")
    units = ("m/s", "m/s", "g/cc")
    figure, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    metrics: dict[str, float] = {}
    for row, (name, unit) in enumerate(zip(names, units)):
        lower, upper = np.percentile(truth[row][mask], (1, 99))
        error = prediction[row] - truth[row]
        error_limit = np.percentile(np.abs(error[mask]), 99)
        prior_rmse = float(np.sqrt(np.mean((prior[row][mask] - truth[row][mask]) ** 2)))
        prediction_rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
        improvement = 100.0 * (prior_rmse - prediction_rmse) / prior_rmse
        for column, (values, title) in enumerate(
            (
                (prior[row], f"{name} 2-Hz prior; RMSE {prior_rmse:.3g} {unit}"),
                (truth[row], f"{name} truth"),
                (prediction[row], f"{name} 20-step prediction"),
            )
        ):
            image = axes[row, column].imshow(
                values, cmap="viridis", aspect="auto", vmin=lower, vmax=upper
            )
            axes[row, column].set_title(title)
            plt.colorbar(image, ax=axes[row, column], shrink=0.75, label=unit)
        image = axes[row, 3].imshow(
            error, cmap="seismic", aspect="auto", vmin=-error_limit, vmax=error_limit
        )
        metrics[f"prior_rmse_{name.lower()}"] = prior_rmse
        metrics[f"rmse_{name.lower()}"] = prediction_rmse
        metrics[f"improvement_{name.lower()}_percent"] = improvement
        axes[row, 3].set_title(
            f"prediction − truth; RMSE {prediction_rmse:.3g} {unit}; {improvement:.1f}% better"
        )
        plt.colorbar(image, ax=axes[row, 3], shrink=0.75, label=unit)
    for axis in axes.ravel():
        axis.set(xlabel="tensor trace", ylabel="tensor time")
    figure.suptitle(
        "Fixed enriched validation-patch QC (not a controlled whole-realization result) — "
        f"ID {metadata['realization_id']}"
    )
    return _save(figure, "live06_validation_patch_inversion_qc"), metrics


def segmentation_sample(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> Path:
    avo = batch["avo"][0].numpy()
    truth = batch["segmentation"][0].numpy()
    prediction = outputs["segmentation"][0].numpy()
    figure, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    for index, name in enumerate(("near", "mid", "far")):
        limit = np.percentile(np.abs(avo[index]), 99)
        image = axes[index].imshow(avo[index], cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        axes[index].set_title(f"({chr(97 + index)}) normalized {name} AVO")
        plt.colorbar(image, ax=axes[index], shrink=0.7)
    axes[3].imshow(truth, cmap=FACIES_CMAP, norm=FACIES_NORM, aspect="auto")
    axes[3].set_title("(d) true shale/sand/CO₂")
    axes[4].imshow(prediction, cmap=FACIES_CMAP, norm=FACIES_NORM, aspect="auto")
    axes[4].set_title("(e) predicted shale/sand/CO₂")
    for axis in axes:
        axis.set(xlabel="tensor trace", ylabel="tensor time")
    figure.suptitle(f"Auxiliary segmentation QC — validation ID {metadata['realization_id']}")
    return _save(figure, "live07_validation_patch_segmentation_qc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_index(
    paths: list[Path],
    checkpoint_epoch: int,
    metadata: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Path:
    messages = {
        "live01_training_loss_dashboard.png": "Train/validation convergence for the complete logged multi-task objective and its principal raw terms.",
        "live02_sampling_validation_dashboard.png": "Inference-time sampling RMSE, segmentation mIoU, checkpoint criterion, scheduler, and curriculum.",
        "live03_weighted_objective_composition.png": "Exact weighted decomposition of the logged total objective.",
        "live04_checkpoint_parameter_activity.png": "Parameter changes between stable epoch-6 and current checkpoints, including the graph branch.",
        "live05_conditional_graph_functional_diagnostic.png": "Actual RGT/AVO edge construction and same-checkpoint functional graph/RGT sensitivity.",
        "live06_validation_patch_inversion_qc.png": "Low-prior/truth/prediction/error QC for a deterministic enriched validation patch.",
        "live07_validation_patch_segmentation_qc.png": "AVO and segmentation truth/prediction QC for the same validation patch.",
        "live08_learned_transformerconv_attention.png": "True learned attention coefficients from both trained TransformerConv layers, separated from fixed AVO edge attributes.",
    }
    records = []
    for path in paths:
        records.append(
            {
                "filename": path.name,
                "checkpoint_epoch": checkpoint_epoch,
                "source": "training_log.csv and stable checkpoints",
                "selection_rule": (
                    "all completed epochs"
                    if path.name.startswith(("live01", "live02", "live03", "live04"))
                    else "smallest validation realization ID; native-scale patch with maximum foreground support"
                ),
                "scientific_message": messages[path.name],
                "private_or_generated_data": True,
                "public_redistribution_needs_verification": True,
                "sha256": _sha256(path),
            }
        )
    index = OUTPUT / "live_training_figure_index.csv"
    pd.DataFrame(records).to_csv(index, index=False)
    report = OUTPUT / "live_training_diagnostics.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint_epoch": checkpoint_epoch,
                "patch": metadata,
                "diagnostics": diagnostics,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _style()
    current_checkpoint = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
    earlier_checkpoint = torch.load(
        RUN / "best_sampling.pt", map_location="cpu", weights_only=False
    )
    checkpoint_epoch = int(current_checkpoint["epoch"])
    table = _clean_log(checkpoint_epoch)
    if int(table.epoch.iloc[-1]) != checkpoint_epoch:
        raise RuntimeError("The stable checkpoint and completed log are not aligned")

    paths = [
        loss_dashboard(table, checkpoint_epoch),
        sampling_dashboard(table, checkpoint_epoch),
        objective_composition(table, checkpoint_epoch),
    ]
    activity_path, activity = parameter_activity(earlier_checkpoint, current_checkpoint)
    paths.append(activity_path)
    model, batch, metadata = _model_and_patch(current_checkpoint)
    outputs = _functional_outputs(model, batch)
    graph_path, graph_diagnostics = graph_mechanism(model, batch, outputs, metadata)
    paths.append(graph_path)
    attention_path, attention_diagnostics = learned_attention(model, batch, metadata)
    inversion_path, inversion_metrics = inversion_sample(batch, outputs, metadata)
    paths.append(inversion_path)
    paths.append(segmentation_sample(batch, outputs, metadata))
    paths.append(attention_path)

    graph_row = activity[activity.module == "graph (all)"].iloc[0]
    diagnostics = {
        **graph_diagnostics,
        **attention_diagnostics,
        **inversion_metrics,
        "graph_parameters": int(graph_row.parameters),
        "graph_changed_percent_epoch6_to_current": float(graph_row.changed_percent),
        "graph_relative_parameter_delta_epoch6_to_current": float(graph_row.relative_delta),
        "best_validation_total_epoch": int(table.loc[table.validation_total.idxmin(), "epoch"]),
        "best_sampling_epoch": int(table.loc[table.sample_criterion.idxmin(), "epoch"]),
    }
    index = _write_index(paths, checkpoint_epoch, metadata, diagnostics)
    print(json.dumps(diagnostics, indent=2))
    print(f"index: {index}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
