"""Machine-readable summaries and slide-ready observability figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROPERTIES = ("vp", "vs", "density")
OPTIMIZED_COMPONENTS = (
    "flow_vp",
    "flow_vs",
    "flow_density",
    "full_property",
    "ssim",
    "segmentation_ce",
    "segmentation_dice",
    "physics",
)


MACHINE_OUTPUT_SCHEMAS = {
    "training_statistics.csv": ["epoch"],
    "raw_loss_components.csv": [
        "epoch",
        "split",
        "component",
        "raw_loss",
        "epoch_1_raw_loss",
        "normalized_to_epoch_1",
    ],
    "weighted_loss_components.csv": [
        "epoch",
        "split",
        "component",
        "raw_loss",
        "effective_coefficient",
        "weighted_contribution",
    ],
    "physics_eligibility_statistics.csv": ["epoch", "split"],
    "gradient_contributions.csv": ["epoch", "objective", "parameter_group"],
    "gradient_cosines.csv": ["epoch", "objective_a", "objective_b"],
    "physics_floor_diagnostics.csv": ["epoch"],
    "graph_floor_diagnostics.csv": ["epoch"],
    "graph_learning_summary.csv": ["epoch", "layer"],
    "fixed_patch_metrics.csv": ["epoch", "patch_role", "property"],
    "whole_realization_metrics.csv": ["epoch", "realization_id", "property"],
    "checkpoint_comparison.csv": ["checkpoint", "epoch", "criterion"],
    "parameter_activity.csv": ["epoch_a", "epoch_b", "module"],
}


def initialize_machine_outputs(directory: str | Path) -> None:
    """Create missing report tables without replacing accumulated diagnostics."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    for filename, columns in MACHINE_OUTPUT_SCHEMAS.items():
        path = output / filename
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def _checkpoint_states(run_directory: Path) -> list[tuple[str, int, dict[str, Any]]]:
    paths = sorted((run_directory / "diagnostic_checkpoints").glob("epoch_*.pt"))
    paths.extend(
        path
        for path in (
            run_directory / "best_fixed_objective.pt",
            run_directory / "best_sampling.pt",
            run_directory / "best_segmentation.pt",
            run_directory / "best_whole_realization.pt",
            run_directory / "last.pt",
        )
        if path.exists()
    )
    records = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        records.append((path.name, int(checkpoint["epoch"]), checkpoint))
    return records


def _relative_parameter_change(
    first: dict[str, Any], second: dict[str, Any], prefixes: tuple[str, ...]
) -> tuple[int, float]:
    first_state, second_state = first["model_state"], second["model_state"]
    names = [
        name
        for name, value in first_state.items()
        if name.startswith(prefixes) and torch.is_floating_point(value)
    ]
    base_square = 0.0
    difference_square = 0.0
    count = 0
    for name in names:
        before = first_state[name].float()
        after = second_state[name].float()
        base_square += float(before.square().sum())
        difference_square += float((after - before).square().sum())
        count += before.numel()
    return count, difference_square**0.5 / (base_square**0.5 + 1e-12)


def update_checkpoint_tables(run_directory: str | Path, output_directory: str | Path) -> None:
    """Summarize selection checkpoints and parameter activity without mutation."""
    run, output = Path(run_directory), Path(output_directory)
    states = _checkpoint_states(run)
    comparison_rows = []
    for name, epoch, checkpoint in states:
        metrics = checkpoint.get("metrics", {})
        comparison_rows.append(
            {
                "checkpoint": name,
                "epoch": epoch,
                "criterion": metrics.get("criterion_name", "last_or_scheduled"),
                "criterion_value": metrics.get(
                    "criterion_value",
                    metrics.get(
                        "validation_fixed_objective",
                        metrics.get("validation_objective", np.nan),
                    ),
                ),
                "predeclared_selection_definition_preserved": True,
            }
        )
    if comparison_rows:
        pd.DataFrame(comparison_rows).drop_duplicates(["checkpoint"], keep="last").sort_values(
            ["epoch", "checkpoint"]
        ).to_csv(output / "checkpoint_comparison.csv", index=False)

    diagnostic = {
        epoch: checkpoint for name, epoch, checkpoint in states if name.startswith("epoch_")
    }
    named = {name: checkpoint for name, _, checkpoint in states}
    pairs: list[tuple[str, int, dict[str, Any], str, int, dict[str, Any]]] = []
    epochs = sorted(diagnostic)
    if len(epochs) > 1:
        pairs.extend(
            (
                f"epoch_{first:04d}",
                first,
                diagnostic[first],
                f"epoch_{second:04d}",
                second,
                diagnostic[second],
            )
            for first, second in zip(epochs[:-1], epochs[1:])
        )
    if 1 in diagnostic:
        for target in (10, 120):
            if target in diagnostic:
                pairs.append(
                    (
                        "epoch_0001",
                        1,
                        diagnostic[1],
                        f"epoch_{target:04d}",
                        target,
                        diagnostic[target],
                    )
                )
        for best_name in (
            "best_fixed_objective.pt",
            "best_sampling.pt",
            "best_segmentation.pt",
            "best_whole_realization.pt",
        ):
            if best_name in named:
                best = named[best_name]
                pairs.append(
                    (
                        "epoch_0001",
                        1,
                        diagnostic[1],
                        best_name,
                        int(best["epoch"]),
                        best,
                    )
                )
                if 120 in diagnostic:
                    pairs.append(
                        (
                            best_name,
                            int(best["epoch"]),
                            best,
                            "epoch_0120",
                            120,
                            diagnostic[120],
                        )
                    )
    groups = {
        "time_embedding": ("time_embedding.",),
        "cnn_encoder": ("condition_embedding.", "encoder."),
        "graph_node_projection": ("graph.node_projection.",),
        "transformerconv_1": ("graph.layers.0.",),
        "transformerconv_2": ("graph.layers.1.",),
        "graph_normalization_reinjection": ("graph.normalizations.",),
        "elastic_flow_decoder": ("decoder.",),
        "segmentation_decoder": ("graph.segmentation.",),
    }
    activity_rows = []
    seen = set()
    for first_name, first_epoch, first, second_name, second_epoch, second in pairs:
        pair_key = (first_name, second_name)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        for module, prefixes in groups.items():
            count, relative = _relative_parameter_change(first, second, prefixes)
            activity_rows.append(
                {
                    "checkpoint_a": first_name,
                    "epoch_a": first_epoch,
                    "checkpoint_b": second_name,
                    "epoch_b": second_epoch,
                    "module": module,
                    "parameter_count": count,
                    "relative_parameter_change": relative,
                    "mechanism_diagnostic_only": True,
                }
            )
    if activity_rows:
        pd.DataFrame(activity_rows).to_csv(output / "parameter_activity.csv", index=False)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(figure)
    return paths


def _empty(axis: plt.Axes, message: str) -> None:
    axis.grid(False)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set(xticks=[], yticks=[])


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _line_by_component(
    axis: plt.Axes,
    table: pd.DataFrame,
    value: str,
    *,
    split: str | None = None,
) -> None:
    selected = table if split is None else table[table["split"] == split]
    for name, group in selected.groupby("component"):
        axis.plot(group["epoch"], group[value], marker="o", ms=3, label=name)
    if not selected.empty:
        axis.legend(frameon=False, fontsize=7, ncol=2)


def _raw_loss_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "raw_loss_components.csv")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    if table.empty:
        _empty(axes[0], "Awaiting epoch diagnostics")
        _empty(axes[1], "Awaiting epoch diagnostics")
    else:
        selected = table[table.component.isin(OPTIMIZED_COMPONENTS)]
        _line_by_component(axes[0], selected, "raw_loss", split="validation")
        axes[0].set(title="(a) Raw validation losses", xlabel="epoch", ylabel="raw loss")
        _line_by_component(
            axes[1], selected, "normalized_to_epoch_1", split="validation"
        )
        axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
        axes[1].set(
            title="(b) Raw losses normalized to epoch 1",
            xlabel="epoch",
            ylabel=r"$L_i(e)/L_i(1)$",
        )
    figure.suptitle("Raw component-loss evolution — weights do not alter these curves")
    return _save(figure, figures, "01_raw_normalized_loss_evolution")


def _weighted_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "weighted_loss_components.csv")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    selected = table[table.get("split", pd.Series(dtype=str)) == "validation"]
    if not selected.empty:
        selected = selected[selected.effective_coefficient > 0.0]
    if selected.empty:
        _empty(axis, "Awaiting epoch diagnostics")
    else:
        pivot = selected.pivot(index="epoch", columns="component", values="weighted_contribution")
        axis.stackplot(pivot.index, *[pivot[c] for c in pivot], labels=pivot.columns, alpha=0.85)
        axis.legend(frameon=False, fontsize=7, ncol=3)
        axis.set(xlabel="epoch", ylabel="weighted scalar contribution")
    axis.set_title("Weighted validation-objective composition")
    return _save(figure, figures, "02_weighted_objective_composition")


def _physics_activity_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "physics_eligibility_statistics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if table.empty:
        _empty(axes[0], "Awaiting epoch diagnostics")
        _empty(axes[1], "Awaiting epoch diagnostics")
    else:
        for split, group in table.groupby("split"):
            axes[0].plot(
                group.epoch,
                group.conditional_raw_physics_loss,
                marker="o",
                label=f"{split}: eligible samples only",
            )
            axes[1].plot(
                group.epoch,
                group.all_step_weighted_physics_contribution,
                marker="o",
                label=f"{split}: all objective evaluations",
            )
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        axes[0].set(xlabel="epoch", ylabel="raw exact-PP loss")
        axes[1].set(xlabel="epoch", ylabel="weighted contribution")
    axes[0].set_title("(a) Conditional on physics eligibility")
    axes[1].set_title("(b) Effective contribution over all steps")
    figure.suptitle("Exact-PP activity: active samples versus optimizer-step average")
    return _save(figure, figures, "03_physics_active_vs_all_step")


def _gradient_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "gradient_contributions.csv")
    paths: list[str] = []
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    selected = table[
        table.get("parameter_group", pd.Series(dtype=str)) == "all_trainable_parameters"
    ]
    if not selected.empty:
        selected = selected[selected.weighted_gradient_norm > 0.0]
    if selected.empty:
        _empty(axis, "Awaiting checkpoint gradient diagnostics")
    else:
        for objective, group in selected.groupby("objective"):
            axis.plot(
                group.epoch,
                group.normalized_effective_gradient_contribution,
                marker="o",
                label=objective,
            )
        axis.legend(frameon=False, ncol=3, fontsize=8)
        axis.set(xlabel="epoch", ylabel="normalized effective gradient pressure", ylim=(0, 1))
    axis.set_title("Effective gradient contribution by objective")
    paths.extend(_save(figure, figures, "04_effective_gradient_contribution"))

    graph_groups = (
        "graph_node_projection",
        "transformerconv_layer_1",
        "transformerconv_layer_2",
        "graph_reinjected_cnn_stream",
    )
    graph = table[table.get("parameter_group", pd.Series(dtype=str)).isin(graph_groups)]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    if graph.empty:
        _empty(axis, "Awaiting granular GNN gradient diagnostics")
    else:
        aggregate = graph.groupby(["epoch", "parameter_group"], as_index=False).agg(
            aligned_weighted_gradient_norm=("weighted_gradient_norm", "sum")
        )
        for group_name, group in aggregate.groupby("parameter_group"):
            axis.plot(
                group.epoch,
                group.aligned_weighted_gradient_norm,
                "o-",
                label=group_name,
            )
        axis.legend(frameon=False, fontsize=8)
        axis.set(xlabel="epoch", ylabel="summed weighted gradient norm", yscale="log")
    axis.set_title("Aligned-objective gradient activity through the GNN path")
    paths.extend(_save(figure, figures, "04b_gnn_gradient_activity"))
    return paths


def _cosine_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "gradient_cosines.csv")
    epochs = sorted(table.epoch.unique()) if not table.empty else []
    epoch = epochs[-1] if epochs else None
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if epoch is None:
        _empty(axis, "Awaiting checkpoint gradient diagnostics")
    else:
        names = list(("elastic_flow", "physics", "segmentation"))
        matrix = np.eye(len(names))
        selected = table[
            (table.epoch == epoch)
            & table.objective_a.isin(names)
            & table.objective_b.isin(names)
            & np.isfinite(table.cosine_similarity)
        ]
        for row in selected.itertuples():
            i, j = names.index(row.objective_a), names.index(row.objective_b)
            matrix[i, j] = matrix[j, i] = row.cosine_similarity
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
        )
        axis.tick_params(axis="x", rotation=25)
        for i in range(len(names)):
            for j in range(len(names)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        plt.colorbar(image, ax=axis, label="gradient cosine")
    axis.set_title(f"Task-gradient cooperation/conflict — epoch {epoch or 'pending'}")
    return _save(figure, figures, "05_task_gradient_cosines")


def _floor_figures(output: Path, figures: Path) -> list[str]:
    paths: list[str] = []
    physics = _read(output / "physics_floor_diagnostics.csv")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if physics.empty:
        _empty(axis, "Awaiting checkpoint floor diagnostics")
    else:
        axis.plot(
            physics.epoch,
            physics.prediction_noiseless,
            "o-",
            label="prediction vs noiseless AVO",
        )
        axis.axhline(physics.prior_noiseless.iloc[0], color="#777777", ls="--", label="2-Hz prior")
        axis.axhline(
            physics.truth_noiseless_operator_floor.iloc[0],
            color="#2a9d8f",
            ls=":",
            label="truth/operator floor",
        )
        axis.axhline(
            physics.truth_noisy_observation_floor.iloc[0],
            color="#e76f51",
            ls=":",
            label="truth/noisy-observation floor",
        )
        axis.legend(frameon=False)
        axis.set(xlabel="epoch", ylabel="normalized exact-PP MSE")
    axis.set_title("Exact-PP consistency through training")
    paths.extend(_save(figure, figures, "06_exact_pp_attainable_floor"))

    graph = _read(output / "graph_floor_diagnostics.csv")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if graph.empty:
        _empty(axis, "Awaiting checkpoint graph diagnostics")
    else:
        axis.plot(graph.epoch, graph.prediction, "o-", label="prediction")
        axis.axhline(graph.prior_reference.iloc[0], color="#777777", ls="--", label="2-Hz prior")
        axis.axhline(graph.truth_reference.iloc[0], color="#2a9d8f", ls=":", label="elastic truth")
        axis.legend(frameon=False)
        axis.set(xlabel="epoch", ylabel="graph edge-smoothness loss")
    axis.set_title("Graph loss relative to non-zero truth reference")
    paths.extend(_save(figure, figures, "07_graph_reference_floor"))
    return paths


def _graph_figures(output: Path, figures: Path) -> list[str]:
    table = _read(output / "graph_learning_summary.csv")
    paths: list[str] = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if table.empty:
        _empty(axes[0], "Awaiting checkpoint graph diagnostics")
        _empty(axes[1], "Awaiting checkpoint graph diagnostics")
    else:
        for layer, group in table.groupby("layer"):
            axes[0].plot(group.epoch, group.attention_concentration, "o-", label=f"layer {layer}")
            axes[1].plot(group.epoch, group.top_decile_attention_mass, "o-", label=f"layer {layer}")
        for axis in axes:
            axis.legend(frameon=False)
            axis.set(xlabel="epoch")
        axes[0].set(ylabel="1 − normalized entropy")
        axes[1].set(ylabel="attention mass")
    axes[0].set_title("(a) Attention concentration")
    axes[1].set_title("(b) Top-decile attention mass")
    figure.suptitle("How SAGE-AVO learns to use supplied stratigraphic graph context")
    paths.extend(_save(figure, figures, "08_graph_attention_evolution"))

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    first = table[table.get("layer", pd.Series(dtype=float)) == 1]
    if first.empty or "graph_reinjection_velocity_rms" not in first:
        _empty(axis, "Awaiting graph reinjection diagnostics")
    else:
        axis.plot(
            first.epoch, first.graph_reinjection_velocity_rms, "o-", label="graph reinjection"
        )
        axis.plot(
            first.epoch, first.rgt_vs_cartesian_velocity_rms, "o-", label="RGT vs Cartesian routing"
        )
        axis.legend(frameon=False)
        axis.set(xlabel="epoch", ylabel="normalized velocity RMS difference")
    axis.set_title("Graph-use mechanism diagnostics (not causal ablations)")
    paths.extend(_save(figure, figures, "09_graph_reinjection_evolution"))
    return paths


def _flow_segmentation_figures(output: Path, figures: Path) -> list[str]:
    paths: list[str] = []
    raw = _read(output / "raw_loss_components.csv")
    metrics = _read(output / "fixed_patch_metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    flow = raw[
        (raw.get("split", pd.Series(dtype=str)) == "validation")
        & raw.get("component", pd.Series(dtype=str)).isin(["flow_vp", "flow_vs", "flow_density"])
    ]
    if flow.empty:
        _empty(axes[0], "Awaiting epoch flow diagnostics")
    else:
        _line_by_component(axes[0], flow, "raw_loss")
        axes[0].set(xlabel="epoch", ylabel="raw velocity-matching MSE")
    if metrics.empty:
        _empty(axes[1], "Awaiting fixed-patch inference")
    else:
        aggregate = metrics.groupby(["epoch", "property"], as_index=False).rmse.mean()
        for prop, group in aggregate.groupby("property"):
            axes[1].plot(group.epoch, group.rmse, "o-", label=prop)
        axes[1].legend(frameon=False)
        axes[1].set(xlabel="epoch", ylabel="physical-unit patch RMSE")
    axes[0].set_title("(a) Conditional-flow losses")
    axes[1].set_title("(b) Fixed-patch property accuracy")
    figure.suptitle("Vp/Vs/density conditional-flow evolution")
    paths.extend(_save(figure, figures, "10_property_flow_evolution"))

    segmentation = _read(output / "segmentation_metrics.csv")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if segmentation.empty:
        _empty(axis, "Awaiting fixed-patch segmentation diagnostics")
    else:
        aggregate = segmentation.groupby("epoch", as_index=False)[
            ["miou", "macro_dice", "class_0_iou", "class_1_iou", "class_2_iou"]
        ].mean()
        for column in aggregate.columns[1:]:
            axis.plot(aggregate.epoch, aggregate[column], "o-", label=column)
        axis.legend(frameon=False, ncol=3)
        axis.set(xlabel="epoch", ylabel="score", ylim=(0, 1))
    axis.set_title("Auxiliary segmentation health on frozen validation patches")
    paths.extend(_save(figure, figures, "11_segmentation_evolution"))
    return paths


def _generic_table_figure(
    output: Path, figures: Path, filename: str, stem: str, title: str
) -> list[str]:
    table = _read(output / filename)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if table.empty or len(table.columns) <= 3:
        _empty(axis, "Awaiting scheduled production checkpoints")
    elif "relative_parameter_change" in table:
        latest = table.sort_values(["epoch_b", "epoch_a"]).groupby("module").tail(1)
        axis.bar(latest.module, latest.relative_parameter_change, color="#277da1")
        axis.tick_params(axis="x", rotation=30)
        axis.set(ylabel=r"$\|w_b-w_a\|/\|w_a\|$")
    else:
        numeric = [
            column
            for column in table.select_dtypes(include=np.number).columns
            if column not in {"epoch", "epoch_a", "epoch_b"}
        ]
        if "epoch" in table and numeric:
            for column in numeric[:6]:
                axis.plot(table.epoch, table[column], "o-", label=column)
            axis.legend(frameon=False, fontsize=8)
            axis.set(xlabel="epoch")
        else:
            _empty(axis, "Table initialized; comparison requires multiple checkpoints")
    axis.set_title(title)
    return _save(figure, figures, stem)


def _whole_realization_figure(output: Path, figures: Path) -> list[str]:
    table = _read(output / "whole_realization_metrics.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    units = {"vp": "m/s", "vs": "m/s", "density": "g/cc"}
    for axis, property_name in zip(axes, PROPERTIES):
        selected = table[table.get("property", pd.Series(dtype=str)) == property_name]
        if selected.empty:
            _empty(axis, f"Awaiting {property_name} whole-realization metrics")
            continue
        for _, group in selected.groupby("realization_id"):
            axis.plot(group.epoch, group.rmse, color="#90a4ae", alpha=0.35, lw=1)
        summary = selected.groupby("epoch", as_index=False).agg(
            rmse=("rmse", "mean"),
            minimum=("rmse", "min"),
            maximum=("rmse", "max"),
            prior=("prior_rmse", "mean"),
        )
        axis.fill_between(
            summary.epoch,
            summary.minimum,
            summary.maximum,
            color="#277da1",
            alpha=0.15,
            label="validation realization range",
        )
        axis.plot(summary.epoch, summary.rmse, "o-", color="#277da1", label="mean prediction")
        axis.axhline(
            float(summary.prior.iloc[0]),
            color="#e76f51",
            ls="--",
            label="2-Hz prior",
        )
        axis.set(
            title=property_name.upper() if property_name != "density" else "Density",
            xlabel="epoch",
            ylabel=f"RMSE ({units[property_name]})",
            xticks=sorted(summary.epoch.unique()),
        )
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Whole-realization validation RMSE relative to the supplied 2-Hz prior")
    return _save(figure, figures, "12_whole_realization_evolution")


def _fixed_patch_sheets(output: Path, figures: Path) -> list[str]:
    paths: list[str] = []
    epoch_directories = sorted((output / "fixed_patch_arrays").glob("epoch_*"))
    if not epoch_directories:
        return paths
    latest = epoch_directories[-1]
    for archive_path in sorted(latest.glob("*.npz")):
        with np.load(archive_path) as archive:
            figure, axes = plt.subplots(4, 5, figsize=(18, 13), constrained_layout=True)
            for index, name in enumerate(("near", "mid", "far")):
                values = archive["avo"][index]
                limit = np.percentile(np.abs(values), 99)
                image = axes[0, index].imshow(
                    values, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit
                )
                axes[0, index].set_title(f"{name} AVO")
                plt.colorbar(image, ax=axes[0, index], shrink=0.65)
            axes[0, 3].imshow(archive["rgt"], cmap="turbo", aspect="auto")
            axes[0, 3].set_title("warped RGT")
            axes[0, 4].imshow(archive["shuey_gradient"], cmap="seismic", aspect="auto")
            axes[0, 4].set_title("compact AVO gradient G")
            for row, name in enumerate(PROPERTIES):
                truth = archive["truth"][row]
                lower, upper = np.percentile(truth, (1, 99))
                for column, key in enumerate(("prior", "truth", "prediction")):
                    axes[row + 1, column].imshow(
                        archive[key][row], cmap="viridis", aspect="auto", vmin=lower, vmax=upper
                    )
                    axes[row + 1, column].set_title(f"{name} {key}")
                residual = archive["residual"][row]
                limit = np.percentile(np.abs(residual), 99)
                axes[row + 1, 3].imshow(
                    residual, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit
                )
                axes[row + 1, 3].set_title(f"{name} prediction − truth")
                if row == 0:
                    axes[row + 1, 4].imshow(
                        archive["graph_embedding_norm"], cmap="magma", aspect="auto"
                    )
                    axes[row + 1, 4].set_title("graph latent norm")
                elif row == 1:
                    axes[row + 1, 4].imshow(
                        archive["segmentation_truth"], cmap="tab10", aspect="auto", vmin=0, vmax=2
                    )
                    axes[row + 1, 4].set_title("segmentation truth")
                else:
                    axes[row + 1, 4].imshow(
                        archive["segmentation_prediction"],
                        cmap="tab10",
                        aspect="auto",
                        vmin=0,
                        vmax=2,
                    )
                    axes[row + 1, 4].set_title("segmentation prediction")
            for axis in axes.ravel():
                axis.set(xlabel="trace", ylabel="sample")
            figure.suptitle(
                f"Frozen validation patch: {archive_path.stem} — {latest.name.replace('_', ' ')}"
            )
            paths.extend(
                _save(figure, figures, f"15_fixed_patch_{archive_path.stem}_{latest.name}")
            )
    return paths


def generate_observability_figures(
    directory: str | Path, figure_directory: str | Path
) -> list[str]:
    """Generate all available plots; pending multi-epoch panels are explicit."""
    output, figures = Path(directory), Path(figure_directory)
    _style()
    paths: list[str] = []
    paths.extend(_raw_loss_figure(output, figures))
    paths.extend(_weighted_figure(output, figures))
    paths.extend(_physics_activity_figure(output, figures))
    paths.extend(_gradient_figure(output, figures))
    paths.extend(_cosine_figure(output, figures))
    paths.extend(_floor_figures(output, figures))
    paths.extend(_graph_figures(output, figures))
    paths.extend(_flow_segmentation_figures(output, figures))
    paths.extend(_whole_realization_figure(output, figures))
    paths.extend(
        _generic_table_figure(
            output,
            figures,
            "checkpoint_comparison.csv",
            "13_checkpoint_comparison",
            "Predeclared checkpoint-criterion comparison",
        )
    )
    paths.extend(
        _generic_table_figure(
            output,
            figures,
            "parameter_activity.csv",
            "14_parameter_activity",
            "Relative parameter activity between checkpoints",
        )
    )
    paths.extend(_fixed_patch_sheets(output, figures))
    return paths


def health_gate(
    directory: str | Path,
    observability_config: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Apply predeclared diagnostic classifications at epoch 5 or 10."""
    output = Path(directory)
    raw = _read(output / "raw_loss_components.csv")
    weighted = _read(output / "weighted_loss_components.csv")
    gradients = _read(output / "gradient_contributions.csv")
    cosines = _read(output / "gradient_cosines.csv")
    physics = _read(output / "physics_floor_diagnostics.csv")
    graph = _read(output / "graph_floor_diagnostics.csv")
    thresholds = observability_config["health_classification"]
    classifications: dict[str, Any] = {}
    mapping = {
        "elastic_flow": "flow",
        "physics": "physics",
        "structure": "structure",
        "segmentation": "segmentation",
    }
    weighted_groups = {
        "elastic_flow": [
            "flow_vp",
            "flow_vs",
            "flow_density",
            "full_vp",
            "full_vs",
            "full_density",
            "ssim",
        ],
        "physics": ["physics"],
        "structure": ["structure"],
        "segmentation": ["segmentation_ce", "segmentation_dice"],
    }
    for objective, raw_component in mapping.items():
        rows = raw[(raw.split == "validation") & (raw.component == raw_component)].sort_values(
            "epoch"
        )
        rows = rows[rows.epoch <= epoch]
        if rows.empty:
            classifications[objective] = {
                "classification": "UNSTABLE",
                "reason": "missing raw diagnostic",
            }
            continue
        first, current = float(rows.raw_loss.iloc[0]), float(rows.raw_loss.iloc[-1])
        reduction = (first - current) / abs(first) if first else 0.0
        previous = float(rows.raw_loss.iloc[-2]) if len(rows) > 1 else first
        recent_reduction = (previous - current) / abs(previous) if previous else 0.0
        weighted_rows = weighted[
            (weighted.split == "validation")
            & (weighted.component.isin(weighted_groups[objective]))
            & (weighted.epoch <= epoch)
        ]
        weighted_contribution = (
            float(
                weighted_rows[
                    weighted_rows.epoch == weighted_rows.epoch.max()
                ].weighted_contribution.sum()
            )
            if not weighted_rows.empty
            else np.nan
        )
        gradient_rows = gradients[
            (gradients.objective.isin(weighted_groups[objective]))
            & (gradients.parameter_group == "all_trainable_parameters")
            & (gradients.epoch <= epoch)
        ]
        gradient_fraction = (
            float(
                gradient_rows[
                    gradient_rows.epoch == gradient_rows.epoch.max()
                ].normalized_effective_gradient_contribution.sum()
            )
            if not gradient_rows.empty
            else np.nan
        )
        related = cosines[
            ((cosines.objective_a == objective) | (cosines.objective_b == objective))
            & (cosines.epoch <= epoch)
        ]
        strongest_conflict = float(related.cosine_similarity.min()) if not related.empty else np.nan
        persistent_conflict = False
        if not related.empty:
            for _, pair in related.groupby(["objective_a", "objective_b"]):
                recent = pair.sort_values("epoch").tail(2).cosine_similarity
                if len(recent) == 2 and bool((recent < thresholds["severe_conflict_cosine"]).all()):
                    persistent_conflict = True
                    break
        distance = np.nan
        if objective == "physics" and not physics.empty:
            row = physics[physics.epoch <= epoch].iloc[-1]
            distance = float(row.normalized_progress_from_prior_to_operator_floor)
        elif objective == "structure" and not graph.empty:
            row = graph[graph.epoch <= epoch].iloc[-1]
            denominator = abs(float(row.prior_reference - row.truth_reference))
            distance = (
                abs(float(row.prediction_minus_truth_reference)) / denominator
                if denominator
                else np.nan
            )
        if not np.isfinite(current):
            classification = "UNSTABLE"
        elif current > first * (1.0 + thresholds["unstable_relative_increase"]):
            classification = "UNSTABLE"
        elif np.isfinite(distance) and distance <= thresholds["near_floor_normalized_distance"]:
            classification = "PLATEAU_NEAR_FLOOR"
        elif (
            np.isfinite(gradient_fraction)
            and gradient_fraction < thresholds["gradient_starved_fraction"]
        ):
            classification = "GRADIENT_STARVED"
        elif persistent_conflict:
            classification = "TASK_CONFLICT"
        elif reduction >= thresholds["improving_relative_reduction"]:
            classification = "IMPROVING"
        else:
            classification = "PLATEAU_WITH_ACTIVE_GRADIENT"
        classifications[objective] = {
            "classification": classification,
            "raw_epoch_1": first,
            "raw_current": current,
            "relative_improvement_from_epoch_1": reduction,
            "relative_improvement_since_previous_diagnostic_epoch": recent_reduction,
            "weighted_scalar_contribution": weighted_contribution,
            "effective_gradient_fraction": gradient_fraction,
            "strongest_gradient_cosine_conflict": strongest_conflict,
            "severe_conflict_persistent_across_two_diagnostics": persistent_conflict,
            "normalized_distance_to_reference": distance,
        }
    stop_classes = {"UNSTABLE", "GRADIENT_STARVED", "TASK_CONFLICT"}
    stop = any(
        value["classification"] in stop_classes
        for key, value in classifications.items()
        if key in {"physics", "structure"}
    )
    report = {
        "epoch": epoch,
        "classifications": classifications,
        "stop_before_continuing": stop,
        "thresholds_predeclared": thresholds,
    }
    if epoch in {5, 10}:
        (output / f"multi_objective_health_epoch{epoch}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        lines = [f"# Multi-objective health at epoch {epoch}", ""]
        for name, values in classifications.items():
            lines.append(f"- {name}: **{values['classification']}**")
        lines.extend(("", f"Stop before continuation: **{stop}**", ""))
        (output / f"multi_objective_health_epoch{epoch}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return report


def write_summary(directory: str | Path, status: str, details: dict[str, Any]) -> Path:
    output = Path(directory)
    path = output / "training_diagnostics_summary.md"
    lines = [
        "# Revision 3.3.2 training diagnostics",
        "",
        f"Status: **{status}**",
        "",
        "Diagnostics are computed by reloading immutable checkpoints in a separate process. They do not update the optimizer, scheduler, RNG state, model parameters, or gradients in the live trainer.",
        "",
        "## Current evidence",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in details.items())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
