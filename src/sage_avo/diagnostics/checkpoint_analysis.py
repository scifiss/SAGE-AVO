"""Separate-process scientific diagnostics for immutable training checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import default_collate

from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.experiments.training import (
    _normalization_tensors,
    curriculum_from_config,
    graph_objective_from_config,
    loss_weights_from_config,
    physics_settings_from_config,
)
from sage_avo.models.sage_avo import angular_features
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.training.checkpoints import load_checkpoint
from sage_avo.training.engine import (
    ContrastiveSettings,
    _forward_objective,
    _move_batch,
)
from sage_avo.training.losses import (
    edge_smoothness,
    graph_structure_loss,
    physics_loss_with_context,
    truth_edge_matching,
)

from .accounting import WEIGHTED_COMPONENTS, effective_component_coefficients


MAJOR_OBJECTIVES = ("elastic_flow", "physics", "structure", "segmentation")
PROPERTIES = ("vp", "vs", "density")


def _boundary_contrast_error(
    prediction: Tensor,
    target: Tensor,
    segmentation: Tensor,
    edge_indices: list[Tensor],
    edge_weights: list[Tensor],
) -> Tensor:
    """Weighted signed edge-vector error on facies/high-truth-contrast edges."""
    total = prediction.new_zeros(())
    for item, edge_index in enumerate(edge_indices):
        source, destination = edge_index
        predicted = prediction[item].reshape(3, -1).transpose(0, 1)
        expected = target[item].reshape(3, -1).transpose(0, 1)
        predicted_difference = predicted[source] - predicted[destination]
        target_difference = expected[source] - expected[destination]
        target_contrast = target_difference.abs().mean(dim=1)
        labels = segmentation[item].reshape(-1)
        boundary = (labels[source] != labels[destination]) | (
            target_contrast >= torch.quantile(target_contrast, 0.75)
        )
        mismatch = (predicted_difference - target_difference).abs().mean(dim=1)
        total = total + (mismatch[boundary] * edge_weights[item][boundary]).mean()
    return total / max(prediction.shape[0], 1)


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _upsert_rows(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        existing = pd.read_csv(path)
        table = pd.concat((existing, new), ignore_index=True)
        table = table.drop_duplicates(keys, keep="last")
    else:
        table = new
    order = [column for column in ("epoch", *keys) if column in table]
    table = table.sort_values(order).reset_index(drop=True) if order else table
    table.to_csv(path, index=False)


def _find_patch(dataset: IndexedRealizationPatches, record: dict[str, Any]) -> int:
    raw_height, raw_width = map(int, record["raw_scale"])
    rows = dataset.index
    match = rows[
        (rows["realization_id"] == int(record["realization_id"]))
        & (rows["top"] == int(record["top"]))
        & (rows["left"] == int(record["left"]))
        & (rows["raw_height"] == raw_height)
        & (rows["raw_width"] == raw_width)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Fixed patch does not resolve uniquely: {record}")
    return int(match.index[0])


def load_fixed_batch(
    dataset_directory: str | Path,
    sample_manifest: dict[str, Any],
    *,
    physics_only: bool = False,
    maximum_patches: int | None = None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Load the predeclared validation samples without augmentation."""
    dataset = IndexedRealizationPatches(dataset_directory, "validation")
    records = [
        record
        for record in sample_manifest["patches"]
        if not physics_only or bool(record["physics_eligible"])
    ]
    if maximum_patches is not None:
        records = records[:maximum_patches]
    items = [dataset[_find_patch(dataset, record)] for record in records]
    return default_collate(items), records


def _parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    named = list(model.named_parameters())

    def selected(*prefixes: str) -> list[tuple[str, nn.Parameter]]:
        return [item for item in named if item[0].startswith(prefixes)]

    shared = [item for item in named if not item[0].startswith(("decoder.", "graph.segmentation."))]
    return {
        "shared_cnn_encoder": selected("time_embedding.", "condition_embedding.", "encoder."),
        "graph_branch": selected(
            "graph.node_projection.",
            "graph.layers.",
            "graph.normalizations.",
        ),
        "graph_node_projection": selected("graph.node_projection."),
        "transformerconv_layer_1": selected("graph.layers.0."),
        "transformerconv_layer_2": selected("graph.layers.1."),
        "elastic_flow_decoder": selected("decoder."),
        "segmentation_decoder": selected("graph.segmentation."),
        "all_shared_trainable_parameters": shared,
        "all_trainable_parameters": named,
    }


def _flatten_gradients(
    gradients: tuple[Tensor | None, ...], parameters: Iterable[nn.Parameter]
) -> Tensor:
    values = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    return torch.cat(values) if values else torch.zeros(0)


def _gradient_diagnostics(
    *,
    model: nn.Module,
    terms: dict[str, Tensor],
    coefficients: dict[str, float],
    epoch: int,
    activation_groups: dict[str, Tensor] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    groups = _parameter_groups(model)
    component_gradients: dict[str, tuple[Tensor | None, ...]] = {}
    active_components = [
        name
        for name in WEIGHTED_COMPONENTS
        if coefficients[name] != 0.0 and terms[name].requires_grad
    ]
    for name in active_components:
        component_gradients[name] = torch.autograd.grad(
            terms[name],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    rows: list[dict[str, Any]] = []
    for group_name, group_named in groups.items():
        if not group_named:
            continue
        names = {name for name, _ in group_named}
        indices = [index for index, (name, _) in enumerate(named) if name in names]
        pressures: dict[str, float] = {}
        norms: dict[str, float] = {}
        for component, gradients in component_gradients.items():
            squared = [
                gradients[index].detach().square().sum()
                for index in indices
                if gradients[index] is not None
            ]
            norm = float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.0
            norms[component] = norm
            pressures[component] = abs(float(coefficients[component])) * norm
        pressure_sum = sum(pressures.values())
        for component in active_components:
            rows.append(
                {
                    "epoch": epoch,
                    "objective": component,
                    "parameter_group": group_name,
                    "raw_gradient_norm": norms[component],
                    "effective_coefficient": coefficients[component],
                    "weighted_gradient_norm": pressures[component],
                    "normalized_effective_gradient_contribution": (
                        pressures[component] / pressure_sum if pressure_sum else np.nan
                    ),
                }
            )

    for group_name, activation in (activation_groups or {}).items():
        pressures = {}
        norms = {}
        for component in active_components:
            gradient = torch.autograd.grad(
                terms[component],
                activation,
                retain_graph=True,
                allow_unused=True,
            )[0]
            norm = float(gradient.detach().square().sum().sqrt()) if gradient is not None else 0.0
            norms[component] = norm
            pressures[component] = abs(float(coefficients[component])) * norm
        pressure_sum = sum(pressures.values())
        for component in active_components:
            rows.append(
                {
                    "epoch": epoch,
                    "objective": component,
                    "parameter_group": group_name,
                    "raw_gradient_norm": norms[component],
                    "effective_coefficient": coefficients[component],
                    "weighted_gradient_norm": pressures[component],
                    "normalized_effective_gradient_contribution": (
                        pressures[component] / pressure_sum if pressure_sum else np.nan
                    ),
                }
            )

    aggregate_terms = {
        "elastic_flow": terms["inversion"],
        "physics": terms["physics"],
        "structure": terms["structure"],
        "segmentation": terms["segmentation"],
    }
    shared_named = groups["all_shared_trainable_parameters"]
    shared_parameters = [parameter for _, parameter in shared_named]
    vectors: dict[str, Tensor] = {}
    for name, term in aggregate_terms.items():
        if term.requires_grad:
            gradients = torch.autograd.grad(
                term,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            vectors[name] = _flatten_gradients(gradients, shared_parameters)
        else:
            vectors[name] = torch.zeros(
                sum(parameter.numel() for parameter in shared_parameters),
                device=shared_parameters[0].device,
            )
    cosine_rows = []
    requested = (
        ("elastic_flow", "physics"),
        ("elastic_flow", "structure"),
        ("elastic_flow", "segmentation"),
        ("physics", "structure"),
        ("physics", "segmentation"),
        ("structure", "segmentation"),
    )
    for first, second in requested:
        a, b = vectors[first], vectors[second]
        denominator = float(a.norm() * b.norm())
        cosine_rows.append(
            {
                "epoch": epoch,
                "objective_a": first,
                "objective_b": second,
                "cosine_similarity": (
                    float(torch.dot(a, b)) / denominator if denominator else np.nan
                ),
                "shared_parameter_definition": (
                    "all trainable parameters except elastic and segmentation heads"
                ),
            }
        )
    return rows, cosine_rows


def _physics_value(
    prediction: Tensor,
    values: dict[str, Tensor],
    normalization: Any,
    specification: Any,
    observed: Tensor,
) -> Tensor:
    return physics_loss_with_context(
        prediction,
        values["physics_context"],
        observed,
        normalization.y_mean.to(prediction.device),
        normalization.y_std.to(prediction.device),
        normalization.x_mean.to(prediction.device),
        normalization.x_std.to(prediction.device),
        mask=values["physics_mask"],
        core_start=values["physics_core_start"],
        sample_origin=values["physics_context_sample_origin"],
        specification=specification,
    )


def _graph_attention_details(
    model: nn.Module,
    values: dict[str, Tensor],
    state: Tensor,
    time: Tensor,
) -> tuple[Any, list[Tensor], Tensor, Tensor, Tensor]:
    height, width = values["rgt"].shape[-2:]
    time_features = model.time_embedding(time[:, None]).unsqueeze(-1).unsqueeze(-1)
    time_features = time_features.expand(-1, -1, height, width)
    condition = model.condition_embedding(torch.cat((values["avo"], values["low"]), dim=1))
    cnn = model.encoder(torch.cat((state, time_features, condition), dim=1))
    tokens = cnn.flatten(2).transpose(1, 2)
    angular, gradient = angular_features(values["avo"], model.representative_angles)
    angle_tokens = angular.flatten(2).transpose(1, 2)
    projected = model.graph.node_projection(torch.cat((tokens, angle_tokens), dim=-1))
    edges = model.graph(
        tokens,
        values["avo"],
        values["rgt"],
    )
    attentions: list[Tensor] = []
    node_features = projected[0]
    edge_index = edges[2][0]
    edge_attribute = edges[3][0].unsqueeze(-1)
    for layer, normalization_layer in zip(model.graph.layers, model.graph.normalizations):
        result, (_, alpha) = layer(
            node_features,
            edge_index,
            edge_attr=edge_attribute,
            return_attention_weights=True,
        )
        attentions.append(alpha.mean(dim=-1))
        node_features = torch.nn.functional.gelu(normalization_layer(result))
    return edges, attentions, cnn, angular, gradient


def _attention_summary(
    *,
    epoch: int,
    layer: int,
    attention: Tensor,
    edge_index: Tensor,
    rgt: Tensor,
    avo: Tensor,
) -> dict[str, Any]:
    probability = attention.detach()
    source, destination = edge_index
    height, width = rgt.shape
    destination_count = height * width
    entropy_per_node = torch.zeros(destination_count, device=probability.device)
    entropy_per_node.scatter_add_(
        0, destination, -probability * torch.log(probability.clamp_min(1e-12))
    )
    degree = torch.zeros(destination_count, device=probability.device)
    degree.scatter_add_(0, destination, torch.ones_like(probability))
    valid = degree > 1
    normalized_entropy = entropy_per_node[valid] / torch.log(degree[valid])
    source_col = source % width
    destination_col = destination % width
    lateral = source_col != destination_col
    rgt_flat = rgt.reshape(-1)
    rgt_mismatch = (rgt_flat[source] - rgt_flat[destination]).abs()
    low_rgt = rgt_mismatch <= torch.quantile(rgt_mismatch, 0.25)
    avo_flat = avo.reshape(3, -1).transpose(0, 1)
    avo_difference = (avo_flat[source] - avo_flat[destination]).square().sum(dim=1).sqrt()
    high_avo_similarity = avo_difference <= torch.quantile(avo_difference, 0.25)
    top_count = max(1, int(round(0.1 * probability.numel())))
    total = probability.sum().clamp_min(1e-12)
    return {
        "epoch": epoch,
        "layer": layer,
        "attention_mean": float(probability.mean()),
        "attention_entropy_normalized": float(normalized_entropy.mean()),
        "attention_concentration": float(1.0 - normalized_entropy.mean()),
        "top_decile_attention_mass": float(probability.topk(top_count).values.sum() / total),
        "lateral_attention_fraction": float(probability[lateral].sum() / total),
        "vertical_attention_fraction": float(probability[~lateral].sum() / total),
        "low_rgt_mismatch_attention_fraction": float(probability[low_rgt].sum() / total),
        "high_avo_similarity_attention_fraction": float(
            probability[high_avo_similarity].sum() / total
        ),
    }


def _elastic_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    mask: np.ndarray,
    *,
    epoch: int,
    patch_role: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(PROPERTIES):
        expected = truth[index][mask]
        inferred = prediction[index][mask]
        baseline = prior[index][mask]
        error = inferred - expected
        residual_sum = float(np.square(error).sum())
        total_sum = float(np.square(expected - expected.mean()).sum())
        correlation = (
            float(np.corrcoef(inferred, expected)[0, 1])
            if np.std(inferred) > 0 and np.std(expected) > 0
            else np.nan
        )
        rows.append(
            {
                "epoch": epoch,
                "patch_role": patch_role,
                "property": name,
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "r2": 1.0 - residual_sum / total_sum if total_sum else np.nan,
                "correlation": correlation,
                "prior_rmse": float(np.sqrt(np.mean(np.square(baseline - expected)))),
            }
        )
    return rows


def _segmentation_metrics(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    ious, dices = [], []
    valid_count = int(mask.sum())
    for label in range(3):
        pred = (predicted == label) & mask
        target = (truth == label) & mask
        intersection = np.count_nonzero(pred & target)
        union = np.count_nonzero(pred | target)
        denominator = np.count_nonzero(pred) + np.count_nonzero(target)
        iou = intersection / union if union else np.nan
        dice = 2.0 * intersection / denominator if denominator else np.nan
        result[f"class_{label}_iou"] = float(iou)
        result[f"class_{label}_dice"] = float(dice)
        result[f"predicted_class_{label}_fraction"] = float(
            np.count_nonzero(pred) / max(valid_count, 1)
        )
        result[f"true_class_{label}_fraction"] = float(
            np.count_nonzero(target) / max(valid_count, 1)
        )
        ious.append(iou)
        dices.append(dice)
    result["miou"] = float(np.nanmean(ious))
    result["macro_dice"] = float(np.nanmean(dices))
    return result


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Return a global SSIM diagnostic over paired valid samples."""
    data_range = max(float(np.ptp(second)), 1e-12)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_first, mean_second = float(first.mean()), float(second.mean())
    variance_first, variance_second = float(first.var()), float(second.var())
    covariance = float(np.mean((first - mean_first) * (second - mean_second)))
    return float(
        ((2.0 * mean_first * mean_second + c1) * (2.0 * covariance + c2))
        / ((mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2))
    )


def _whole_realization_diagnostics(
    *,
    model: nn.Module,
    dataset_directory: Path,
    realization_ids: list[int],
    normalization: dict[str, list[float]],
    config: dict[str, Any],
    epoch: int,
    device: torch.device,
    output_directory: Path,
) -> list[dict[str, Any]]:
    """Evaluate the frozen validation sections in a checkpoint-only process."""
    rows: list[dict[str, Any]] = []
    specification = physics_settings_from_config(config).specification
    if specification is None:
        raise RuntimeError("Whole-realization physics requires the v003 forward specification")
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    array_directory = output_directory / "whole_realization_arrays" / f"epoch_{epoch:04d}"
    array_directory.mkdir(parents=True, exist_ok=True)
    for realization_id in realization_ids:
        path = dataset_directory / "realizations" / f"realization_{realization_id:07d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            avo = np.asarray(archive["avo"], dtype=np.float32)
            avo_clean = np.asarray(archive["avo_clean"], dtype=np.float32)
            low = np.asarray(archive["low"], dtype=np.float32)
            rgt = np.asarray(archive["rgt"], dtype=np.float32)
            truth = np.asarray(archive["elastic"], dtype=np.float32)
            labels = np.asarray(archive["segmentation"], dtype=np.int64)
            valid = np.asarray(archive["valid_mask"], dtype=bool)
        prediction, predicted_labels = infer_full_realization(
            model,
            avo=avo,
            low=low,
            rgt=rgt,
            normalization=normalization,
            patch_shape=tuple(int(value) for value in config["patches"]["shape"]),
            stride=tuple(int(value) for value in config["patches"]["stride"]),
            steps=int(config["observability"]["diagnostics"]["flow_integration_steps"]),
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            valid_mask=valid,
            guidance_scale=0.0,
        )
        segmentation = _segmentation_metrics(predicted_labels, labels, valid)
        for channel, name in enumerate(PROPERTIES):
            expected = truth[channel][valid]
            inferred = prediction[channel][valid]
            baseline = low[channel][valid]
            error = inferred - expected
            total_sum = float(np.square(expected - expected.mean()).sum())
            rows.append(
                {
                    "epoch": epoch,
                    "realization_id": realization_id,
                    "property": name,
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mae": float(np.mean(np.abs(error))),
                    "r2": (
                        1.0 - float(np.square(error).sum()) / total_sum if total_sum else np.nan
                    ),
                    "correlation": float(np.corrcoef(inferred, expected)[0, 1]),
                    "ssim": _global_ssim(inferred, expected),
                    "prior_rmse": float(np.sqrt(np.mean(np.square(baseline - expected)))),
                    **segmentation,
                }
            )
        with torch.no_grad():
            prediction_tensor = torch.from_numpy(prediction[None]).to(device)
            modeled = (
                forward_avo_three_band_spec_torch(
                    prediction_tensor[:, 0],
                    prediction_tensor[:, 1],
                    prediction_tensor[:, 2],
                    specification,
                    sample_origin=0,
                )[0]
                .cpu()
                .numpy()
            )
        normalized_modeled = (modeled - x_mean) / x_std
        normalized_clean = (avo_clean - x_mean) / x_std
        normalized_noisy = (avo - x_mean) / x_std
        physics_clean = float(
            np.mean(np.square(normalized_modeled[:, valid] - normalized_clean[:, valid]))
        )
        physics_noisy = float(
            np.mean(np.square(normalized_modeled[:, valid] - normalized_noisy[:, valid]))
        )
        for row in rows[-3:]:
            row["forward_avo_rmse_clean_normalized"] = float(np.sqrt(physics_clean))
            row["forward_avo_rmse_noisy_normalized"] = float(np.sqrt(physics_noisy))
        np.savez_compressed(
            array_directory / f"realization_{realization_id:07d}.npz",
            prior=low,
            truth=truth,
            prediction=prediction,
            residual=prediction - truth,
            segmentation_truth=labels,
            segmentation_prediction=predicted_labels,
            valid_mask=valid,
        )
    return rows


def analyze_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_directory: str | Path,
    run_directory: str | Path,
    sample_manifest_path: str | Path,
    output_directory: str | Path,
    device: str = "cpu",
    maximum_patches: int | None = None,
    flow_steps: int | None = None,
    include_whole_realizations: bool = True,
) -> dict[str, Any]:
    """Reload one checkpoint and write diagnostic-only machine-readable products."""
    checkpoint_path = Path(checkpoint_path)
    dataset_directory = Path(dataset_directory)
    run_directory = Path(run_directory)
    output = Path(output_directory)
    sample_manifest = json.loads(Path(sample_manifest_path).read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    epoch = int(checkpoint["epoch"])
    torch_device = torch.device(device)
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(torch_device)
    normalization_mapping = json.loads(
        (dataset_directory / "normalization.json").read_text(encoding="utf-8")
    )
    model.set_norm_stats(normalization_mapping)
    load_checkpoint(checkpoint_path, model, restore_rng=False, map_location=torch_device)
    model.eval()
    before_hash = _state_hash(model)
    batch, records = load_fixed_batch(
        dataset_directory,
        sample_manifest,
        physics_only=True,
        maximum_patches=maximum_patches,
    )
    values = _move_batch(batch, torch_device)
    time = torch.full(
        (values["target"].shape[0],),
        float(config["observability"]["diagnostics"]["deterministic_time"]),
        device=torch_device,
    )
    total_epochs = int(config["training"]["epochs"])
    base_weights = loss_weights_from_config(
        config, float(config["training"]["loss_weights"]["physics"])
    )
    weights = curriculum_from_config(config).weights_for_epoch(
        base_weights, epoch - 1, total_epochs
    )
    normalization = _normalization_tensors(normalization_mapping)
    run_manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    class_weights = torch.tensor(
        run_manifest["observability"]["training_class_weights"],
        dtype=torch.float32,
        device=torch_device,
    )
    physics = physics_settings_from_config(config)
    graph_objective = graph_objective_from_config(config)
    reinjected_streams: list[Tensor] = []

    def capture_reinjected_stream(_: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        reinjected_streams.append(inputs[0])

    hook = model.decoder.register_forward_pre_hook(capture_reinjected_stream)
    try:
        total, terms = _forward_objective(
            model,
            values,
            time,
            normalization,
            weights,
            class_weights,
            physics,
            ContrastiveSettings(
                temperature=float(config["training"]["contrastive_loss"]["temperature"]),
                max_samples=int(config["training"]["contrastive_loss"]["max_samples"]),
            ),
            deterministic_contrastive=True,
            contrastive_generator=None,
            adaptive_weighter=None,
            graph_objective=graph_objective,
        )
    finally:
        hook.remove()
    if len(reinjected_streams) != 1:
        raise RuntimeError(
            "Expected one graph-reinjected decoder input during diagnostic objective; "
            f"observed {len(reinjected_streams)}"
        )
    coefficients = effective_component_coefficients(weights)
    gradient_rows, cosine_rows = _gradient_diagnostics(
        model=model,
        terms=terms,
        coefficients=coefficients,
        epoch=epoch,
        activation_groups={"graph_reinjected_cnn_stream": reinjected_streams[0]},
    )

    state = (1.0 - time[:, None, None, None]) * values["low"] + time[:, None, None, None] * values[
        "target"
    ]
    model_output = model(state, time, values["avo"], values["low"], values["rgt"])
    predicted_full = values["low"] + model_output.velocity
    physics_values = {
        "prior_noiseless": _physics_value(
            values["low"], values, normalization, physics.specification, values["physics_avo"]
        ),
        "prediction_noiseless": _physics_value(
            predicted_full,
            values,
            normalization,
            physics.specification,
            values["physics_avo"],
        ),
        "truth_noiseless_operator_floor": _physics_value(
            values["target"],
            values,
            normalization,
            physics.specification,
            values["physics_avo"],
        ),
        "truth_noisy_observation_floor": _physics_value(
            values["target"], values, normalization, physics.specification, values["avo"]
        ),
    }
    floor = float(physics_values["truth_noiseless_operator_floor"])
    prior_level = float(physics_values["prior_noiseless"])
    prediction_level = float(physics_values["prediction_noiseless"])
    physics_row = {
        "epoch": epoch,
        **{name: float(value) for name, value in physics_values.items()},
        "normalized_progress_from_prior_to_operator_floor": (
            (prediction_level - floor) / (prior_level - floor) if prior_level != floor else np.nan
        ),
        "training_physics_target": "noiseless Stage-02 AVO on native 50x100 patches",
    }

    graph_prior = edge_smoothness(
        values["low"], model_output.edge_indices, model_output.edge_weights
    )
    graph_truth = edge_smoothness(
        values["target"], model_output.edge_indices, model_output.edge_weights
    )
    graph_prediction = edge_smoothness(
        predicted_full, model_output.edge_indices, model_output.edge_weights
    )
    selected_graph_objective = graph_structure_loss(
        predicted_full,
        values["target"],
        values["rgt"],
        values["segmentation"],
        model_output.edge_indices,
        model_output.edge_weights,
        graph_objective,
    )
    truth_edge_mismatch = truth_edge_matching(
        predicted_full,
        values["target"],
        model_output.edge_indices,
        model_output.edge_weights,
    )
    boundary_contrast_error = _boundary_contrast_error(
        predicted_full,
        values["target"],
        values["segmentation"],
        model_output.edge_indices,
        model_output.edge_weights,
    )
    graph_row = {
        "epoch": epoch,
        "selected_objective_mode": graph_objective.mode,
        "selected_objective_value": float(selected_graph_objective),
        "prior_reference": float(graph_prior),
        "truth_reference": float(graph_truth),
        "prediction": float(graph_prediction),
        "prediction_minus_truth_reference": float(graph_prediction - graph_truth),
        "truth_edge_mismatch": float(truth_edge_mismatch),
        "fault_boundary_contrast_error": float(boundary_contrast_error),
        "reference_interpretation": "truth level is not assumed to be zero",
    }

    graph_result, attentions, cnn, angular, gradient = _graph_attention_details(
        model, values, state, time
    )
    edge_indices = graph_result[2]
    graph_rows: list[dict[str, Any]] = []
    for layer_index, attention in enumerate(attentions, start=1):
        graph_rows.append(
            _attention_summary(
                epoch=epoch,
                layer=layer_index,
                attention=attention,
                edge_index=edge_indices[0],
                rgt=values["rgt"][0],
                avo=values["avo"][0],
            )
        )
    bypass_velocity = model.decoder(cnn)
    original_mode = model.graph.graph_mode
    try:
        model.graph.graph_mode = "cartesian"
        cartesian = model(state, time, values["avo"], values["low"], values["rgt"])
    finally:
        model.graph.graph_mode = original_mode
    graph_rows[0].update(
        {
            "graph_embedding_rms": float(model_output.embeddings.square().mean().sqrt()),
            "graph_reinjection_velocity_rms": float(
                (model_output.velocity - bypass_velocity).square().mean().sqrt()
            ),
            "rgt_vs_cartesian_velocity_rms": float(
                (model_output.velocity - cartesian.velocity).square().mean().sqrt()
            ),
            "mechanism_diagnostic_only": True,
        }
    )

    steps = int(
        flow_steps
        if flow_steps is not None
        else config["observability"]["diagnostics"]["flow_integration_steps"]
    )
    with torch.no_grad():
        sampled = model.sample(values["avo"], values["low"], values["rgt"], steps=steps)
        final_output = model(
            sampled,
            torch.ones(sampled.shape[0], device=torch_device),
            values["avo"],
            values["low"],
            values["rgt"],
        )
    mean = np.asarray(normalization_mapping["y_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization_mapping["y_std"], dtype=np.float32)[:, None, None]
    metric_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    arrays_directory = output / "fixed_patch_arrays" / f"epoch_{epoch:04d}"
    arrays_directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        prediction = sampled[index].detach().cpu().numpy() * std + mean
        truth = values["target"][index].detach().cpu().numpy() * std + mean
        prior = values["low"][index].detach().cpu().numpy() * std + mean
        mask = values["mask"][index, 0].detach().cpu().numpy() > 0.5
        metric_rows.extend(
            _elastic_metrics(
                prediction,
                truth,
                prior,
                mask,
                epoch=epoch,
                patch_role=str(record["role"]),
            )
        )
        predicted_labels = final_output.segmentation_logits[index].argmax(0).cpu().numpy()
        true_labels = values["segmentation"][index].cpu().numpy()
        segmentation_rows.append(
            {
                "epoch": epoch,
                "patch_role": record["role"],
                **_segmentation_metrics(predicted_labels, true_labels, mask),
            }
        )
        np.savez_compressed(
            arrays_directory / f"{record['role']}.npz",
            avo=values["avo"][index].detach().cpu().numpy(),
            rgt=values["rgt"][index].detach().cpu().numpy(),
            prior=prior,
            truth=truth,
            prediction=prediction,
            residual=prediction - truth,
            segmentation_truth=true_labels,
            segmentation_prediction=predicted_labels,
            shuey_intercept=angular[index, 3].detach().cpu().numpy(),
            shuey_gradient=gradient[index, 0].detach().cpu().numpy(),
            graph_embedding_norm=model_output.embeddings[index]
            .detach()
            .square()
            .sum(dim=1)
            .sqrt()
            .reshape(values["rgt"].shape[-2:])
            .cpu()
            .numpy(),
        )

    diagnostic_directory = output
    _upsert_rows(
        diagnostic_directory / "gradient_contributions.csv",
        gradient_rows,
        ["epoch", "objective", "parameter_group"],
    )
    _upsert_rows(
        diagnostic_directory / "gradient_cosines.csv",
        cosine_rows,
        ["epoch", "objective_a", "objective_b"],
    )
    _upsert_rows(
        diagnostic_directory / "physics_floor_diagnostics.csv",
        [physics_row],
        ["epoch"],
    )
    _upsert_rows(
        diagnostic_directory / "graph_floor_diagnostics.csv",
        [graph_row],
        ["epoch"],
    )
    _upsert_rows(
        diagnostic_directory / "graph_learning_summary.csv",
        graph_rows,
        ["epoch", "layer"],
    )
    _upsert_rows(
        diagnostic_directory / "fixed_patch_metrics.csv",
        metric_rows,
        ["epoch", "patch_role", "property"],
    )
    _upsert_rows(
        diagnostic_directory / "segmentation_metrics.csv",
        segmentation_rows,
        ["epoch", "patch_role"],
    )
    whole_rows: list[dict[str, Any]] = []
    if include_whole_realizations:
        whole_rows = _whole_realization_diagnostics(
            model=model,
            dataset_directory=dataset_directory,
            realization_ids=[int(value) for value in sample_manifest["whole_realization_ids"]],
            normalization=normalization_mapping,
            config=config,
            epoch=epoch,
            device=torch_device,
            output_directory=output,
        )
        _upsert_rows(
            diagnostic_directory / "whole_realization_metrics.csv",
            whole_rows,
            ["epoch", "realization_id", "property"],
        )
    after_hash = _state_hash(model)
    if after_hash != before_hash:
        raise RuntimeError("Diagnostic extraction modified the checkpoint model state")
    report = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "fixed_patch_roles": [record["role"] for record in records],
        "fixed_validation_only": True,
        "test_data_used": False,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "model_state_unchanged": True,
        "optimizer_loaded_or_modified": False,
        "total_diagnostic_objective": float(total),
        "flow_integration_steps": steps,
        "whole_realizations_evaluated": sorted({int(row["realization_id"]) for row in whole_rows}),
        "physics": physics_row,
        "graph": graph_row,
    }
    report_path = output / f"checkpoint_diagnostics_epoch_{epoch:04d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
