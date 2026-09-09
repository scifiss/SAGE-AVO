"""Read-only counterfactual sweeps of task-specific elastic attention strength."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.diagnostics.checkpoint_analysis import (
    _attention_summary,
    _graph_attention_details,
    _physics_value,
    _state_hash,
    load_fixed_batch,
)
from sage_avo.experiments.training import (
    _normalization_tensors,
    _validation_sample_metrics,
    curriculum_from_config,
    graph_objective_from_config,
    loss_weights_from_config,
    physics_settings_from_config,
)
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.checkpoints import load_checkpoint
from sage_avo.training.engine import ContrastiveSettings, _move_batch, validate_epoch
from sage_avo.training.selection import weighted_objective_contributions


SWEEP_COLUMNS = (
    "label",
    "requested_strength",
    "realized_strength_min",
    "realized_strength_max",
    "tangential_rgt_strength",
    "tangential_avo_strength",
    "normal_rgt_strength",
    "normal_avo_strength",
    "validation_fixed_objective",
    "validation_raw_physics",
    "sample_criterion",
    "sample_rmse_vp_normalized",
    "sample_rmse_vs_normalized",
    "sample_rmse_density_normalized",
    "sample_miou",
    "sample_class_0_iou",
    "sample_class_1_iou",
    "sample_class_2_iou",
    "probe_exact_pp_prediction_noiseless",
    "elastic_attention_entropy_mean",
    "elastic_attention_concentration_mean",
    "elastic_attention_prior_correlation_mean",
    "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inverse_softplus(value: float, *, dtype: torch.dtype) -> float:
    """Return a finite raw value; requested zero becomes numerical zero."""
    if value < 0:
        raise ValueError("Elastic structural strength must be non-negative")
    if value == 0:
        return math.log(float(torch.finfo(dtype).tiny))
    return math.log(math.expm1(value))


def _raw_strength_matrix(
    values: Sequence[Sequence[float]], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    positive = torch.as_tensor(values, dtype=dtype, device=device)
    if tuple(positive.shape) != (2, 2):
        raise ValueError("Elastic strength intervention must have shape [2,2]")
    raw = torch.empty_like(positive)
    for relation in range(2):
        for feature in range(2):
            raw[relation, feature] = _inverse_softplus(
                float(positive[relation, feature]), dtype=dtype
            )
    return raw


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temporary.replace(path)


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


@torch.no_grad()
def run_elastic_strength_sweep(
    *,
    checkpoint_path: str | Path,
    dataset_directory: str | Path,
    run_directory: str | Path,
    fixed_selection_path: str | Path,
    probe_manifest_path: str | Path,
    output_directory: str | Path,
    strengths: Sequence[float],
    strength_matrices: Sequence[tuple[str, Sequence[Sequence[float]]]] = (),
    device_name: str,
    flow_steps: int,
) -> dict[str, Any]:
    """Evaluate frozen weights under in-memory elastic-strength interventions.

    No optimizer is constructed, and the checkpoint file is never written. Completed
    settings are checkpointed as CSV rows so the sweep can resume after interruption.
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_directory = Path(dataset_directory)
    run_directory = Path(run_directory)
    fixed_selection_path = Path(fixed_selection_path)
    probe_manifest_path = Path(probe_manifest_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_csv = output_directory / "elastic_strength_sweep.csv"
    output_json = output_directory / "elastic_strength_sweep_manifest.json"

    checkpoint_file_before = _file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    epoch = int(checkpoint["epoch"])
    print_torch_runtime()
    device = select_torch_device(
        device_name,
        require_cuda=str(device_name).startswith("cuda"),
        context="read-only elastic-attention strength sweep",
    )
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
    normalization_mapping = json.loads(
        (dataset_directory / "normalization.json").read_text(encoding="utf-8")
    )
    model.set_norm_stats(normalization_mapping)
    load_checkpoint(checkpoint_path, model, restore_rng=False, map_location=device)
    model.eval()
    strength_parameter = getattr(model.graph, "elastic_attention_raw_strengths", None)
    if strength_parameter is None or tuple(strength_parameter.shape) != (2, 2):
        raise RuntimeError("Checkpoint does not expose four task-specific elastic strengths")
    original_raw = strength_parameter.detach().clone()
    original_positive = torch.nn.functional.softplus(original_raw).detach().cpu().tolist()
    model_hash_before = _state_hash(model)

    selection = json.loads(fixed_selection_path.read_text(encoding="utf-8"))
    validation_source = IndexedRealizationPatches(dataset_directory, "validation")
    validation_indices = [int(value) for value in selection["validation_indices"]]
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    run_manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    class_weights = torch.tensor(
        run_manifest["observability"]["training_class_weights"],
        dtype=torch.float32,
        device=device,
    )
    normalization = _normalization_tensors(normalization_mapping)
    physics = physics_settings_from_config(config)
    if physics.specification is None:
        raise RuntimeError("The exact shared forward specification is required for this sweep")
    graph_objective = graph_objective_from_config(config)
    base_weights = loss_weights_from_config(
        config, float(config["training"]["loss_weights"]["physics"])
    )
    total_epochs = int(config["training"]["epochs"])
    final_weights = curriculum_from_config(config).weights_for_epoch(
        base_weights, total_epochs - 1, total_epochs
    )
    validation_time_grid = tuple(
        float(value) for value in config["training"]["validation_time_grid"]
    )
    contrastive = ContrastiveSettings(
        temperature=float(config["training"]["contrastive_loss"]["temperature"]),
        max_samples=int(config["training"]["contrastive_loss"]["max_samples"]),
    )
    guidance = config["training"]["physics_guided_sampling"]
    guidance_scale = float(guidance["guidance_scale"]) if bool(guidance["enabled"]) else 0.0

    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    probe_batch, _ = load_fixed_batch(
        dataset_directory,
        probe_manifest,
        physics_only=True,
        maximum_patches=2,
    )
    probe_values = _move_batch(probe_batch, device)
    probe_time = torch.full(
        (probe_values["target"].shape[0],),
        float(config["observability"]["diagnostics"]["deterministic_time"]),
        device=device,
    )

    existing: list[dict[str, Any]] = []
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    completed_labels = {str(row["label"]) for row in existing}
    rows: list[dict[str, Any]] = existing
    requested: list[tuple[str, float | str, torch.Tensor | None]] = [
        ("native_epoch3", "native", None)
    ]
    requested.extend(
        (
            f"fixed_{value:g}",
            float(value),
            _raw_strength_matrix(
                ((value, value), (value, value)),
                dtype=strength_parameter.dtype,
                device=device,
            ),
        )
        for value in strengths
    )
    requested.extend(
        (
            str(label),
            "matrix",
            _raw_strength_matrix(values, dtype=strength_parameter.dtype, device=device),
        )
        for label, values in strength_matrices
    )

    try:
        for label, requested_strength, raw_intervention in requested:
            if label in completed_labels:
                print(f"[strength-sweep] skip completed setting {label}", flush=True)
                continue
            strength_parameter.copy_(original_raw)
            if raw_intervention is not None:
                strength_parameter.copy_(raw_intervention)
            realized = torch.nn.functional.softplus(strength_parameter)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            validation = validate_epoch(
                model,
                validation_loader,
                normalization,
                final_weights,
                time_grid=validation_time_grid,
                class_weights=class_weights,
                physics=physics,
                contrastive=contrastive,
                adaptive_weighter=None,
                max_batches=len(validation_loader),
                graph_objective=graph_objective,
            )
            fixed = weighted_objective_contributions(validation, final_weights)
            sampled = _validation_sample_metrics(
                model,
                validation_loader,
                device,
                steps=int(flow_steps),
                max_batches=len(validation_loader),
                guidance_scale=guidance_scale,
            )

            state = (1.0 - probe_time[:, None, None, None]) * probe_values["low"] + probe_time[
                :, None, None, None
            ] * probe_values["target"]
            output = model(
                state,
                probe_time,
                probe_values["avo"],
                probe_values["low"],
                probe_values["rgt"],
            )
            predicted_full = probe_values["low"] + output.velocity
            exact_pp = _physics_value(
                predicted_full,
                probe_values,
                normalization,
                physics.specification,
                probe_values["physics_avo"],
            )
            _, details, _, _, _ = _graph_attention_details(
                model, probe_values, state, probe_time
            )
            elastic_summaries = [
                _attention_summary(
                    epoch=epoch,
                    stream=str(detail.get("stream", "shared")),
                    relation=str(detail["relation"]),
                    layer=int(detail["layer"]),
                    attention=detail["attention"],
                    edge_index=detail["edge_index"],
                    rgt=probe_values["rgt"][0],
                    avo=probe_values["avo"][0],
                    attention_prior=detail.get("attention_prior"),
                    structural_strengths=detail.get("structural_strengths"),
                )
                for detail in details
                if str(detail.get("stream", "shared")) == "elastic"
            ]
            if not elastic_summaries:
                raise RuntimeError("No elastic-stream attention diagnostics were produced")
            row = {
                "label": label,
                "requested_strength": requested_strength,
                "realized_strength_min": float(realized.min()),
                "realized_strength_max": float(realized.max()),
                "tangential_rgt_strength": float(realized[0, 0]),
                "tangential_avo_strength": float(realized[0, 1]),
                "normal_rgt_strength": float(realized[1, 0]),
                "normal_avo_strength": float(realized[1, 1]),
                "validation_fixed_objective": float(fixed["total"]),
                "validation_raw_physics": float(validation.physics),
                "sample_criterion": float(sampled["criterion"]),
                "sample_rmse_vp_normalized": float(sampled["normalized_rmse"][0]),
                "sample_rmse_vs_normalized": float(sampled["normalized_rmse"][1]),
                "sample_rmse_density_normalized": float(sampled["normalized_rmse"][2]),
                "sample_miou": float(sampled["miou"]),
                "sample_class_0_iou": float(sampled["class_iou"][0]),
                "sample_class_1_iou": float(sampled["class_iou"][1]),
                "sample_class_2_iou": float(sampled["class_iou"][2]),
                "probe_exact_pp_prediction_noiseless": float(exact_pp),
                "elastic_attention_entropy_mean": _finite_mean(
                    [item["attention_entropy_normalized"] for item in elastic_summaries]
                ),
                "elastic_attention_concentration_mean": _finite_mean(
                    [item["attention_concentration"] for item in elastic_summaries]
                ),
                "elastic_attention_prior_correlation_mean": _finite_mean(
                    [item.get("attention_prior_correlation", np.nan) for item in elastic_summaries]
                ),
                "cuda_peak_allocated_mib": (
                    float(torch.cuda.max_memory_allocated(device) / 2**20)
                    if device.type == "cuda"
                    else 0.0
                ),
                "cuda_peak_reserved_mib": (
                    float(torch.cuda.max_memory_reserved(device) / 2**20)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            rows.append(row)
            _write_csv(output_csv, rows)
            print(
                "[strength-sweep] "
                f"{label} fixed={row['validation_fixed_objective']:.8f} "
                f"sample={row['sample_criterion']:.8f} "
                f"physics={row['validation_raw_physics']:.8f} "
                f"class1={row['sample_class_1_iou']:.8f}",
                flush=True,
            )
    finally:
        strength_parameter.copy_(original_raw)

    model_hash_after = _state_hash(model)
    checkpoint_file_after = _file_sha256(checkpoint_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("Strength sweep failed to restore the in-memory model state")
    if checkpoint_file_after != checkpoint_file_before:
        raise RuntimeError("Strength sweep modified the immutable checkpoint file")
    report = {
        "status": "COMPLETE",
        "scientific_role": "read_only_counterfactual_inference_sweep_not_training",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "checkpoint_file_sha256_before": checkpoint_file_before,
        "checkpoint_file_sha256_after": checkpoint_file_after,
        "checkpoint_file_unchanged": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after_restore": model_hash_after,
        "model_state_restored": True,
        "native_positive_elastic_strengths": original_positive,
        "requested_fixed_strengths": [float(value) for value in strengths],
        "requested_strength_matrices": [
            {"label": str(label), "values": values} for label, values in strength_matrices
        ],
        "validation_patch_count": len(validation_indices),
        "probe_patch_count": int(probe_values["target"].shape[0]),
        "flow_steps": int(flow_steps),
        "optimizer_constructed": False,
        "training_performed": False,
        "rows": len(rows),
        "csv": str(output_csv),
    }
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_json)
    return report
