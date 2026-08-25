#!/usr/bin/env python3
"""Replay the historical step-88 masked-physics failure and verify its correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from sage_avo.config import load_config, seed_everything
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import build_patch_sampling_weights
from sage_avo.experiments import training as experiment_training
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.models.variants import (
    build_sage_avo_variant,
    sage_avo_model_kwargs,
    variant_definition,
)
from sage_avo.training.engine import (
    ContrastiveSettings,
    _forward_objective,
    _move_batch,
    train_step,
)
from sage_avo.training.flow import straight_path
from sage_avo.training.losses import masked_mse


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
DATASET = (
    PRIVATE
    / "stage_artifacts"
    / "stage03"
    / "ds_v00331_production100_support_aware"
    / "dataset"
)
CONFIG_PATH = REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"
DATASET_MANIFEST_SHA256 = "1afe64debc9b0901afde88b327a1c04088c1a2a8e51efafadd38bc5e6ba845ee"
EXPECTED_STEP88 = {
    "realization_id": [3400053, 3400055, 3400003, 3400020, 3400024, 3400074, 3400055, 3400074],
    "top": [185, 197, 155, 0, 124, 214, 152, 197],
    "left": [54, 19, 50, 40, 4, 40, 6, 9],
    "raw_shape": [[40, 80], [64, 128], [50, 100], [50, 100], [64, 128], [50, 100], [40, 80], [64, 128]],
}


def _sha256_tensor(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _all_finite(values: Any) -> bool:
    if isinstance(values, Tensor):
        return bool(torch.isfinite(values).all())
    if isinstance(values, dict):
        return all(_all_finite(value) for value in values.values())
    if isinstance(values, (tuple, list)):
        return all(_all_finite(value) for value in values)
    if isinstance(values, float):
        return math.isfinite(values)
    return True


def _setup(device: torch.device) -> dict[str, Any]:
    if file_sha256(DATASET / "dataset_manifest.json") != DATASET_MANIFEST_SHA256:
        raise RuntimeError("Frozen Stage-03 manifest hash mismatch")
    config = load_config(CONFIG_PATH)
    training = config["training"]
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic_torch=True)
    augmentation_generator = torch.Generator().manual_seed(seed + 11)
    sampler_generator = torch.Generator().manual_seed(seed + 13)
    time_generator = torch.Generator().manual_seed(seed + 17)
    contrastive_generator = torch.Generator(device=device).manual_seed(seed + 19)
    dataset = IndexedRealizationPatches(
        DATASET,
        "train",
        augment=bool(training["augmentation"]["enabled"]),
        augmentation_config=experiment_training._augmentation(config),
        augmentation_generator=augmentation_generator,
    )
    sampler = WeightedRandomSampler(
        build_patch_sampling_weights(dataset, experiment_training._sampling(config)),
        num_samples=len(dataset),
        replacement=True,
        generator=sampler_generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    definition = variant_definition(
        "full", physics_weight=float(training["loss_weights"]["physics"])
    )
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
    normalization = json.loads((DATASET / "normalization.json").read_text(encoding="utf-8"))
    model.set_norm_stats(normalization)
    base_weights = experiment_training.loss_weights_from_config(
        config, float(definition.physics_weight)
    )
    weights = experiment_training.curriculum_from_config(config).weights_for_epoch(
        base_weights, 0, int(training["epochs"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    class_weights = experiment_training._class_weights(
        dataset,
        classes=int(config["model"]["classes"]),
        foreground_boost=float(training["class_weight_foreground_boost"]),
    ).to(device)
    return {
        "config": config,
        "training": training,
        "loader": loader,
        "model": model,
        "normalization": experiment_training._normalization_tensors(normalization),
        "weights": weights,
        "optimizer": optimizer,
        "class_weights": class_weights,
        "time_generator": time_generator,
        "physics": experiment_training.physics_settings_from_config(config),
        "contrastive": ContrastiveSettings(
            temperature=float(training["contrastive_loss"]["temperature"]),
            max_samples=int(training["contrastive_loss"]["max_samples"]),
        ),
        "contrastive_generator": contrastive_generator,
    }


def _modeled_physics(
    model: torch.nn.Module,
    values: dict[str, Tensor],
    time_value: Tensor,
    normalization: Any,
    specification: Any,
) -> tuple[Tensor, Tensor]:
    state, _ = straight_path(values["low"], values["target"], time_value)
    output = model(state, time_value, values["avo"], values["low"], values["rgt"])
    predicted = output.velocity + values["low"]
    context = values["physics_context"].clone()
    height = predicted.shape[2]
    for item, start_value in enumerate(values["physics_core_start"]):
        start = int(start_value.item())
        context[item, :, start : start + height] = predicted[item]
    physical = (
        context * normalization.y_std.to(context.device)
        + normalization.y_mean.to(context.device)
    )
    modeled = forward_avo_three_band_spec_torch(
        physical[:, 0],
        physical[:, 1],
        physical[:, 2],
        specification,
        sample_origin=values["physics_context_sample_origin"],
    )
    core = torch.empty_like(values["physics_avo"])
    for item, start_value in enumerate(values["physics_core_start"]):
        start = int(start_value.item())
        core[item] = modeled[item, :, start : start + height]
    normalized = (
        core - normalization.x_mean.to(core.device)
    ) / normalization.x_std.to(core.device)
    return normalized, predicted


def run(arguments: argparse.Namespace) -> None:
    device = torch.device(arguments.device)
    runtime = _setup(device)
    started = time.perf_counter()
    batch88: dict[str, Tensor] | None = None
    finite_steps = 0
    for step, batch in enumerate(runtime["loader"], 1):
        if step == 88:
            batch88 = batch
            break
        metrics = train_step(
            runtime["model"],
            batch,
            runtime["optimizer"],
            runtime["normalization"],
            runtime["weights"],
            runtime["class_weights"],
            float(runtime["training"]["gradient_clip"]),
            runtime["time_generator"],
            runtime["physics"],
            runtime["contrastive"],
            runtime["contrastive_generator"],
            None,
        )
        if not _all_finite(metrics.__dict__):
            raise RuntimeError(f"Corrected replay became nonfinite before step 88 at step {step}")
        finite_steps += 1
    if batch88 is None:
        raise RuntimeError("Training loader did not reach step 88")
    observed = {
        key: batch88[key].tolist() for key in ("realization_id", "top", "left", "raw_shape")
    }
    if observed != EXPECTED_STEP88:
        raise RuntimeError(f"Step-88 batch mismatch: {observed}")

    values = _move_batch(batch88, device)
    time_value = torch.rand(
        values["target"].shape[0], generator=runtime["time_generator"]
    ).to(device)
    modeled, _ = _modeled_physics(
        runtime["model"],
        values,
        time_value,
        runtime["normalization"],
        runtime["physics"].specification,
    )
    target = values["physics_avo"]
    expanded = values["physics_mask"].expand_as(modeled).to(modeled.dtype)
    residual = modeled - target
    squared = residual.square()
    denominator = expanded.sum() + 1e-8
    legacy = (squared * expanded).sum() / denominator
    corrected = masked_mse(modeled, target, values["physics_mask"])
    eligible = expanded != 0
    reference = (residual[eligible].square() * expanded[eligible]).sum() / denominator
    corrected_gradient = torch.autograd.grad(corrected, modeled, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, modeled, retain_graph=True)[0]

    total, terms = _forward_objective(
        runtime["model"],
        values,
        time_value,
        runtime["normalization"],
        runtime["weights"],
        runtime["class_weights"],
        runtime["physics"],
        runtime["contrastive"],
        deterministic_contrastive=False,
        contrastive_generator=runtime["contrastive_generator"],
        adaptive_weighter=None,
    )
    runtime["optimizer"].zero_grad(set_to_none=True)
    total.backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in runtime["model"].parameters()
    )

    inactive_values = dict(values)
    inactive_values["physics_mask"] = torch.zeros_like(values["physics_mask"])
    inactive_total, inactive_terms = _forward_objective(
        runtime["model"],
        inactive_values,
        time_value,
        runtime["normalization"],
        runtime["weights"],
        runtime["class_weights"],
        runtime["physics"],
        runtime["contrastive"],
        deterministic_contrastive=True,
        contrastive_generator=None,
        adaptive_weighter=None,
    )
    runtime["optimizer"].zero_grad(set_to_none=True)
    inactive_total.backward()
    inactive_gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in runtime["model"].parameters()
    )
    eligible_sample_mask = values["physics_eligible"].bool()
    overflow_locations = ~torch.isfinite(squared)
    report = {
        "schema_version": 1,
        "revision": "3.3.2a",
        "status": "passed",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "device": str(device),
        "historical_step": 88,
        "preceding_corrected_optimizer_steps": finite_steps,
        "batch_exactly_reproduced": True,
        "batch": observed,
        "time": [float(value) for value in time_value],
        "physics_eligible": values["physics_eligible"].bool().tolist(),
        "eligible_sample_count": int(eligible_sample_mask.sum()),
        "effective_denominator": float(denominator),
        "before_correction": {
            "legacy_square_then_mask_loss": "NaN" if torch.isnan(legacy) else float(legacy),
            "legacy_loss_nonfinite": not bool(torch.isfinite(legacy)),
            "squared_residual_nonfinite_count": int(overflow_locations.sum()),
            "nonfinite_count_by_item": [
                int(overflow_locations[item].sum()) for item in range(len(overflow_locations))
            ],
            "failing_item": 1,
            "failing_item_realization_id": int(values["realization_id"][1]),
            "failing_item_raw_shape": values["raw_shape"][1].tolist(),
        },
        "after_correction": {
            "physics_loss": float(corrected),
            "eligible_only_reference": float(reference),
            "loss_absolute_difference": float((corrected - reference).abs()),
            "gradient_max_absolute_difference": float(
                (corrected_gradient - reference_gradient).abs().max()
            ),
            "inactive_gradient_nonzero_count": int(
                torch.count_nonzero(corrected_gradient[~eligible]).item()
            ),
            "eligible_modeled_response_sha256": _sha256_tensor(modeled[eligible_sample_mask]),
            "total_objective": float(total),
            "raw_components": {name: float(value.detach()) for name, value in terms.items()},
            "all_components_finite": _all_finite(terms),
            "total_finite": bool(torch.isfinite(total)),
            "all_model_gradients_finite": gradient_finite,
            "all_parameters_finite": _all_finite(runtime["model"].state_dict()),
            "all_optimizer_state_finite": _all_finite(runtime["optimizer"].state_dict()),
        },
        "all_inactive_batch": {
            "raw_physics_loss": float(inactive_terms["physics"]),
            "weighted_physics_contribution": float(
                inactive_terms["physics"] * runtime["weights"].physics
            ),
            "total_objective": float(inactive_total),
            "total_finite": bool(torch.isfinite(inactive_total)),
            "all_gradients_finite": inactive_gradients_finite,
        },
        "scientific_contract": (
            "Only native physics-eligible entries contribute; mask weighting and "
            "denominator semantics are unchanged."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    required = (
        report["before_correction"]["legacy_loss_nonfinite"]
        and report["after_correction"]["loss_absolute_difference"] == 0.0
        and report["after_correction"]["gradient_max_absolute_difference"] == 0.0
        and report["after_correction"]["inactive_gradient_nonzero_count"] == 0
        and report["after_correction"]["all_components_finite"]
        and report["after_correction"]["total_finite"]
        and report["after_correction"]["all_model_gradients_finite"]
        and report["after_correction"]["all_parameters_finite"]
        and report["after_correction"]["all_optimizer_state_finite"]
        and report["all_inactive_batch"]["raw_physics_loss"] == 0.0
        and report["all_inactive_batch"]["weighted_physics_contribution"] == 0.0
        and report["all_inactive_batch"]["total_finite"]
        and report["all_inactive_batch"]["all_gradients_finite"]
    )
    report["status"] = "passed" if required else "failed"
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    if not required:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output", required=True)
    root.add_argument("--device", default="cuda")
    return root


if __name__ == "__main__":
    run(parser().parse_args())
