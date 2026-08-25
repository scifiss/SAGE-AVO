#!/usr/bin/env python3
"""Record a deterministic short production-training trajectory for comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from sage_avo.config import load_config, seed_everything
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import build_patch_sampling_weights
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.training import (
    _augmentation,
    _class_weights,
    _normalization_tensors,
    _sampling,
    curriculum_from_config,
    loss_weights_from_config,
    physics_settings_from_config,
)
from sage_avo.models.variants import (
    build_sage_avo_variant,
    sage_avo_model_kwargs,
    variant_definition,
)
from sage_avo.training.engine import ContrastiveSettings, train_step


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET_MANIFEST_SHA256 = "1afe64debc9b0901afde88b327a1c04088c1a2a8e51efafadd38bc5e6ba845ee"
PRIVATE = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
DATASET = (
    PRIVATE / "stage_artifacts" / "stage03" / "ds_v00331_production100_support_aware" / "dataset"
)


def _tensor_hash(records: list[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in records:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_hash(model: torch.nn.Module) -> str:
    return _tensor_hash([(name, value) for name, value in model.state_dict().items()])


def _gradient_hash(model: torch.nn.Module) -> str:
    return _tensor_hash(
        [
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
    )


def _optimizer_hash(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    named = {parameter: name for name, parameter in model.named_parameters()}
    records: list[tuple[str, Tensor]] = []
    scalar_records: list[str] = []
    for parameter, state in optimizer.state.items():
        name = named[parameter]
        for key, value in sorted(state.items()):
            if isinstance(value, Tensor):
                records.append((f"{name}:{key}", value))
            else:
                scalar_records.append(f"{name}:{key}:{value!r}")
    digest = hashlib.sha256()
    digest.update(_tensor_hash(records).encode())
    digest.update("|".join(scalar_records).encode())
    for group in optimizer.param_groups:
        digest.update(
            json.dumps(
                {key: value for key, value in group.items() if key != "params"},
                sort_keys=True,
                default=str,
            ).encode()
        )
    return digest.hexdigest()


def _batch_hash(batch: dict[str, Tensor]) -> str:
    return _tensor_hash(
        [(name, value) for name, value in sorted(batch.items()) if isinstance(value, Tensor)]
    )


def _gradient_norm(model: torch.nn.Module) -> float:
    return float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )


def run(args: argparse.Namespace) -> None:
    if file_sha256(DATASET / "dataset_manifest.json") != DATASET_MANIFEST_SHA256:
        raise RuntimeError("Frozen Stage-03 manifest hash mismatch")
    config = load_config(REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic_torch=True)
    device = torch.device(args.device)
    training = config["training"]
    augmentation_generator = torch.Generator().manual_seed(seed + 11)
    sampler_generator = torch.Generator().manual_seed(seed + 13)
    time_generator = torch.Generator().manual_seed(seed + 17)
    contrastive_generator = torch.Generator(device=device).manual_seed(seed + 19)
    dataset = IndexedRealizationPatches(
        DATASET,
        "train",
        augment=bool(training["augmentation"]["enabled"]),
        augmentation_config=_augmentation(config),
        augmentation_generator=augmentation_generator,
    )
    weights = build_patch_sampling_weights(dataset, _sampling(config))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
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
    base_weights = loss_weights_from_config(config, float(definition.physics_weight))
    effective_weights = curriculum_from_config(config).weights_for_epoch(
        base_weights, 0, int(training["epochs"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
        eta_min=float(training["scheduler_eta_min"]),
    )
    class_weights = _class_weights(
        dataset,
        classes=int(config["model"]["classes"]),
        foreground_boost=float(training["class_weight_foreground_boost"]),
    ).to(device)
    initial_hash = _model_hash(model)
    steps: list[dict[str, Any]] = []
    iterator = iter(loader)
    for step_index in range(int(args.steps)):
        batch = next(iterator)
        batch_signature = _batch_hash(batch)
        metrics = train_step(
            model,
            batch,
            optimizer,
            _normalization_tensors(normalization),
            effective_weights,
            class_weights,
            float(training["gradient_clip"]),
            time_generator,
            physics_settings_from_config(config),
            ContrastiveSettings(
                temperature=float(training["contrastive_loss"]["temperature"]),
                max_samples=int(training["contrastive_loss"]["max_samples"]),
            ),
            contrastive_generator,
            None,
        )
        steps.append(
            {
                "step": step_index + 1,
                "batch_hash": batch_signature,
                "realization_ids": batch["realization_id"].tolist(),
                "top": batch["top"].tolist(),
                "left": batch["left"].tolist(),
                "raw_shape": batch["raw_shape"].tolist(),
                "metrics": metrics.__dict__,
                "gradient_norm_after_clipping": _gradient_norm(model),
                "gradient_sha256": _gradient_hash(model),
                "parameter_sha256": _model_hash(model),
                "optimizer_sha256": _optimizer_hash(model, optimizer),
            }
        )
    scheduler.step()
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "seed": seed,
        "device": str(device),
        "steps": int(args.steps),
        "batch_size": int(training["batch_size"]),
        "augmentation_enabled": bool(training["augmentation"]["enabled"]),
        "weighted_sampling_enabled": bool(training["weighted_patch_sampling"]["enabled"]),
        "amp_enabled": False,
        "initial_parameter_sha256": initial_hash,
        "step_records": steps,
        "final_parameter_sha256": _model_hash(model),
        "final_optimizer_sha256": _optimizer_hash(model, optimizer),
        "scheduler_state": scheduler.state_dict(),
        "generator_state_sha256": {
            "augmentation": hashlib.sha256(
                augmentation_generator.get_state().numpy().tobytes()
            ).hexdigest(),
            "sampler": hashlib.sha256(sampler_generator.get_state().numpy().tobytes()).hexdigest(),
            "time": hashlib.sha256(time_generator.get_state().numpy().tobytes()).hexdigest(),
            "contrastive": hashlib.sha256(
                contrastive_generator.get_state().cpu().numpy().tobytes()
            ).hexdigest(),
        },
    }
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output", required=True)
    root.add_argument("--steps", type=int, default=3)
    root.add_argument("--device", default="cpu")
    return root


if __name__ == "__main__":
    run(parser().parse_args())
