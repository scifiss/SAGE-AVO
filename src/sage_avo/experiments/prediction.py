"""Generate comparable full-test predictions for every benchmark condition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage_avo.config import seed_everything
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.evaluation.inference import infer_full_realization, load_normalization
from sage_avo.models.variants import (
    LEARNED_VARIANTS,
    build_sage_avo_variant,
    sage_avo_model_kwargs,
)
from sage_avo.training.checkpoints import load_checkpoint

from .manifest import build_run_manifest, write_json


V003_CHECKPOINT_FILES = {
    "fixed_objective": "best_fixed_objective.pt",
    "sampling": "best_sampling.pt",
    "segmentation": "best_segmentation.pt",
    "whole_realization": "best_whole_realization.pt",
}


def preferred_inference_checkpoint(config: dict[str, Any], run_directory: Path) -> Path:
    """Resolve the checkpoint named by the experiment's selection contract."""
    if int(config.get("schema_version", 2)) < 3:
        return run_directory / "best_sampling.pt"
    checkpointing = config["training"]["checkpointing"]
    criterion = str(checkpointing["preferred_final_criterion"])
    try:
        filename = V003_CHECKPOINT_FILES[criterion]
    except KeyError as error:
        raise ValueError(
            "training.checkpointing.preferred_final_criterion must be one of "
            f"{tuple(V003_CHECKPOINT_FILES)}; found {criterion!r}"
        ) from error
    return run_directory / filename


def _realization_path(dataset_directory: Path, realization_id: int) -> Path:
    directory = dataset_directory / "realizations"
    canonical = directory / f"realization_{realization_id:07d}.npz"
    legacy = directory / f"realization_{realization_id:04d}.npz"
    return canonical if canonical.exists() or not legacy.exists() else legacy


def load_controlled_model(
    variant: str,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    normalization: dict[str, list[float]] | None = None,
) -> torch.nn.Module:
    model = build_sage_avo_variant(
        variant,
        **sage_avo_model_kwargs(config),
    ).to(device)
    if normalization is not None:
        model.set_norm_stats(normalization)
    load_checkpoint(checkpoint, model, map_location=device)
    if normalization is None:
        if torch.any(model.X_std_buf <= 0) or torch.any(model.Y_std_buf <= 0):
            raise ValueError("Checkpoint contains invalid normalization buffers")
        model.normalization_ready.fill_(True)
    model.eval()
    return model


def predict_controlled_variant(
    *,
    repository: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    dataset_directory: str | Path,
    experiment_directory: str | Path,
    variant: str,
    device_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    prediction_directory: str | Path | None = None,
    require_cuda: bool = False,
    inference_batch_size: int | None = None,
) -> Path:
    """Write one prediction artifact per complete test realization."""
    if variant not in ("low_prior",) + LEARNED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    seed = int(config["experiment"]["seed"])
    seed_everything(
        seed,
        deterministic_torch=bool(config["hardware"].get("deterministic_algorithms", True)),
    )
    dataset_root = Path(dataset_directory)
    experiment_root = Path(experiment_directory)
    prediction_root = Path(prediction_directory) if prediction_directory is not None else experiment_root
    output = prediction_root / "predictions" / variant
    output.mkdir(parents=True, exist_ok=True)
    split_ids = json.loads((dataset_root / "split_ids.json").read_text(encoding="utf-8"))
    normalization = load_normalization(dataset_root)
    prior = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))["prior"]
    checkpoint: Path | None = None
    model = None
    if variant != "low_prior":
        checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else preferred_inference_checkpoint(config, experiment_root / "runs" / variant)
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Controlled checkpoint not found: {checkpoint}")
        print_torch_runtime()
        device = select_torch_device(
            device_name,
            require_cuda=require_cuda,
            context=f"controlled inference ({variant})",
        )
        model = load_controlled_model(variant, config, checkpoint, device, normalization)
    patch_shape = tuple(int(value) for value in config["patches"]["shape"])
    stride = tuple(int(value) for value in config["patches"]["stride"])
    for realization_id in split_ids["test"]:
        with np.load(_realization_path(dataset_root, realization_id)) as archive:
            low = archive["low"]
            if variant == "low_prior":
                elastic = low
                segmentation = None
            else:
                elastic, segmentation = infer_full_realization(
                    model,
                    avo=archive["avo"],
                    low=low,
                    rgt=archive["rgt"],
                    normalization=normalization,
                    patch_shape=patch_shape,
                    stride=stride,
                    steps=int(config["training"]["sample_steps_test"]),
                    batch_size=int(inference_batch_size or config["training"]["batch_size"]),
                    device=device,
                    valid_mask=archive["valid_mask"] if "valid_mask" in archive else None,
                    guidance_scale=(
                        float(config["training"]["physics_guided_sampling"]["guidance_scale"])
                        if bool(config["training"]["physics_guided_sampling"]["enabled"])
                        else 0.0
                    ),
                )
        payload = {"elastic": elastic.astype(np.float32), "realization_id": realization_id}
        if segmentation is not None:
            payload["segmentation"] = segmentation
        np.savez_compressed(output / f"realization_{realization_id:07d}.npz", **payload)
    manifest = build_run_manifest(
        repository=repository,
        config_path=config_path,
        seed=seed,
        split_ids=split_ids,
        model_variant=variant,
        checkpoint=str(checkpoint.name) if checkpoint is not None else None,
        training_epochs=0 if variant == "low_prior" else int(config["training"]["epochs"]),
        normalization=normalization,
        prior_settings=prior,
        metric_definitions=config["evaluation"],
        status="complete",
    )
    manifest["source_snapshot"] = config.get("source_snapshot")
    write_json(output / "manifest.json", manifest)
    return output
