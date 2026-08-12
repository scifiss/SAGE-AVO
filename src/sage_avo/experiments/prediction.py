"""Generate comparable full-test predictions for every benchmark condition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage_avo.config import seed_everything
from sage_avo.evaluation.inference import infer_full_realization, load_normalization
from sage_avo.models.variants import LEARNED_VARIANTS, build_sage_avo_variant

from .manifest import build_run_manifest, write_json


def _load_model(
    variant: str,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_config = config["model"]
    model = build_sage_avo_variant(
        variant,
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


def predict_controlled_variant(
    *,
    repository: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    dataset_directory: str | Path,
    experiment_directory: str | Path,
    variant: str,
    device_name: str | None = None,
) -> Path:
    """Write one prediction artifact per complete test realization."""
    if variant not in ("low_prior",) + LEARNED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataset_root = Path(dataset_directory)
    experiment_root = Path(experiment_directory)
    output = experiment_root / "predictions" / variant
    output.mkdir(parents=True, exist_ok=True)
    split_ids = json.loads((dataset_root / "split_ids.json").read_text(encoding="utf-8"))
    normalization = load_normalization(dataset_root)
    prior = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))["prior"]
    checkpoint: Path | None = None
    model = None
    if variant != "low_prior":
        checkpoint = experiment_root / "runs" / variant / "best_sampling.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Controlled checkpoint not found: {checkpoint}")
        device = torch.device(
            device_name
            or ("cuda" if torch.cuda.is_available() and config["hardware"]["preferred_device"] == "cuda" else "cpu")
        )
        model = _load_model(variant, config, checkpoint, device)
    patch_shape = tuple(int(value) for value in config["patches"]["shape"])
    stride = tuple(int(value) for value in config["patches"]["stride"])
    for realization_id in split_ids["test"]:
        with np.load(dataset_root / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
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
                    batch_size=int(config["training"]["batch_size"]),
                    device=device,
                )
        payload = {"elastic": elastic.astype(np.float32), "realization_id": realization_id}
        if segmentation is not None:
            payload["segmentation"] = segmentation
        np.savez_compressed(output / f"realization_{realization_id:04d}.npz", **payload)
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
    write_json(output / "manifest.json", manifest)
    return output
