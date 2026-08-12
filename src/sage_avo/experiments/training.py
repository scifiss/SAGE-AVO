"""Controlled learned-variant training with one checkpoint criterion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sage_avo.config import seed_everything
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.models.variants import build_sage_avo_variant, variant_definition
from sage_avo.training.checkpoints import save_checkpoint
from sage_avo.training.engine import PhysicsNormalization, train_epoch
from sage_avo.training.flow import straight_path
from sage_avo.training.losses import LossWeights, masked_mse

from .manifest import build_run_manifest, write_json


def _normalization_tensors(normalization: dict[str, list[float]]) -> PhysicsNormalization:
    def tensor(name: str) -> Tensor:
        return torch.tensor(normalization[name], dtype=torch.float32).view(1, 3, 1, 1)

    return PhysicsNormalization(
        x_mean=tensor("x_mean"),
        x_std=tensor("x_std"),
        y_mean=tensor("y_mean"),
        y_std=tensor("y_std"),
    )


def _class_weights(dataset: IndexedRealizationPatches, classes: int = 3) -> Tensor:
    counts = np.zeros(classes, dtype=np.float64)
    for index in range(len(dataset)):
        labels = dataset[index]["segmentation"].numpy()
        counts += np.bincount(labels.reshape(-1), minlength=classes)
    inverse = counts.sum() / np.maximum(counts, 1.0)
    normalized = inverse / inverse.mean()
    return torch.tensor(normalized, dtype=torch.float32)


@torch.no_grad()
def _validation_flow_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}
        for fraction in (0.0, 0.5, 1.0):
            time = torch.full((batch["target"].shape[0],), fraction, device=device)
            state, velocity = straight_path(batch["low"], batch["target"], time)
            output = model(state, time, batch["avo"], batch["low"], batch["rgt"])
            values.append(float(masked_mse(output.velocity, velocity, batch["mask"])))
    return float(np.mean(values))


@torch.no_grad()
def _validation_sample_score(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    steps: int,
    max_batches: int,
) -> tuple[float, list[float], float]:
    model.eval()
    squared_error = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)
    intersections = np.zeros(3, dtype=np.float64)
    unions = np.zeros(3, dtype=np.float64)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        values = {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}
        prediction = model.sample(values["avo"], values["low"], values["rgt"], steps=steps)
        mask = values["mask"].expand_as(prediction)
        error = ((prediction - values["target"]) ** 2 * mask).sum(dim=(0, 2, 3))
        squared_error += error.cpu().numpy()
        count += mask.sum(dim=(0, 2, 3)).cpu().numpy()
        final_time = torch.ones(prediction.shape[0], device=device)
        labels = model(
            prediction, final_time, values["avo"], values["low"], values["rgt"]
        ).segmentation_logits.argmax(dim=1)
        for label in range(3):
            predicted = labels == label
            target = values["segmentation"] == label
            intersections[label] += (predicted & target).sum().item()
            unions[label] += (predicted | target).sum().item()
    normalized_rmse = np.sqrt(squared_error / np.maximum(count, 1.0))
    miou = float(np.mean(intersections / np.maximum(unions, 1.0)))
    criterion = float(np.mean(normalized_rmse) - 0.1 * miou)
    return criterion, normalized_rmse.tolist(), miou


def train_controlled_variant(
    *,
    repository: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    dataset_directory: str | Path,
    experiment_directory: str | Path,
    variant: str,
    device_name: str | None = None,
) -> Path:
    """Train one learned condition against the shared immutable data artifacts."""
    definition = variant_definition(
        variant, physics_weight=float(config["training"]["loss_weights"]["physics"])
    )
    if definition.graph_mode is None:
        raise ValueError("The low-prior condition requires no training")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device(
        device_name
        or ("cuda" if torch.cuda.is_available() and config["hardware"]["preferred_device"] == "cuda" else "cpu")
    )
    dataset_root = Path(dataset_directory)
    experiment_root = Path(experiment_directory)
    run_directory = experiment_root / "runs" / variant
    run_directory.mkdir(parents=True, exist_ok=True)
    normalization = json.loads((dataset_root / "normalization.json").read_text(encoding="utf-8"))
    split_ids = json.loads((dataset_root / "split_ids.json").read_text(encoding="utf-8"))
    prior = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))["prior"]
    train_dataset = IndexedRealizationPatches(dataset_root, "train")
    validation_dataset = IndexedRealizationPatches(dataset_root, "validation")
    training_config = config["training"]
    generator = torch.Generator().manual_seed(seed)
    time_generator = torch.Generator().manual_seed(seed + 17)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model_config = config["model"]
    model = build_sage_avo_variant(
        variant,
        hidden_channels=int(model_config["hidden_channels"]),
        graph_layers=int(model_config["graph_layers"]),
        graph_heads=int(model_config["graph_heads"]),
        max_rgt_shift=int(model_config["max_rgt_shift_samples"]),
        classes=int(model_config["classes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training_config["epochs"]), eta_min=1e-6
    )
    configured_weights = training_config["loss_weights"]
    weights = LossWeights(
        flow=float(configured_weights["flow"]),
        full_property=float(configured_weights["full_property"]),
        segmentation=float(configured_weights["segmentation"]),
        physics=float(definition.physics_weight),
        structure=float(configured_weights["structure"]),
    )
    class_weights = _class_weights(train_dataset).to(device)
    physics_normalization = _normalization_tensors(normalization)
    metric_definitions = {
        "checkpoint_criterion": training_config["checkpoint_criterion"],
        "validation_time_grid": [0.0, 0.5, 1.0],
        "validation_sampling_batches": int(training_config["validation_sample_batches"]),
    }
    manifest = build_run_manifest(
        repository=repository,
        config_path=config_path,
        seed=seed,
        split_ids=split_ids,
        model_variant=variant,
        checkpoint=None,
        training_epochs=int(training_config["epochs"]),
        normalization=normalization,
        prior_settings=prior,
        metric_definitions=metric_definitions,
        status="running",
    )
    write_json(run_directory / "manifest.json", manifest)
    log_path = run_directory / "training_log.csv"
    best_flow = float("inf")
    best_sample = float("inf")
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_total",
                "train_flow",
                "train_full_property",
                "train_segmentation",
                "train_physics",
                "train_structure",
                "validation_flow",
                "sample_criterion",
                "sample_rmse_vp_normalized",
                "sample_rmse_vs_normalized",
                "sample_rmse_density_normalized",
                "sample_miou",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for epoch in range(1, int(training_config["epochs"]) + 1):
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                physics_normalization,
                weights,
                class_weights,
                float(training_config["gradient_clip"]),
                time_generator,
            )
            validation_flow = _validation_flow_loss(model, validation_loader, device)
            if validation_flow < best_flow:
                best_flow = validation_flow
                save_checkpoint(
                    run_directory / "best_flow.pt",
                    model,
                    optimizer,
                    epoch,
                    {"validation_flow": validation_flow},
                    config,
                )
            sample_criterion = np.nan
            sample_rmse = [np.nan] * 3
            sample_miou = np.nan
            if epoch % int(training_config["validation_sample_every"]) == 0:
                sample_criterion, sample_rmse, sample_miou = _validation_sample_score(
                    model,
                    validation_loader,
                    device,
                    steps=int(training_config["sample_steps_validation"]),
                    max_batches=int(training_config["validation_sample_batches"]),
                )
                if sample_criterion < best_sample:
                    best_sample = sample_criterion
                    save_checkpoint(
                        run_directory / "best_sampling.pt",
                        model,
                        optimizer,
                        epoch,
                        {
                            "sample_criterion": sample_criterion,
                            "normalized_rmse": sample_rmse,
                            "miou": sample_miou,
                        },
                        config,
                    )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_total": train_metrics.total,
                    "train_flow": train_metrics.flow,
                    "train_full_property": train_metrics.full_property,
                    "train_segmentation": train_metrics.segmentation,
                    "train_physics": train_metrics.physics,
                    "train_structure": train_metrics.structure,
                    "validation_flow": validation_flow,
                    "sample_criterion": sample_criterion,
                    "sample_rmse_vp_normalized": sample_rmse[0],
                    "sample_rmse_vs_normalized": sample_rmse[1],
                    "sample_rmse_density_normalized": sample_rmse[2],
                    "sample_miou": sample_miou,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            stream.flush()
            scheduler.step()
    save_checkpoint(
        run_directory / "last.pt",
        model,
        optimizer,
        int(training_config["epochs"]),
        {"validation_flow": validation_flow},
        config,
    )
    manifest["checkpoint"] = "best_sampling.pt"
    manifest["status"] = "complete"
    manifest["best_validation_flow"] = best_flow
    manifest["best_sample_criterion"] = best_sample
    write_json(run_directory / "manifest.json", manifest)
    return run_directory
