#!/usr/bin/env python3
"""Read-only, common-protocol audit of the immutable v002 baseline.

All outputs are written to a versioned audit directory. The v002 checkpoints,
logs, Stage-02 corpus, and Stage-03 dataset are opened read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sage_avo.config import load_config, seed_everything
from sage_avo.data import IndexedRealizationPatches
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.evaluation.metrics import elastic_metrics, elastic_metrics_with_ssim, ssim_2d
from sage_avo.experiments.prediction import load_controlled_model
from sage_avo.experiments.training import (
    _class_weights,
    _normalization_tensors,
    curriculum_from_config,
    loss_weights_from_config,
    physics_settings_from_config,
)
from sage_avo.training.engine import ContrastiveSettings, validate_epoch
from sage_avo.training.selection import weighted_objective_contributions


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
ARCHIVE = PRIVATE_ROOT / "archives" / "v002_completed_baseline_20260816"
DATASET = (
    PRIVATE_ROOT
    / "stage_artifacts"
    / "stage03"
    / "ds_v002_production100_multiscale"
    / "dataset"
)
DEFAULT_OUTPUT = PRIVATE_ROOT / "revision3" / "v002_common_protocol_audit"
CHECKPOINTS = ("best_sampling.pt", "best_flow.pt", "last.pt")
PROPERTIES = ("vp", "vs", "density")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, tensor in sorted(checkpoint["model_state"].items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_inventory() -> pd.DataFrame:
    rows = []
    for name in CHECKPOINTS:
        path = ARCHIVE / "checkpoints" / name
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "checkpoint": name,
                "epoch": int(checkpoint["epoch"]),
                "file_sha256": _sha256(path),
                "model_state_sha256": _model_state_sha256(path),
                "embedded_metrics": json.dumps(
                    checkpoint.get("metrics", {}), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _fixed_patch_indices(dataset: IndexedRealizationPatches) -> list[int]:
    """Choose the first native-grid row for every validation realization.

    This rule uses neither labels nor model error, so it is deterministic and
    cannot enrich the audit suite for foreground or favorable predictions.
    """
    table = dataset.index
    native = table[
        (table.raw_height == table.output_height)
        & (table.raw_width == table.output_width)
    ]
    selected = []
    for realization_id in sorted(native.realization_id.unique()):
        selected.append(int(native[native.realization_id == realization_id].index[0]))
    return selected


def _segmentation_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    classes: int = 3,
) -> dict[str, float]:
    valid = mask.astype(bool)
    output: dict[str, float] = {}
    ious = []
    dices = []
    for label in range(classes):
        predicted = (prediction == label) & valid
        observed = (target == label) & valid
        intersection = int(np.count_nonzero(predicted & observed))
        union = int(np.count_nonzero(predicted | observed))
        denominator = int(np.count_nonzero(predicted) + np.count_nonzero(observed))
        iou = intersection / union if union else float("nan")
        dice = 2 * intersection / denominator if denominator else float("nan")
        output[f"class_{label}_iou"] = float(iou)
        output[f"class_{label}_dice"] = float(dice)
        ious.append(iou)
        dices.append(dice)
    output["miou"] = float(np.nanmean(ious))
    output["macro_dice"] = float(np.nanmean(dices))
    return output


def _stack_metrics_with_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Aggregate point metrics globally and 2-D SSIM equally by patch."""
    output = elastic_metrics(prediction, target, mask)
    output["ssim"] = float(
        np.mean(
            [
                ssim_2d(prediction[index], target[index], mask[index])
                for index in range(prediction.shape[0])
            ]
        )
    )
    return output


@torch.no_grad()
def _evaluate_patch_suite(
    model: torch.nn.Module,
    dataset: IndexedRealizationPatches,
    indices: list[int],
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    priors = []
    masks = []
    predicted_labels = []
    target_labels = []
    coordinates = []
    model.eval()
    for batch in loader:
        values = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        prediction = model.sample(
            values["avo"],
            values["low"],
            values["rgt"],
            steps=steps,
            guidance_scale=0.0,
            avo_mask=values["mask"],
        )
        logits = model(
            prediction,
            torch.ones(prediction.shape[0], device=device),
            values["avo"],
            values["low"],
            values["rgt"],
        ).segmentation_logits
        predictions.append(prediction.cpu().numpy())
        targets.append(values["target"].cpu().numpy())
        priors.append(values["low"].cpu().numpy())
        masks.append(values["mask"].cpu().numpy())
        predicted_labels.append(logits.argmax(dim=1).cpu().numpy())
        target_labels.append(values["segmentation"].cpu().numpy())
        for realization_id, top, left, raw_shape in zip(
            batch["realization_id"], batch["top"], batch["left"], batch["raw_shape"]
        ):
            coordinates.append(
                {
                    "realization_id": int(realization_id),
                    "top": int(top),
                    "left": int(left),
                    "raw_height": int(raw_shape[0]),
                    "raw_width": int(raw_shape[1]),
                }
            )

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    prior = np.concatenate(priors)
    mask = np.concatenate(masks)[:, 0]
    labels = np.concatenate(predicted_labels)
    observed_labels = np.concatenate(target_labels)
    y_mean = np.asarray(dataset.normalization["y_mean"])[None, :, None, None]
    y_std = np.asarray(dataset.normalization["y_std"])[None, :, None, None]
    prediction_physical = prediction * y_std + y_mean
    target_physical = target * y_std + y_mean
    prior_physical = prior * y_std + y_mean

    metrics: dict[str, float] = {}
    normalized_rmse = []
    for channel, name in enumerate(PROPERTIES):
        normalized = _stack_metrics_with_ssim(
            prediction[:, channel], target[:, channel], mask
        )
        physical = _stack_metrics_with_ssim(
            prediction_physical[:, channel], target_physical[:, channel], mask
        )
        baseline = _stack_metrics_with_ssim(
            prior_physical[:, channel], target_physical[:, channel], mask
        )
        normalized_rmse.append(normalized["rmse"])
        for key, value in normalized.items():
            metrics[f"{name}_{key}_normalized"] = value
        for key, value in physical.items():
            metrics[f"{name}_{key}_physical"] = value
        for key, value in baseline.items():
            metrics[f"{name}_prior_{key}_physical"] = value
        metrics[f"{name}_rmse_improvement_vs_prior_percent"] = float(
            100.0 * (baseline["rmse"] - physical["rmse"]) / baseline["rmse"]
        )
    segmentation = _segmentation_counts(labels, observed_labels, mask)
    metrics.update({f"segmentation_{key}": value for key, value in segmentation.items()})
    metrics["sampling_criterion"] = float(
        np.mean(normalized_rmse) - 0.1 * segmentation["miou"]
    )
    metrics["patch_count"] = float(len(indices))
    return metrics, pd.DataFrame(coordinates)


def _evaluate_whole_realization(
    model: torch.nn.Module,
    dataset_root: Path,
    realization_id: int,
    normalization: dict[str, list[float]],
    config: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    path = dataset_root / "realizations" / f"realization_{realization_id:07d}.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    prediction, labels = infer_full_realization(
        model,
        avo=arrays["avo"],
        low=arrays["low"],
        rgt=arrays["rgt"],
        normalization=normalization,
        patch_shape=tuple(int(value) for value in config["patches"]["shape"]),
        stride=tuple(int(value) for value in config["patches"]["stride"]),
        steps=steps,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        valid_mask=arrays["valid_mask"],
        guidance_scale=0.0,
    )
    metrics: dict[str, float] = {"realization_id": float(realization_id)}
    for channel, name in enumerate(PROPERTIES):
        values = elastic_metrics_with_ssim(
            prediction[channel], arrays["elastic"][channel], arrays["valid_mask"]
        )
        prior = elastic_metrics_with_ssim(
            arrays["low"][channel], arrays["elastic"][channel], arrays["valid_mask"]
        )
        for key, value in values.items():
            metrics[f"{name}_{key}"] = value
        for key, value in prior.items():
            metrics[f"{name}_prior_{key}"] = value
        metrics[f"{name}_rmse_improvement_vs_prior_percent"] = float(
            100.0 * (prior["rmse"] - values["rmse"]) / prior["rmse"]
        )
    segmentation = _segmentation_counts(
        labels, arrays["segmentation"], arrays["valid_mask"]
    )
    metrics.update({f"segmentation_{key}": value for key, value in segmentation.items()})
    return metrics, {
        "elastic": prediction,
        "segmentation": labels,
        "realization_id": np.asarray(realization_id),
    }


def _historical_metric_reconciliation(
    dataset: IndexedRealizationPatches,
) -> dict[str, Any]:
    training_indices = list(range(min(16 * 8, len(dataset))))
    historical = dataset.index.iloc[training_indices]
    smallest_id = int(dataset.index.realization_id.min())
    native = dataset.index[
        (dataset.index.realization_id == smallest_id)
        & (dataset.index.raw_height == dataset.index.output_height)
        & (dataset.index.raw_width == dataset.index.output_width)
    ]
    foreground = []
    for index in native.index:
        values = dataset.sampling_fields(int(index))
        foreground.append(int(torch.count_nonzero(values["segmentation"])))
    posthoc_index = int(native.index[int(np.argmax(foreground))])
    row = dataset.index.iloc[posthoc_index]
    return {
        "training_logged_sampling_protocol": {
            "loader_batches": 16,
            "batch_size": 8,
            "patch_count": len(training_indices),
            "unique_realization_ids": sorted(
                int(value) for value in historical.realization_id.unique()
            ),
            "coordinate_rows": historical[
                ["realization_id", "top", "left", "raw_height", "raw_width"]
            ].to_dict(orient="records"),
            "aggregation": "global masked squared-error sum/count per property",
        },
        "completed120_posthoc_protocol": {
            "patch_index": posthoc_index,
            "realization_id": int(row.realization_id),
            "top": int(row.top),
            "left": int(row.left),
            "raw_height": int(row.raw_height),
            "raw_width": int(row.raw_width),
            "selection": (
                "smallest validation realization ID, native patch with maximum "
                "non-background support"
            ),
            "aggregation": "one enriched patch in physical units",
        },
        "conclusion": (
            "The two reported Vp RMSE rankings used different samples and aggregation. "
            "A per-property linear denormalization multiplies RMSE by a positive constant, "
            "so it cannot reverse checkpoint ordering when samples and masks are identical."
        ),
    }


def audit(output: Path, *, device_name: str, skip_whole: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    print(f"audit output: {output}", flush=True)
    config_path = ARCHIVE / "configuration" / "sage_avo_s01.yaml"
    config = load_config(config_path)
    seed_everything(int(config["experiment"]["seed"]), deterministic_torch=True)
    device = torch.device(device_name)
    dataset = IndexedRealizationPatches(DATASET, "validation")
    training_dataset = IndexedRealizationPatches(DATASET, "train")
    indices = _fixed_patch_indices(dataset)
    coordinates_path = output / "fixed_patch_suite_coordinates.csv"
    inventory = _checkpoint_inventory()
    inventory.to_csv(output / "checkpoint_inventory.csv", index=False)
    print("checkpoint inventory complete", flush=True)
    state_groups = inventory.groupby("model_state_sha256", sort=False)
    representatives = [group.iloc[0].checkpoint for _, group in state_groups]

    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    base_weights = loss_weights_from_config(
        config, float(config["training"]["loss_weights"]["physics"])
    )
    final_weights = curriculum_from_config(config).weights_for_epoch(
        base_weights, int(config["training"]["epochs"]) - 1, int(config["training"]["epochs"])
    )
    normalization = _normalization_tensors(dataset.normalization)
    class_weights = _class_weights(
        training_dataset,
        classes=int(config["model"]["classes"]),
        foreground_boost=float(config["training"]["class_weight_foreground_boost"]),
    ).to(device)
    print(f"training-derived class weights: {class_weights.cpu().tolist()}", flush=True)
    physics = physics_settings_from_config(config)
    raw_rows = []
    patch_rows = []
    models: dict[str, torch.nn.Module] = {}
    for checkpoint_name in representatives:
        print(f"evaluating {checkpoint_name}", flush=True)
        checkpoint_path = ARCHIVE / "checkpoints" / checkpoint_name
        model = load_controlled_model(
            "full", config, checkpoint_path, device, dataset.normalization
        )
        models[checkpoint_name] = model
        raw = validate_epoch(
            model,
            loader,
            normalization,
            final_weights,
            time_grid=tuple(float(v) for v in config["training"]["validation_time_grid"]),
            class_weights=class_weights,
            physics=physics,
            contrastive=ContrastiveSettings(),
            max_batches=16,
        )
        raw_values = asdict(raw)
        print(f"  fixed objective: {raw.total:.8f}", flush=True)
        raw_rows.append(
            {
                "checkpoint": checkpoint_name,
                **raw_values,
                **{
                    f"fixed_contribution_{key}": value
                    for key, value in weighted_objective_contributions(
                        raw, final_weights
                    ).items()
                },
            }
        )
        patch_metrics, coordinates = _evaluate_patch_suite(
            model,
            dataset,
            indices,
            device,
            steps=20,
            batch_size=int(config["training"]["batch_size"]),
        )
        coordinates.to_csv(coordinates_path, index=False)
        patch_rows.append({"checkpoint": checkpoint_name, **patch_metrics})
        print(
            f"  patch sampling criterion: {patch_metrics['sampling_criterion']:.8f}",
            flush=True,
        )

    aliases = inventory.set_index("checkpoint")["model_state_sha256"].to_dict()
    for checkpoint_name in CHECKPOINTS:
        if checkpoint_name in representatives:
            continue
        representative = next(
            name for name in representatives if aliases[name] == aliases[checkpoint_name]
        )
        raw_rows.append(
            {
                **next(row for row in raw_rows if row["checkpoint"] == representative),
                "checkpoint": checkpoint_name,
                "evaluation_reused_from_identical_model_state": representative,
            }
        )
        patch_rows.append(
            {
                **next(row for row in patch_rows if row["checkpoint"] == representative),
                "checkpoint": checkpoint_name,
                "evaluation_reused_from_identical_model_state": representative,
            }
        )
    pd.DataFrame(raw_rows).to_csv(output / "fixed_objective_components.csv", index=False)
    pd.DataFrame(patch_rows).to_csv(output / "fixed_patch_metrics.csv", index=False)

    whole_rows = []
    split_ids = json.loads((DATASET / "split_ids.json").read_text(encoding="utf-8"))
    whole_cases = {
        "validation": min(int(value) for value in split_ids["validation"]),
        "test": min(int(value) for value in split_ids["test"]),
    }
    if not skip_whole:
        predictions = output / "whole_realization_predictions"
        predictions.mkdir(exist_ok=True)
        for checkpoint_name, model in models.items():
            for split, realization_id in whole_cases.items():
                print(
                    f"whole inference {checkpoint_name} {split} {realization_id}",
                    flush=True,
                )
                metrics, payload = _evaluate_whole_realization(
                    model,
                    DATASET,
                    realization_id,
                    dataset.normalization,
                    config,
                    device,
                    steps=20,
                )
                whole_rows.append(
                    {"checkpoint": checkpoint_name, "split": split, **metrics}
                )
                np.savez_compressed(
                    predictions
                    / f"{checkpoint_name.removesuffix('.pt')}_{split}_{realization_id}.npz",
                    **payload,
                )
        for checkpoint_name in CHECKPOINTS:
            if checkpoint_name in representatives:
                continue
            representative = next(
                name for name in representatives if aliases[name] == aliases[checkpoint_name]
            )
            for row in [item for item in whole_rows if item["checkpoint"] == representative]:
                whole_rows.append(
                    {
                        **row,
                        "checkpoint": checkpoint_name,
                        "evaluation_reused_from_identical_model_state": representative,
                    }
                )
        pd.DataFrame(whole_rows).to_csv(
            output / "whole_realization_metrics.csv", index=False
        )

    training_log = pd.read_csv(ARCHIVE / "metrics" / "training_log.csv")
    observed_best = {
        "dynamic_weighted_validation_total": int(
            training_log.loc[training_log.validation_total.idxmin(), "epoch"]
        ),
        "sampling_criterion": int(
            training_log.loc[training_log.sample_criterion.idxmin(), "epoch"]
        ),
        "segmentation_miou": int(
            training_log.loc[training_log.sample_miou.idxmax(), "epoch"]
        ),
        "raw_validation_flow": int(
            training_log.loc[training_log.validation_flow.idxmin(), "epoch"]
        ),
        "raw_validation_ssim": int(
            training_log.loc[training_log.validation_ssim.idxmin(), "epoch"]
        ),
    }
    report = {
        "status": "complete" if not skip_whole else "patch_only",
        "immutable_archive": str(ARCHIVE),
        "archive_manifest_sha256": _sha256(ARCHIVE / "artifact_manifest.json"),
        "evaluation_output": str(output),
        "device": str(device),
        "sample_steps": 20,
        "fixed_patch_rule": "first native-grid patch row per sorted validation realization ID",
        "fixed_patch_count": len(indices),
        "raw_component_protocol": {
            "validation_batches": 16,
            "batch_size": int(config["training"]["batch_size"]),
            "time_grid": config["training"]["validation_time_grid"],
            "fixed_final_weights": asdict(final_weights),
            "note": (
                "Raw physics component retains the historical v002 patch-local operator "
                "for audit only; v003 supersedes it with halo and absolute mute context."
            ),
        },
        "whole_realization_cases": whole_cases,
        "checkpoint_observations": observed_best,
        "unpersisted_checkpoints": [29, 94],
        "name_audit": {
            "best_flow.pt": (
                "Selected by the epoch-dependent weighted validation total, not by raw "
                "validation flow. The accurate v003 criterion name is best_fixed_objective.pt."
            ),
            "best_sampling.pt": "Selected by mean normalized elastic RMSE minus 0.1*mIoU.",
            "last.pt": "Final epoch training state.",
        },
        "metric_reconciliation": _historical_metric_reconciliation(dataset),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-whole", action="store_true")
    arguments = parser.parse_args()
    audit(arguments.output, device_name=arguments.device, skip_whole=arguments.skip_whole)


if __name__ == "__main__":
    main()
