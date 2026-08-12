"""Per-realization, pooled, and paired controlled-ablation statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import elastic_metrics_with_ssim, segmentation_metrics


PROPERTIES = ("vp", "vs", "density")
ALL_VARIANTS = ("low_prior", "full", "no_gnn", "no_rgt", "no_physics")
LOWER_IS_BETTER = {"rmse", "mae"}


def _prediction_path(experiment_directory: Path, variant: str, realization_id: int) -> Path:
    directory = experiment_directory / "predictions" / variant
    canonical = directory / f"realization_{realization_id:07d}.npz"
    legacy = directory / f"realization_{realization_id:04d}.npz"
    return canonical if canonical.exists() or not legacy.exists() else legacy


def _realization_path(dataset_directory: Path, realization_id: int) -> Path:
    directory = dataset_directory / "realizations"
    canonical = directory / f"realization_{realization_id:07d}.npz"
    legacy = directory / f"realization_{realization_id:04d}.npz"
    return canonical if canonical.exists() or not legacy.exists() else legacy


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path) as archive:
        elastic = archive["elastic"]
        segmentation = archive["segmentation"] if "segmentation" in archive.files else None
    return elastic, segmentation


def _summary_rows(
    per_realization: pd.DataFrame,
    pooled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pooled_lookup = {
        (row["variant"], row["domain"], row["metric"]): row["pooled"] for row in pooled_rows
    }
    rows: list[dict[str, Any]] = []
    for keys, group in per_realization.groupby(["variant", "domain", "metric"], dropna=False):
        variant, domain, metric = keys
        values = group["value"].to_numpy(float)
        rows.append(
            {
                "variant": variant,
                "domain": domain,
                "metric": metric,
                "pooled": pooled_lookup.get((variant, domain, metric), np.nan),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_realizations": int(len(values)),
            }
        )
    return rows


def _bootstrap_interval(
    values: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def evaluate_controlled_ablation(
    *,
    experiment_directory: str | Path,
    dataset_directory: str | Path,
    bootstrap_repetitions: int = 2000,
    bootstrap_confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Evaluate all five conditions on identical complete test realizations."""
    experiment_root = Path(experiment_directory)
    dataset_root = Path(dataset_directory)
    split_ids = json.loads((dataset_root / "split_ids.json").read_text(encoding="utf-8"))
    test_ids = [int(value) for value in split_ids["test"]]
    records: list[dict[str, Any]] = []
    pooled_truth = {name: [] for name in PROPERTIES}
    pooled_prediction = {
        variant: {name: [] for name in PROPERTIES} for variant in ALL_VARIANTS
    }
    pooled_segmentation_truth: list[np.ndarray] = []
    pooled_segmentation_prediction = {variant: [] for variant in ALL_VARIANTS if variant != "low_prior"}
    for realization_id in test_ids:
        with np.load(_realization_path(dataset_root, realization_id)) as archive:
            truth = archive["elastic"]
            segmentation_truth = archive["segmentation"]
            mask_name = "valid_mask" if "valid_mask" in archive.files else "mask"
            mask = archive[mask_name].astype(bool)
        for channel, name in enumerate(PROPERTIES):
            pooled_truth[name].append(truth[channel][mask])
        pooled_segmentation_truth.append(segmentation_truth[mask])
        for variant in ALL_VARIANTS:
            prediction_path = _prediction_path(experiment_root, variant, realization_id)
            if not prediction_path.exists():
                raise FileNotFoundError(f"Missing controlled prediction: {prediction_path}")
            prediction, segmentation_prediction = _load_prediction(prediction_path)
            for channel, name in enumerate(PROPERTIES):
                metrics = elastic_metrics_with_ssim(prediction[channel], truth[channel], mask)
                for metric in ("rmse", "mae", "r2", "ssim"):
                    records.append(
                        {
                            "variant": variant,
                            "realization_id": realization_id,
                            "domain": name,
                            "metric": metric,
                            "value": metrics[metric],
                        }
                    )
                pooled_prediction[variant][name].append(prediction[channel][mask])
            if segmentation_prediction is not None:
                segmentation = segmentation_metrics(segmentation_prediction[mask], segmentation_truth[mask])
                for metric, value in segmentation.items():
                    records.append(
                        {
                            "variant": variant,
                            "realization_id": realization_id,
                            "domain": "segmentation",
                            "metric": metric,
                            "value": value,
                        }
                    )
                pooled_segmentation_prediction[variant].append(segmentation_prediction[mask])

    per_realization = pd.DataFrame(records)
    pooled_rows: list[dict[str, Any]] = []
    for variant in ALL_VARIANTS:
        for name in PROPERTIES:
            prediction = np.concatenate(pooled_prediction[variant][name])
            truth = np.concatenate(pooled_truth[name])
            residual = prediction - truth
            denominator = np.sum((truth - truth.mean()) ** 2)
            pooled = {
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1.0 - np.sum(residual**2) / denominator),
            }
            for metric, value in pooled.items():
                pooled_rows.append(
                    {
                        "variant": variant,
                        "domain": name,
                        "metric": metric,
                        "pooled": value,
                        "mean": np.nan,
                        "std": np.nan,
                        "n_realizations": len(test_ids),
                    }
                )
        if variant != "low_prior":
            predicted_segmentation = np.concatenate(pooled_segmentation_prediction[variant])
            true_segmentation = np.concatenate(pooled_segmentation_truth)
            for metric, value in segmentation_metrics(predicted_segmentation, true_segmentation).items():
                pooled_rows.append(
                    {
                        "variant": variant,
                        "domain": "segmentation",
                        "metric": metric,
                        "pooled": value,
                        "mean": np.nan,
                        "std": np.nan,
                        "n_realizations": len(test_ids),
                    }
                )
    summary = pd.DataFrame(_summary_rows(per_realization, pooled_rows))

    paired_rows: list[dict[str, Any]] = []
    full = per_realization[per_realization["variant"] == "full"]
    for comparator in ("low_prior", "no_gnn", "no_rgt", "no_physics"):
        other = per_realization[per_realization["variant"] == comparator]
        merged = full.merge(
            other,
            on=["realization_id", "domain", "metric"],
            suffixes=("_full", "_comparator"),
        )
        for (domain, metric), group in merged.groupby(["domain", "metric"]):
            if metric in LOWER_IS_BETTER:
                differences = group["value_comparator"].to_numpy() - group["value_full"].to_numpy()
            else:
                differences = group["value_full"].to_numpy() - group["value_comparator"].to_numpy()
            lower, upper = _bootstrap_interval(
                differences,
                bootstrap_repetitions,
                bootstrap_confidence,
                seed + len(paired_rows),
            )
            paired_rows.append(
                {
                    "full_vs": comparator,
                    "domain": domain,
                    "metric": metric,
                    "mean_improvement": float(np.mean(differences)),
                    "std_improvement": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
                    "n_realizations": int(len(differences)),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "confidence": bootstrap_confidence,
                    "positive_means_full_is_better": True,
                }
            )
    paired = pd.DataFrame(paired_rows)
    full_vp = per_realization[
        (per_realization["variant"] == "full")
        & (per_realization["domain"] == "vp")
        & (per_realization["metric"] == "rmse")
    ].sort_values("value")
    median_value = full_vp["value"].median()
    representative_id = int(
        full_vp.iloc[np.argmin(np.abs(full_vp["value"].to_numpy() - median_value))]["realization_id"]
    )
    return summary, per_realization, paired, representative_id
