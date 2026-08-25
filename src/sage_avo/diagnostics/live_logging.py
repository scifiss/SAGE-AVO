"""Append-only epoch logs for observability without touching optimization state."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import torch

from sage_avo.training.engine import StepMetrics
from sage_avo.training.losses import LossWeights

from .accounting import EpochLossObserver, raw_and_weighted_rows


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class BatchProgressLogger:
    """Report detached batch progress without touching training or RNG state."""

    epoch: int
    total_epochs: int
    total_batches: int
    physics_weight: float
    interval_batches: int = 50
    _started: float = field(default_factory=time.perf_counter, init=False)
    _last_report_time: float = field(default=0.0, init=False)
    _last_report_batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.epoch < 1 or self.total_epochs < self.epoch:
            raise ValueError("epoch must lie in [1, total_epochs]")
        if self.total_batches < 1:
            raise ValueError("total_batches must be positive")
        if self.interval_batches < 1 or self.interval_batches > 50:
            raise ValueError("interval_batches must lie in [1, 50]")
        self._last_report_time = self._started

    def __call__(self, batch: dict[str, torch.Tensor], metrics: StepMetrics) -> None:
        completed = self._last_report_batch + 1
        # The observer is called exactly once after each completed optimizer step.
        self._last_report_batch = completed
        nonfinite = not math.isfinite(metrics.total) or not math.isfinite(metrics.physics)
        scheduled = completed == 1 or completed % self.interval_batches == 0
        final = completed == self.total_batches
        if not (scheduled or final or nonfinite):
            return

        now = time.perf_counter()
        elapsed = now - self._started
        interval_count = completed - getattr(self, "_reported_batch", 0)
        interval_elapsed = now - self._last_report_time
        rolling_seconds = interval_elapsed / max(interval_count, 1)
        mean_seconds = elapsed / completed
        eta = mean_seconds * max(self.total_batches - completed, 0)
        eligible = batch.get("physics_eligible")
        batch_size = int(batch["target"].shape[0])
        eligible_count = int(eligible.sum().item()) if eligible is not None else batch_size
        physics_active = float(self.physics_weight) > 0.0 and eligible_count > 0

        if torch.cuda.is_available():
            allocated_mib = torch.cuda.memory_allocated() / 2**20
            reserved_mib = torch.cuda.memory_reserved() / 2**20
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            memory = (
                f"gpu_mib={allocated_mib:.0f}/{reserved_mib:.0f} "
                f"peak_mib={peak_mib:.0f}"
            )
        else:
            memory = "gpu_mib=n/a peak_mib=n/a"
        status = " NONFINITE_LOSS" if nonfinite else ""
        print(
            f"[train-progress] epoch={self.epoch}/{self.total_epochs} "
            f"batch={completed}/{self.total_batches} elapsed={_duration(elapsed)} "
            f"rolling_s_per_batch={rolling_seconds:.3f} eta={_duration(eta)} "
            f"weighted_total={metrics.total:.8g} raw_physics={metrics.physics:.8g} "
            f"physics_active={str(physics_active).lower()} "
            f"physics_eligible={eligible_count}/{batch_size} {memory}{status}",
            flush=True,
        )
        self._last_report_time = now
        self._reported_batch = completed


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _first_epoch_raw(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    result: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (str(row["split"]), str(row["component"]))
            result.setdefault(key, float(row["raw_loss"]))
    return result


def log_epoch_observability(
    *,
    directory: str | Path,
    epoch: int,
    train_metrics: StepMetrics,
    validation_metrics: StepMetrics,
    weights: LossWeights,
    train_observer: EpochLossObserver,
    validation_observer: EpochLossObserver,
    learning_rate: float,
) -> None:
    """Write raw, weighted, curriculum, and physics-eligibility epoch records."""
    output = Path(directory)
    raw_path = output / "raw_loss_components.csv"
    raw_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    for split, metrics in (
        ("train", train_metrics),
        ("validation", validation_metrics),
    ):
        split_raw, split_weighted = raw_and_weighted_rows(
            epoch=epoch,
            split=split,
            metrics=metrics,
            weights=weights,
        )
        raw_rows.extend(split_raw)
        weighted_rows.extend(split_weighted)
    baselines = _first_epoch_raw(raw_path)
    for row in raw_rows:
        key = (str(row["split"]), str(row["component"]))
        baseline = baselines.get(key, float(row["raw_loss"]))
        row["epoch_1_raw_loss"] = baseline
        row["normalized_to_epoch_1"] = (
            float(row["raw_loss"]) / baseline if baseline != 0.0 else float("nan")
        )
    _append_rows(raw_path, raw_rows)
    _append_rows(output / "weighted_loss_components.csv", weighted_rows)

    physics_rows = []
    for split, observer in (
        ("train", train_observer),
        ("validation", validation_observer),
    ):
        physics_rows.append(
            {
                "epoch": int(epoch),
                "split": split,
                **observer.summary(),
            }
        )
    _append_rows(output / "physics_eligibility_statistics.csv", physics_rows)
    _append_rows(
        output / "training_statistics.csv",
        [
            {
                "epoch": int(epoch),
                "learning_rate": float(learning_rate),
                "train_total_weighted_objective": float(train_metrics.total),
                "validation_total_weighted_objective": float(validation_metrics.total),
                "density_coefficient": float(weights.density),
                "ssim_coefficient": float(weights.ssim),
                "physics_coefficient": float(weights.physics),
                "structure_coefficient": float(weights.structure),
                "flow_velocity_coefficient": float(weights.flow_velocity),
                "full_property_coefficient": float(weights.full_property),
                "segmentation_coefficient": float(weights.segmentation),
                "contrastive_coefficient": float(weights.contrastive),
            }
        ],
    )
