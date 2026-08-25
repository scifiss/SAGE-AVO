"""Loss and physics-eligibility accounting with no optimizer-side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

from sage_avo.training.engine import StepMetrics
from sage_avo.training.losses import LossWeights


RAW_COMPONENTS = (
    "inversion",
    "flow",
    "flow_vp",
    "flow_vs",
    "flow_density",
    "full_vp",
    "full_vs",
    "full_density",
    "full_property",
    "ssim",
    "segmentation_ce",
    "segmentation_dice",
    "segmentation",
    "physics",
    "structure",
    "contrastive",
)

WEIGHTED_COMPONENTS = (
    "flow_vp",
    "flow_vs",
    "flow_density",
    "full_vp",
    "full_vs",
    "full_density",
    "ssim",
    "segmentation_ce",
    "segmentation_dice",
    "physics",
    "structure",
    "contrastive",
)


def effective_component_coefficients(weights: LossWeights) -> dict[str, float]:
    """Return the exact scalar multiplying each independently logged raw term."""
    denominator = 2.0 + float(weights.density)
    return {
        "flow_vp": weights.inversion * weights.flow_velocity / denominator,
        "flow_vs": weights.inversion * weights.flow_velocity / denominator,
        "flow_density": (weights.inversion * weights.flow_velocity * weights.density / denominator),
        "full_vp": weights.inversion * weights.full_property / denominator,
        "full_vs": weights.inversion * weights.full_property / denominator,
        "full_density": (weights.inversion * weights.full_property * weights.density / denominator),
        "ssim": weights.inversion * weights.ssim,
        "segmentation_ce": (weights.segmentation * weights.segmentation_cross_entropy),
        "segmentation_dice": weights.segmentation * weights.segmentation_dice,
        "physics": weights.physics,
        "structure": weights.structure,
        "contrastive": weights.contrastive,
    }


def raw_and_weighted_rows(
    *,
    epoch: int,
    split: str,
    metrics: StepMetrics,
    weights: LossWeights,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficients = effective_component_coefficients(weights)
    raw_rows = []
    weighted_rows = []
    for name in RAW_COMPONENTS:
        raw = float(getattr(metrics, name))
        raw_rows.append({"epoch": int(epoch), "split": split, "component": name, "raw_loss": raw})
    for name in WEIGHTED_COMPONENTS:
        raw = float(getattr(metrics, name))
        coefficient = float(coefficients[name])
        weighted_rows.append(
            {
                "epoch": int(epoch),
                "split": split,
                "component": name,
                "raw_loss": raw,
                "effective_coefficient": coefficient,
                "weighted_contribution": coefficient * raw,
            }
        )
    return raw_rows, weighted_rows


@dataclass
class EpochLossObserver:
    """Read batch metadata and detached metrics after an optimization/eval step."""

    physics_weight: float
    objective_evaluations: int = 0
    sample_evaluations: int = 0
    physics_eligible_samples: int = 0
    active_physics_evaluations: int = 0
    inactive_physics_evaluations: int = 0
    _conditional_weighted_sum: float = 0.0
    _all_step_physics_sum: float = 0.0
    _active_values: list[float] = field(default_factory=list)

    def __call__(self, batch: dict[str, Tensor], metrics: StepMetrics) -> None:
        self.observe(batch, metrics)

    def observe(self, batch: dict[str, Tensor], metrics: StepMetrics) -> None:
        eligible = batch.get("physics_eligible")
        batch_size = int(batch["target"].shape[0])
        eligible_count = int(eligible.sum().item()) if eligible is not None else batch_size
        active = eligible_count > 0
        raw_physics = float(metrics.physics)
        self.objective_evaluations += 1
        self.sample_evaluations += batch_size
        self.physics_eligible_samples += eligible_count
        self._all_step_physics_sum += raw_physics
        if active:
            self.active_physics_evaluations += 1
            self._active_values.append(raw_physics)
            self._conditional_weighted_sum += raw_physics * eligible_count
        else:
            self.inactive_physics_evaluations += 1

    def summary(self) -> dict[str, float | int | str]:
        conditional = (
            self._conditional_weighted_sum / self.physics_eligible_samples
            if self.physics_eligible_samples
            else float("nan")
        )
        all_step = (
            self._all_step_physics_sum / self.objective_evaluations
            if self.objective_evaluations
            else float("nan")
        )
        return {
            "optimizer_or_eval_steps": self.objective_evaluations,
            "sample_evaluations": self.sample_evaluations,
            "physics_active_steps": self.active_physics_evaluations,
            "physics_inactive_steps": self.inactive_physics_evaluations,
            "physics_eligible_samples": self.physics_eligible_samples,
            "physics_eligible_fraction_seen": (
                self.physics_eligible_samples / self.sample_evaluations
                if self.sample_evaluations
                else float("nan")
            ),
            "conditional_raw_physics_loss": conditional,
            "conditional_weighted_physics_per_eligible_sample": (
                float(self.physics_weight) * conditional
            ),
            "all_step_raw_physics_loss": all_step,
            "all_step_weighted_physics_contribution": (float(self.physics_weight) * all_step),
            "mixed_batch_reduction": (
                "physics mask excludes ineligible pixels within mixed batches; "
                "only epoch averaging across fully inactive steps introduces zeros"
            ),
            "active_batch_raw_physics_mean": (
                float(np.mean(self._active_values)) if self._active_values else float("nan")
            ),
        }
