"""Stable v003 checkpoint criteria independent of the training curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .losses import LossWeights


CHECKPOINT_CRITERIA = {
    "fixed_objective": (
        "final fixed weights applied to raw validation components: inversion + "
        "segmentation + exact-PP physics + graph structure"
    ),
    "sampling": "mean normalized elastic RMSE minus 0.1 times segmentation mIoU",
    "segmentation": "maximum deterministic sampled macro mIoU",
    "whole_realization": (
        "mean per-realization normalized elastic RMSE minus 0.1 times mean "
        "per-realization segmentation mIoU on fixed complete validation sections"
    ),
}


@dataclass
class CheckpointSelectionState:
    """Serializable best values/epochs restored exactly when training resumes."""

    best_values: dict[str, float] = field(
        default_factory=lambda: {
            "fixed_objective": float("inf"),
            "sampling": float("inf"),
            "segmentation": float("-inf"),
            "whole_realization": float("inf"),
        }
    )
    best_epochs: dict[str, int | None] = field(
        default_factory=lambda: {name: None for name in CHECKPOINT_CRITERIA}
    )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "CheckpointSelectionState":
        state = cls()
        if not mapping:
            return state
        for name in CHECKPOINT_CRITERIA:
            if name in mapping.get("best_values", {}):
                state.best_values[name] = float(mapping["best_values"][name])
            if name in mapping.get("best_epochs", {}):
                value = mapping["best_epochs"][name]
                state.best_epochs[name] = None if value is None else int(value)
        return state

    def update(self, criterion: str, value: float, epoch: int) -> bool:
        if criterion not in CHECKPOINT_CRITERIA:
            raise KeyError(f"Unknown checkpoint criterion {criterion!r}")
        if not np.isfinite(value):
            return False
        improved = (
            value > self.best_values[criterion]
            if criterion == "segmentation"
            else value < self.best_values[criterion]
        )
        if improved:
            self.best_values[criterion] = float(value)
            self.best_epochs[criterion] = int(epoch)
        return improved

    def to_dict(self) -> dict[str, dict[str, float | int | None]]:
        return asdict(self)


def weighted_objective_contributions(
    metrics: Any,
    weights: LossWeights,
) -> dict[str, float]:
    """Recompose an objective from raw terms and one explicit weight set."""
    density = float(weights.density)
    flow = (
        float(metrics.flow_vp)
        + float(metrics.flow_vs)
        + density * float(metrics.flow_density)
    ) / (2.0 + density)
    full = (
        float(metrics.full_vp)
        + float(metrics.full_vs)
        + density * float(metrics.full_density)
    ) / (2.0 + density)
    inversion = (
        float(weights.flow_velocity) * flow
        + float(weights.full_property) * full
        + float(weights.ssim) * float(metrics.ssim)
    )
    contributions = {
        "inversion": float(weights.inversion) * inversion,
        "segmentation": float(weights.segmentation) * float(metrics.segmentation),
        "contrastive": float(weights.contrastive) * float(metrics.contrastive),
        "physics": float(weights.physics) * float(metrics.physics),
        "structure": float(weights.structure) * float(metrics.structure),
    }
    contributions["total"] = float(sum(contributions.values()))
    return contributions


def checkpoint_metadata(
    criterion: str,
    value: float,
    epoch: int,
) -> dict[str, float | int | str]:
    if criterion not in CHECKPOINT_CRITERIA:
        raise KeyError(f"Unknown checkpoint criterion {criterion!r}")
    return {
        "criterion_name": criterion,
        "criterion_formula": CHECKPOINT_CRITERIA[criterion],
        "criterion_value": float(value),
        "criterion_epoch": int(epoch),
    }


def whole_realization_criterion(
    normalized_rmse: list[float] | np.ndarray,
    miou: float,
) -> float:
    """Return the declared v003 fixed whole-section selection criterion."""
    values = np.asarray(normalized_rmse, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or not np.isfinite(miou):
        return float("nan")
    return float(values.mean() - 0.1 * float(miou))
