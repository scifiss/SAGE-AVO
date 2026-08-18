"""Epoch-dependent weights from the final SAGE-AVO 005 experiment."""

from __future__ import annotations

from dataclasses import dataclass

from .losses import LossWeights


def training_progress(epoch_index: int, total_epochs: int) -> float:
    """Return inclusive zero-to-one progress for a zero-based epoch index."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if epoch_index < 0 or epoch_index >= total_epochs:
        raise ValueError("epoch_index must lie in [0, total_epochs)")
    return epoch_index / max(total_epochs - 1, 1)


def linear_schedule(start: float, end: float, progress: float) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    return float(start) + (float(end) - float(start)) * progress


@dataclass(frozen=True)
class Curriculum:
    """Available training curricula, independently configurable."""

    density_start: float = 2.0
    density_end: float = 3.5
    ssim_start: float = 0.15
    ssim_end: float = 0.05
    physics_multiplier_start: float = 1.0
    physics_multiplier_end: float = 0.70
    structure_multiplier_start: float = 1.0
    structure_multiplier_end: float = 0.75

    def weights_for_epoch(
        self,
        base: LossWeights,
        epoch_index: int,
        total_epochs: int,
    ) -> LossWeights:
        progress = training_progress(epoch_index, total_epochs)
        return LossWeights(
            inversion=base.inversion,
            flow_velocity=base.flow_velocity,
            full_property=base.full_property,
            ssim=linear_schedule(self.ssim_start, self.ssim_end, progress),
            segmentation=base.segmentation,
            segmentation_cross_entropy=base.segmentation_cross_entropy,
            segmentation_dice=base.segmentation_dice,
            contrastive=base.contrastive,
            physics=base.physics
            * linear_schedule(
                self.physics_multiplier_start,
                self.physics_multiplier_end,
                progress,
            ),
            structure=base.structure
            * linear_schedule(
                self.structure_multiplier_start,
                self.structure_multiplier_end,
                progress,
            ),
            density=linear_schedule(self.density_start, self.density_end, progress),
        )
