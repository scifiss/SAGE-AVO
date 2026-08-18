"""Registered geological and AVO augmentation from final experiment 005."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.50
    avo_gain_probability: float = 0.40
    avo_gain_minimum: float = 0.98
    avo_gain_maximum: float = 1.02
    avo_noise_probability: float = 0.35
    avo_noise_standard_deviation: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "horizontal_flip_probability",
            "avo_gain_probability",
            "avo_noise_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.avo_gain_minimum <= 0 or self.avo_gain_maximum < self.avo_gain_minimum:
            raise ValueError("Invalid AVO gain interval")
        if self.avo_noise_standard_deviation < 0:
            raise ValueError("AVO noise standard deviation cannot be negative")


def _draw(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator))


def augment_patch(
    item: dict[str, Tensor],
    config: AugmentationConfig = AugmentationConfig(),
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Apply registered geometry and mild normalized-domain AVO perturbations."""
    augmented = dict(item)
    if _draw(generator) < config.horizontal_flip_probability:
        for name in ("avo", "target", "low", "rgt", "mask", "segmentation"):
            augmented[name] = torch.flip(augmented[name], dims=(-1,))
        if "dip" in augmented:
            augmented["dip"] = -torch.flip(augmented["dip"], dims=(-1,))

    if _draw(generator) < config.avo_gain_probability:
        gain = config.avo_gain_minimum + _draw(generator) * (
            config.avo_gain_maximum - config.avo_gain_minimum
        )
        augmented["avo"] = augmented["avo"] * gain

    if _draw(generator) < config.avo_noise_probability:
        noise = torch.randn(
            augmented["avo"].shape,
            generator=generator,
            device=augmented["avo"].device,
            dtype=augmented["avo"].dtype,
        )
        augmented["avo"] = (
            augmented["avo"] + config.avo_noise_standard_deviation * noise
        )
    return augmented
