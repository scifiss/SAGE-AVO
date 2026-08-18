"""Weighted patch sampling from facies, structure, and AVO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class SamplingDataset(Protocol):
    def __len__(self) -> int: ...

    def sampling_fields(self, index: int) -> dict[str, Tensor]: ...


@dataclass(frozen=True)
class PatchSamplingConfig:
    """Weights used by the configured replacement sampler."""

    foreground_boost: float = 5.0
    structure_boost: float = 2.5
    avo_gradient_boost: float = 2.0
    foreground_fraction_threshold: float = 0.01
    upper_quantile: float = 0.98
    representative_angles_degrees: tuple[float, float, float] = (10.0, 24.0, 38.0)


def avo_gradient(
    avo: Tensor,
    angles_degrees: tuple[float, float, float] = (10.0, 24.0, 38.0),
) -> Tensor:
    """Return the three-band least-squares gradient in ``sin²(theta)``."""
    if avo.shape[0] != 3:
        raise ValueError("avo must have shape [3,H,W]")
    angles = torch.as_tensor(angles_degrees, dtype=avo.dtype, device=avo.device)
    sin_squared = torch.sin(torch.deg2rad(angles)).square().view(3, 1, 1)
    centered_angles = sin_squared - sin_squared.mean(dim=0, keepdim=True)
    centered_avo = avo - avo.mean(dim=0, keepdim=True)
    return (centered_angles * centered_avo).sum(dim=0) / (
        centered_angles.square().sum(dim=0) + 1e-6
    )


def structural_gradient_score(rgt: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Mean centered-difference RGT gradient magnitude for one patch."""
    if rgt.ndim != 2:
        raise ValueError("rgt must have shape [H,W]")
    vertical = torch.zeros_like(rgt)
    horizontal = torch.zeros_like(rgt)
    vertical[1:-1] = 0.5 * (rgt[2:] - rgt[:-2])
    horizontal[:, 1:-1] = 0.5 * (rgt[:, 2:] - rgt[:, :-2])
    magnitude = torch.sqrt(vertical.square() + horizontal.square())
    if valid_mask is None:
        return magnitude.mean()
    valid = valid_mask.to(device=rgt.device, dtype=rgt.dtype)
    return (magnitude * valid).sum() / (valid.sum() + 1e-8)


def build_patch_sampling_weights(
    dataset: SamplingDataset,
    config: PatchSamplingConfig = PatchSamplingConfig(),
) -> Tensor:
    """Compute replacement-sampling probabilities for all foreground classes.

    ``foreground_boost`` applies to ``segmentation > 0`` and therefore weights
    both reservoir sand and class-2 plume pixels.
    """
    values: list[Tensor] = []
    for index in range(len(dataset)):
        fields = dataset.sampling_fields(index)
        segmentation = fields["segmentation"].float()
        rgt = fields["rgt"].float()
        avo = fields["avo"].float()
        valid = fields.get("mask")
        if valid is None:
            valid = torch.ones_like(segmentation)
        elif valid.ndim == 3:
            valid = valid[0]
        valid = (valid > 0.5).to(segmentation.dtype)
        foreground_fraction = (((segmentation > 0).float() * valid).sum() / (valid.sum() + 1e-8))
        structure_score = structural_gradient_score(rgt, valid)
        gradient = avo_gradient(avo, config.representative_angles_degrees).abs()
        gradient_score = (gradient * valid).sum() / (valid.sum() + 1e-8)
        weight = torch.ones((), dtype=torch.float64)
        if float(foreground_fraction) > config.foreground_fraction_threshold:
            weight = weight + config.foreground_boost * foreground_fraction.double()
        weight = weight + config.structure_boost * structure_score.double()
        weight = weight + config.avo_gradient_boost * gradient_score.double()
        values.append(weight)
    if not values:
        raise ValueError("Cannot construct weights for an empty dataset")
    weights = torch.stack(values)
    maximum = torch.quantile(weights.float(), config.upper_quantile).double()
    weights = torch.clamp(weights, max=maximum)
    return weights / weights.mean().clamp(min=1e-8)
