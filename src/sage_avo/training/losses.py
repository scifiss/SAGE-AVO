"""Explicit multitask losses for SAGE-AVO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from sage_avo.forward.torch_forward import forward_avo_three_band_torch


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.expand_as(prediction)
    return ((prediction - target).square() * expanded).sum() / (expanded.sum() + 1e-8)


def multiclass_dice_loss(logits: Tensor, target: Tensor, classes: int = 3) -> Tensor:
    probabilities = logits.softmax(dim=1)
    encoded = F.one_hot(target.long(), classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * encoded).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + encoded.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def edge_smoothness(
    full_properties: Tensor,
    edge_indices: list[Tensor],
    edge_weights: list[Tensor],
) -> Tensor:
    """Penalize property contrast most strongly along high-weight graph edges."""
    batch, channels, _, _ = full_properties.shape
    if not edge_indices:
        return full_properties.new_zeros(())
    total = full_properties.new_zeros(())
    for item in range(batch):
        flattened = full_properties[item].reshape(channels, -1).transpose(0, 1)
        source, destination = edge_indices[item]
        contrast = (flattened[source] - flattened[destination]).abs().mean(dim=1)
        total = total + (contrast * edge_weights[item]).mean()
    return total / max(batch, 1)


def physics_loss(
    normalized_prediction: Tensor,
    normalized_avo: Tensor,
    y_mean: Tensor,
    y_std: Tensor,
    x_mean: Tensor,
    x_std: Tensor,
) -> Tensor:
    """Compare observed and differentiably forward-modeled three-band AVO."""
    physical = normalized_prediction * y_std + y_mean
    modeled = forward_avo_three_band_torch(physical[:, 0], physical[:, 1], physical[:, 2])
    normalized_modeled = (modeled - x_mean) / x_std
    return F.mse_loss(normalized_modeled, normalized_avo)


@dataclass(frozen=True)
class LossWeights:
    flow: float = 1.0
    full_property: float = 0.2
    segmentation: float = 0.3
    physics: float = 0.5
    structure: float = 0.5


def multitask_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    predicted_full: Tensor,
    target_full: Tensor,
    segmentation_logits: Tensor,
    segmentation_target: Tensor,
    mask: Tensor,
    structural_loss: Tensor,
    physics_consistency: Tensor,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine flow, full-property, segmentation, physics, and structure terms."""
    terms = {
        "flow": masked_mse(predicted_velocity, target_velocity, mask),
        "full_property": masked_mse(predicted_full, target_full, mask),
        "segmentation": F.cross_entropy(segmentation_logits, segmentation_target.long(), weight=class_weights)
        + 0.5 * multiclass_dice_loss(segmentation_logits, segmentation_target),
        "physics": physics_consistency,
        "structure": structural_loss,
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
