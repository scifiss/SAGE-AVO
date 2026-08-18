"""Complete multitask objective used by the final SAGE-AVO workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sage_avo.forward.torch_forward import (
    CURRENT_ANGLE_BANDS,
    forward_avo_three_band_torch,
    forward_avo_three_band_spec_torch,
)
from sage_avo.forward.specification import ForwardModelSpecification


def _single_channel_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != reference.ndim:
        raise ValueError("mask must have shape [B,H,W] or [B,C,H,W]")
    return mask[:, :1].to(device=reference.device, dtype=reference.dtype)


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over valid pixels and all requested channels."""
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != prediction.ndim:
        raise ValueError("mask must have shape [B,H,W] or [B,C,H,W]")
    if mask.shape[1] == 1:
        expanded = mask.expand_as(prediction)
    elif mask.shape[1] == prediction.shape[1]:
        expanded = mask
    else:
        raise ValueError("mask must have one channel or match the prediction channels")
    expanded = expanded.to(device=prediction.device, dtype=prediction.dtype)
    return ((prediction - target).square() * expanded).sum() / (expanded.sum() + 1e-8)


def weighted_elastic_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    density_weight: float,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """Vp/Vs/density MSE with the configured density emphasis."""
    if prediction.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("Elastic tensors must contain Vp, Vs, and density channels")
    per_channel = tuple(
        masked_mse(prediction[:, index : index + 1], target[:, index : index + 1], mask)
        for index in range(3)
    )
    vp, vs, density = per_channel
    combined = (vp + vs + float(density_weight) * density) / (2.0 + float(density_weight))
    return combined, per_channel


def masked_ssim_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Channel-wise SSIM loss averaged only over the valid property mask."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    channels = prediction.shape[1]
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    gaussian = torch.exp(-((coordinates - window_size // 2) ** 2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    window = window_2d.view(1, 1, window_size, window_size).expand(channels, 1, -1, -1)
    padding = window_size // 2
    expanded = _single_channel_mask(mask, prediction).expand_as(prediction)
    support = F.conv2d(expanded, window, padding=padding, groups=channels)
    safe_support = support.clamp_min(1e-8)
    mean_prediction = (
        F.conv2d(prediction * expanded, window, padding=padding, groups=channels)
        / safe_support
    )
    mean_target = (
        F.conv2d(target * expanded, window, padding=padding, groups=channels) / safe_support
    )
    prediction_variance = (
        F.conv2d(prediction.square() * expanded, window, padding=padding, groups=channels)
        / safe_support
        - mean_prediction.square()
    )
    target_variance = (
        F.conv2d(target.square() * expanded, window, padding=padding, groups=channels)
        / safe_support
        - mean_target.square()
    )
    covariance = (
        F.conv2d(prediction * target * expanded, window, padding=padding, groups=channels)
        / safe_support
        - mean_prediction * mean_target
    )
    c1, c2 = 0.01**2, 0.03**2
    similarity = (
        (2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)
    ) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (prediction_variance + target_variance + c2)
        + 1e-12
    )
    return ((1.0 - similarity) * expanded).sum() / (expanded.sum() + 1e-8)


def multiclass_dice_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    classes: int = 3,
) -> Tensor:
    probabilities = logits.softmax(dim=1)
    safe_target = target.long().clamp(0, classes - 1)
    encoded = F.one_hot(safe_target, classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    if valid_mask is None:
        mask = torch.ones_like(target, dtype=probabilities.dtype).unsqueeze(1)
    else:
        mask = _single_channel_mask(valid_mask, probabilities)
    probabilities = probabilities * mask
    encoded = encoded * mask
    intersection = (probabilities * encoded).sum(dim=(0, 2, 3))
    denominator = probabilities.square().sum(dim=(0, 2, 3)) + encoded.square().sum(
        dim=(0, 2, 3)
    )
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    class_weights: Tensor | None = None,
    cross_entropy_weight: float = 1.0,
    dice_weight: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Masked weighted cross-entropy plus masked multiclass Dice."""
    safe_target = target.long().clamp(0, logits.shape[1] - 1)
    per_pixel = F.cross_entropy(
        logits,
        safe_target,
        weight=class_weights,
        reduction="none",
    )
    mask = _single_channel_mask(valid_mask, logits).squeeze(1)
    cross_entropy = (per_pixel * mask).sum() / (mask.sum() + 1e-8)
    dice = multiclass_dice_loss(logits, safe_target, valid_mask=mask, classes=logits.shape[1])
    total = float(cross_entropy_weight) * cross_entropy + float(dice_weight) * dice
    return total, cross_entropy, dice


def legacy_instance_contrastive_loss(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    temperature: float = 0.07,
    max_samples: int = 1024,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Compute the optional self-instance contrastive term.

    Each sampled node is treated as its own class. This capability is disabled
    in the production configuration and is not a supervised facies contrastive
    objective.
    """
    batch, nodes, channels = embeddings.shape
    mask = valid_mask
    if mask.ndim == 4:
        mask = mask[:, 0]
    valid = mask.reshape(batch * nodes) > 0.5
    selected = embeddings.reshape(batch * nodes, channels)[valid]
    if selected.shape[0] < 2:
        return embeddings.new_zeros(())
    if selected.shape[0] > max_samples:
        if deterministic:
            indices = torch.linspace(
                0,
                selected.shape[0] - 1,
                max_samples,
                device=selected.device,
            ).long()
        else:
            indices = torch.randperm(
                selected.shape[0],
                device=selected.device,
                generator=generator,
            )[:max_samples]
        selected = selected[indices]
    selected = F.normalize(selected, dim=1)
    similarities = selected @ selected.transpose(0, 1) / float(temperature)
    labels = torch.arange(selected.shape[0], device=selected.device)
    return F.cross_entropy(similarities, labels)


class AdaptiveTaskWeighter(nn.Module):
    """Optional homoscedastic task weighting capability."""

    def __init__(self, task_names: Sequence[str]) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("At least one active task is required")
        self.task_names = tuple(task_names)
        self.log_variances = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        missing = set(self.task_names) - set(losses)
        if missing:
            raise KeyError(f"Missing adaptive task losses: {sorted(missing)}")
        total = self.log_variances.new_zeros(())
        for index, name in enumerate(self.task_names):
            precision = torch.exp(-self.log_variances[index])
            total = total + precision * losses[name] + self.log_variances[index]
        return total


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
    *,
    mask: Tensor | None = None,
    angles_degrees: Sequence[float] = tuple(float(value) for value in range(3, 46)),
    bands_degrees: tuple[tuple[float, float], ...] = CURRENT_ANGLE_BANDS,
    wavelet_hz: float = 14.0,
    dt_seconds: float = 0.004,
    wavelet_samples: int = 81,
    apply_mute: bool = True,
    mute_start: tuple[float, float] = (30.0, 0.0),
    mute_end: tuple[float, float] = (45.0, 0.1),
    taper_samples: int = 5,
) -> Tensor:
    """Compare observed and differentiably forward-modeled three-band AVO."""
    physical = normalized_prediction * y_std + y_mean
    angles = torch.as_tensor(
        tuple(angles_degrees),
        device=physical.device,
        dtype=physical.dtype,
    )
    modeled = forward_avo_three_band_torch(
        physical[:, 0],
        physical[:, 1],
        physical[:, 2],
        angles_degrees=angles,
        wavelet_hz=wavelet_hz,
        dt_seconds=dt_seconds,
        wavelet_samples=wavelet_samples,
        bands_degrees=bands_degrees,
        apply_mute=apply_mute,
        mute_start=mute_start,
        mute_end=mute_end,
        taper_samples=taper_samples,
    )
    normalized_modeled = (modeled - x_mean) / x_std
    if mask is None:
        return F.mse_loss(normalized_modeled, normalized_avo)
    return masked_mse(normalized_modeled, normalized_avo, mask)


def physics_loss_with_context(
    normalized_prediction: Tensor,
    normalized_context: Tensor,
    normalized_observed_avo: Tensor,
    y_mean: Tensor,
    y_std: Tensor,
    x_mean: Tensor,
    x_std: Tensor,
    *,
    mask: Tensor,
    core_start: Tensor,
    sample_origin: Tensor,
    specification: ForwardModelSpecification,
) -> Tensor:
    """Forward native-grid predictions with truth context supplying the vertical halo."""
    batch, _, core_height, core_width = normalized_prediction.shape
    if normalized_context.ndim != 4 or normalized_context.shape[0] != batch:
        raise ValueError("normalized_context must have shape [B,3,H+2*halo,W]")
    if normalized_context.shape[3] != core_width:
        raise ValueError("Physics context and prediction must share their trace grid")
    context = normalized_context.clone()
    for item in range(batch):
        start = int(core_start[item].item())
        context[item, :, start : start + core_height] = normalized_prediction[item]
    physical = context * y_std + y_mean
    modeled_context = forward_avo_three_band_spec_torch(
        physical[:, 0],
        physical[:, 1],
        physical[:, 2],
        specification,
        sample_origin=sample_origin,
    )
    modeled = torch.empty_like(normalized_observed_avo)
    for item in range(batch):
        start = int(core_start[item].item())
        modeled[item] = modeled_context[item, :, start : start + core_height]
    normalized_modeled = (modeled - x_mean) / x_std
    return masked_mse(normalized_modeled, normalized_observed_avo, mask)


@dataclass(frozen=True)
class LossWeights:
    """Effective weights for one epoch of the configured objective."""

    inversion: float = 1.0
    flow_velocity: float = 0.65
    full_property: float = 0.20
    ssim: float = 0.15
    segmentation: float = 0.30
    segmentation_cross_entropy: float = 1.0
    segmentation_dice: float = 0.50
    contrastive: float = 0.0
    physics: float = 0.50
    structure: float = 0.50
    density: float = 2.0


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
    contrastive_consistency: Tensor,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine all independently configurable objective terms."""
    flow_mse, flow_channels = weighted_elastic_mse(
        predicted_velocity,
        target_velocity,
        mask,
        weights.density,
    )
    full_property, full_channels = weighted_elastic_mse(
        predicted_full,
        target_full,
        mask,
        weights.density,
    )
    ssim = masked_ssim_loss(predicted_velocity, target_velocity, mask)
    segmentation, segmentation_ce, segmentation_dice = segmentation_loss(
        segmentation_logits,
        segmentation_target,
        mask,
        class_weights=class_weights,
        cross_entropy_weight=weights.segmentation_cross_entropy,
        dice_weight=weights.segmentation_dice,
    )
    inversion = (
        weights.flow_velocity * flow_mse
        + weights.full_property * full_property
        + weights.ssim * ssim
    )
    total = (
        weights.inversion * inversion
        + weights.segmentation * segmentation
        + weights.contrastive * contrastive_consistency
        + weights.physics * physics_consistency
        + weights.structure * structural_loss
    )
    terms = {
        "inversion": inversion,
        "flow": flow_mse,
        "flow_vp": flow_channels[0],
        "flow_vs": flow_channels[1],
        "flow_density": flow_channels[2],
        "full_property": full_property,
        "full_vp": full_channels[0],
        "full_vs": full_channels[1],
        "full_density": full_channels[2],
        "ssim": ssim,
        "segmentation": segmentation,
        "segmentation_ce": segmentation_ce,
        "segmentation_dice": segmentation_dice,
        "contrastive": contrastive_consistency,
        "physics": physics_consistency,
        "structure": structural_loss,
    }
    return total, terms
