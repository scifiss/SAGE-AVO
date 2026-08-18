"""Training and deterministic-validation primitives for complete SAGE-AVO."""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from sage_avo.forward.torch_forward import CURRENT_ANGLE_BANDS
from sage_avo.forward.specification import ForwardModelSpecification

from .flow import straight_path
from .losses import (
    AdaptiveTaskWeighter,
    LossWeights,
    edge_smoothness,
    legacy_instance_contrastive_loss,
    multitask_loss,
    physics_loss,
    physics_loss_with_context,
)


@dataclass(frozen=True)
class PhysicsNormalization:
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor


@dataclass(frozen=True)
class PhysicsSettings:
    angles_degrees: tuple[float, ...] = tuple(float(value) for value in range(3, 46))
    bands_degrees: tuple[tuple[float, float], ...] = CURRENT_ANGLE_BANDS
    wavelet_hz: float = 14.0
    dt_seconds: float = 0.004
    wavelet_samples: int = 81
    apply_mute: bool = True
    mute_start: tuple[float, float] = (30.0, 0.0)
    mute_end: tuple[float, float] = (45.0, 0.1)
    taper_samples: int = 5
    specification: ForwardModelSpecification | None = None


@dataclass(frozen=True)
class ContrastiveSettings:
    temperature: float = 0.07
    max_samples: int = 1024


@dataclass(frozen=True)
class StepMetrics:
    total: float
    inversion: float
    flow: float
    flow_vp: float
    flow_vs: float
    flow_density: float
    full_property: float
    full_vp: float
    full_vs: float
    full_density: float
    ssim: float
    segmentation: float
    segmentation_ce: float
    segmentation_dice: float
    contrastive: float
    physics: float
    structure: float


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


def _forward_objective(
    model: nn.Module,
    values: dict[str, Tensor],
    time: Tensor,
    normalization: PhysicsNormalization,
    weights: LossWeights,
    class_weights: Tensor | None,
    physics: PhysicsSettings,
    contrastive: ContrastiveSettings,
    *,
    deterministic_contrastive: bool,
    contrastive_generator: torch.Generator | None,
    adaptive_weighter: AdaptiveTaskWeighter | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    state, target_velocity = straight_path(values["low"], values["target"], time)
    output = model(state, time, values["avo"], values["low"], values["rgt"])
    predicted_full = output.velocity + values["low"]
    structural = (
        edge_smoothness(predicted_full, output.edge_indices, output.edge_weights)
        if weights.structure > 0
        else predicted_full.new_zeros(())
    )
    if weights.physics <= 0:
        physical_consistency = predicted_full.new_zeros(())
    elif physics.specification is not None and "physics_context" in values:
        physical_consistency = physics_loss_with_context(
            predicted_full,
            values["physics_context"],
            values["physics_avo"],
            normalization.y_mean.to(predicted_full.device),
            normalization.y_std.to(predicted_full.device),
            normalization.x_mean.to(predicted_full.device),
            normalization.x_std.to(predicted_full.device),
            mask=values["physics_mask"],
            core_start=values["physics_core_start"],
            sample_origin=values["physics_context_sample_origin"],
            specification=physics.specification,
        )
    else:
        physical_consistency = physics_loss(
            predicted_full,
            values["avo"],
            normalization.y_mean.to(predicted_full.device),
            normalization.y_std.to(predicted_full.device),
            normalization.x_mean.to(predicted_full.device),
            normalization.x_std.to(predicted_full.device),
            mask=values["mask"],
            angles_degrees=physics.angles_degrees,
            bands_degrees=physics.bands_degrees,
            wavelet_hz=physics.wavelet_hz,
            dt_seconds=physics.dt_seconds,
            wavelet_samples=physics.wavelet_samples,
            apply_mute=physics.apply_mute,
            mute_start=physics.mute_start,
            mute_end=physics.mute_end,
            taper_samples=physics.taper_samples,
        )
    contrastive_consistency = (
        legacy_instance_contrastive_loss(
            output.embeddings,
            values["mask"],
            temperature=contrastive.temperature,
            max_samples=contrastive.max_samples,
            deterministic=deterministic_contrastive,
            generator=contrastive_generator,
        )
        if weights.contrastive > 0
        else predicted_full.new_zeros(())
    )
    total, terms = multitask_loss(
        output.velocity,
        target_velocity,
        predicted_full,
        values["target"],
        output.segmentation_logits,
        values["segmentation"],
        values["mask"],
        structural,
        physical_consistency,
        contrastive_consistency,
        weights,
        class_weights,
    )
    if adaptive_weighter is not None:
        total = adaptive_weighter(terms)
    return total, terms


def _metrics(total: Tensor, terms: dict[str, Tensor]) -> StepMetrics:
    return StepMetrics(
        total=float(total.detach()),
        **{
            item.name: float(terms[item.name].detach())
            for item in fields(StepMetrics)
            if item.name != "total"
        },
    )


def train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    normalization: PhysicsNormalization,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
    gradient_clip: float = 1.0,
    time_generator: torch.Generator | None = None,
    physics: PhysicsSettings = PhysicsSettings(),
    contrastive: ContrastiveSettings = ContrastiveSettings(),
    contrastive_generator: torch.Generator | None = None,
    adaptive_weighter: AdaptiveTaskWeighter | None = None,
) -> StepMetrics:
    """Run one stochastic-time SAGE-AVO optimization step."""
    device = next(model.parameters()).device
    values = _move_batch(batch, device)
    time = torch.rand(values["target"].shape[0], generator=time_generator).to(device)
    total, terms = _forward_objective(
        model,
        values,
        time,
        normalization,
        weights,
        class_weights,
        physics,
        contrastive,
        deterministic_contrastive=False,
        contrastive_generator=contrastive_generator,
        adaptive_weighter=adaptive_weighter,
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    return _metrics(total, terms)


def _average_metrics(results: list[StepMetrics]) -> StepMetrics:
    if not results:
        raise ValueError("Loader produced no batches")
    return StepMetrics(
        **{
            item.name: sum(getattr(result, item.name) for result in results) / len(results)
            for item in fields(StepMetrics)
        }
    )


def train_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    normalization: PhysicsNormalization,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
    gradient_clip: float = 1.0,
    time_generator: torch.Generator | None = None,
    physics: PhysicsSettings = PhysicsSettings(),
    contrastive: ContrastiveSettings = ContrastiveSettings(),
    contrastive_generator: torch.Generator | None = None,
    adaptive_weighter: AdaptiveTaskWeighter | None = None,
    max_batches: int | None = None,
) -> StepMetrics:
    model.train()
    results: list[StepMetrics] = []
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        results.append(
            train_step(
                model,
                batch,
                optimizer,
                normalization,
                weights,
                class_weights,
                gradient_clip,
                time_generator,
                physics,
                contrastive,
                contrastive_generator,
                adaptive_weighter,
            )
        )
    return _average_metrics(results)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
    normalization: PhysicsNormalization,
    weights: LossWeights,
    *,
    time_grid: Sequence[float] = (0.2, 0.5, 0.8),
    class_weights: Tensor | None = None,
    physics: PhysicsSettings = PhysicsSettings(),
    contrastive: ContrastiveSettings = ContrastiveSettings(),
    adaptive_weighter: AdaptiveTaskWeighter | None = None,
    max_batches: int | None = None,
) -> StepMetrics:
    """Evaluate the full objective at a deterministic interior-time grid."""
    model.eval()
    device = next(model.parameters()).device
    results: list[StepMetrics] = []
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        values = _move_batch(batch, device)
        for fraction in time_grid:
            time = torch.full(
                (values["target"].shape[0],),
                float(fraction),
                device=device,
                dtype=values["target"].dtype,
            )
            total, terms = _forward_objective(
                model,
                values,
                time,
                normalization,
                weights,
                class_weights,
                physics,
                contrastive,
                deterministic_contrastive=True,
                contrastive_generator=None,
                adaptive_weighter=adaptive_weighter,
            )
            results.append(_metrics(total, terms))
    return _average_metrics(results)
