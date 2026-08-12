"""One-step and epoch-level training primitives without notebook globals."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .flow import straight_path
from .losses import LossWeights, edge_smoothness, multitask_loss, physics_loss


@dataclass(frozen=True)
class PhysicsNormalization:
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor


@dataclass(frozen=True)
class StepMetrics:
    total: float
    flow: float
    full_property: float
    segmentation: float
    physics: float
    structure: float


def train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    normalization: PhysicsNormalization,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
    gradient_clip: float = 1.0,
    time_generator: torch.Generator | None = None,
) -> StepMetrics:
    """Run one reproducible SAGE-AVO optimization step."""
    device = next(model.parameters()).device
    values = {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}
    time = torch.rand(values["target"].shape[0], generator=time_generator).to(device)
    state, target_velocity = straight_path(values["low"], values["target"], time)
    output = model(state, time, values["avo"], values["low"], values["rgt"])
    predicted_full = output.velocity + values["low"]
    structural = (
        edge_smoothness(predicted_full, output.edge_indices, output.edge_weights)
        if weights.structure > 0
        else predicted_full.new_zeros(())
    )
    physics = (
        physics_loss(
            predicted_full,
            values["avo"],
            normalization.y_mean.to(device),
            normalization.y_std.to(device),
            normalization.x_mean.to(device),
            normalization.x_std.to(device),
        )
        if weights.physics > 0
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
        physics,
        weights,
        class_weights,
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    detached = {name: float(value.detach()) for name, value in terms.items()}
    return StepMetrics(float(total.detach()), **detached)


def train_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    normalization: PhysicsNormalization,
    weights: LossWeights = LossWeights(),
    class_weights: Tensor | None = None,
    gradient_clip: float = 1.0,
    time_generator: torch.Generator | None = None,
) -> StepMetrics:
    """Average explicit step metrics over one epoch."""
    model.train()
    results = [
        train_step(
            model,
            batch,
            optimizer,
            normalization,
            weights,
            class_weights,
            gradient_clip,
            time_generator,
        )
        for batch in batches
    ]
    if not results:
        raise ValueError("Training loader produced no batches")
    fields = StepMetrics.__dataclass_fields__
    return StepMetrics(**{name: sum(getattr(item, name) for item in results) / len(results) for name in fields})
