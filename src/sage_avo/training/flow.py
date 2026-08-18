"""Deterministic conditional residual-transport definitions."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def straight_path(low: Tensor, target: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """Return the interpolated state and constant velocity ``target - low``."""
    if low.shape != target.shape:
        raise ValueError("low and target must share a shape")
    if time.ndim != 1 or time.shape[0] != low.shape[0]:
        raise ValueError("time must have one value per batch item")
    fraction = time.reshape((-1,) + (1,) * (low.ndim - 1))
    return (1.0 - fraction) * low + fraction * target, target - low


def heun_integrate(
    initial: Tensor,
    velocity: Callable[[Tensor, Tensor], Tensor],
    *,
    steps: int,
    correction: Callable[[Tensor, int], Tensor] | None = None,
) -> Tensor:
    """Integrate ``dx/dt = velocity(x, t)`` over ``[0, 1]`` with Heun/RK2.

    ``correction`` is invoked after each completed step and is used by SAGE-AVO
    for optional differentiable seismic guidance.  It is deliberately outside
    the velocity evaluation so a zero guidance setting follows the exact
    unguided Heun trajectory.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    state = initial
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (state.shape[0],),
            index * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        next_time = torch.full(
            (state.shape[0],),
            (index + 1) * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        first = velocity(state, time)
        provisional = state + step_size * first
        second = velocity(provisional, next_time)
        state = state + 0.5 * step_size * (first + second)
        if correction is not None:
            state = correction(state, index)
    return state
