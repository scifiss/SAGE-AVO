"""Deterministic conditional residual-transport definitions."""

from __future__ import annotations

from torch import Tensor


def straight_path(low: Tensor, target: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """Return the interpolated state and constant velocity ``target - low``."""
    if low.shape != target.shape:
        raise ValueError("low and target must share a shape")
    if time.ndim != 1 or time.shape[0] != low.shape[0]:
        raise ValueError("time must have one value per batch item")
    fraction = time.reshape((-1,) + (1,) * (low.ndim - 1))
    return (1.0 - fraction) * low + fraction * target, target - low
