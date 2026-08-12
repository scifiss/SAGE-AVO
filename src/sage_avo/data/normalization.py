"""Explicit train-only normalization statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """Channel-wise mean and standard deviation for inputs and targets."""

    x_mean: tuple[float, ...]
    x_std: tuple[float, ...]
    y_mean: tuple[float, ...]
    y_std: tuple[float, ...]

    def save(self, path: str | Path) -> None:
        """Write portable JSON statistics."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def _channel_stats(array: np.ndarray, mask: np.ndarray | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if array.ndim != 4:
        raise ValueError("Expected [sample, channel, height, width]")
    means: list[float] = []
    stds: list[float] = []
    for channel in range(array.shape[1]):
        values = array[:, channel]
        valid = np.isfinite(values)
        if mask is not None:
            channel_mask = mask[:, min(channel, mask.shape[1] - 1)].astype(bool)
            valid &= channel_mask
        selected = values[valid]
        if selected.size == 0:
            raise ValueError(f"Channel {channel} has no valid samples")
        means.append(float(selected.mean()))
        stds.append(float(max(selected.std(), 1e-8)))
    return tuple(means), tuple(stds)


def compute_normalization_stats(
    inputs: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> NormalizationStats:
    """Compute statistics from the training split only."""
    x_mean, x_std = _channel_stats(inputs)
    y_mean, y_std = _channel_stats(targets, target_mask)
    return NormalizationStats(x_mean, x_std, y_mean, y_std)
