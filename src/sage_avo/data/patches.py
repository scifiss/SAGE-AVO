"""Multiscale patch extraction with retained physical-scale metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class PatchMetadata:
    realization_id: int
    top: int
    left: int
    raw_height: int
    raw_width: int
    output_height: int
    output_width: int

    @property
    def scale_factors(self) -> tuple[float, float]:
        return self.raw_height / self.output_height, self.raw_width / self.output_width


def resize_channels_first(patch: np.ndarray, output_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize a ``[channel, height, width]`` patch without losing channel identity."""
    if patch.ndim != 3:
        raise ValueError("patch must have shape [channel, height, width]")
    factors = (1.0, output_shape[0] / patch.shape[1], output_shape[1] / patch.shape[2])
    return zoom(patch, factors, order=order).astype(patch.dtype, copy=False)


def extract_patch(
    volume: np.ndarray,
    top: int,
    left: int,
    raw_shape: tuple[int, int],
    output_shape: tuple[int, int],
    realization_id: int,
    interpolation_order: int = 1,
) -> tuple[np.ndarray, PatchMetadata]:
    """Extract and resize a patch while returning its original sampling scale."""
    height, width = raw_shape
    if volume.ndim != 3:
        raise ValueError("volume must have shape [channel, height, width]")
    if top < 0 or left < 0 or top + height > volume.shape[1] or left + width > volume.shape[2]:
        raise ValueError("Requested patch lies outside the volume")
    raw = volume[:, top : top + height, left : left + width]
    resized = resize_channels_first(raw, output_shape, order=interpolation_order)
    metadata = PatchMetadata(
        realization_id, top, left, height, width, output_shape[0], output_shape[1]
    )
    return resized, metadata
