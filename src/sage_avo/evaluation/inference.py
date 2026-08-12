"""Common full-realization tiling and stitching for every learned variant."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def tile_starts(length: int, patch: int, stride: int) -> list[int]:
    """Cover an axis exactly, including an end-aligned final tile."""
    if patch > length or stride < 1:
        raise ValueError("patch must fit the axis and stride must be positive")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def blend_window(shape: tuple[int, int]) -> np.ndarray:
    """Return a clipped Hann window without zero-coverage image boundaries."""
    vertical = np.hanning(shape[0]) if shape[0] > 2 else np.ones(shape[0])
    horizontal = np.hanning(shape[1]) if shape[1] > 2 else np.ones(shape[1])
    return np.maximum(np.outer(vertical, horizontal), 1e-3).astype(np.float32)


@torch.no_grad()
def infer_full_realization(
    model: nn.Module,
    *,
    avo: np.ndarray,
    low: np.ndarray,
    rgt: np.ndarray,
    normalization: dict[str, list[float]],
    patch_shape: tuple[int, int],
    stride: tuple[int, int],
    steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer and Hann-stitch elastic properties and segmentation probabilities."""
    model.eval()
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    normalized_avo = (np.asarray(avo, dtype=np.float32) - x_mean) / x_std
    normalized_low = (np.asarray(low, dtype=np.float32) - y_mean) / y_std
    height, width = rgt.shape
    positions = [
        (top, left)
        for top in tile_starts(height, patch_shape[0], stride[0])
        for left in tile_starts(width, patch_shape[1], stride[1])
    ]
    window = blend_window(patch_shape)
    elastic_sum = np.zeros((3, height, width), dtype=np.float64)
    probability_sum = np.zeros((3, height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        avo_batch = torch.from_numpy(
            np.stack(
                [normalized_avo[:, t : t + patch_shape[0], x : x + patch_shape[1]] for t, x in batch_positions]
            )
        ).to(device)
        low_batch = torch.from_numpy(
            np.stack(
                [normalized_low[:, t : t + patch_shape[0], x : x + patch_shape[1]] for t, x in batch_positions]
            )
        ).to(device)
        rgt_batch = torch.from_numpy(
            np.stack([rgt[t : t + patch_shape[0], x : x + patch_shape[1]] for t, x in batch_positions]).astype(np.float32)
        ).to(device)
        prediction = model.sample(avo_batch, low_batch, rgt_batch, steps=steps)
        final_time = torch.ones(prediction.shape[0], device=device)
        logits = model(prediction, final_time, avo_batch, low_batch, rgt_batch).segmentation_logits
        physical = prediction.cpu().numpy() * y_std[None] + y_mean[None]
        probabilities = logits.softmax(dim=1).cpu().numpy()
        for item, (top, left) in enumerate(batch_positions):
            spatial = np.s_[top : top + patch_shape[0], left : left + patch_shape[1]]
            elastic_sum[(slice(None),) + spatial] += physical[item] * window[None]
            probability_sum[(slice(None),) + spatial] += probabilities[item] * window[None]
            weight_sum[spatial] += window
    elastic = elastic_sum / np.maximum(weight_sum[None], 1e-12)
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-12)
    return elastic.astype(np.float32), probabilities.argmax(axis=0).astype(np.uint8)


def load_normalization(dataset_directory: str | Path) -> dict[str, list[float]]:
    return json.loads((Path(dataset_directory) / "normalization.json").read_text(encoding="utf-8"))
