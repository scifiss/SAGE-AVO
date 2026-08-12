"""RGT-steered graph construction shared by models and visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphEdges:
    source: np.ndarray
    destination: np.ndarray
    weight: np.ndarray

    def strongest(self, fraction: float = 0.15) -> "GraphEdges":
        """Return the highest-weight edge fraction for visualization only."""
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        count = max(1, int(np.ceil(self.weight.size * fraction)))
        indices = np.argsort(self.weight)[-count:]
        return GraphEdges(self.source[indices], self.destination[indices], self.weight[indices])


def build_rgt_graph(
    rgt: np.ndarray,
    avo_gradient: np.ndarray | None = None,
    max_shift: int = 3,
    include_vertical: bool = True,
) -> GraphEdges:
    """Connect adjacent traces at the closest RGT and weight AVO similarity.

    Both directions are included. Edge weight is one when no AVO gradient is
    supplied; otherwise it decays exponentially with gradient contrast.
    """
    if rgt.ndim != 2:
        raise ValueError("rgt must be 2-D")
    if avo_gradient is not None and avo_gradient.shape != rgt.shape:
        raise ValueError("avo_gradient and rgt must share a shape")
    height, width = rgt.shape
    source: list[int] = []
    destination: list[int] = []

    for column in range(width - 1):
        for row in range(height):
            candidates = np.arange(max(0, row - max_shift), min(height, row + max_shift + 1))
            target_row = int(candidates[np.argmin(np.abs(rgt[candidates, column + 1] - rgt[row, column]))])
            first = row * width + column
            second = target_row * width + column + 1
            source.extend((first, second))
            destination.extend((second, first))

    if include_vertical:
        for column in range(width):
            for row in range(height - 1):
                first = row * width + column
                second = (row + 1) * width + column
                source.extend((first, second))
                destination.extend((second, first))

    src = np.asarray(source, dtype=np.int64)
    dst = np.asarray(destination, dtype=np.int64)
    if avo_gradient is None:
        weights = np.ones(src.size, dtype=np.float32)
    else:
        flattened = np.asarray(avo_gradient, dtype=np.float64).reshape(-1)
        contrast = np.abs(flattened[src] - flattened[dst])
        scale = max(float(np.std(contrast)), 1e-6)
        weights = np.exp(-contrast / scale).astype(np.float32)
    return GraphEdges(src, dst, weights)
