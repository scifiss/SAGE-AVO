"""Reproducible field-conditioned synthetic geology.

The generator deliberately exposes its assumptions. It creates diverse members
of one field-conditioned geological family; it does not claim independent
regional geological coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .conventions import delta_from_sand_probability


@dataclass(frozen=True)
class SyntheticGeology:
    sand_probability: np.ndarray
    porosity: np.ndarray
    rgt: np.ndarray
    delta: np.ndarray
    plume_mask: np.ndarray
    scale_metadata: dict[str, float | int]


def _normalized_correlated_noise(
    shape: tuple[int, int], rng: np.random.Generator, sigma: tuple[float, float]
) -> np.ndarray:
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    return (field - field.mean()) / max(field.std(), 1e-8)


def _warp(array: np.ndarray, vertical_displacement: np.ndarray, horizontal_displacement: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(array.shape, dtype=float)
    return map_coordinates(
        array,
        [rows - vertical_displacement, columns - horizontal_displacement],
        order=1,
        mode="nearest",
    )


def _plume_mask(
    sand_probability: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    height, width = sand_probability.shape
    rows, columns = np.indices((height, width))
    result = np.zeros((height, width), dtype=bool)
    reservoir = sand_probability > 0.5
    candidates = np.argwhere(reservoir)
    if candidates.size == 0:
        return result
    for _ in range(count):
        center_row, center_column = candidates[rng.integers(candidates.shape[0])]
        radius_x = rng.uniform(0.06, 0.18) * width
        radius_z = rng.uniform(0.02, 0.08) * height
        ellipse = (
            ((columns - center_column) / max(radius_x, 1.0)) ** 2
            + ((rows - center_row) / max(radius_z, 1.0)) ** 2
            <= 1.0
        )
        result |= ellipse & reservoir
    return result


def make_synthetic_geology(
    sand_probability_base: np.ndarray,
    porosity_base: np.ndarray,
    rgt_base: np.ndarray,
    seed: int,
    max_faults: int = 7,
) -> SyntheticGeology:
    """Perturb facies, porosity, and RGT coherently and add reservoir plumes."""
    arrays = [np.asarray(value, dtype=float) for value in (sand_probability_base, porosity_base, rgt_base)]
    if any(value.ndim != 2 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("All base arrays must be 2-D with a common shape")
    sand_base, porosity_base_array, rgt_base_array = arrays
    rng = np.random.default_rng(seed)
    height, width = sand_base.shape

    columns = np.indices((height, width), dtype=float)[1]
    fold_amplitude = rng.uniform(-0.06, 0.06) * height
    fold_cycles = rng.uniform(0.5, 2.0)
    vertical = fold_amplitude * np.sin(2.0 * np.pi * fold_cycles * columns / max(width - 1, 1))
    vertical += 2.5 * _normalized_correlated_noise((height, width), rng, (12.0, 30.0))
    horizontal = 1.5 * _normalized_correlated_noise((height, width), rng, (20.0, 20.0))

    n_faults = int(rng.integers(0, max_faults + 1))
    for _ in range(n_faults):
        fault_column = int(rng.integers(max(1, width // 10), max(2, 9 * width // 10)))
        throw = rng.uniform(-0.04, 0.04) * height
        vertical[:, fault_column:] += throw

    sand = _warp(sand_base, vertical, horizontal)
    porosity = _warp(porosity_base_array, vertical, horizontal)
    rgt = _warp(rgt_base_array, vertical, horizontal)

    sand += rng.uniform(0.05, 0.18) * _normalized_correlated_noise((height, width), rng, (3.0, 12.0))
    porosity += rng.uniform(0.005, 0.025) * _normalized_correlated_noise((height, width), rng, (4.0, 10.0))
    sand = np.clip(sand, 0.0, 1.0)
    porosity = np.clip(porosity, 0.02, 0.45)
    rgt = np.maximum.accumulate(rgt, axis=0)
    plume = _plume_mask(sand, rng, count=int(rng.integers(1, 3)))

    return SyntheticGeology(
        sand_probability=sand.astype(np.float32),
        porosity=porosity.astype(np.float32),
        rgt=rgt.astype(np.float32),
        delta=delta_from_sand_probability(sand).astype(np.float32),
        plume_mask=plume.astype(np.uint8),
        scale_metadata={"seed": seed, "fault_count": n_faults, "field_conditioned_family": 1},
    )
