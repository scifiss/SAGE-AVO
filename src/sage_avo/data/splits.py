"""Leakage-resistant realization-level data splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealizationSplit:
    """Integer realization identifiers assigned to disjoint data splits."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_realizations(
    n_realizations: int,
    fractions: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 12345,
) -> RealizationSplit:
    """Shuffle and split whole realization IDs, preventing patch leakage."""
    if n_realizations < 3:
        raise ValueError("At least three realizations are required")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain three values summing to one")
    if any(value <= 0 for value in fractions):
        raise ValueError("All split fractions must be positive")

    ids = np.arange(n_realizations, dtype=np.int64)
    np.random.default_rng(seed).shuffle(ids)
    n_train = int(fractions[0] * n_realizations)
    n_validation = int(fractions[1] * n_realizations)
    return RealizationSplit(
        train=ids[:n_train],
        validation=ids[n_train : n_train + n_validation],
        test=ids[n_train + n_validation :],
    )
