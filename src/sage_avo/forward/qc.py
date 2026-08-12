"""Forward-operator agreement and field consistency metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardAgreement:
    scale: tuple[float, ...]
    correlation: tuple[float, ...]
    normalized_rmse: tuple[float, ...]


def compare_forward_outputs(reference: np.ndarray, candidate: np.ndarray) -> ForwardAgreement:
    """Compare equivalent ``[band, time, trace]`` forward outputs per band.

    Use this same function for Torch-versus-Madagascar validation and for field
    forward QC. It does not imply that either input is ground truth.
    """
    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("reference and candidate must be matching [band, time, trace] arrays")
    scales: list[float] = []
    correlations: list[float] = []
    normalized_rmse: list[float] = []
    for band in range(first.shape[0]):
        x = first[band].reshape(-1)
        y = second[band].reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        scale = float(np.dot(x, y) / max(np.dot(y, y), 1e-12))
        scaled = scale * y
        correlation = float(np.corrcoef(x, scaled)[0, 1]) if x.std() > 0 and scaled.std() > 0 else float("nan")
        nrmse = float(np.sqrt(np.mean((x - scaled) ** 2)) / max(x.std(), 1e-12))
        scales.append(scale)
        correlations.append(correlation)
        normalized_rmse.append(nrmse)
    return ForwardAgreement(tuple(scales), tuple(correlations), tuple(normalized_rmse))
