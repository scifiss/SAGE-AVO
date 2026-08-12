"""Dependency-light scientific metrics with explicit masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _paired(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(prediction, dtype=float)
    second = np.asarray(target, dtype=float)
    if first.shape != second.shape:
        raise ValueError("prediction and target must share a shape")
    valid = np.isfinite(first) & np.isfinite(second)
    if mask is not None:
        valid &= np.broadcast_to(mask, first.shape).astype(bool)
    return first[valid], second[valid]


def elastic_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return RMSE, MAE, R², and correlation over valid samples."""
    predicted, observed = _paired(prediction, target, mask)
    if predicted.size == 0:
        raise ValueError("No valid samples")
    residual = predicted - observed
    variance = np.sum((observed - observed.mean()) ** 2)
    correlation = np.corrcoef(predicted, observed)[0, 1] if predicted.std() and observed.std() else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / variance) if variance > 0 else float("nan"),
        "correlation": float(correlation),
    }


def ssim_2d(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    sigma: float = 1.5,
) -> float:
    """Compute a Gaussian-window structural similarity index for one 2-D map."""
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(target, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must be matching 2-D arrays")
    data_range = float(np.nanpercentile(observed, 99) - np.nanpercentile(observed, 1))
    data_range = max(data_range, 1e-8)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_x = gaussian_filter(predicted, sigma=sigma, mode="reflect")
    mean_y = gaussian_filter(observed, sigma=sigma, mode="reflect")
    variance_x = gaussian_filter(predicted**2, sigma=sigma, mode="reflect") - mean_x**2
    variance_y = gaussian_filter(observed**2, sigma=sigma, mode="reflect") - mean_y**2
    covariance = gaussian_filter(predicted * observed, sigma=sigma, mode="reflect") - mean_x * mean_y
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    similarity = numerator / np.maximum(denominator, 1e-12)
    valid = np.isfinite(similarity)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("No valid samples for SSIM")
    return float(np.mean(similarity[valid]))


def elastic_metrics_with_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return RMSE, MAE, R², correlation, and 2-D SSIM."""
    output = elastic_metrics(prediction, target, mask)
    output["ssim"] = ssim_2d(prediction, target, mask)
    return output


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    classes: int = 3,
) -> dict[str, float]:
    """Return macro intersection-over-union and Dice/F1."""
    predicted = np.asarray(prediction).reshape(-1)
    observed = np.asarray(target).reshape(-1)
    if predicted.shape != observed.shape:
        raise ValueError("prediction and target must share a shape")
    ious: list[float] = []
    dices: list[float] = []
    output: dict[str, float] = {}
    for label in range(classes):
        pred_mask = predicted == label
        true_mask = observed == label
        intersection = np.sum(pred_mask & true_mask)
        union = np.sum(pred_mask | true_mask)
        total = np.sum(pred_mask) + np.sum(true_mask)
        iou = float(intersection / union) if union else float("nan")
        dice = float(2 * intersection / total) if total else float("nan")
        ious.append(iou)
        dices.append(dice)
        output[f"class_{label}_iou"] = iou
    output["miou"] = float(np.nanmean(ious))
    output["macro_dice"] = float(np.nanmean(dices))
    return output
