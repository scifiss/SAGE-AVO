"""Observable dual-domain barrier primitives for v00332w."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def robust_normalize(values, valid, window=9):
    """Local-median residual divided by local MAD; finite and deterministic."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    filled = np.where(valid, values, np.nan)
    # Per-row interpolation avoids allowing invalid support to create barriers.
    for row in range(len(filled)):
        good = np.flatnonzero(np.isfinite(filled[row]))
        if len(good):
            filled[row] = np.interp(np.arange(filled.shape[1]), good, filled[row, good])
        else:
            filled[row] = 0.0
    center = median_filter(filled, size=(1, window), mode="nearest")
    residual = abs(filled - center)
    mad = median_filter(residual, size=(1, window), mode="nearest")
    global_scale = np.median(residual[valid]) if valid.any() else 1.0
    scale = 1.4826 * np.maximum(mad, max(global_scale, 1e-8))
    return residual / scale


def inverse_shift_fields(inverse_t, valid):
    """Physical shift and abrupt-change evidence on adjacent trace boundaries."""
    inverse_t = np.asarray(inverse_t, float)
    valid = np.asarray(valid, bool)
    if inverse_t.shape != valid.shape or inverse_t.shape[1] < 3:
        raise ValueError("Require matching inverse_t/valid grids with at least three traces")
    shift = np.diff(inverse_t, axis=1)
    shift_valid = valid[:, :-1] & valid[:, 1:]
    local = median_filter(shift, size=(1, 9), mode="nearest")
    jump = abs(shift - local)
    second = np.zeros_like(shift)
    second[:, 1:-1] = abs(shift[:, 2:] - 2 * shift[:, 1:-1] + shift[:, :-2])
    return {
        "shift": shift,
        "valid": shift_valid,
        "shift_jump": jump,
        "shift_second_difference": second,
        "normalized_shift_jump": robust_normalize(shift, shift_valid),
        "normalized_shift_second_difference": robust_normalize(second, shift_valid),
    }


def barrier_safe(observables, thresholds):
    """Simple interpretable pre-union barrier; no truth-label input."""
    reflector_match = (
        observables["flattened_waveform_cosine"] >= thresholds["flattened_waveform_cosine"]
        and observables["flattened_phase_cosine"] >= thresholds["flattened_phase_cosine"]
        and observables["strength_ratio"] >= thresholds["strength_ratio"]
    )
    barrier = (
        observables["normalized_shift_jump"] > thresholds["normalized_shift_jump"]
        or observables["normalized_shift_second_difference"]
        > thresholds["normalized_shift_second_difference"]
        or observables["original_waveform_cosine"] < thresholds["original_waveform_cosine"]
        or observables["original_phase_cosine"] < thresholds["original_phase_cosine"]
    )
    return reflector_match, bool(reflector_match and not barrier)
