"""Lightweight relative-geologic-time utilities.

Production dip estimation may use PySeistr/PWD. These functions provide a
dependency-light public implementation for testing and demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class PwdRgtResult:
    """Outputs from two-pass plane-wave-destruction structural processing."""

    dip: np.ndarray
    rgt: np.ndarray
    structure_oriented_seismic: np.ndarray
    reference_sample: int

    def validate(self) -> None:
        if self.dip.ndim != 2:
            raise ValueError("dip must be a 2-D array")
        if self.rgt.shape != self.dip.shape or self.structure_oriented_seismic.shape != self.dip.shape:
            raise ValueError("PWD/RGT arrays must share one [time, cdp] shape")
        if not 0 <= self.reference_sample < self.dip.shape[0]:
            raise ValueError("reference_sample lies outside the time axis")


@dataclass(frozen=True)
class PwdDipResult:
    """Two-pass PWD dip and its structure-oriented seismic image."""

    dip: np.ndarray
    structure_oriented_seismic: np.ndarray

    def validate(self) -> None:
        if self.dip.ndim != 2 or self.structure_oriented_seismic.shape != self.dip.shape:
            raise ValueError("PWD dip arrays must share one [time, cdp] shape")


def _pava_1d(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing projection by pool-adjacent violators."""
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(np.asarray(values, dtype=float)):
        levels.append(float(value))
        weights.append(1.0)
        starts.append(index)
        ends.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (weights[-2] * levels[-2] + weights[-1] * levels[-1]) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    output = np.empty(len(values), dtype=float)
    for level, start, end in zip(levels, starts, ends):
        output[start : end + 1] = level
    return output


def repair_rgt_monotonicity(
    rgt: np.ndarray,
    *,
    minimum_step: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project every RGT trace onto a strictly increasing coordinate.

    The raw PWD/RGT result should be retained separately. This function applies
    a documented least-squares isotonic repair for safe horizon inversion and
    graph construction, and reports the magnitude of the adjustment.
    """
    raw = np.asarray(rgt, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("rgt must be a finite 2-D array")
    if minimum_step < 0:
        raise ValueError("minimum_step must be non-negative")
    rows = np.arange(raw.shape[0], dtype=float)
    repaired = np.empty_like(raw)
    for column in range(raw.shape[1]):
        shifted = raw[:, column] - minimum_step * rows
        repaired[:, column] = _pava_1d(shifted) + minimum_step * rows
    adjustment = repaired - raw
    qc = {
        "adjustment_rmse": float(np.sqrt(np.mean(adjustment**2))),
        "adjustment_max_absolute": float(np.max(np.abs(adjustment))),
        "minimum_step": float(np.min(np.diff(repaired, axis=0))),
    }
    return repaired.astype(np.float32), qc


def monotonicity_report(rgt: np.ndarray, tolerance: float = -1e-3) -> dict[str, float]:
    """Report the fraction and worst magnitude of non-monotonic vertical steps."""
    if rgt.ndim != 2:
        raise ValueError("rgt must be a 2-D array")
    steps = np.diff(rgt, axis=0)
    finite = np.isfinite(steps)
    bad = finite & (steps < tolerance)
    return {
        "fraction_bad": float(bad.sum() / max(finite.sum(), 1)),
        "worst_step": float(np.nanmin(steps)) if finite.any() else float("nan"),
    }


def integrate_dip_to_rgt(
    dip: np.ndarray,
    vertical_increment: float = 1.0,
    smooth_sigma: tuple[float, float] = (1.0, 2.0),
) -> np.ndarray:
    """Construct a monotonic RGT approximation from a 2-D slope field.

    The vertical coordinate supplies the dominant monotonic term. A laterally
    integrated, smoothed slope correction bends equal-RGT surfaces along dip.
    For publication processing, compare this approximation with the chosen PWD
    implementation and retain the monotonicity QC.
    """
    if dip.ndim != 2:
        raise ValueError("dip must be a 2-D array")
    clean = np.nan_to_num(dip, copy=True)
    lateral_shift = np.cumsum(clean, axis=1)
    lateral_shift -= lateral_shift.mean(axis=1, keepdims=True)
    lateral_shift = gaussian_filter(lateral_shift, smooth_sigma)
    vertical = np.arange(dip.shape[0], dtype=np.float64)[:, None] * vertical_increment
    rgt = vertical - lateral_shift
    minimum_step = max(vertical_increment * 1e-4, 1e-8)
    return np.maximum.accumulate(rgt + minimum_step * np.arange(rgt.shape[0])[:, None], axis=0)


def estimate_pwd_rgt(
    seismic: np.ndarray,
    time_ms: np.ndarray,
    *,
    gaussian_sigma: tuple[float, float] = (1.0, 0.0),
    dip_order: int = 2,
    dip_iterations: int = 14,
    dip_rect: tuple[int, int, int] = (6, 14, 1),
    structure_radius: int = 2,
    structure_order: int = 2,
    structure_epsilon: float = 0.01,
    rgt_epsilon: float = 0.08,
    reference_sample: int | None = None,
) -> PwdRgtResult:
    """Estimate local slope and RGT with the production PySeistr sequence.

    This is the algorithm used by the source S01 notebook: first-pass PWD dip,
    structure-oriented mean filtering, refined PWD dip, then RGT integration.
    The dependency-light :func:`integrate_dip_to_rgt` remains useful for tests
    and demonstrations, but it is not silently substituted here.
    """
    image = np.asarray(seismic, dtype=np.float32)
    times = np.asarray(time_ms, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("seismic must have shape [time, cdp]")
    if times.ndim != 1 or times.size != image.shape[0]:
        raise ValueError("time_ms must match the seismic time axis")
    if times.size < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("time_ms must be strictly increasing")
    if not np.isfinite(image).all():
        raise ValueError("seismic contains non-finite samples; repair dead CDPs before PWD")

    dip_result = estimate_pwd_dip(
        image,
        gaussian_sigma=gaussian_sigma,
        dip_order=dip_order,
        dip_iterations=dip_iterations,
        dip_rect=dip_rect,
        structure_radius=structure_radius,
        structure_order=structure_order,
        structure_epsilon=structure_epsilon,
    )
    refined_dip = dip_result.dip
    structure_oriented = dip_result.structure_oriented_seismic
    try:
        from pyseistr.rgt import rgt as rgt_from_dip
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Production PWD/RGT processing requires pyseistr. Install the field "
            "extra with `python -m pip install -e \".[field]\"`."
        ) from error
    if reference_sample is None:
        reference_sample = int(np.argmax(np.mean(np.abs(structure_oriented), axis=1)))
    rgt_map = rgt_from_dip(
        refined_dip,
        o1=float(times[0]) / 1000.0,
        d1=float(np.median(np.diff(times))) / 1000.0,
        order=2,
        i0=int(reference_sample),
        eps=float(rgt_epsilon),
        verb=False,
    )
    result = PwdRgtResult(
        dip=np.asarray(refined_dip, dtype=np.float32),
        rgt=np.asarray(rgt_map, dtype=np.float32),
        structure_oriented_seismic=np.asarray(structure_oriented, dtype=np.float32),
        reference_sample=int(reference_sample),
    )
    result.validate()
    return result


def estimate_pwd_dip(
    seismic: np.ndarray,
    *,
    gaussian_sigma: tuple[float, float] = (1.0, 0.0),
    dip_order: int = 2,
    dip_iterations: int = 14,
    dip_rect: tuple[int, int, int] = (6, 14, 1),
    structure_radius: int = 2,
    structure_order: int = 2,
    structure_epsilon: float = 0.01,
) -> PwdDipResult:
    """Run the historical two-pass PySeistr PWD dip sequence without RGT integration."""
    image = np.asarray(seismic, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("seismic must be a finite [time, cdp] array")
    try:
        from pyseistr import somean2d
        from pyseistr.dip2d import dip2dc
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Production PWD dip processing requires pyseistr. Install the field "
            "extra with `python -m pip install -e \".[field]\"`."
        ) from error
    first_input = gaussian_filter(image, sigma=gaussian_sigma)
    first_dip = dip2dc(
        first_input,
        order=dip_order,
        niter=dip_iterations,
        rect=list(dip_rect),
        verb=0,
    )
    structure_oriented = somean2d(
        image,
        first_dip,
        structure_radius,
        structure_order,
        structure_epsilon,
    )
    refined_input = gaussian_filter(
        np.asarray(structure_oriented, dtype=np.float32), sigma=gaussian_sigma
    )
    refined_dip = dip2dc(
        refined_input,
        order=dip_order,
        niter=dip_iterations,
        rect=list(dip_rect),
        verb=0,
    )
    result = PwdDipResult(
        dip=np.asarray(refined_dip, dtype=np.float32),
        structure_oriented_seismic=np.asarray(structure_oriented, dtype=np.float32),
    )
    result.validate()
    return result


def save_pwd_rgt(path: str | Path, result: PwdRgtResult) -> None:
    """Cache production PWD/RGT outputs without embedding them in a notebook."""
    result.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        dip=result.dip,
        rgt=result.rgt,
        structure_oriented_seismic=result.structure_oriented_seismic,
        reference_sample=np.asarray(result.reference_sample, dtype=np.int32),
    )


def load_pwd_rgt(path: str | Path) -> PwdRgtResult:
    """Load and validate cached production PWD/RGT outputs."""
    with np.load(Path(path), allow_pickle=False) as archive:
        result = PwdRgtResult(
            dip=archive["dip"],
            rgt=archive["rgt"],
            structure_oriented_seismic=archive["structure_oriented_seismic"],
            reference_sample=int(archive["reference_sample"]),
        )
    result.validate()
    return result
