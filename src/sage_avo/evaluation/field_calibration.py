"""Explicit field-domain diagnostics, transfer, and inference guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import hilbert


@dataclass(frozen=True)
class FieldTransferSpecification:
    """Scientist-supplied amplitude, phase, polarity, and optional FIR transfer."""

    transfer_id: str
    gain_by_band: tuple[float, float, float]
    phase_degrees_by_band: tuple[float, float, float]
    polarity_by_band: tuple[int, int, int]
    filter_taps_by_band: tuple[tuple[float, ...], ...] | None = None

    def validate(self) -> None:
        if not self.transfer_id:
            raise ValueError("transfer_id must not be empty")
        if any(value <= 0.0 for value in self.gain_by_band):
            raise ValueError("Every field-transfer gain must be positive")
        if any(value not in {-1, 1} for value in self.polarity_by_band):
            raise ValueError("Field-transfer polarities must be -1 or +1")
        if self.filter_taps_by_band is not None:
            if len(self.filter_taps_by_band) != 3:
                raise ValueError("filter_taps_by_band must contain three filters")
            for taps in self.filter_taps_by_band:
                if not taps or not np.isfinite(taps).all():
                    raise ValueError("Every field-transfer FIR filter must be finite and non-empty")


@dataclass(frozen=True)
class FieldDiagnosticThresholds:
    """Explicit acceptance limits; no implicit calibration thresholds exist."""

    minimum_polarity_correlation: float
    maximum_absolute_phase_degrees: float
    amplitude_scale_minimum: float
    amplitude_scale_maximum: float
    minimum_normalized_percentile_overlap: float
    maximum_spatial_rms_coefficient_of_variation: float
    maximum_spectral_peak_difference_hz: float

    def validate(self) -> None:
        if not -1.0 <= self.minimum_polarity_correlation <= 1.0:
            raise ValueError("minimum_polarity_correlation must lie in [-1, 1]")
        if self.maximum_absolute_phase_degrees < 0.0:
            raise ValueError("maximum_absolute_phase_degrees must be non-negative")
        if not 0.0 < self.amplitude_scale_minimum <= self.amplitude_scale_maximum:
            raise ValueError("Amplitude-scale bounds must be positive and ordered")
        if not 0.0 <= self.minimum_normalized_percentile_overlap <= 1.0:
            raise ValueError("Percentile overlap threshold must lie in [0, 1]")
        if self.maximum_spatial_rms_coefficient_of_variation < 0.0:
            raise ValueError("Spatial RMS variation threshold must be non-negative")
        if self.maximum_spectral_peak_difference_hz < 0.0:
            raise ValueError("Spectral peak tolerance must be non-negative")


def apply_field_transfer(
    field_avo: np.ndarray,
    specification: FieldTransferSpecification,
) -> np.ndarray:
    """Apply only the explicitly saved field-to-synthetic domain transfer."""
    specification.validate()
    data = np.asarray(field_avo, dtype=float)
    if data.ndim != 3 or data.shape[0] != 3 or not np.isfinite(data).all():
        raise ValueError("field_avo must be finite with shape [3, time, trace]")
    output = np.empty_like(data)
    for band in range(3):
        phase = specification.phase_degrees_by_band[band]
        rotated = (
            np.real(hilbert(data[band], axis=0) * np.exp(1j * np.deg2rad(phase)))
            if phase
            else data[band]
        )
        if specification.filter_taps_by_band is not None:
            rotated = convolve1d(
                rotated,
                np.asarray(specification.filter_taps_by_band[band], dtype=float),
                axis=0,
                mode="constant",
                cval=0.0,
            )
        output[band] = (
            specification.polarity_by_band[band]
            * specification.gain_by_band[band]
            * rotated
        )
    return output.astype(np.float32)


def _spectral_summary(data: np.ndarray, dt_seconds: float) -> dict[str, float]:
    demeaned = np.asarray(data, dtype=float) - np.mean(data, axis=0, keepdims=True)
    spectrum = np.mean(np.abs(np.fft.rfft(demeaned, axis=0)) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(demeaned.shape[0], d=dt_seconds)
    if spectrum.size:
        spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total <= 0.0:
        return {
            "peak_frequency_hz": 0.0,
            "energy_band_low_hz": 0.0,
            "energy_band_high_hz": 0.0,
        }
    cumulative = np.cumsum(spectrum) / total
    return {
        "peak_frequency_hz": float(frequencies[int(np.argmax(spectrum))]),
        "energy_band_low_hz": float(frequencies[np.searchsorted(cumulative, 0.05)]),
        "energy_band_high_hz": float(frequencies[np.searchsorted(cumulative, 0.95)]),
    }


def _percentile_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_limits = np.percentile(first, (1.0, 99.0))
    second_limits = np.percentile(second, (1.0, 99.0))
    intersection = max(
        0.0,
        min(first_limits[1], second_limits[1])
        - max(first_limits[0], second_limits[0]),
    )
    union = max(first_limits[1], second_limits[1]) - min(
        first_limits[0], second_limits[0]
    )
    return float(intersection / union) if union > 0.0 else float(first_limits[0] == second_limits[0])


def _spatial_rms_stability(data: np.ndarray, windows: int) -> float:
    pieces = [piece for piece in np.array_split(data, windows, axis=1) if piece.size]
    rms = np.asarray([np.sqrt(np.mean(piece**2)) for piece in pieces])
    return float(rms.std() / max(rms.mean(), 1e-12))


def field_domain_diagnostics(
    field_avo: np.ndarray,
    synthetic_reference_avo: np.ndarray,
    *,
    dt_seconds: float,
    synthetic_x_mean: tuple[float, float, float] | list[float],
    synthetic_x_std: tuple[float, float, float] | list[float],
    thresholds: FieldDiagnosticThresholds | None = None,
    spatial_windows: int = 8,
) -> dict[str, Any]:
    """Compare transferred field AVA with a paired synthetic-domain reference.

    Phase and polarity are meaningful only when both inputs share a registered
    grid. Passing arrays with different shapes leaves those checks unresolved
    and therefore cannot produce a passing calibration report.
    """
    field = np.asarray(field_avo, dtype=float)
    reference = np.asarray(synthetic_reference_avo, dtype=float)
    if field.ndim != 3 or reference.ndim != 3 or field.shape[0] != 3 or reference.shape[0] != 3:
        raise ValueError("Field and reference AVA must have shape [3, time, trace]")
    if dt_seconds <= 0.0 or spatial_windows < 1:
        raise ValueError("dt_seconds and spatial_windows must be positive")
    means = np.asarray(synthetic_x_mean, dtype=float)
    scales = np.asarray(synthetic_x_std, dtype=float)
    if means.shape != (3,) or scales.shape != (3,) or np.any(scales <= 0.0):
        raise ValueError("Synthetic normalization must contain three positive scales")
    paired = field.shape == reference.shape
    band_reports = []
    all_checks: list[bool] = []
    if thresholds is not None:
        thresholds.validate()
    for band in range(3):
        field_values = field[band]
        reference_values = reference[band]
        field_rms = float(np.sqrt(np.mean(field_values**2)))
        reference_rms = float(np.sqrt(np.mean(reference_values**2)))
        amplitude_scale = field_rms / max(reference_rms, 1e-12)
        field_spectrum = _spectral_summary(field_values, dt_seconds)
        reference_spectrum = _spectral_summary(reference_values, dt_seconds)
        if paired:
            first = field_values.reshape(-1)
            second = reference_values.reshape(-1)
            correlation = float(np.corrcoef(first, second)[0, 1])
            field_fft = np.fft.rfft(field_values - field_values.mean(axis=0), axis=0)
            reference_fft = np.fft.rfft(
                reference_values - reference_values.mean(axis=0), axis=0
            )
            cross_spectrum = np.sum(field_fft[1:] * np.conj(reference_fft[1:]))
            phase_degrees = float(np.angle(cross_spectrum, deg=True))
        else:
            correlation = phase_degrees = float("nan")
        normalized_field = (field_values - means[band]) / scales[band]
        normalized_reference = (reference_values - means[band]) / scales[band]
        overlap = _percentile_overlap(normalized_field, normalized_reference)
        spatial_cv = _spatial_rms_stability(field_values, spatial_windows)
        peak_difference = abs(
            field_spectrum["peak_frequency_hz"]
            - reference_spectrum["peak_frequency_hz"]
        )
        checks: dict[str, bool] = {}
        if thresholds is not None:
            checks = {
                "polarity": bool(
                    paired
                    and np.isfinite(correlation)
                    and correlation >= thresholds.minimum_polarity_correlation
                ),
                "phase": bool(
                    paired
                    and np.isfinite(phase_degrees)
                    and abs(phase_degrees)
                    <= thresholds.maximum_absolute_phase_degrees
                ),
                "amplitude_scale": bool(
                    thresholds.amplitude_scale_minimum
                    <= amplitude_scale
                    <= thresholds.amplitude_scale_maximum
                ),
                "percentile_overlap": bool(
                    overlap >= thresholds.minimum_normalized_percentile_overlap
                ),
                "spatial_stability": bool(
                    spatial_cv
                    <= thresholds.maximum_spatial_rms_coefficient_of_variation
                ),
                "spectrum": bool(
                    peak_difference <= thresholds.maximum_spectral_peak_difference_hz
                ),
            }
            all_checks.extend(checks.values())
        band_reports.append(
            {
                "band_index": band,
                "paired_grid": paired,
                "polarity_correlation": correlation,
                "phase_difference_degrees": phase_degrees,
                "field_rms": field_rms,
                "reference_rms": reference_rms,
                "field_to_reference_amplitude_scale": amplitude_scale,
                "normalized_percentile_overlap": overlap,
                "spatial_rms_coefficient_of_variation": spatial_cv,
                "spectral_peak_difference_hz": peak_difference,
                "field_spectrum": field_spectrum,
                "reference_spectrum": reference_spectrum,
                "checks": checks,
            }
        )
    return {
        "status": (
            "pass"
            if thresholds is not None and all(all_checks) and all_checks
            else "fail"
            if thresholds is not None
            else "diagnostic_only"
        ),
        "paired_grid": paired,
        "dt_seconds": float(dt_seconds),
        "spatial_windows": int(spatial_windows),
        "thresholds": asdict(thresholds) if thresholds is not None else None,
        "bands": band_reports,
    }


def save_field_calibration_manifest(
    path: str | Path,
    *,
    diagnostics: dict[str, Any],
    transfer: FieldTransferSpecification,
    forward_specification_sha256: str,
    approved_by: str,
) -> dict[str, Any]:
    """Persist an explicitly approved calibration that passes all diagnostics."""
    transfer.validate()
    if diagnostics.get("status") != "pass":
        raise ValueError("A field calibration manifest requires passing explicit diagnostics")
    if not approved_by.strip():
        raise ValueError("approved_by must identify the calibration authority")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "approved_by": approved_by,
        "forward_specification_sha256": forward_specification_sha256,
        "transfer": asdict(transfer),
        "diagnostics": diagnostics,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_passing_field_calibration(
    path: str | Path,
    *,
    expected_forward_specification_sha256: str,
) -> dict[str, Any]:
    """Load a calibration or block field inference with an exact reason."""
    source = Path(path)
    if not source.exists():
        raise RuntimeError(
            "Field inference is blocked: no saved calibration manifest is available"
        )
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("diagnostics", {}).get("status") != "pass":
        raise RuntimeError("Field inference is blocked: calibration diagnostics did not pass")
    if not str(manifest.get("approved_by", "")).strip():
        raise RuntimeError("Field inference is blocked: calibration has no recorded approval")
    if manifest.get("forward_specification_sha256") != expected_forward_specification_sha256:
        raise RuntimeError(
            "Field inference is blocked: calibration and model forward specifications differ"
        )
    return manifest


def prepare_calibrated_field_input(
    field_avo: np.ndarray,
    *,
    calibration_manifest: str | Path,
    expected_forward_specification_sha256: str,
    normalization: dict[str, list[float]],
) -> np.ndarray:
    """Validate, transfer, then normalize field AVA for model inference."""
    transferred = prepare_calibrated_field_observation(
        field_avo,
        calibration_manifest=calibration_manifest,
        expected_forward_specification_sha256=expected_forward_specification_sha256,
    )
    mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    standard_deviation = np.asarray(
        normalization["x_std"], dtype=np.float32
    )[:, None, None]
    if np.any(standard_deviation <= 0.0):
        raise ValueError("Training input standard deviations must be positive")
    return (transferred - mean) / standard_deviation


def prepare_calibrated_field_observation(
    field_avo: np.ndarray,
    *,
    calibration_manifest: str | Path,
    expected_forward_specification_sha256: str,
) -> np.ndarray:
    """Validate and transfer field AVA before a normalizing inference wrapper."""
    manifest = load_passing_field_calibration(
        calibration_manifest,
        expected_forward_specification_sha256=expected_forward_specification_sha256,
    )
    transfer = FieldTransferSpecification(**manifest["transfer"])
    return apply_field_transfer(field_avo, transfer)
