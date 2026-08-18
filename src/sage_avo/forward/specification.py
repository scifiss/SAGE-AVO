"""One serializable forward-model contract shared by Stage 02 and Stage 04."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy.signal import hilbert

from .stacks import AngleBand, validate_bands
from .wavelet import ricker


@dataclass(frozen=True)
class WaveletSpecification:
    """Wavelet metadata retained for every synthetic observation."""

    wavelet_id: str = "ricker_14hz_zero_phase"
    kind: str = "ricker"
    peak_frequency_hz: float = 14.0
    bandwidth_hz: float | None = None
    phase_degrees: float = 0.0
    amplitude: float = 1.0
    samples: int = 81
    angle_minimum_degrees: float | None = None
    angle_maximum_degrees: float | None = None

    def samples_array(self, dt_seconds: float) -> np.ndarray:
        if self.kind != "ricker":
            raise ValueError(f"Unsupported wavelet kind {self.kind!r}")
        base = ricker(self.peak_frequency_hz, dt_seconds, self.samples)
        if self.phase_degrees:
            analytic = hilbert(base)
            base = np.real(analytic * np.exp(1j * np.deg2rad(self.phase_degrees)))
            base = base / max(np.sum(np.abs(base)), 1e-12)
        return (self.amplitude * base).astype(np.float64)

    def applies_to(self, angle_degrees: float) -> bool:
        lower = -np.inf if self.angle_minimum_degrees is None else self.angle_minimum_degrees
        upper = np.inf if self.angle_maximum_degrees is None else self.angle_maximum_degrees
        return bool(lower <= angle_degrees <= upper)


@dataclass(frozen=True)
class ForwardModelSpecification:
    """Complete physics, sampling, convolution, mute, and stacking contract."""

    specification_id: str
    angles_degrees: tuple[float, ...]
    bands: tuple[AngleBand, ...]
    dt_seconds: float
    wavelets: tuple[WaveletSpecification, ...]
    convolution_mode: str = "constant_zero_same"
    apply_mute: bool = True
    mute_start: tuple[float, float] = (30.0, 0.0)
    mute_end: tuple[float, float] = (45.0, 0.1)
    taper_samples: int = 5
    mute_time_origin_seconds: float = 0.0
    band_endpoint_convention: str = "inclusive_shared_endpoints"
    amplitude_normalization: str = "none_before_training_statistics"

    def __post_init__(self) -> None:
        if not self.specification_id:
            raise ValueError("specification_id must not be empty")
        angles = np.asarray(self.angles_degrees, dtype=float)
        if angles.ndim != 1 or angles.size == 0 or not np.all(np.diff(angles) > 0):
            raise ValueError("angles_degrees must be a non-empty increasing sequence")
        if angles[0] < 0.0 or angles[-1] > 55.0:
            raise ValueError("Configured PP angles must remain within 0–55 degrees")
        validate_bands(self.bands)
        if self.dt_seconds <= 0.0 or self.taper_samples < 0:
            raise ValueError("dt_seconds must be positive and taper_samples non-negative")
        if self.convolution_mode != "constant_zero_same":
            raise ValueError("Only explicit constant-zero same-length convolution is supported")
        if not self.wavelets:
            raise ValueError("At least one wavelet must be configured")
        for angle in angles:
            matches = [wavelet for wavelet in self.wavelets if wavelet.applies_to(float(angle))]
            if len(matches) != 1:
                raise ValueError(
                    f"Angle {angle:g} must match exactly one wavelet; found {len(matches)}"
                )

    @property
    def maximum_wavelet_half_length(self) -> int:
        return max(wavelet.samples // 2 for wavelet in self.wavelets)

    def wavelet_for_angle(self, angle_degrees: float) -> WaveletSpecification:
        matches = [wavelet for wavelet in self.wavelets if wavelet.applies_to(angle_degrees)]
        if len(matches) != 1:
            raise ValueError(f"Angle {angle_degrees:g} does not map uniquely to a wavelet")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "angles_degrees": list(self.angles_degrees),
            "bands": [asdict(band) for band in self.bands],
            "dt_seconds": self.dt_seconds,
            "wavelets": [asdict(wavelet) for wavelet in self.wavelets],
            "convolution_mode": self.convolution_mode,
            "apply_mute": self.apply_mute,
            "mute_start": list(self.mute_start),
            "mute_end": list(self.mute_end),
            "taper_samples": self.taper_samples,
            "mute_time_origin_seconds": self.mute_time_origin_seconds,
            "band_endpoint_convention": self.band_endpoint_convention,
            "amplitude_normalization": self.amplitude_normalization,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _angles(definition: dict[str, Any]) -> tuple[float, ...]:
    start = float(definition["start"])
    stop = float(definition["stop"])
    step = float(definition["step"])
    count = int(round((stop - start) / step)) + 1
    values = tuple(start + index * step for index in range(count))
    if not np.isclose(values[-1], stop):
        raise ValueError("Angle start/stop/step do not form an inclusive grid")
    return values


def forward_specification_from_mapping(config: dict[str, Any]) -> ForwardModelSpecification:
    """Build the shared contract from a v003 Stage-02 or Stage-04 mapping."""
    forward = config.get("forward_model", config.get("forward"))
    if not isinstance(forward, dict):
        raise KeyError("Configuration must contain forward_model or forward")
    bands_mapping = forward.get("bands", forward.get("production_bands"))
    # Mapping insertion order is not a scientific contract: resolved JSON
    # snapshots may be written with sorted keys.  Canonicalize bands by their
    # physical lower angle so YAML and JSON replays are identical.
    ordered_bands = sorted(
        bands_mapping.items(),
        key=lambda item: (float(item[1][0]), float(item[1][1]), str(item[0])),
    )
    bands = tuple(
        AngleBand(str(name), float(limits[0]), float(limits[1]))
        for name, limits in ordered_bands
    )
    wavelet_mapping = forward.get("wavelet_bank")
    if wavelet_mapping is None:
        legacy = forward["wavelet"]
        wavelet_mapping = [
            {
                "wavelet_id": f"ricker_{float(legacy['frequency_hz']):g}hz_zero_phase",
                "kind": legacy.get("type", "ricker"),
                "peak_frequency_hz": legacy["frequency_hz"],
                "phase_degrees": 0.0,
                "amplitude": 1.0,
                "samples": legacy["samples"],
            }
        ]
    wavelets = tuple(WaveletSpecification(**item) for item in wavelet_mapping)
    mute = forward["front_mute"]
    return ForwardModelSpecification(
        specification_id=str(forward.get("specification_id", "legacy_forward_mapping")),
        angles_degrees=_angles(forward["angles_degrees"]),
        bands=bands,
        dt_seconds=float(forward["dt_seconds"]),
        wavelets=wavelets,
        convolution_mode=str(forward.get("convolution_mode", "constant_zero_same")),
        apply_mute=bool(mute["enabled"]),
        mute_start=tuple(float(value) for value in mute["start"]),
        mute_end=tuple(float(value) for value in mute["end"]),
        taper_samples=int(mute["taper_samples"]),
        mute_time_origin_seconds=float(mute.get("time_origin_seconds", 0.0)),
        band_endpoint_convention=str(
            forward.get("band_endpoint_convention", "inclusive_shared_endpoints")
        ),
        amplitude_normalization=str(
            forward.get("amplitude_normalization", "none_before_training_statistics")
        ),
    )
