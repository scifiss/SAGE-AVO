"""Observation-domain perturbations applied only after physical forward modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert


@dataclass(frozen=True)
class ObservationPerturbationConfig:
    enabled: bool = False
    white_noise_fraction_by_band: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colored_noise_fraction_by_band: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colored_noise_sigma_samples: tuple[float, float] = (2.0, 4.0)
    coherent_noise_fraction: float = 0.0
    coherent_frequency_cycles_per_sample: tuple[float, float] = (0.01, 0.05)
    coherent_slope_samples_per_trace: tuple[float, float] = (-0.2, 0.2)
    gain_range_by_band: tuple[tuple[float, float], ...] = (
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    )
    phase_degrees_by_band: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    polarity_flip_probability_by_band: tuple[float, float, float] = (0.0, 0.0, 0.0)
    far_angle_weakening_range: tuple[float, float] = (1.0, 1.0)
    far_angle_missing_probability: float = 0.0

    def validate(self) -> None:
        triples = (
            self.white_noise_fraction_by_band,
            self.colored_noise_fraction_by_band,
            self.gain_range_by_band,
            self.phase_degrees_by_band,
            self.polarity_flip_probability_by_band,
        )
        if any(len(value) != 3 for value in triples):
            raise ValueError("Every band-dependent observation setting must have length three")
        probabilities = (*self.polarity_flip_probability_by_band, self.far_angle_missing_probability)
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Observation probabilities must lie between zero and one")
        if any(value < 0.0 for value in self.white_noise_fraction_by_band):
            raise ValueError("Noise fractions must be non-negative")
        if any(value < 0.0 for value in self.colored_noise_fraction_by_band):
            raise ValueError("Noise fractions must be non-negative")


@dataclass(frozen=True)
class PerturbedObservation:
    stacks: np.ndarray
    metadata: dict[str, object]


def observation_config_from_mapping(mapping: dict[str, object]) -> ObservationPerturbationConfig:
    """Parse the explicit v003 observation-perturbation configuration."""
    config = ObservationPerturbationConfig(
        enabled=bool(mapping.get("enabled", False)),
        white_noise_fraction_by_band=tuple(mapping["white_noise_fraction_by_band"]),
        colored_noise_fraction_by_band=tuple(mapping["colored_noise_fraction_by_band"]),
        colored_noise_sigma_samples=tuple(mapping["colored_noise_sigma_samples"]),
        coherent_noise_fraction=float(mapping["coherent_noise_fraction"]),
        coherent_frequency_cycles_per_sample=tuple(
            mapping["coherent_frequency_cycles_per_sample"]
        ),
        coherent_slope_samples_per_trace=tuple(mapping["coherent_slope_samples_per_trace"]),
        gain_range_by_band=tuple(tuple(value) for value in mapping["gain_range_by_band"]),
        phase_degrees_by_band=tuple(
            tuple(value) for value in mapping["phase_degrees_by_band"]
        ),
        polarity_flip_probability_by_band=tuple(
            mapping["polarity_flip_probability_by_band"]
        ),
        far_angle_weakening_range=tuple(mapping["far_angle_weakening_range"]),
        far_angle_missing_probability=float(mapping["far_angle_missing_probability"]),
    )
    config.validate()
    return config


def _phase_rotate(data: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return data
    analytic = hilbert(data, axis=0)
    return np.real(analytic * np.exp(1j * np.deg2rad(degrees)))


def apply_observation_perturbations(
    clean_stacks: np.ndarray,
    rng: np.random.Generator,
    config: ObservationPerturbationConfig,
) -> PerturbedObservation:
    """Apply gain/phase/polarity/noise/far-angle effects after forward modeling."""
    config.validate()
    clean = np.asarray(clean_stacks, dtype=float)
    if clean.ndim != 3 or clean.shape[0] != 3:
        raise ValueError("clean_stacks must have shape [3, time, trace]")
    if not config.enabled:
        return PerturbedObservation(
            clean.astype(np.float32, copy=True),
            {"enabled": False, "config": asdict(config)},
        )
    output = clean.copy()
    reference_std = np.std(clean, axis=(1, 2))
    gains = []
    phases = []
    polarities = []
    for band in range(3):
        gain = float(rng.uniform(*config.gain_range_by_band[band]))
        phase = float(rng.uniform(*config.phase_degrees_by_band[band]))
        polarity = -1.0 if rng.random() < config.polarity_flip_probability_by_band[band] else 1.0
        output[band] = polarity * gain * _phase_rotate(output[band], phase)
        gains.append(gain)
        phases.append(phase)
        polarities.append(int(polarity))

    coherent = np.zeros(clean.shape[1:], dtype=float)
    if config.coherent_noise_fraction > 0.0:
        rows, columns = np.indices(clean.shape[1:])
        frequency = float(rng.uniform(*config.coherent_frequency_cycles_per_sample))
        slope = float(rng.uniform(*config.coherent_slope_samples_per_trace))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        coherent = np.sin(2.0 * np.pi * frequency * (rows - slope * columns) + phase)
    else:
        frequency = slope = phase = 0.0

    for band in range(3):
        white_fraction = config.white_noise_fraction_by_band[band]
        if white_fraction > 0.0:
            output[band] += rng.standard_normal(clean.shape[1:]) * (
                white_fraction * reference_std[band]
            )
        colored_fraction = config.colored_noise_fraction_by_band[band]
        if colored_fraction > 0.0:
            colored = gaussian_filter(
                rng.standard_normal(clean.shape[1:]),
                sigma=config.colored_noise_sigma_samples,
                mode="reflect",
            )
            colored /= max(colored.std(), 1e-12)
            output[band] += colored * colored_fraction * reference_std[band]
        if config.coherent_noise_fraction > 0.0:
            output[band] += (
                coherent * config.coherent_noise_fraction * reference_std[band]
            )

    far_missing = bool(rng.random() < config.far_angle_missing_probability)
    far_scale = 0.0 if far_missing else float(rng.uniform(*config.far_angle_weakening_range))
    output[2] *= far_scale
    metadata = {
        "enabled": True,
        "config": asdict(config),
        "reference_standard_deviation": reference_std.tolist(),
        "realized_gain_by_band": gains,
        "realized_phase_degrees_by_band": phases,
        "realized_polarity_by_band": polarities,
        "coherent_frequency_cycles_per_sample": frequency,
        "coherent_slope_samples_per_trace": slope,
        "coherent_phase_radians": phase,
        "far_angle_missing": far_missing,
        "far_angle_scale": far_scale,
    }
    return PerturbedObservation(output.astype(np.float32), metadata)
