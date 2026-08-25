"""Controlled model variants for the five-condition benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sage_avo.forward.specification import forward_specification_from_mapping

from .sage_avo import SAGEAVO


LEARNED_VARIANTS = ("full", "no_gnn", "no_rgt", "no_physics")
ALL_VARIANTS = ("low_prior",) + LEARNED_VARIANTS


@dataclass(frozen=True)
class VariantDefinition:
    name: str
    graph_mode: str | None
    physics_weight: float | None


def sage_avo_model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve one shared training/inference model contract from configuration."""
    model = config["model"]
    training = config["training"]
    shared_specification = (
        forward_specification_from_mapping(config) if "forward_model" in config else None
    )
    if shared_specification is None:
        forward = training["physics_forward"]
        angles_definition = forward["angles_degrees"]
        start = float(angles_definition["start"])
        stop = float(angles_definition["stop"])
        step = float(angles_definition["step"])
        count = int(round((stop - start) / step)) + 1
        physics_angles = tuple(start + index * step for index in range(count))
        physics_bands = tuple(
            tuple(float(value) for value in band) for band in forward["bands_degrees"]
        )
        wavelet_hz = float(forward["wavelet_hz"])
        dt_seconds = float(forward["dt_seconds"])
        wavelet_samples = int(forward["wavelet_samples"])
        apply_mute = bool(forward["front_mute"]["enabled"])
        mute_start = tuple(float(value) for value in forward["front_mute"]["start"])
        mute_end = tuple(float(value) for value in forward["front_mute"]["end"])
        taper_samples = int(forward["front_mute"]["taper_samples"])
    else:
        physics_angles = shared_specification.angles_degrees
        physics_bands = tuple(
            (band.minimum_degrees, band.maximum_degrees)
            for band in shared_specification.bands
        )
        wavelet_hz = shared_specification.wavelets[0].peak_frequency_hz
        dt_seconds = shared_specification.dt_seconds
        wavelet_samples = shared_specification.wavelets[0].samples
        apply_mute = shared_specification.apply_mute
        mute_start = shared_specification.mute_start
        mute_end = shared_specification.mute_end
        taper_samples = shared_specification.taper_samples
    guidance = training["physics_guided_sampling"]
    representative_angles = tuple(
        (minimum + maximum) / 2.0 for minimum, maximum in physics_bands
    )
    configured_representatives = tuple(
        float(value) for value in model["representative_angles_degrees"]
    )
    if configured_representatives != representative_angles:
        raise ValueError(
            "model.representative_angles_degrees must equal the configured "
            f"angle-band midpoints {representative_angles}"
        )
    return {
        "hidden_channels": int(model["hidden_channels"]),
        "graph_layers": int(model["graph_layers"]),
        "graph_heads": int(model["graph_heads"]),
        "max_rgt_shift": int(model["max_rgt_shift_samples"]),
        "classes": int(model["classes"]),
        "representative_angles": representative_angles,
        "physics_angles_degrees": physics_angles,
        "physics_bands_degrees": physics_bands,
        "physics_wavelet_hz": wavelet_hz,
        "physics_dt_seconds": dt_seconds,
        "physics_wavelet_samples": wavelet_samples,
        "physics_apply_mute": apply_mute,
        "physics_mute_start": mute_start,
        "physics_mute_end": mute_end,
        "physics_taper_samples": taper_samples,
        "guidance_start_fraction": float(guidance["start_fraction"]),
        "guidance_interval_steps": int(guidance["interval_steps"]),
        "residual_trust_region_scales": (
            tuple(
                float(value)
                for value in training["residual_trust_region"]["normalized_scales"]
            )
            if bool(training.get("residual_trust_region", {}).get("enabled", False))
            else None
        ),
    }


def variant_definition(name: str, physics_weight: float = 0.5) -> VariantDefinition:
    """Return the only intended difference for a controlled condition."""
    definitions = {
        "low_prior": VariantDefinition("low_prior", None, None),
        "full": VariantDefinition("full", "rgt", physics_weight),
        "no_gnn": VariantDefinition("no_gnn", "none", physics_weight),
        "no_rgt": VariantDefinition("no_rgt", "cartesian", physics_weight),
        "no_physics": VariantDefinition("no_physics", "rgt", 0.0),
    }
    if name not in definitions:
        raise ValueError(f"Unknown variant {name!r}; expected one of {tuple(definitions)}")
    return definitions[name]


def build_sage_avo_variant(
    name: str,
    *,
    hidden_channels: int = 64,
    graph_layers: int = 2,
    graph_heads: int = 4,
    max_rgt_shift: int = 3,
    classes: int = 3,
    representative_angles: tuple[float, float, float] = (10.0, 24.0, 38.0),
    physics_angles_degrees: tuple[float, ...] = tuple(float(value) for value in range(3, 46)),
    physics_bands_degrees: tuple[tuple[float, float], ...] = (
        (3.0, 17.0),
        (17.0, 31.0),
        (31.0, 45.0),
    ),
    physics_wavelet_hz: float = 14.0,
    physics_dt_seconds: float = 0.004,
    physics_wavelet_samples: int = 81,
    physics_apply_mute: bool = True,
    physics_mute_start: tuple[float, float] = (30.0, 0.0),
    physics_mute_end: tuple[float, float] = (45.0, 0.1),
    physics_taper_samples: int = 5,
    guidance_start_fraction: float = 1.0 / 3.0,
    guidance_interval_steps: int = 3,
    residual_trust_region_scales: tuple[float, float, float] | None = None,
) -> SAGEAVO:
    """Build a learned ablation with all shared hyperparameters held constant."""
    definition = variant_definition(name)
    if definition.graph_mode is None:
        raise ValueError("low_prior has no neural model")
    return SAGEAVO(
        hidden_channels=hidden_channels,
        graph_layers=graph_layers,
        graph_heads=graph_heads,
        max_rgt_shift=max_rgt_shift,
        graph_mode=definition.graph_mode,
        classes=classes,
        representative_angles=representative_angles,
        physics_angles_degrees=physics_angles_degrees,
        physics_bands_degrees=physics_bands_degrees,
        physics_wavelet_hz=physics_wavelet_hz,
        physics_dt_seconds=physics_dt_seconds,
        physics_wavelet_samples=physics_wavelet_samples,
        physics_apply_mute=physics_apply_mute,
        physics_mute_start=physics_mute_start,
        physics_mute_end=physics_mute_end,
        physics_taper_samples=physics_taper_samples,
        guidance_start_fraction=guidance_start_fraction,
        guidance_interval_steps=guidance_interval_steps,
        residual_trust_region_scales=residual_trust_region_scales,
    )
