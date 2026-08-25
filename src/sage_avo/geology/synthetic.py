"""Reproducible field-conditioned synthetic geology.

The generator deliberately exposes its assumptions. It creates diverse members
of one field-conditioned geological family; it does not claim independent
regional geological coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

from .conventions import delta_from_sand_probability
from .field_conditioning import ElasticModelSet, predict_elastic_fields
from .fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    calibrated_differential_gassmann_substitution,
    constrained_local_gassmann_substitution,
    poisson_ratio_from_moduli,
)
from .rock_physics import (
    ElasticProperties,
    elastic_from_gpa,
    elastic_moduli_gpa,
    hertz_mindlin_gassmann,
    local_inverse_gassmann_substitution,
    matched_hm_delta_substitution,
)


@dataclass(frozen=True)
class SyntheticGeology:
    sand_probability: np.ndarray
    porosity: np.ndarray
    rgt: np.ndarray
    delta: np.ndarray
    plume_mask: np.ndarray
    scale_metadata: dict[str, float | int]


@dataclass(frozen=True)
class Deformation:
    """One coherent map used to deform every geological channel."""

    vertical_displacement: np.ndarray
    horizontal_displacement: np.ndarray
    metadata: dict[str, float | int | list[dict[str, float]]]


@dataclass(frozen=True)
class FluidScenario:
    """Gassmann substitution result and its explicit saturation support."""

    elastic: np.ndarray
    plume_mask: np.ndarray
    co2_saturation: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class FieldConditionedRealization:
    """Complete geological state before seismic forward modeling."""

    elastic: np.ndarray
    elastic_brine: np.ndarray
    delta: np.ndarray
    sand_probability: np.ndarray
    porosity: np.ndarray
    rgt: np.ndarray
    strat_fraction: np.ndarray
    reservoir_mask: np.ndarray
    segmentation: np.ndarray
    plume_mask: np.ndarray
    co2_saturation: np.ndarray
    vertical_displacement: np.ndarray
    horizontal_displacement: np.ndarray
    metadata: dict[str, object]


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


def make_deformation(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    fold_probability: float = 0.8,
    fold_amplitude_samples: tuple[float, float] = (10.0, 30.0),
    fold_cycles: tuple[float, float] = (0.3, 3.0),
    secondary_fold_amplitude_samples: tuple[float, float] = (2.0, 8.0),
    secondary_fold_cycles: tuple[float, float] = (2.0, 4.0),
    maximum_faults: int = 7,
    fault_throw_samples: tuple[float, float] = (-40.0, 40.0),
    fault_dip_samples_per_trace: tuple[float, float] = (-0.5, 0.5),
) -> Deformation:
    """Create the coherent fold/fault displacement field."""
    height, width = shape
    rows, columns = np.indices(shape, dtype=np.float32)
    vertical = np.zeros(shape, dtype=np.float32)
    folds_applied = bool(rng.uniform() <= fold_probability)
    if folds_applied:
        amplitude_1 = float(rng.uniform(*fold_amplitude_samples))
        cycles_1 = float(rng.uniform(*fold_cycles))
        amplitude_2 = float(rng.uniform(*secondary_fold_amplitude_samples))
        cycles_2 = float(rng.uniform(*secondary_fold_cycles))
        vertical += amplitude_1 * np.sin(2.0 * np.pi * cycles_1 * columns / max(width, 1))
        vertical += amplitude_2 * np.sin(2.0 * np.pi * cycles_2 * columns / max(width, 1))
    else:
        amplitude_1 = cycles_1 = amplitude_2 = cycles_2 = 0.0

    faults: list[dict[str, float]] = []
    fault_count = int(rng.integers(0, maximum_faults + 1))
    for _ in range(fault_count):
        fault_column = float(rng.uniform(0.2 * width, 0.8 * width))
        throw = float(rng.uniform(*fault_throw_samples))
        dip = float(rng.uniform(*fault_dip_samples_per_trace))
        side = columns > fault_column + dip * rows
        vertical[side] += throw
        faults.append({"column": fault_column, "throw_samples": throw, "dip": dip})

    horizontal = 1.5 * _normalized_correlated_noise(shape, rng, (20.0, 20.0))
    return Deformation(
        vertical_displacement=vertical,
        horizontal_displacement=horizontal.astype(np.float32),
        metadata={
            "folds_applied": int(folds_applied),
            "fold_amplitude_1_samples": amplitude_1,
            "fold_cycles_1": cycles_1,
            "fold_amplitude_2_samples": amplitude_2,
            "fold_cycles_2": cycles_2,
            "fault_count": fault_count,
            "faults": faults,
        },
    )


def warp_with_deformation(array: np.ndarray, deformation: Deformation, *, order: int = 1) -> np.ndarray:
    """Warp a 2-D or channels-first 3-D array with a shared deformation."""
    values = np.asarray(array)
    if values.ndim == 2:
        if order == 1:
            return _warp(values, deformation.vertical_displacement, deformation.horizontal_displacement)
        rows, columns = np.indices(values.shape, dtype=float)
        return map_coordinates(
            values,
            [rows - deformation.vertical_displacement, columns - deformation.horizontal_displacement],
            order=order,
            mode="nearest",
        )
    if values.ndim == 3:
        return np.stack(
            [warp_with_deformation(channel, deformation, order=order) for channel in values], axis=0
        )
    raise ValueError("array must have shape [time, trace] or [channel, time, trace]")


def apply_co2_fluid_substitution(
    elastic_brine: np.ndarray,
    porosity: np.ndarray,
    sand_probability: np.ndarray,
    reservoir_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    plume_count: tuple[int, int] = (1, 2),
    lateral_radius_samples: tuple[float, float] = (15.0, 40.0),
    vertical_radius_samples: tuple[float, float] = (5.0, 15.0),
    minimum_sand_thickness_samples: int = 15,
    sand_threshold: float = 0.5,
    co2_saturation: tuple[float, float] = (0.3, 0.8),
    critical_porosity: float = 0.36,
    coordination_factor: float = 2.8,
    quartz_bulk_modulus_gpa: float = 39.0,
    clay_bulk_modulus_gpa: float = 21.0,
    quartz_shear_modulus_gpa: float = 45.0,
    clay_shear_modulus_gpa: float = 6.85,
    quartz_density_g_cc: float = 2.65,
    clay_density_g_cc: float = 2.60,
    overburden_density_kg_m3: float = 1600.0,
    gravity_m_s2: float = 9.8,
    depth_origin_m: float = 2000.0,
    depth_increment_m: float = 4.0,
    brine_bulk_modulus_gpa: float = 2.2,
    co2_bulk_modulus_gpa: float = 0.1,
    brine_density_g_cc: float = 1.03,
    co2_density_g_cc: float = 0.65,
    brie_exponent: float = 3.0,
) -> FluidScenario:
    """Place reservoir-confined plumes and apply the HM/Gassmann compatibility model."""
    properties = np.asarray(elastic_brine, dtype=float)
    if properties.ndim != 3 or properties.shape[0] != 3:
        raise ValueError("elastic_brine must have shape [3, time, trace]")
    shape = properties.shape[1:]
    facies_sand = (sand_probability >= sand_threshold) & np.asarray(reservoir_mask, dtype=bool)
    core = binary_erosion(facies_sand, structure=np.ones((minimum_sand_thickness_samples, 1)))
    candidates = np.argwhere(core)
    plume = np.zeros(shape, dtype=bool)
    saturation = np.zeros(shape, dtype=np.float32)
    requested = int(rng.integers(plume_count[0], plume_count[1] + 1))
    rows, columns = np.indices(shape)
    if candidates.size:
        for _ in range(requested):
            center_row, center_column = candidates[int(rng.integers(candidates.shape[0]))]
            radius_x = float(rng.uniform(*lateral_radius_samples))
            radius_z = float(rng.uniform(*vertical_radius_samples))
            ellipse = (
                ((columns - center_column) / radius_x) ** 2
                + ((rows - center_row) / radius_z) ** 2
                <= 1.0
            )
            member = ellipse & facies_sand & ~plume
            plume |= member
            saturation[member] = float(rng.uniform(*co2_saturation))

    substituted = hertz_mindlin_gassmann(
        1.0 - np.asarray(sand_probability, dtype=float),
        porosity,
        saturation,
        critical_porosity=critical_porosity,
        coordination_factor=coordination_factor,
        quartz_bulk_modulus_gpa=quartz_bulk_modulus_gpa,
        clay_bulk_modulus_gpa=clay_bulk_modulus_gpa,
        quartz_shear_modulus_gpa=quartz_shear_modulus_gpa,
        clay_shear_modulus_gpa=clay_shear_modulus_gpa,
        quartz_density_g_cc=quartz_density_g_cc,
        clay_density_g_cc=clay_density_g_cc,
        overburden_density_kg_m3=overburden_density_kg_m3,
        gravity_m_s2=gravity_m_s2,
        depth_origin_m=depth_origin_m,
        depth_increment_m=depth_increment_m,
        brine_bulk_modulus_gpa=brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=co2_bulk_modulus_gpa,
        brine_density_g_cc=brine_density_g_cc,
        co2_density_g_cc=co2_density_g_cc,
        brie_exponent=brie_exponent,
    )
    output = properties.copy()
    output[0] = np.where(plume, substituted.vp, output[0])
    output[1] = np.where(plume, substituted.vs, output[1])
    output[2] = np.where(plume, substituted.density, output[2])
    return FluidScenario(
        elastic=output.astype(np.float32),
        plume_mask=plume.astype(np.uint8),
        co2_saturation=saturation,
        metadata={
            "requested_plumes": requested,
            "plume_pixels": int(plume.sum()),
            "brie_exponent": float(brie_exponent),
            "minimum_sand_thickness_samples": int(minimum_sand_thickness_samples),
        },
    )


def apply_co2_fluid_substitution_v003(
    elastic_brine: np.ndarray,
    porosity: np.ndarray,
    sand_probability: np.ndarray,
    reservoir_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "local_inverse_gassmann",
    fluid_calibration: CalibratedDryFrameModel | None = None,
    depth_m: np.ndarray | None = None,
    plume_count: tuple[int, int] = (1, 2),
    lateral_radius_samples: tuple[float, float] = (15.0, 40.0),
    vertical_radius_samples: tuple[float, float] = (5.0, 15.0),
    minimum_sand_thickness_samples: int = 7,
    sand_threshold: float = 0.3,
    co2_saturation: tuple[float, float] = (0.3, 0.8),
    saturation_correlation_sigma_samples: tuple[float, float] = (3.0, 8.0),
    saturation_variation_fraction: float = 0.15,
    vp_bounds_m_s: tuple[float, float] = (1200.0, 6500.0),
    vs_bounds_m_s: tuple[float, float] = (500.0, 4000.0),
    density_bounds_g_cc: tuple[float, float] = (1.5, 3.2),
    **rock_physics: float,
) -> FluidScenario:
    """Place plumes and apply one explicit v003 fluid-substitution mode.

    ``calibrated_differential_gassmann`` is the calibrated production mode.
    ``constrained_local_gassmann`` is retained for bounded comparative QC.
    ``local_inverse_gassmann`` is compatibility-only because its mineral
    projection and dry-frame clipping make it ineligible for production.
    ``matched_hm_delta`` retains only the differential HM response.
    ``legacy_absolute_hm`` is available solely for artifact reproduction.
    """
    allowed = {
        "calibrated_differential_gassmann",
        "constrained_local_gassmann",
        "local_inverse_gassmann",
        "matched_hm_delta",
        "legacy_absolute_hm",
    }
    if mode not in allowed:
        raise ValueError(f"Unknown fluid-substitution mode {mode!r}; expected one of {sorted(allowed)}")
    properties = np.asarray(elastic_brine, dtype=float)
    if properties.ndim != 3 or properties.shape[0] != 3:
        raise ValueError("elastic_brine must have shape [3, time, trace]")
    shape = properties.shape[1:]
    facies_sand = (np.asarray(sand_probability) >= sand_threshold) & np.asarray(
        reservoir_mask, dtype=bool
    )
    core = binary_erosion(
        facies_sand,
        structure=np.ones((minimum_sand_thickness_samples, 1)),
    )
    candidates = np.argwhere(core)
    plume = np.zeros(shape, dtype=bool)
    saturation = np.zeros(shape, dtype=np.float32)
    requested = int(rng.integers(plume_count[0], plume_count[1] + 1))
    rows, columns = np.indices(shape)
    if candidates.size:
        for _ in range(requested):
            center_row, center_column = candidates[int(rng.integers(candidates.shape[0]))]
            radius_x = float(rng.uniform(*lateral_radius_samples))
            radius_z = float(rng.uniform(*vertical_radius_samples))
            ellipse = (
                ((columns - center_column) / radius_x) ** 2
                + ((rows - center_row) / radius_z) ** 2
                <= 1.0
            )
            member = ellipse & facies_sand & ~plume
            base_saturation = float(rng.uniform(*co2_saturation))
            correlated = _normalized_correlated_noise(
                shape,
                rng,
                saturation_correlation_sigma_samples,
            )
            local_saturation = base_saturation * (
                1.0 + saturation_variation_fraction * correlated
            )
            local_saturation = np.clip(local_saturation, *co2_saturation)
            plume |= member
            saturation[member] = local_saturation[member]

    hm_keys = {
        "critical_porosity",
        "coordination_factor",
        "quartz_bulk_modulus_gpa",
        "clay_bulk_modulus_gpa",
        "quartz_shear_modulus_gpa",
        "clay_shear_modulus_gpa",
        "quartz_density_g_cc",
        "clay_density_g_cc",
        "overburden_density_kg_m3",
        "gravity_m_s2",
        "depth_origin_m",
        "depth_increment_m",
        "brine_bulk_modulus_gpa",
        "co2_bulk_modulus_gpa",
        "brine_density_g_cc",
        "co2_density_g_cc",
        "brie_exponent",
    }
    hm_parameters = {key: value for key, value in rock_physics.items() if key in hm_keys}
    shaliness = 1.0 - np.asarray(sand_probability, dtype=float)
    if mode in {"calibrated_differential_gassmann", "constrained_local_gassmann"}:
        if fluid_calibration is None:
            raise ValueError(f"{mode} requires a loaded well-calibration artifact")
        if depth_m is None or np.asarray(depth_m).shape != shape:
            raise ValueError(f"{mode} requires a depth_m grid matching the elastic section")
        default_physics = FluidRockPhysics()
        physics = FluidRockPhysics(
            **{
                name: float(rock_physics.get(name, getattr(default_physics, name)))
                for name in FluidRockPhysics.__dataclass_fields__
            }
        )
        candidate_function = (
            calibrated_differential_gassmann_substitution
            if mode == "calibrated_differential_gassmann"
            else constrained_local_gassmann_substitution
        )
        output = properties.copy()
        if plume.any():
            result = candidate_function(
                properties[0][plume],
                properties[1][plume],
                properties[2][plume],
                np.asarray(porosity)[plume],
                shaliness[plume],
                saturation[plume],
                np.asarray(depth_m)[plume],
                fluid_calibration,
                physics,
            )
            output[0][plume] = result.elastic.vp
            output[1][plume] = result.elastic.vs
            output[2][plume] = result.elastic.density
            ratio = result.dry_bulk_gpa / result.frame_shear_gpa
            poisson = poisson_ratio_from_moduli(
                result.dry_bulk_gpa, result.frame_shear_gpa
            )
            diagnostics = {
                "calibration_id": fluid_calibration.calibration_id,
                "calibration_method": fluid_calibration.metadata["method"],
                "rf_brine_vp_vs_density_correction": 0.0,
                "zero_saturation_recovers_original_rf_brine": True,
                "shear_modulus_changed_by_fluid": False,
                "feasibility_projection_used": False,
                "dry_bulk_clipping_used": False,
                "elastic_output_clipping_used": False,
                "direct_independent_elastic_delta_transfer": False,
                "effective_porosity_method": "two_phase_mineral_brine_density_closure",
                "effective_porosity_formula": (
                    "phi_eff = (rho_mineral - rho_RF) / "
                    "(rho_mineral - rho_brine)"
                ),
                "geological_porosity_role": (
                    "preserved geological-model and ML-input coordinate; not replaced"
                ),
                "rock_physics_porosity_role": (
                    "density-closure coordinate used only for dry-frame/fluid calculations"
                ),
                "rock_physics_porosity_uncertainty": (
                    "depends on mineral mixture and the sampled scenario brine density"
                ),
                "geological_porosity_percentiles": np.quantile(
                    result.input_porosity, [0.01, 0.5, 0.99]
                ).tolist(),
                "effective_porosity_percentiles": np.quantile(
                    result.effective_porosity, [0.01, 0.5, 0.99]
                ).tolist(),
                "porosity_adjustment_percentiles": np.quantile(
                    result.effective_porosity - result.input_porosity,
                    [0.01, 0.5, 0.99],
                ).tolist(),
                "dry_bulk_gpa_percentiles": np.quantile(
                    result.dry_bulk_gpa, [0.01, 0.5, 0.99]
                ).tolist(),
                "dry_to_frame_shear_percentiles": np.quantile(
                    ratio, [0.01, 0.5, 0.99]
                ).tolist(),
                "dry_poisson_ratio_percentiles": np.quantile(
                    poisson, [0.01, 0.5, 0.99]
                ).tolist(),
                "delta_bulk_gpa_percentiles": np.quantile(
                    result.delta_bulk_gpa, [0.01, 0.5, 0.99]
                ).tolist(),
                "nearest_calibration_distance_percentiles": np.quantile(
                    result.nearest_calibration_distance, [0.01, 0.5, 0.99]
                ).tolist(),
            }
        else:
            diagnostics = {
                "calibration_id": fluid_calibration.calibration_id,
                "plume_empty": True,
                "feasibility_projection_used": False,
                "dry_bulk_clipping_used": False,
                "elastic_output_clipping_used": False,
            }
        substituted = ElasticProperties(output[0], output[1], output[2])
    elif mode == "local_inverse_gassmann":
        local_keys = {
            "quartz_bulk_modulus_gpa",
            "clay_bulk_modulus_gpa",
            "brine_bulk_modulus_gpa",
            "co2_bulk_modulus_gpa",
            "brine_density_g_cc",
            "co2_density_g_cc",
            "brie_exponent",
            "compatibility_margin",
        }
        local_parameters = {
            key: value for key, value in rock_physics.items() if key in local_keys
        }
        result = local_inverse_gassmann_substitution(
            properties[0],
            properties[1],
            properties[2],
            porosity,
            shaliness,
            saturation,
            **local_parameters,
        )
        substituted = result.elastic
        diagnostics = {
            "adjusted_effective_mineral_fraction": result.adjusted_mineral_fraction,
            "scientific_compatibility": (
                "compatibility-only; ineligible for production because "
                "mineral feasibility projection and dry-bulk clipping are applied"
            ),
            "shear_modulus_change_max_gpa": float(
                np.max(
                    np.abs(
                        elastic_moduli_gpa(
                            substituted.vp,
                            substituted.vs,
                            substituted.density,
                        )[1]
                        - result.shear_gpa
                    )
                )
            ),
        }
    elif mode == "matched_hm_delta":
        substituted = matched_hm_delta_substitution(
            ElasticProperties(properties[0], properties[1], properties[2]),
            shaliness,
            porosity,
            saturation,
            **hm_parameters,
        )
        diagnostics = {"matched_dry_frame": True}
    else:
        substituted = hertz_mindlin_gassmann(
            shaliness,
            porosity,
            saturation,
            **hm_parameters,
        )
        diagnostics = {
            "scientific_compatibility": "absolute HM overwrite; compatibility-only and production-ineligible"
        }

    output = properties.copy()
    if mode in {"calibrated_differential_gassmann", "constrained_local_gassmann"}:
        output[0] = np.where(plume, substituted.vp, output[0])
        output[1] = np.where(plume, substituted.vs, output[1])
        output[2] = np.where(plume, substituted.density, output[2])
        if plume.any() and (
            np.any((output[0][plume] < vp_bounds_m_s[0]) | (output[0][plume] > vp_bounds_m_s[1]))
            or np.any((output[1][plume] < vs_bounds_m_s[0]) | (output[1][plume] > vs_bounds_m_s[1]))
            or np.any(
                (output[2][plume] < density_bounds_g_cc[0])
                | (output[2][plume] > density_bounds_g_cc[1])
            )
        ):
            raise ValueError("Calibrated fluid result lies outside declared elastic validity bounds")
    else:
        output[0] = np.where(plume, np.clip(substituted.vp, *vp_bounds_m_s), output[0])
        output[1] = np.where(plume, np.clip(substituted.vs, *vs_bounds_m_s), output[1])
        output[2] = np.where(
            plume,
            np.clip(substituted.density, *density_bounds_g_cc),
            output[2],
        )
    if not np.isfinite(output).all() or np.any(output[0] <= output[1]) or np.any(output[1] <= 0):
        raise ValueError("Fluid substitution produced invalid elastic properties")
    return FluidScenario(
        elastic=output.astype(np.float32),
        plume_mask=plume.astype(np.uint8),
        co2_saturation=saturation,
        metadata={
            "mode": mode,
            "requested_plumes": requested,
            "plume_pixels": int(plume.sum()),
            "minimum_sand_thickness_samples": int(minimum_sand_thickness_samples),
            "saturation_correlation_sigma_samples": list(
                saturation_correlation_sigma_samples
            ),
            "saturation_variation_fraction": float(saturation_variation_fraction),
            **diagnostics,
        },
    )


def apply_correlated_elastic_heterogeneity(
    elastic: np.ndarray,
    reservoir_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_samples: tuple[float, float] = (3.0, 10.0),
    bulk_fraction_std: float = 0.025,
    shear_fraction_std: float = 0.018,
    density_fraction_std: float = 0.004,
) -> tuple[np.ndarray, dict[str, object]]:
    """Perturb coupled moduli/density fields before seismic forward modeling."""
    properties = np.asarray(elastic, dtype=float)
    if properties.ndim != 3 or properties.shape[0] != 3:
        raise ValueError("elastic must have shape [3, time, trace]")
    bulk, shear = elastic_moduli_gpa(*properties)
    latent = _normalized_correlated_noise(properties.shape[1:], rng, sigma_samples)
    secondary = _normalized_correlated_noise(properties.shape[1:], rng, sigma_samples)
    support = np.asarray(reservoir_mask, dtype=bool)
    bulk_scale = 1.0 + bulk_fraction_std * latent
    shear_scale = 1.0 + shear_fraction_std * (0.8 * latent + 0.2 * secondary)
    density_scale = 1.0 + density_fraction_std * (0.6 * latent + 0.4 * secondary)
    perturbed = elastic_from_gpa(
        np.where(support, bulk * bulk_scale, bulk),
        np.where(support, shear * shear_scale, shear),
        np.where(support, properties[2] * density_scale, properties[2]),
    )
    output = np.stack((perturbed.vp, perturbed.vs, perturbed.density)).astype(np.float32)
    return output, {
        "mode": "correlated_bulk_shear_density",
        "sigma_samples": list(sigma_samples),
        "bulk_fraction_std": float(bulk_fraction_std),
        "shear_fraction_std": float(shear_fraction_std),
        "density_fraction_std": float(density_fraction_std),
    }


def make_field_conditioned_realization(
    *,
    sand_probability_base: np.ndarray,
    porosity_base: np.ndarray,
    rgt_base: np.ndarray,
    strat_fraction_base: np.ndarray,
    reservoir_mask_base: np.ndarray,
    elastic_background_base: np.ndarray,
    elastic_blend_weight_base: np.ndarray,
    reservoir_model: ElasticModelSet,
    seed: int,
    geology_config: dict[str, object],
    fluid_config: dict[str, object],
    fluid_calibration: CalibratedDryFrameModel | None = None,
    fluid_property_metadata: dict[str, object] | None = None,
    depth_m: np.ndarray | None = None,
) -> FieldConditionedRealization:
    """Generate one coherent member of the Stage-01-conditioned geological family."""
    rng = np.random.default_rng(seed)
    shape = np.asarray(sand_probability_base).shape
    deformation = make_deformation(
        shape,
        rng,
        fold_probability=float(geology_config["fold_probability"]),
        fold_amplitude_samples=tuple(geology_config["fold_amplitude_samples"]),
        fold_cycles=tuple(geology_config["fold_cycles"]),
        secondary_fold_amplitude_samples=tuple(geology_config["secondary_fold_amplitude_samples"]),
        secondary_fold_cycles=tuple(geology_config["secondary_fold_cycles"]),
        maximum_faults=int(geology_config["maximum_faults"]),
        fault_throw_samples=tuple(geology_config["fault_throw_samples"]),
        fault_dip_samples_per_trace=tuple(geology_config["fault_dip_samples_per_trace"]),
    )
    sand = warp_with_deformation(sand_probability_base, deformation)
    porosity = warp_with_deformation(porosity_base, deformation)
    rgt = warp_with_deformation(rgt_base, deformation)
    strat_fraction = warp_with_deformation(strat_fraction_base, deformation)
    reservoir = warp_with_deformation(reservoir_mask_base.astype(float), deformation) >= 0.5
    background = warp_with_deformation(elastic_background_base, deformation)
    blend_weight = warp_with_deformation(elastic_blend_weight_base, deformation)

    sigma = tuple(float(value) for value in geology_config["heterogeneity_sigma_samples"])
    sand += float(rng.uniform(*geology_config["sand_heterogeneity_std"])) * _normalized_correlated_noise(
        shape, rng, sigma
    )
    porosity += float(rng.uniform(*geology_config["porosity_heterogeneity_std"])) * _normalized_correlated_noise(
        shape, rng, sigma
    )
    sand = np.clip(sand, 0.0, 1.0)
    porosity = np.clip(porosity, 0.002, 0.35)
    delta = delta_from_sand_probability(sand)
    rgt = np.maximum.accumulate(rgt, axis=0)
    strat_fraction = np.clip(strat_fraction, 0.0, 1.0)

    reservoir_prediction = predict_elastic_fields(
        reservoir_model, delta, porosity, strat_fraction
    )
    weight = np.clip(blend_weight, 0.0, 1.0) * reservoir[None]
    brine = (1.0 - weight) * background + weight * reservoir_prediction
    heterogeneity_metadata: dict[str, object] = {"enabled": False}
    elastic_heterogeneity = geology_config.get("elastic_heterogeneity")
    if isinstance(elastic_heterogeneity, dict) and bool(
        elastic_heterogeneity.get("enabled", False)
    ):
        brine, heterogeneity_metadata = apply_correlated_elastic_heterogeneity(
            brine,
            reservoir,
            rng,
            sigma_samples=tuple(elastic_heterogeneity["sigma_samples"]),
            bulk_fraction_std=float(elastic_heterogeneity["bulk_fraction_std"]),
            shear_fraction_std=float(elastic_heterogeneity["shear_fraction_std"]),
            density_fraction_std=float(elastic_heterogeneity["density_fraction_std"]),
        )
        heterogeneity_metadata["enabled"] = True

    fluid_mode = str(fluid_config.get("mode", "legacy_absolute_hm"))
    fluid_enabled = bool(fluid_config.get("enabled", True))
    if not fluid_enabled or fluid_mode == "disabled_core_elastic":
        fluid = FluidScenario(
            elastic=brine.astype(np.float32),
            plume_mask=np.zeros(shape, dtype=np.uint8),
            co2_saturation=np.zeros(shape, dtype=np.float32),
            metadata={
                "mode": "disabled_core_elastic",
                "enabled": False,
                "requested_plumes": 0,
                "plume_pixels": 0,
                "scientific_scope": (
                    "field-conditioned facies and elastic variability; no CO2-specific target"
                ),
                "co2_class_present": False,
            },
        )
    else:
        fluid_function = (
            apply_co2_fluid_substitution
            if fluid_mode == "legacy_absolute_hm" and "mode" not in fluid_config
            else apply_co2_fluid_substitution_v003
        )
        fluid = fluid_function(
            brine,
            porosity,
            sand,
            reservoir,
            rng,
            plume_count=tuple(fluid_config["plume_count"]),
            lateral_radius_samples=tuple(fluid_config["plume_lateral_radius_samples"]),
            vertical_radius_samples=tuple(fluid_config["plume_vertical_radius_samples"]),
            minimum_sand_thickness_samples=int(fluid_config["minimum_sand_thickness_samples"]),
            sand_threshold=float(geology_config["sand_facies_probability_threshold"]),
            co2_saturation=tuple(fluid_config["co2_saturation"]),
            **(
                {
                    "mode": fluid_mode,
                    "fluid_calibration": fluid_calibration,
                    "depth_m": depth_m,
                    "saturation_correlation_sigma_samples": tuple(
                        fluid_config["saturation_correlation_sigma_samples"]
                    ),
                    "saturation_variation_fraction": float(
                        fluid_config["saturation_variation_fraction"]
                    ),
                    "vp_bounds_m_s": tuple(fluid_config["vp_bounds_m_s"]),
                    "vs_bounds_m_s": tuple(fluid_config["vs_bounds_m_s"]),
                    "density_bounds_g_cc": tuple(fluid_config["density_bounds_g_cc"]),
                    "compatibility_margin": float(fluid_config["compatibility_margin"]),
                }
                if fluid_function is apply_co2_fluid_substitution_v003
                else {}
            ),
            critical_porosity=float(fluid_config["critical_porosity"]),
            coordination_factor=float(fluid_config["coordination_factor"]),
            quartz_bulk_modulus_gpa=float(fluid_config["quartz_bulk_modulus_gpa"]),
            clay_bulk_modulus_gpa=float(fluid_config["clay_bulk_modulus_gpa"]),
            quartz_shear_modulus_gpa=float(fluid_config["quartz_shear_modulus_gpa"]),
            clay_shear_modulus_gpa=float(fluid_config["clay_shear_modulus_gpa"]),
            quartz_density_g_cc=float(fluid_config["quartz_density_g_cc"]),
            clay_density_g_cc=float(fluid_config["clay_density_g_cc"]),
            overburden_density_kg_m3=float(fluid_config["overburden_density_kg_m3"]),
            gravity_m_s2=float(fluid_config["gravity_m_s2"]),
            depth_origin_m=float(fluid_config["depth_origin_m"]),
            depth_increment_m=float(fluid_config["depth_increment_m"]),
            brine_bulk_modulus_gpa=float(fluid_config["brine_bulk_modulus_gpa"]),
            co2_bulk_modulus_gpa=float(fluid_config["co2_bulk_modulus_gpa"]),
            brine_density_g_cc=float(fluid_config["brine_density_g_cc"]),
            co2_density_g_cc=float(fluid_config["co2_density_g_cc"]),
            brie_exponent=float(fluid_config["brie_exponent"]),
        )
    segmentation = ((sand >= float(geology_config["sand_facies_probability_threshold"])) & reservoir).astype(
        np.uint8
    )
    segmentation[fluid.plume_mask.astype(bool)] = 2
    return FieldConditionedRealization(
        elastic=fluid.elastic,
        elastic_brine=brine.astype(np.float32),
        delta=delta.astype(np.float32),
        sand_probability=sand.astype(np.float32),
        porosity=porosity.astype(np.float32),
        rgt=rgt.astype(np.float32),
        strat_fraction=strat_fraction.astype(np.float32),
        reservoir_mask=reservoir.astype(np.uint8),
        segmentation=segmentation,
        plume_mask=fluid.plume_mask,
        co2_saturation=fluid.co2_saturation,
        vertical_displacement=deformation.vertical_displacement.astype(np.float32),
        horizontal_displacement=deformation.horizontal_displacement.astype(np.float32),
        metadata={
            "seed": seed,
            "deformation": deformation.metadata,
            "elastic_heterogeneity": heterogeneity_metadata,
            "fluid": {
                **fluid.metadata,
                **(
                    {"property_state": fluid_property_metadata}
                    if fluid_property_metadata is not None
                    else {}
                ),
            },
        },
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
