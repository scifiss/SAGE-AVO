"""Well-calibrated, projection-free fluid-substitution utilities.

These functions do not clip porosity, dry-frame moduli, elastic properties, or
fluid responses. Invalid states remain explicit so production validation can
reject them instead of silently changing the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .rock_physics import ElasticProperties, brie_fluid_mixture, elastic_moduli_gpa


@dataclass(frozen=True)
class FluidRockPhysics:
    """Explicit units and constants used by the calibrated fluid workflow."""

    quartz_bulk_modulus_gpa: float = 39.0
    clay_bulk_modulus_gpa: float = 21.0
    quartz_shear_modulus_gpa: float = 45.0
    clay_shear_modulus_gpa: float = 6.85
    quartz_density_g_cc: float = 2.65
    clay_density_g_cc: float = 2.60
    brine_bulk_modulus_gpa: float = 2.20
    co2_bulk_modulus_gpa: float = 0.10
    brine_density_g_cc: float = 1.03
    co2_density_g_cc: float = 0.65
    brie_exponent: float = 3.0


@dataclass(frozen=True)
class CalibratedDryFrameModel:
    """Deterministic local-neighbour dry-frame model calibrated to well logs.

    Features are effective porosity (fraction), DELTA/shaliness (fraction),
    and depth (km).  Targets are log dry bulk and log shear modulus in GPa.
    The log representation enforces positive predicted moduli without clipping.
    """

    calibration_id: str
    feature_names: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    features_standardized: np.ndarray
    log_dry_bulk_gpa: np.ndarray
    log_shear_gpa: np.ndarray
    well_ids: np.ndarray
    neighbor_count: int
    metadata: dict[str, Any]

    def predict(
        self,
        effective_porosity: np.ndarray,
        shaliness: np.ndarray,
        depth_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict dry bulk, shear, and nearest-calibration distance."""
        phi = np.asarray(effective_porosity, dtype=float)
        shale = np.asarray(shaliness, dtype=float)
        depth = np.asarray(depth_m, dtype=float)
        if not (phi.shape == shale.shape == depth.shape):
            raise ValueError("Fluid-calibration feature arrays must have identical shapes")
        query = np.column_stack((phi.ravel(), shale.ravel(), depth.ravel() / 1000.0))
        standardized = (query - self.feature_center) / self.feature_scale
        tree = cKDTree(self.features_standardized)
        count = min(int(self.neighbor_count), len(self.features_standardized))
        distances, indices = tree.query(standardized, k=count)
        if count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-8) ** 2
        weights /= np.sum(weights, axis=1, keepdims=True)
        log_dry = np.sum(weights * self.log_dry_bulk_gpa[indices], axis=1)
        log_shear = np.sum(weights * self.log_shear_gpa[indices], axis=1)
        dry = np.exp(log_dry).reshape(phi.shape)
        shear = np.exp(log_shear).reshape(phi.shape)
        nearest = distances[:, 0].reshape(phi.shape)
        return dry, shear, nearest


@dataclass(frozen=True)
class CalibratedFluidResult:
    """Elastic result and the full physical state used to obtain it."""

    elastic: ElasticProperties
    rf_bulk_gpa: np.ndarray
    rf_shear_gpa: np.ndarray
    target_bulk_gpa: np.ndarray
    target_shear_gpa: np.ndarray
    dry_bulk_gpa: np.ndarray
    frame_shear_gpa: np.ndarray
    mineral_bulk_gpa: np.ndarray
    mineral_shear_gpa: np.ndarray
    mineral_density_g_cc: np.ndarray
    effective_porosity: np.ndarray
    input_porosity: np.ndarray
    fluid_bulk_gpa: np.ndarray
    delta_bulk_gpa: np.ndarray
    delta_density_g_cc: np.ndarray
    nearest_calibration_distance: np.ndarray
    method: str


def _require_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _require_fraction(name: str, values: np.ndarray) -> np.ndarray:
    array = _require_finite(name, values)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be supplied as a fraction in [0, 1]")
    return array


def mineral_properties_vrh_strict(
    shaliness: np.ndarray,
    physics: FluidRockPhysics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quartz/clay VRH moduli and arithmetic density without clipping."""
    shale = _require_fraction("shaliness", shaliness)
    quartz = 1.0 - shale
    bulk_voigt = quartz * physics.quartz_bulk_modulus_gpa + shale * physics.clay_bulk_modulus_gpa
    bulk_reuss = 1.0 / (
        quartz / physics.quartz_bulk_modulus_gpa + shale / physics.clay_bulk_modulus_gpa
    )
    shear_voigt = (
        quartz * physics.quartz_shear_modulus_gpa + shale * physics.clay_shear_modulus_gpa
    )
    shear_reuss = 1.0 / (
        quartz / physics.quartz_shear_modulus_gpa + shale / physics.clay_shear_modulus_gpa
    )
    density = quartz * physics.quartz_density_g_cc + shale * physics.clay_density_g_cc
    return 0.5 * (bulk_voigt + bulk_reuss), 0.5 * (shear_voigt + shear_reuss), density


def density_derived_effective_porosity(
    density_brine_g_cc: np.ndarray,
    mineral_density_g_cc: np.ndarray,
    brine_density_g_cc: float,
) -> np.ndarray:
    """Close the two-phase mineral/brine density equation analytically."""
    density = _require_finite("RF brine density", density_brine_g_cc)
    mineral = _require_finite("mineral density", mineral_density_g_cc)
    denominator = mineral - float(brine_density_g_cc)
    if np.any(denominator <= 0.0):
        raise ValueError("Mineral density must exceed brine density")
    phi = (mineral - density) / denominator
    if np.any((phi <= 0.0) | (phi >= 1.0)):
        raise ValueError("Density-derived effective porosity lies outside (0, 1)")
    return phi


def inverse_gassmann_dry_bulk_strict(
    saturated_bulk_gpa: np.ndarray,
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    fluid_bulk_gpa: float | np.ndarray,
) -> np.ndarray:
    """Invert Gassmann with no projection, clipping, or denominator repair."""
    saturated = _require_finite("saturated bulk modulus", saturated_bulk_gpa)
    phi = _require_fraction("porosity", porosity)
    mineral = _require_finite("mineral bulk modulus", mineral_bulk_gpa)
    fluid = _require_finite("fluid bulk modulus", np.asarray(fluid_bulk_gpa))
    if np.any(phi <= 0.0) or np.any(mineral <= 0.0) or np.any(fluid <= 0.0):
        raise ValueError("Strict inverse Gassmann requires positive porosity and moduli")
    numerator = saturated * (phi * mineral / fluid + 1.0 - phi) - mineral
    denominator = phi * mineral / fluid + saturated / mineral - 1.0 - phi
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("Strict inverse Gassmann encountered a singular denominator")
    return numerator / denominator


def forward_gassmann_bulk_strict(
    dry_bulk_gpa: np.ndarray,
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    fluid_bulk_gpa: float | np.ndarray,
) -> np.ndarray:
    """Forward Gassmann without hidden denominator floors or modulus clipping."""
    dry = _require_finite("dry bulk modulus", dry_bulk_gpa)
    phi = _require_fraction("porosity", porosity)
    mineral = _require_finite("mineral bulk modulus", mineral_bulk_gpa)
    fluid = _require_finite("fluid bulk modulus", np.asarray(fluid_bulk_gpa))
    if np.any(phi <= 0.0) or np.any(dry <= 0.0) or np.any(mineral <= 0.0) or np.any(fluid <= 0.0):
        raise ValueError("Strict forward Gassmann requires positive porosity and moduli")
    denominator = phi / fluid + (1.0 - phi) / mineral - dry / mineral**2
    if np.any(denominator <= 0.0):
        raise ValueError("Strict forward Gassmann encountered a non-positive denominator")
    return dry + (1.0 - dry / mineral) ** 2 / denominator


def poisson_ratio_from_moduli(bulk_gpa: np.ndarray, shear_gpa: np.ndarray) -> np.ndarray:
    """Return isotropic Poisson ratio from bulk and shear modulus."""
    bulk = np.asarray(bulk_gpa, dtype=float)
    shear = np.asarray(shear_gpa, dtype=float)
    return (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))


def elastic_from_gpa_strict(
    bulk_gpa: np.ndarray,
    shear_gpa: np.ndarray,
    density_g_cc: np.ndarray,
) -> ElasticProperties:
    """Convert GPa/g/cc to m/s and reject, rather than mask, invalid states."""
    bulk = _require_finite("bulk modulus", bulk_gpa)
    shear = _require_finite("shear modulus", shear_gpa)
    density = _require_finite("density", density_g_cc)
    p_modulus = bulk + 4.0 * shear / 3.0
    if np.any((bulk <= 0.0) | (shear <= 0.0) | (density <= 0.0) | (p_modulus <= 0.0)):
        raise ValueError("Elastic state contains a non-positive physical modulus or density")
    vp = 1000.0 * np.sqrt(p_modulus / density)
    vs = 1000.0 * np.sqrt(shear / density)
    if np.any(vp <= vs):
        raise ValueError("Elastic state violates Vp > Vs")
    return ElasticProperties(vp=vp, vs=vs, density=density)


def constrained_local_gassmann_substitution(
    vp_brine_m_s: np.ndarray,
    vs_brine_m_s: np.ndarray,
    density_brine_g_cc: np.ndarray,
    input_porosity: np.ndarray,
    shaliness: np.ndarray,
    co2_saturation: np.ndarray,
    depth_m: np.ndarray,
    calibration: CalibratedDryFrameModel,
    physics: FluidRockPhysics,
) -> CalibratedFluidResult:
    """Candidate A: close effective porosity to density, then invert locally.

    The RF Vp, Vs, and density are not corrected.  The sole fitted uncertain
    variable is effective porosity, obtained analytically from the local
    mineral/brine density equation.  The supplied calibration is used to
    quantify well-support distance, not to replace the local dry frame.
    """
    vp = _require_finite("RF brine Vp", vp_brine_m_s)
    vs = _require_finite("RF brine Vs", vs_brine_m_s)
    density = _require_finite("RF brine density", density_brine_g_cc)
    phi_input = _require_fraction("input porosity", input_porosity)
    saturation = _require_fraction("CO2 saturation", co2_saturation)
    shale = _require_fraction("shaliness", shaliness)
    depth = _require_finite("depth", depth_m)
    if not (vp.shape == vs.shape == density.shape == phi_input.shape == saturation.shape == shale.shape == depth.shape):
        raise ValueError("Candidate-A arrays must have identical shapes")
    mineral_bulk, mineral_shear, mineral_density = mineral_properties_vrh_strict(shale, physics)
    phi_effective = density_derived_effective_porosity(
        density,
        mineral_density,
        physics.brine_density_g_cc,
    )
    rf_bulk, rf_shear = elastic_moduli_gpa(vp, vs, density)
    dry_bulk = inverse_gassmann_dry_bulk_strict(
        rf_bulk,
        phi_effective,
        mineral_bulk,
        physics.brine_bulk_modulus_gpa,
    )
    if np.any((dry_bulk <= 0.0) | (dry_bulk >= mineral_bulk)):
        raise ValueError("Candidate A produced a dry bulk modulus outside (0, Kmin)")
    _, frame_shear, nearest = calibration.predict(phi_effective, shale, depth)
    fluid_bulk, fluid_density = brie_fluid_mixture(
        saturation,
        brine_bulk_modulus_gpa=physics.brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=physics.co2_bulk_modulus_gpa,
        brine_density_g_cc=physics.brine_density_g_cc,
        co2_density_g_cc=physics.co2_density_g_cc,
        brie_exponent=physics.brie_exponent,
    )
    target_bulk = forward_gassmann_bulk_strict(
        dry_bulk,
        phi_effective,
        mineral_bulk,
        fluid_bulk,
    )
    delta_density = phi_effective * (fluid_density - physics.brine_density_g_cc)
    target_density = density + delta_density
    zero = saturation == 0.0
    target_bulk = np.where(zero, rf_bulk, target_bulk)
    target_density = np.where(zero, density, target_density)
    elastic = elastic_from_gpa_strict(target_bulk, rf_shear, target_density)
    return CalibratedFluidResult(
        elastic=elastic,
        rf_bulk_gpa=rf_bulk,
        rf_shear_gpa=rf_shear,
        target_bulk_gpa=target_bulk,
        target_shear_gpa=rf_shear,
        dry_bulk_gpa=dry_bulk,
        frame_shear_gpa=frame_shear,
        mineral_bulk_gpa=mineral_bulk,
        mineral_shear_gpa=mineral_shear,
        mineral_density_g_cc=mineral_density,
        effective_porosity=phi_effective,
        input_porosity=phi_input,
        fluid_bulk_gpa=fluid_bulk,
        delta_bulk_gpa=target_bulk - rf_bulk,
        delta_density_g_cc=target_density - density,
        nearest_calibration_distance=nearest,
        method="constrained_local_gassmann",
    )


def calibrated_differential_gassmann_substitution(
    vp_brine_m_s: np.ndarray,
    vs_brine_m_s: np.ndarray,
    density_brine_g_cc: np.ndarray,
    input_porosity: np.ndarray,
    shaliness: np.ndarray,
    co2_saturation: np.ndarray,
    depth_m: np.ndarray,
    calibration: CalibratedDryFrameModel,
    physics: FluidRockPhysics,
) -> CalibratedFluidResult:
    """Candidate B: transfer a calibrated same-frame Gassmann delta in K-rho space."""
    vp = _require_finite("RF brine Vp", vp_brine_m_s)
    vs = _require_finite("RF brine Vs", vs_brine_m_s)
    density = _require_finite("RF brine density", density_brine_g_cc)
    phi_input = _require_fraction("input porosity", input_porosity)
    saturation = _require_fraction("CO2 saturation", co2_saturation)
    shale = _require_fraction("shaliness", shaliness)
    depth = _require_finite("depth", depth_m)
    if not (vp.shape == vs.shape == density.shape == phi_input.shape == saturation.shape == shale.shape == depth.shape):
        raise ValueError("Candidate-B arrays must have identical shapes")
    mineral_bulk, mineral_shear, mineral_density = mineral_properties_vrh_strict(shale, physics)
    phi_effective = density_derived_effective_porosity(
        density,
        mineral_density,
        physics.brine_density_g_cc,
    )
    dry_bulk, frame_shear, nearest = calibration.predict(phi_effective, shale, depth)
    if np.any((dry_bulk <= 0.0) | (dry_bulk >= mineral_bulk)):
        raise ValueError("Candidate B calibration produced a dry bulk modulus outside (0, Kmin)")
    rf_bulk, rf_shear = elastic_moduli_gpa(vp, vs, density)
    brine_frame_bulk = forward_gassmann_bulk_strict(
        dry_bulk,
        phi_effective,
        mineral_bulk,
        physics.brine_bulk_modulus_gpa,
    )
    fluid_bulk, fluid_density = brie_fluid_mixture(
        saturation,
        brine_bulk_modulus_gpa=physics.brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=physics.co2_bulk_modulus_gpa,
        brine_density_g_cc=physics.brine_density_g_cc,
        co2_density_g_cc=physics.co2_density_g_cc,
        brie_exponent=physics.brie_exponent,
    )
    co2_frame_bulk = forward_gassmann_bulk_strict(
        dry_bulk,
        phi_effective,
        mineral_bulk,
        fluid_bulk,
    )
    delta_bulk = co2_frame_bulk - brine_frame_bulk
    delta_density = phi_effective * (fluid_density - physics.brine_density_g_cc)
    target_bulk = rf_bulk + delta_bulk
    target_density = density + delta_density
    zero = saturation == 0.0
    target_bulk = np.where(zero, rf_bulk, target_bulk)
    target_density = np.where(zero, density, target_density)
    elastic = elastic_from_gpa_strict(target_bulk, rf_shear, target_density)
    return CalibratedFluidResult(
        elastic=elastic,
        rf_bulk_gpa=rf_bulk,
        rf_shear_gpa=rf_shear,
        target_bulk_gpa=target_bulk,
        target_shear_gpa=rf_shear,
        dry_bulk_gpa=dry_bulk,
        frame_shear_gpa=frame_shear,
        mineral_bulk_gpa=mineral_bulk,
        mineral_shear_gpa=mineral_shear,
        mineral_density_g_cc=mineral_density,
        effective_porosity=phi_effective,
        input_porosity=phi_input,
        fluid_bulk_gpa=fluid_bulk,
        delta_bulk_gpa=target_bulk - rf_bulk,
        delta_density_g_cc=target_density - density,
        nearest_calibration_distance=nearest,
        method="calibrated_differential_gassmann",
    )


def save_calibrated_dry_frame(
    model: CalibratedDryFrameModel,
    path: str | Path,
) -> tuple[Path, Path]:
    """Save calibration arrays plus a human-readable metadata sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        calibration_id=np.asarray(model.calibration_id),
        feature_names=np.asarray(model.feature_names),
        feature_center=model.feature_center,
        feature_scale=model.feature_scale,
        features_standardized=model.features_standardized,
        log_dry_bulk_gpa=model.log_dry_bulk_gpa,
        log_shear_gpa=model.log_shear_gpa,
        well_ids=np.asarray(model.well_ids, dtype="U32"),
        neighbor_count=np.asarray(model.neighbor_count),
    )
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(model.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination, metadata_path


def load_calibrated_dry_frame(path: str | Path) -> CalibratedDryFrameModel:
    """Load a calibration artifact without permitting pickled payloads."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata_path = source.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return CalibratedDryFrameModel(
            calibration_id=str(archive["calibration_id"]),
            feature_names=tuple(str(item) for item in archive["feature_names"]),
            feature_center=np.asarray(archive["feature_center"], dtype=float),
            feature_scale=np.asarray(archive["feature_scale"], dtype=float),
            features_standardized=np.asarray(archive["features_standardized"], dtype=float),
            log_dry_bulk_gpa=np.asarray(archive["log_dry_bulk_gpa"], dtype=float),
            log_shear_gpa=np.asarray(archive["log_shear_gpa"], dtype=float),
            well_ids=np.asarray(archive["well_ids"]),
            neighbor_count=int(archive["neighbor_count"]),
            metadata=metadata,
        )
