"""Compact, explicit rock- and fluid-physics utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticProperties:
    vp: np.ndarray
    vs: np.ndarray
    density: np.ndarray


def moduli_from_velocities(vp: np.ndarray, vs: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bulk and shear moduli in a unit system consistent with inputs."""
    vp_array = np.asarray(vp, dtype=float)
    vs_array = np.asarray(vs, dtype=float)
    rho_array = np.asarray(density, dtype=float)
    shear = rho_array * vs_array**2
    bulk = rho_array * vp_array**2 - 4.0 * shear / 3.0
    return bulk, shear


def velocities_from_moduli(bulk: np.ndarray, shear: np.ndarray, density: np.ndarray) -> ElasticProperties:
    """Convert bulk/shear moduli and density to Vp/Vs."""
    rho = np.maximum(np.asarray(density, dtype=float), 1e-8)
    vp = np.sqrt(np.maximum((bulk + 4.0 * shear / 3.0) / rho, 0.0))
    vs = np.sqrt(np.maximum(shear / rho, 0.0))
    return ElasticProperties(vp, vs, rho)


def gassmann_substitute(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    porosity: np.ndarray,
    mineral_bulk: float,
    initial_fluid_bulk: float,
    substituted_fluid_bulk: np.ndarray | float,
    initial_fluid_density: float,
    substituted_fluid_density: np.ndarray | float,
) -> ElasticProperties:
    """Apply Gassmann fluid substitution while retaining the dry-frame shear modulus."""
    phi = np.clip(np.asarray(porosity, dtype=float), 1e-4, 0.6)
    saturated_bulk, shear = moduli_from_velocities(vp, vs, density)
    denominator = (
        phi / initial_fluid_bulk
        + (1.0 - phi) / mineral_bulk
        - saturated_bulk / mineral_bulk**2
    )
    dry_bulk = (saturated_bulk * (phi * mineral_bulk / initial_fluid_bulk + 1.0 - phi) - mineral_bulk) / (
        phi * mineral_bulk / initial_fluid_bulk + saturated_bulk / mineral_bulk - 1.0
    )
    fluid_bulk = np.asarray(substituted_fluid_bulk, dtype=float)
    substituted_bulk = dry_bulk + (1.0 - dry_bulk / mineral_bulk) ** 2 / (
        phi / fluid_bulk + (1.0 - phi) / mineral_bulk - dry_bulk / mineral_bulk**2
    )
    fluid_density = np.asarray(substituted_fluid_density, dtype=float)
    new_density = np.asarray(density, dtype=float) + phi * (fluid_density - initial_fluid_density)
    result = velocities_from_moduli(substituted_bulk, shear, new_density)
    invalid = ~np.isfinite(denominator) | ~np.isfinite(result.vp) | ~np.isfinite(result.vs)
    return ElasticProperties(
        np.where(invalid, vp, result.vp),
        np.where(invalid, vs, result.vs),
        np.where(invalid, density, result.density),
    )


def hertz_mindlin_gassmann(
    shaliness: np.ndarray,
    porosity: np.ndarray,
    co2_saturation: np.ndarray,
    *,
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
) -> ElasticProperties:
    """Historical Hertz--Mindlin dry frame followed by Gassmann substitution.

    Moduli are in GPa, density in g/cc, and returned velocities in m/s. The
    depth/effective-pressure relation is an explicit scenario assumption, not
    inferred from the time axis.
    """
    vsh = np.clip(np.asarray(shaliness, dtype=float), 0.0, 1.0)
    phi = np.clip(np.asarray(porosity, dtype=float), 0.01, critical_porosity - 0.01)
    saturation = np.clip(np.asarray(co2_saturation, dtype=float), 0.0, 1.0)
    if not (vsh.shape == phi.shape == saturation.shape):
        raise ValueError("shaliness, porosity, and saturation must have matching shapes")
    height, width = phi.shape
    depth = np.broadcast_to(
        (depth_origin_m + np.arange(height) * depth_increment_m)[:, None], (height, width)
    )
    bulk_mineral = 0.5 * (
        (1.0 - vsh) * quartz_bulk_modulus_gpa
        + vsh * clay_bulk_modulus_gpa
        + 1.0
        / ((1.0 - vsh) / quartz_bulk_modulus_gpa + vsh / clay_bulk_modulus_gpa + 1e-8)
    )
    shear_mineral = 0.5 * (
        (1.0 - vsh) * quartz_shear_modulus_gpa
        + vsh * clay_shear_modulus_gpa
        + 1.0
        / ((1.0 - vsh) / quartz_shear_modulus_gpa + vsh / clay_shear_modulus_gpa + 1e-8)
    )
    poisson = (3.0 * bulk_mineral - 2.0 * shear_mineral) / (
        6.0 * bulk_mineral + 2.0 * shear_mineral
    )
    effective_pressure_gpa = overburden_density_kg_m3 * gravity_m_s2 * depth / 1e9
    coordination = coordination_factor / critical_porosity
    bulk_contact = (
        coordination**2
        * (1.0 - critical_porosity) ** 2
        * shear_mineral**2
        * effective_pressure_gpa
        / (18.0 * np.pi**2 * (1.0 - poisson) ** 2)
    ) ** (1.0 / 3.0)
    shear_contact = (5.0 - 4.0 * poisson) / (10.0 - 5.0 * poisson) * (
        3.0
        * coordination**2
        * (1.0 - critical_porosity) ** 2
        * shear_mineral**2
        * effective_pressure_gpa
        / (2.0 * np.pi**2 * (1.0 - poisson) ** 2)
    ) ** (1.0 / 3.0)
    dry_bulk = 1.0 / (
        phi / critical_porosity / (bulk_contact + 4.0 * shear_contact / 3.0)
        + (1.0 - phi / critical_porosity) / (bulk_mineral + 4.0 * shear_contact / 3.0)
    ) - 4.0 * shear_contact / 3.0
    fluid_bulk = (brine_bulk_modulus_gpa - co2_bulk_modulus_gpa) * (
        1.0 - saturation
    ) ** brie_exponent + co2_bulk_modulus_gpa
    mineral_density = (1.0 - vsh) * quartz_density_g_cc + vsh * clay_density_g_cc
    fluid_density = saturation * co2_density_g_cc + (1.0 - saturation) * brine_density_g_cc
    saturated_density = (1.0 - phi) * mineral_density + phi * fluid_density
    saturated_bulk = dry_bulk + (1.0 - dry_bulk / bulk_mineral) ** 2 / (
        phi / fluid_bulk + (1.0 - phi) / bulk_mineral - dry_bulk / bulk_mineral**2
    )
    vp = np.sqrt(np.maximum((saturated_bulk + 4.0 * shear_contact / 3.0) / saturated_density, 0.0))
    vs = np.sqrt(np.maximum(shear_contact / saturated_density, 0.0))
    return ElasticProperties(vp * 1000.0, vs * 1000.0, saturated_density)
