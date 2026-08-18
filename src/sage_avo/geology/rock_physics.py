"""Compact, explicit rock- and fluid-physics utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticProperties:
    vp: np.ndarray
    vs: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class LocalGassmannResult:
    """Local RF-background substitution with diagnostic dry-frame state."""

    elastic: ElasticProperties
    saturated_bulk_gpa: np.ndarray
    dry_bulk_gpa: np.ndarray
    shear_gpa: np.ndarray
    mineral_bulk_gpa: np.ndarray
    fluid_bulk_gpa: np.ndarray
    adjusted_mineral_fraction: float


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


def elastic_moduli_gpa(
    vp_m_s: np.ndarray,
    vs_m_s: np.ndarray,
    density_g_cc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return saturated bulk and shear moduli in GPa from explicit field units."""
    vp_km_s = np.asarray(vp_m_s, dtype=float) / 1000.0
    vs_km_s = np.asarray(vs_m_s, dtype=float) / 1000.0
    density = np.asarray(density_g_cc, dtype=float)
    shear = density * vs_km_s**2
    bulk = density * vp_km_s**2 - 4.0 * shear / 3.0
    return bulk, shear


def elastic_from_gpa(
    bulk_gpa: np.ndarray,
    shear_gpa: np.ndarray,
    density_g_cc: np.ndarray,
) -> ElasticProperties:
    """Convert GPa and g/cc to m/s while preserving the stated unit contract."""
    density = np.maximum(np.asarray(density_g_cc, dtype=float), 1e-8)
    vp = 1000.0 * np.sqrt(
        np.maximum((np.asarray(bulk_gpa) + 4.0 * np.asarray(shear_gpa) / 3.0) / density, 0.0)
    )
    vs = 1000.0 * np.sqrt(np.maximum(np.asarray(shear_gpa) / density, 0.0))
    return ElasticProperties(vp, vs, density)


def mineral_bulk_modulus_vrh(
    shaliness: np.ndarray,
    *,
    quartz_bulk_modulus_gpa: float = 39.0,
    clay_bulk_modulus_gpa: float = 21.0,
) -> np.ndarray:
    """Voigt-Reuss-Hill mineral bulk modulus for the DELTA=shaliness convention."""
    shale = np.clip(np.asarray(shaliness, dtype=float), 0.0, 1.0)
    voigt = (1.0 - shale) * quartz_bulk_modulus_gpa + shale * clay_bulk_modulus_gpa
    reuss = 1.0 / (
        (1.0 - shale) / quartz_bulk_modulus_gpa + shale / clay_bulk_modulus_gpa
    )
    return 0.5 * (voigt + reuss)


def brie_fluid_mixture(
    co2_saturation: np.ndarray,
    *,
    brine_bulk_modulus_gpa: float = 2.2,
    co2_bulk_modulus_gpa: float = 0.1,
    brine_density_g_cc: float = 1.03,
    co2_density_g_cc: float = 0.65,
    brie_exponent: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Brie bulk modulus and arithmetic density for a CO2/brine mixture."""
    saturation = np.clip(np.asarray(co2_saturation, dtype=float), 0.0, 1.0)
    bulk = (brine_bulk_modulus_gpa - co2_bulk_modulus_gpa) * (
        1.0 - saturation
    ) ** brie_exponent + co2_bulk_modulus_gpa
    density = (
        (1.0 - saturation) * brine_density_g_cc
        + saturation * co2_density_g_cc
    )
    return bulk, density


def inverse_gassmann_dry_bulk(
    saturated_bulk_gpa: np.ndarray,
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    fluid_bulk_gpa: float | np.ndarray,
) -> np.ndarray:
    """Invert Gassmann using its complete algebraic denominator."""
    saturated = np.asarray(saturated_bulk_gpa, dtype=float)
    phi = np.clip(np.asarray(porosity, dtype=float), 1e-6, 0.6)
    mineral = np.asarray(mineral_bulk_gpa, dtype=float)
    fluid = np.asarray(fluid_bulk_gpa, dtype=float)
    numerator = saturated * (phi * mineral / fluid + 1.0 - phi) - mineral
    denominator = phi * mineral / fluid + saturated / mineral - 1.0 - phi
    return numerator / np.where(np.abs(denominator) > 1e-10, denominator, np.nan)


def forward_gassmann_bulk(
    dry_bulk_gpa: np.ndarray,
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    fluid_bulk_gpa: float | np.ndarray,
) -> np.ndarray:
    """Forward Gassmann bulk modulus in GPa."""
    dry = np.asarray(dry_bulk_gpa, dtype=float)
    phi = np.clip(np.asarray(porosity, dtype=float), 1e-6, 0.6)
    mineral = np.asarray(mineral_bulk_gpa, dtype=float)
    fluid = np.asarray(fluid_bulk_gpa, dtype=float)
    denominator = phi / fluid + (1.0 - phi) / mineral - dry / mineral**2
    return dry + (1.0 - dry / mineral) ** 2 / np.maximum(denominator, 1e-10)


def local_inverse_gassmann_substitution(
    vp_brine_m_s: np.ndarray,
    vs_brine_m_s: np.ndarray,
    density_brine_g_cc: np.ndarray,
    porosity: np.ndarray,
    shaliness: np.ndarray,
    co2_saturation: np.ndarray,
    *,
    quartz_bulk_modulus_gpa: float = 39.0,
    clay_bulk_modulus_gpa: float = 21.0,
    brine_bulk_modulus_gpa: float = 2.2,
    co2_bulk_modulus_gpa: float = 0.1,
    brine_density_g_cc: float = 1.03,
    co2_density_g_cc: float = 0.65,
    brie_exponent: float = 3.0,
    compatibility_margin: float = 0.995,
) -> LocalGassmannResult:
    """Substitute fluid from the local RF brine state via inverse Gassmann.

    Some empirical RF backgrounds are softer than the lower Gassmann bound
    implied by a pure quartz/clay mineral mixture.  For those samples the
    mineral modulus is reduced only as much as required to admit a finite,
    non-negative local dry frame.  This adjustment is reported explicitly;
    the local saturated state is never overwritten by an unrelated absolute
    Hertz--Mindlin model.
    """
    vp = np.asarray(vp_brine_m_s, dtype=float)
    vs = np.asarray(vs_brine_m_s, dtype=float)
    density = np.asarray(density_brine_g_cc, dtype=float)
    phi = np.clip(np.asarray(porosity, dtype=float), 1e-4, 0.6)
    saturation = np.clip(np.asarray(co2_saturation, dtype=float), 0.0, 1.0)
    shale = np.clip(np.asarray(shaliness, dtype=float), 0.0, 1.0)
    if not (vp.shape == vs.shape == density.shape == phi.shape == saturation.shape == shale.shape):
        raise ValueError("All local Gassmann arrays must have identical shapes")
    if not 0.0 < compatibility_margin < 1.0:
        raise ValueError("compatibility_margin must lie between zero and one")

    saturated_bulk, shear = elastic_moduli_gpa(vp, vs, density)
    mineral_prior = mineral_bulk_modulus_vrh(
        shale,
        quartz_bulk_modulus_gpa=quartz_bulk_modulus_gpa,
        clay_bulk_modulus_gpa=clay_bulk_modulus_gpa,
    )
    compatibility_denominator = 1.0 / np.maximum(saturated_bulk, 1e-8) - (
        phi / brine_bulk_modulus_gpa
    )
    compatibility_limit = np.where(
        compatibility_denominator > 1e-10,
        (1.0 - phi) / compatibility_denominator,
        mineral_prior,
    )
    minimum_mineral = saturated_bulk * 1.001
    effective_mineral = np.maximum(
        minimum_mineral,
        np.minimum(mineral_prior, compatibility_margin * compatibility_limit),
    )
    adjusted = effective_mineral < mineral_prior * (1.0 - 1e-8)
    dry_bulk = inverse_gassmann_dry_bulk(
        saturated_bulk,
        phi,
        effective_mineral,
        brine_bulk_modulus_gpa,
    )
    dry_bulk = np.clip(np.nan_to_num(dry_bulk, nan=0.0), 0.0, 0.999 * effective_mineral)
    fluid_bulk, fluid_density = brie_fluid_mixture(
        saturation,
        brine_bulk_modulus_gpa=brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=co2_bulk_modulus_gpa,
        brine_density_g_cc=brine_density_g_cc,
        co2_density_g_cc=co2_density_g_cc,
        brie_exponent=brie_exponent,
    )
    substituted_bulk = forward_gassmann_bulk(
        dry_bulk,
        phi,
        effective_mineral,
        fluid_bulk,
    )
    substituted_density = density + phi * (fluid_density - brine_density_g_cc)
    elastic = elastic_from_gpa(substituted_bulk, shear, substituted_density)
    zero = saturation <= 0.0
    elastic = ElasticProperties(
        np.where(zero, vp, elastic.vp),
        np.where(zero, vs, elastic.vs),
        np.where(zero, density, elastic.density),
    )
    return LocalGassmannResult(
        elastic=elastic,
        saturated_bulk_gpa=substituted_bulk,
        dry_bulk_gpa=dry_bulk,
        shear_gpa=shear,
        mineral_bulk_gpa=effective_mineral,
        fluid_bulk_gpa=fluid_bulk,
        adjusted_mineral_fraction=float(np.mean(adjusted)),
    )


def matched_hm_delta_substitution(
    elastic_brine: ElasticProperties,
    shaliness: np.ndarray,
    porosity: np.ndarray,
    co2_saturation: np.ndarray,
    **hm_parameters: float,
) -> ElasticProperties:
    """Apply only the matched HM CO2-minus-brine differential to an RF background."""
    saturation = np.asarray(co2_saturation, dtype=float)
    hm_brine = hertz_mindlin_gassmann(
        shaliness,
        porosity,
        np.zeros_like(saturation),
        **hm_parameters,
    )
    hm_co2 = hertz_mindlin_gassmann(
        shaliness,
        porosity,
        saturation,
        **hm_parameters,
    )
    zero = saturation <= 0.0
    return ElasticProperties(
        np.where(zero, elastic_brine.vp, elastic_brine.vp + hm_co2.vp - hm_brine.vp),
        np.where(zero, elastic_brine.vs, elastic_brine.vs + hm_co2.vs - hm_brine.vs),
        np.where(
            zero,
            elastic_brine.density,
            elastic_brine.density + hm_co2.density - hm_brine.density,
        ),
    )


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
    """Hertz--Mindlin dry frame followed by Gassmann substitution.

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
