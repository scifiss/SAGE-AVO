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
