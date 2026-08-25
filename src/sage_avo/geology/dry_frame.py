"""Projection-free dry-frame families used by the Revision-3.3 support gate.

The functions in this module describe scenario families, not a posterior over
field rock physics.  Inputs and outputs are explicit GPa/fraction units, and
invalid states are reported instead of clipped into admissible ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedDryFrame:
    """One dry-frame family matched to a supplied dry-shear modulus."""

    bulk_gpa: np.ndarray
    shear_gpa: np.ndarray
    effective_pressure_mpa: np.ndarray
    relative_shear_misfit: np.ndarray
    valid: np.ndarray
    family: str


def _require_same_shape(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    converted = {name: np.asarray(value, dtype=float) for name, value in arrays.items()}
    shapes = {value.shape for value in converted.values()}
    if len(shapes) != 1:
        raise ValueError(f"Dry-frame arrays must have identical shapes: {arrays.keys()}")
    if not all(np.isfinite(value).all() for value in converted.values()):
        raise ValueError("Dry-frame inputs must be finite")
    return converted


def hertz_mindlin_end_member(
    mineral_bulk_gpa: np.ndarray,
    mineral_shear_gpa: np.ndarray,
    effective_pressure_mpa: np.ndarray,
    *,
    critical_porosity: float = 0.36,
    coordination_number: float = 7.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Hertz--Mindlin contact end member at critical porosity.

    The pressure is an effective-pressure scenario variable.  It is not
    inferred from depth and must not be described as measured S01 pressure.
    """
    values = _require_same_shape(
        mineral_bulk_gpa=mineral_bulk_gpa,
        mineral_shear_gpa=mineral_shear_gpa,
        effective_pressure_mpa=effective_pressure_mpa,
    )
    bulk = values["mineral_bulk_gpa"]
    shear = values["mineral_shear_gpa"]
    pressure_gpa = values["effective_pressure_mpa"] / 1000.0
    if not 0.0 < critical_porosity < 1.0:
        raise ValueError("critical_porosity must lie in (0, 1)")
    if coordination_number <= 0.0:
        raise ValueError("coordination_number must be positive")
    if np.any((bulk <= 0.0) | (shear <= 0.0) | (pressure_gpa <= 0.0)):
        raise ValueError("Mineral moduli and effective pressure must be positive")
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    contact_bulk = (
        coordination_number**2
        * (1.0 - critical_porosity) ** 2
        * shear**2
        * pressure_gpa
        / (18.0 * np.pi**2 * (1.0 - poisson) ** 2)
    ) ** (1.0 / 3.0)
    contact_shear = (5.0 - 4.0 * poisson) / (5.0 * (2.0 - poisson)) * (
        3.0
        * coordination_number**2
        * (1.0 - critical_porosity) ** 2
        * shear**2
        * pressure_gpa
        / (2.0 * np.pi**2 * (1.0 - poisson) ** 2)
    ) ** (1.0 / 3.0)
    return contact_bulk, contact_shear


def hashin_shtrikman_sand_frame(
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    mineral_shear_gpa: np.ndarray,
    effective_pressure_mpa: np.ndarray,
    *,
    family: str,
    critical_porosity: float = 0.36,
    coordination_number: float = 7.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return modified HS lower (soft) or upper (stiff) sand-frame moduli."""
    values = _require_same_shape(
        porosity=porosity,
        mineral_bulk_gpa=mineral_bulk_gpa,
        mineral_shear_gpa=mineral_shear_gpa,
        effective_pressure_mpa=effective_pressure_mpa,
    )
    phi = values["porosity"]
    mineral_bulk = values["mineral_bulk_gpa"]
    mineral_shear = values["mineral_shear_gpa"]
    pressure = values["effective_pressure_mpa"]
    if family not in {"soft_sand", "stiff_sand"}:
        raise ValueError("family must be 'soft_sand' or 'stiff_sand'")
    if np.any((phi <= 0.0) | (phi >= critical_porosity)):
        raise ValueError("Sand-frame porosity must lie in (0, critical_porosity)")
    contact_bulk, contact_shear = hertz_mindlin_end_member(
        mineral_bulk,
        mineral_shear,
        pressure,
        critical_porosity=critical_porosity,
        coordination_number=coordination_number,
    )
    fraction = phi / critical_porosity
    if family == "soft_sand":
        bulk_reference = contact_shear
        zeta_reference = contact_shear / 6.0 * (
            9.0 * contact_bulk + 8.0 * contact_shear
        ) / (contact_bulk + 2.0 * contact_shear)
        dry_bulk = 1.0 / (
            fraction / (contact_bulk + 4.0 * bulk_reference / 3.0)
            + (1.0 - fraction) / (mineral_bulk + 4.0 * bulk_reference / 3.0)
        ) - 4.0 * bulk_reference / 3.0
        dry_shear = 1.0 / (
            fraction / (contact_shear + zeta_reference)
            + (1.0 - fraction) / (mineral_shear + zeta_reference)
        ) - zeta_reference
    else:
        zeta_reference = mineral_shear / 6.0 * (
            9.0 * mineral_bulk + 8.0 * mineral_shear
        ) / (mineral_bulk + 2.0 * mineral_shear)
        dry_bulk = 1.0 / (
            fraction / (contact_bulk + 4.0 * mineral_shear / 3.0)
            + (1.0 - fraction) / (mineral_bulk + 4.0 * mineral_shear / 3.0)
        ) - 4.0 * mineral_shear / 3.0
        dry_shear = 1.0 / (
            fraction / (contact_shear + zeta_reference)
            + (1.0 - fraction) / (mineral_shear + zeta_reference)
        ) - zeta_reference
    return dry_bulk, dry_shear


def match_hashin_shtrikman_family_to_shear(
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    mineral_shear_gpa: np.ndarray,
    target_shear_gpa: np.ndarray,
    *,
    family: str,
    effective_pressure_range_mpa: tuple[float, float] = (0.1, 200.0),
    pressure_samples: int = 512,
    maximum_relative_shear_misfit: float = 0.15,
    critical_porosity: float = 0.36,
    coordination_number: float = 7.8,
) -> MatchedDryFrame:
    """Match an HS/Hertz--Mindlin family to fluid-independent dry shear."""
    values = _require_same_shape(
        porosity=porosity,
        mineral_bulk_gpa=mineral_bulk_gpa,
        mineral_shear_gpa=mineral_shear_gpa,
        target_shear_gpa=target_shear_gpa,
    )
    if effective_pressure_range_mpa[0] <= 0.0:
        raise ValueError("Effective-pressure search must be strictly positive")
    if effective_pressure_range_mpa[1] <= effective_pressure_range_mpa[0]:
        raise ValueError("Effective-pressure search range is not increasing")
    pressures = np.geomspace(*effective_pressure_range_mpa, num=pressure_samples)
    shape = values["porosity"].shape
    bulk_trials = np.empty((pressure_samples, *shape), dtype=float)
    shear_trials = np.empty_like(bulk_trials)
    for index, pressure in enumerate(pressures):
        pressure_array = np.full(shape, pressure, dtype=float)
        bulk_trials[index], shear_trials[index] = hashin_shtrikman_sand_frame(
            values["porosity"],
            values["mineral_bulk_gpa"],
            values["mineral_shear_gpa"],
            pressure_array,
            family=family,
            critical_porosity=critical_porosity,
            coordination_number=coordination_number,
        )
    target = values["target_shear_gpa"]
    indices = np.argmin(np.abs(np.log(shear_trials / target[None, ...])), axis=0)
    selected_bulk = np.take_along_axis(bulk_trials, indices[None, ...], axis=0)[0]
    selected_shear = np.take_along_axis(shear_trials, indices[None, ...], axis=0)[0]
    selected_pressure = pressures[indices]
    relative_misfit = np.abs(selected_shear - target) / target
    valid = (
        np.isfinite(selected_bulk)
        & np.isfinite(selected_shear)
        & (selected_bulk > 0.0)
        & (selected_bulk < values["mineral_bulk_gpa"])
        & (selected_shear > 0.0)
        & (relative_misfit <= maximum_relative_shear_misfit)
    )
    return MatchedDryFrame(
        bulk_gpa=selected_bulk,
        shear_gpa=selected_shear,
        effective_pressure_mpa=selected_pressure,
        relative_shear_misfit=relative_misfit,
        valid=valid,
        family=family,
    )


def constant_cement_power_law_bulk(
    porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    mineral_shear_gpa: np.ndarray,
    target_shear_gpa: np.ndarray,
    bulk_to_shear_exponent_ratio: float | np.ndarray,
    *,
    critical_porosity: float = 0.36,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an empirical constant-cement compaction trend tied to dry shear.

    This is explicitly a power-law constant-cement trend family, not the
    Dvorkin contact-cement model.  The exponent ratio must be calibrated to
    confirmed bulk-modulus anchors before use.
    """
    values = _require_same_shape(
        porosity=porosity,
        mineral_bulk_gpa=mineral_bulk_gpa,
        mineral_shear_gpa=mineral_shear_gpa,
        target_shear_gpa=target_shear_gpa,
    )
    phi = values["porosity"]
    if np.any((phi <= 0.0) | (phi >= critical_porosity)):
        raise ValueError("Constant-cement porosity must lie in (0, critical_porosity)")
    void_fraction = 1.0 - phi / critical_porosity
    shear_exponent = np.log(
        values["target_shear_gpa"] / values["mineral_shear_gpa"]
    ) / np.log(void_fraction)
    ratio = np.asarray(bulk_to_shear_exponent_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("Bulk/shear exponent ratio must be positive and finite")
    bulk = values["mineral_bulk_gpa"] * void_fraction ** (ratio * shear_exponent)
    return bulk, shear_exponent


def dry_poisson_ratio(bulk_gpa: np.ndarray, shear_gpa: np.ndarray) -> np.ndarray:
    """Return dry-frame isotropic Poisson ratio."""
    bulk = np.asarray(bulk_gpa, dtype=float)
    shear = np.asarray(shear_gpa, dtype=float)
    return (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
