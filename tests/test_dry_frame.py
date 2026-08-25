import numpy as np

from sage_avo.geology.dry_frame import (
    constant_cement_power_law_bulk,
    dry_poisson_ratio,
    hashin_shtrikman_sand_frame,
    match_hashin_shtrikman_family_to_shear,
)


def test_soft_and_stiff_sand_frames_are_finite_and_ordered() -> None:
    phi = np.array([0.08, 0.12, 0.18])
    bulk = np.array([32.0, 30.0, 28.0])
    shear = np.array([25.0, 22.0, 19.0])
    pressure = np.full(3, 30.0)
    soft_bulk, soft_shear = hashin_shtrikman_sand_frame(
        phi, bulk, shear, pressure, family="soft_sand"
    )
    stiff_bulk, stiff_shear = hashin_shtrikman_sand_frame(
        phi, bulk, shear, pressure, family="stiff_sand"
    )
    assert np.isfinite([soft_bulk, soft_shear, stiff_bulk, stiff_shear]).all()
    assert np.all(stiff_bulk >= soft_bulk)
    assert np.all(stiff_shear >= soft_shear)


def test_soft_sand_pressure_match_recovers_target_shear() -> None:
    phi = np.array([0.08, 0.10])
    bulk = np.array([31.0, 29.0])
    shear = np.array([23.0, 20.0])
    pressure = np.array([25.0, 50.0])
    _, target = hashin_shtrikman_sand_frame(
        phi, bulk, shear, pressure, family="soft_sand"
    )
    matched = match_hashin_shtrikman_family_to_shear(
        phi, bulk, shear, target, family="soft_sand", pressure_samples=1024
    )
    assert matched.valid.all()
    assert np.max(matched.relative_shear_misfit) < 0.005


def test_constant_cement_trend_is_projection_free_and_physical() -> None:
    phi = np.array([0.08, 0.12])
    mineral_bulk = np.array([31.0, 29.0])
    mineral_shear = np.array([23.0, 20.0])
    target_shear = np.array([9.0, 7.0])
    dry_bulk, exponent = constant_cement_power_law_bulk(
        phi,
        mineral_bulk,
        mineral_shear,
        target_shear,
        np.array([1.2, 1.8]),
    )
    poisson = dry_poisson_ratio(dry_bulk, target_shear)
    assert np.all(exponent > 0.0)
    assert np.all((dry_bulk > 0.0) & (dry_bulk < mineral_bulk))
    assert np.isfinite(poisson).all()
