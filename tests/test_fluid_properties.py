import numpy as np
import pytest

from sage_avo.geology.fluid_properties import (
    batzle_wang_brine,
    sample_fluid_scenario,
    span_wagner_co2,
)


def test_batzle_wang_reference_state_is_physical() -> None:
    state = batzle_wang_brine(30.0, 80.0, 0.08)
    assert 1.0 < state.density_g_cc < 1.2
    assert 1400.0 < state.acoustic_velocity_m_s < 1900.0
    assert np.isclose(
        state.bulk_modulus_gpa,
        state.density_g_cc * state.acoustic_velocity_m_s**2 * 1.0e-6,
    )


@pytest.mark.parametrize(
    ("pressure", "temperature", "salinity"),
    [
        (4.9, 80.0, 0.08),
        (60.1, 80.0, 0.08),
        (30.0, 19.9, 0.08),
        (30.0, 100.1, 0.08),
        (30.0, 80.0, 0.321),
    ],
)
def test_batzle_wang_rejects_extrapolation(
    pressure: float,
    temperature: float,
    salinity: float,
) -> None:
    with pytest.raises(ValueError):
        batzle_wang_brine(pressure, temperature, salinity)


def test_span_wagner_supercritical_state_is_physical() -> None:
    state = span_wagner_co2(30.0, 80.0)
    assert state.phase.lower().startswith("supercritical")
    assert 0.1 < state.density_g_cc < 1.2
    assert state.bulk_modulus_gpa > 0.0


def test_scenario_sampling_is_reproducible() -> None:
    scenario = {
        "seed_offset": 3_200_000,
        "pressure_mpa": [24.0, 36.0],
        "temperature_c": [55.0, 95.0],
        "salinity_mass_fraction": [0.006, 0.12],
        "brie_exponent": [2.0, 4.0],
    }
    first = sample_fluid_scenario(17, scenario)
    second = sample_fluid_scenario(17, scenario)
    assert first == second
    assert first["brine"]["pressure_mpa"] == first["co2"]["pressure_mpa"]
