"""Pressure/temperature/composition-aware pore-fluid properties.

Brine follows the empirical NaCl correlations of Batzle and Wang (1992).
Carbon-dioxide properties are evaluated with CoolProp's HEOS implementation of
the Span-Wagner reference equation.  Every public function uses explicit units
and rejects states outside the declared correlation or single-phase envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


BATZLE_WANG_DOI = "10.1190/1.1443207"
SPAN_WAGNER_DOI = "10.1063/1.555991"


@dataclass(frozen=True)
class FluidPropertyState:
    """One evaluated pore-fluid state with explicit units and provenance."""

    fluid: str
    pressure_mpa: float
    temperature_c: float
    salinity_mass_fraction: float | None
    density_g_cc: float
    acoustic_velocity_m_s: float
    bulk_modulus_gpa: float
    phase: str
    model: str
    model_version: str
    reference_doi: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)


_WATER_VELOCITY_COEFFICIENTS = np.asarray(
    [
        [1.40285e3, 1.52400, 3.43700e-3, -1.19700e-5],
        [4.87100, -1.11000e-2, 1.73900e-4, -1.62800e-6],
        [-4.78300e-2, 2.74700e-4, -2.13500e-6, 1.23700e-8],
        [1.48700e-4, -6.50300e-7, -1.45500e-8, 1.32700e-10],
        [-2.19700e-7, 7.98700e-10, 5.23000e-11, -4.61400e-13],
    ],
    dtype=float,
)


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def batzle_wang_brine(
    pressure_mpa: float,
    temperature_c: float,
    salinity_mass_fraction: float,
) -> FluidPropertyState:
    """Evaluate NaCl-brine density and adiabatic bulk modulus.

    The conservative declared empirical support is 5--60 MPa, 20--100 degrees C, and
    0--0.32 NaCl mass fraction.  Values outside that support are rejected rather
    than extrapolated.  Bulk modulus is ``rho * Vp**2`` in consistent units.
    """
    pressure = _finite_scalar("pressure_mpa", pressure_mpa)
    temperature = _finite_scalar("temperature_c", temperature_c)
    salinity = _finite_scalar("salinity_mass_fraction", salinity_mass_fraction)
    if not 5.0 <= pressure <= 60.0:
        raise ValueError("Batzle-Wang pressure must lie in [5, 60] MPa")
    if not 20.0 <= temperature <= 100.0:
        raise ValueError("Batzle-Wang temperature must lie in [20, 100] degC")
    if not 0.0 <= salinity <= 0.32:
        raise ValueError("Batzle-Wang NaCl mass fraction must lie in [0, 0.32]")

    density_water = 1.0 + 1.0e-6 * (
        -80.0 * temperature
        - 3.3 * temperature**2
        + 0.00175 * temperature**3
        + 489.0 * pressure
        - 2.0 * temperature * pressure
        + 0.016 * temperature**2 * pressure
        - 1.3e-5 * temperature**3 * pressure
        - 0.333 * pressure**2
        - 0.002 * temperature * pressure**2
    )
    correction = (
        300.0 * pressure
        - 2400.0 * pressure * salinity
        + temperature
        * (
            80.0
            + 3.0 * temperature
            - 3300.0 * salinity
            - 13.0 * pressure
            + 47.0 * pressure * salinity
        )
    )
    density_brine = density_water + salinity * (
        0.668 + 0.44 * salinity + 1.0e-6 * correction
    )

    water_velocity = sum(
        _WATER_VELOCITY_COEFFICIENTS[i, j]
        * temperature**i
        * pressure**j
        for i in range(5)
        for j in range(4)
    )
    salinity_linear = (
        1170.0
        - 9.6 * temperature
        + 0.055 * temperature**2
        - 8.5e-5 * temperature**3
        + 2.6 * pressure
        - 0.0029 * temperature * pressure
        - 0.0476 * pressure**2
    )
    salinity_three_halves = 780.0 - 10.0 * pressure + 0.16 * pressure**2
    brine_velocity = (
        water_velocity
        + salinity_linear * salinity
        + salinity_three_halves * salinity**1.5
        - 1820.0 * salinity**2
    )
    bulk_modulus = density_brine * brine_velocity**2 * 1.0e-6
    if density_brine <= 0.0 or brine_velocity <= 0.0 or bulk_modulus <= 0.0:
        raise ValueError("Batzle-Wang evaluation produced a non-physical brine state")
    return FluidPropertyState(
        fluid="NaCl brine",
        pressure_mpa=pressure,
        temperature_c=temperature,
        salinity_mass_fraction=salinity,
        density_g_cc=float(density_brine),
        acoustic_velocity_m_s=float(brine_velocity),
        bulk_modulus_gpa=float(bulk_modulus),
        phase="liquid",
        model="Batzle-Wang NaCl brine correlation",
        model_version="1992",
        reference_doi=BATZLE_WANG_DOI,
    )


def span_wagner_co2(
    pressure_mpa: float,
    temperature_c: float,
    *,
    require_supercritical: bool = True,
) -> FluidPropertyState:
    """Evaluate pure-CO2 properties with CoolProp HEOS/Span-Wagner.

    Two-phase, unknown, critical-point, and extrapolated states are rejected.
    The Revision-3.2 storage scenario additionally requires a supercritical
    phase; callers may disable that stricter requirement for unit tests only.
    """
    pressure = _finite_scalar("pressure_mpa", pressure_mpa)
    temperature = _finite_scalar("temperature_c", temperature_c)
    pressure_pa = pressure * 1.0e6
    temperature_k = temperature + 273.15
    try:
        import CoolProp
        from CoolProp.CoolProp import PhaseSI, PropsSI
    except ImportError as error:
        raise ImportError(
            "CO2 EOS evaluation requires CoolProp; install the project dependencies"
        ) from error

    maximum_pressure = float(PropsSI("PMAX", "CarbonDioxide"))
    minimum_temperature = float(PropsSI("TMIN", "CarbonDioxide"))
    maximum_temperature = float(PropsSI("TMAX", "CarbonDioxide"))
    if pressure_pa <= 0.0 or pressure_pa > maximum_pressure:
        raise ValueError("CO2 pressure lies outside the CoolProp fluid limit")
    if not minimum_temperature <= temperature_k <= maximum_temperature:
        raise ValueError("CO2 temperature lies outside the CoolProp fluid limit")
    phase = str(PhaseSI("P", pressure_pa, "T", temperature_k, "HEOS::CarbonDioxide"))
    rejected = {"twophase", "unknown", "critical_point"}
    if phase.lower() in rejected:
        raise ValueError(f"CO2 phase {phase!r} is not an accepted single-phase state")
    supercritical_phases = {"supercritical", "supercritical_gas", "supercritical_liquid"}
    if require_supercritical and phase.lower() not in supercritical_phases:
        raise ValueError(f"CO2 phase {phase!r} is not supercritical")
    density_kg_m3 = float(
        PropsSI("Dmass", "P", pressure_pa, "T", temperature_k, "HEOS::CarbonDioxide")
    )
    acoustic_velocity = float(
        PropsSI("A", "P", pressure_pa, "T", temperature_k, "HEOS::CarbonDioxide")
    )
    bulk_modulus = density_kg_m3 * acoustic_velocity**2 / 1.0e9
    if density_kg_m3 <= 0.0 or acoustic_velocity <= 0.0 or bulk_modulus <= 0.0:
        raise ValueError("Span-Wagner evaluation produced a non-physical CO2 state")
    return FluidPropertyState(
        fluid="pure CO2",
        pressure_mpa=pressure,
        temperature_c=temperature,
        salinity_mass_fraction=None,
        density_g_cc=density_kg_m3 / 1000.0,
        acoustic_velocity_m_s=acoustic_velocity,
        bulk_modulus_gpa=bulk_modulus,
        phase=phase,
        model="CoolProp HEOS Span-Wagner CO2 equation of state",
        model_version=str(CoolProp.__version__),
        reference_doi=SPAN_WAGNER_DOI,
    )


def sample_fluid_scenario(
    realization_id: int,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Sample one reproducible Revision-3.2 fluid state from declared ranges."""
    seed = int(scenario["seed_offset"]) + int(realization_id)
    rng = np.random.default_rng(seed)

    def draw(name: str) -> float:
        lower, upper = (float(value) for value in scenario[name])
        return float(rng.uniform(lower, upper))

    pressure = draw("pressure_mpa")
    temperature = draw("temperature_c")
    salinity = draw("salinity_mass_fraction")
    brie_exponent = draw("brie_exponent")
    brine = batzle_wang_brine(pressure, temperature, salinity)
    co2 = span_wagner_co2(pressure, temperature, require_supercritical=True)
    return {
        "sampling_seed": seed,
        "sampling_rule": "independent uniform draws in configured closed ranges",
        "realization_id": int(realization_id),
        "brie_exponent": brie_exponent,
        "brine": brine.to_dict(),
        "co2": co2.to_dict(),
    }
