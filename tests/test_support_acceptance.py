"""Unit tests for deterministic whole-candidate support acceptance."""

from __future__ import annotations

import numpy as np

from sage_avo.geology import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    deterministic_candidate_seed,
    evaluate_candidate_support,
    support_contract_from_mapping,
)


def _calibration() -> CalibratedDryFrameModel:
    features = np.asarray(
        [
            [-0.02, -0.02, -0.02],
            [0.00, 0.00, 0.00],
            [0.02, 0.02, 0.02],
        ]
    )
    return CalibratedDryFrameModel(
        calibration_id="unit-calibration",
        feature_names=("effective_porosity", "shaliness", "depth_km"),
        feature_center=np.asarray([0.15, 0.20, 2.50]),
        feature_scale=np.ones(3),
        features_standardized=features,
        log_dry_bulk_gpa=np.log(np.asarray([10.0, 10.0, 10.0])),
        log_shear_gpa=np.log(np.asarray([8.0, 8.0, 8.0])),
        well_ids=np.asarray(["A", "B", "C"]),
        neighbor_count=2,
        metadata={
            "time_depth_linear_coefficients": {
                "slope_m_per_ms": 1.0,
                "intercept_m": 0.0,
            }
        },
    )


def _contract(calibration: CalibratedDryFrameModel):
    return support_contract_from_mapping(
        {
            "calibration_id": calibration.calibration_id,
            "nearest_neighbor_training_quantile": 0.99,
            "physical_depth_domain_m": [2400.0, 3250.0],
            "physical_porosity_domain_fraction": [0.01, 0.30],
            "minimum_overall_coverage": 0.95,
            "minimum_class_coverage": 0.90,
            "facies_shaliness_boundary": 0.50,
            "depth_class_boundaries_m": [2850.0, 3100.0],
            "dry_bulk_to_shear_range": [0.30, 4.00],
            "dry_poisson_ratio_range": [0.00, 0.45],
            "maximum_fixed_shear_error_gpa": 1e-5,
            "maximum_outside_plume_change": 0.0,
            "scenario_pressure_range_mpa": [24.0, 36.0],
            "scenario_temperature_range_c": [55.0, 95.0],
            "scenario_salinity_range_fraction": [0.006, 0.12],
            "scenario_brie_exponent_range": [2.0, 4.0],
        },
        calibration,
    )


def _fluid_metadata() -> dict[str, object]:
    return {
        "feasibility_projection_used": False,
        "dry_bulk_clipping_used": False,
        "elastic_output_clipping_used": False,
        "direct_independent_elastic_delta_transfer": False,
        "shear_modulus_changed_by_fluid": False,
        "property_state": {
            "brine": {
                "pressure_mpa": 30.0,
                "temperature_c": 80.0,
                "salinity_mass_fraction": 0.06,
            },
            "co2": {"phase": "supercritical"},
            "brie_exponent": 3.0,
        },
    }


def test_retry_seed_is_stable_and_signed_int64_safe() -> None:
    assert deterministic_candidate_seed(1234, 0, namespace="test") == 1234
    first = deterministic_candidate_seed(1234, 1, namespace="test")
    assert first == deterministic_candidate_seed(1234, 1, namespace="test")
    assert first != deterministic_candidate_seed(1234, 2, namespace="test")
    assert 0 <= first < 2**63


def test_support_accepts_physical_in_domain_candidate() -> None:
    calibration = _calibration()
    physics = FluidRockPhysics(brine_density_g_cc=1.03)
    # This density closes to phi_eff=0.15 for a 20% shale VRH mixture.
    mineral_density = 0.8 * physics.quartz_density_g_cc + 0.2 * physics.clay_density_g_cc
    density = mineral_density - 0.15 * (mineral_density - physics.brine_density_g_cc)
    elastic = np.empty((3, 2, 2), dtype=float)
    elastic[0] = 3500.0
    elastic[1] = 1900.0
    elastic[2] = density
    report = evaluate_candidate_support(
        elastic=elastic,
        elastic_brine=elastic.copy(),
        shaliness=np.full((2, 2), 0.20),
        plume_mask=np.ones((2, 2), dtype=bool),
        co2_saturation=np.zeros((2, 2)),
        time_ms=np.asarray([2500.0, 2500.0]),
        fluid_metadata=_fluid_metadata(),
        calibration=calibration,
        physics=physics,
        contract=_contract(calibration),
    )
    assert report.accepted
    assert report.rejection_reasons == ()


def test_support_rejects_complete_out_of_depth_candidate() -> None:
    calibration = _calibration()
    physics = FluidRockPhysics(brine_density_g_cc=1.03)
    mineral_density = 0.8 * physics.quartz_density_g_cc + 0.2 * physics.clay_density_g_cc
    density = mineral_density - 0.15 * (mineral_density - physics.brine_density_g_cc)
    elastic = np.stack(
        (
            np.full((2, 2), 3500.0),
            np.full((2, 2), 1900.0),
            np.full((2, 2), density),
        )
    )
    report = evaluate_candidate_support(
        elastic=elastic,
        elastic_brine=elastic.copy(),
        shaliness=np.full((2, 2), 0.20),
        plume_mask=np.ones((2, 2), dtype=bool),
        co2_saturation=np.zeros((2, 2)),
        time_ms=np.asarray([2390.0, 2500.0]),
        fluid_metadata=_fluid_metadata(),
        calibration=calibration,
        physics=physics,
        contract=_contract(calibration),
    )
    assert not report.accepted
    assert "physical_depth_domain" in report.rejection_reasons
