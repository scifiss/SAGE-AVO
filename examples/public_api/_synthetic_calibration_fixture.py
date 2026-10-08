"""Deliberately synthetic physical fixture; never an approved field calibration."""

import numpy as np

from sage_avo import api


def calibration_and_support():
    features = np.asarray([[-0.02, -0.02, -0.02], [0.0, 0.0, 0.0], [0.02, 0.02, 0.02]])
    calibration = api.CalibratedDryFrameModel(
        calibration_id="synthetic_demonstration_not_field_approved",
        feature_names=("effective_porosity", "shaliness", "depth_km"),
        feature_center=np.asarray([0.15, 0.20, 2.50]),
        feature_scale=np.ones(3),
        features_standardized=features,
        log_dry_bulk_gpa=np.log(np.full(3, 10.0)),
        log_shear_gpa=np.log(np.full(3, 8.0)),
        well_ids=np.asarray(["synthetic_A", "synthetic_B", "synthetic_C"]),
        neighbor_count=2,
        metadata={"scientific_status": "unit_fixture_only"},
    )
    contract = api.support_contract_from_mapping(
        {
            "calibration_id": calibration.calibration_id,
            "nearest_neighbor_training_quantile": 0.99,
            "physical_depth_domain_m": [2400.0, 2600.0],
            "physical_porosity_domain_fraction": [0.10, 0.20],
            "minimum_overall_coverage": 1.0,
            "minimum_class_coverage": 1.0,
            "facies_shaliness_boundary": 0.50,
            "depth_class_boundaries_m": [2450.0, 2550.0],
            "dry_bulk_to_shear_range": [0.5, 2.0],
            "dry_poisson_ratio_range": [0.0, 0.45],
            "maximum_fixed_shear_error_gpa": 1e-5,
            "maximum_outside_plume_change": 0.0,
            "scenario_pressure_range_mpa": [24.0, 36.0],
            "scenario_temperature_range_c": [55.0, 95.0],
            "scenario_salinity_range_fraction": [0.006, 0.12],
            "scenario_brie_exponent_range": [2.0, 4.0],
        },
        calibration,
    )
    return calibration, contract
