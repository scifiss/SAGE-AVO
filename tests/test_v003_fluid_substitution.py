import numpy as np

from sage_avo.geology.fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    calibrated_differential_gassmann_substitution,
)
from sage_avo.geology.rock_physics import (
    ElasticProperties,
    elastic_moduli_gpa,
    local_inverse_gassmann_substitution,
    matched_hm_delta_substitution,
)
from sage_avo.geology.synthetic import apply_co2_fluid_substitution_v003


def _calibration() -> CalibratedDryFrameModel:
    features = np.array(
        [
            [0.08, 0.20, 2.6],
            [0.10, 0.30, 2.7],
            [0.12, 0.40, 2.8],
            [0.14, 0.50, 2.9],
            [0.09, 0.60, 3.0],
            [0.11, 0.70, 3.1],
        ]
    )
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    return CalibratedDryFrameModel(
        calibration_id="unit_test_calibration",
        feature_names=("effective_porosity_fraction", "DELTA_shaliness_fraction", "depth_km"),
        feature_center=center,
        feature_scale=scale,
        features_standardized=(features - center) / scale,
        log_dry_bulk_gpa=np.log(np.array([8.5, 8.0, 7.5, 7.0, 7.8, 7.2])),
        log_shear_gpa=np.log(np.array([7.2, 7.0, 6.8, 6.5, 6.7, 6.4])),
        well_ids=np.array(["A", "A", "B", "B", "C", "C"]),
        neighbor_count=3,
        metadata={"method": "unit-test physical dry frames"},
    )


def _local_state(shape: tuple[int, int] = (21, 17)) -> tuple[np.ndarray, ...]:
    return (
        np.full(shape, 3200.0),
        np.full(shape, 1750.0),
        np.full(shape, 2.45),
        np.full(shape, 0.08),
        np.full(shape, 0.30),
    )


def test_local_inverse_gassmann_zero_saturation_reproduces_rf_brine():
    vp, vs, density, porosity, shaliness = _local_state()
    result = local_inverse_gassmann_substitution(
        vp,
        vs,
        density,
        porosity,
        shaliness,
        np.zeros_like(vp),
    )
    np.testing.assert_allclose(result.elastic.vp, vp, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(result.elastic.vs, vs, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(result.elastic.density, density, rtol=0.0, atol=1e-10)


def test_local_inverse_gassmann_preserves_shear_and_has_smooth_saturation_response():
    vp, vs, density, porosity, shaliness = _local_state((1, 1))
    sweep = np.linspace(0.0, 0.9, 19)
    outputs = [
        local_inverse_gassmann_substitution(
            vp,
            vs,
            density,
            porosity,
            shaliness,
            np.full_like(vp, saturation),
        )
        for saturation in sweep
    ]
    vp_values = np.array([result.elastic.vp.item() for result in outputs])
    vs_values = np.array([result.elastic.vs.item() for result in outputs])
    density_values = np.array([result.elastic.density.item() for result in outputs])
    assert np.all(np.diff(vp_values) <= 1e-8)
    assert np.all(np.diff(density_values) < 0.0)
    assert np.max(np.abs(np.diff(vp_values, n=2))) < 60.0
    assert (vs_values.max() - vs_values.min()) / vs_values[0] < 0.01
    for result in outputs:
        _, substituted_shear = elastic_moduli_gpa(
            result.elastic.vp,
            result.elastic.vs,
            result.elastic.density,
        )
        np.testing.assert_allclose(substituted_shear, result.shear_gpa, rtol=1e-12)
        assert np.isfinite(result.elastic.vp).all()
        assert np.all(result.elastic.vp > result.elastic.vs)
        assert np.all(result.elastic.vs > 0.0)


def test_v003_fluid_scenario_changes_only_plume_pixels():
    shape = (31, 41)
    vp, vs, density, porosity, _ = _local_state(shape)
    brine = np.stack((vp, vs, density)).astype(np.float32)
    scenario = apply_co2_fluid_substitution_v003(
        brine,
        porosity,
        np.full(shape, 0.9),
        np.ones(shape, dtype=bool),
        np.random.default_rng(12345),
        mode="local_inverse_gassmann",
        plume_count=(1, 1),
        lateral_radius_samples=(8.0, 8.0),
        vertical_radius_samples=(5.0, 5.0),
        minimum_sand_thickness_samples=1,
        co2_saturation=(0.5, 0.5),
    )
    plume = scenario.plume_mask.astype(bool)
    assert plume.any()
    np.testing.assert_array_equal(scenario.elastic[:, ~plume], brine[:, ~plume])
    assert np.all(scenario.elastic[0] > scenario.elastic[1])
    assert np.isfinite(scenario.elastic).all()
    assert np.all(scenario.elastic[2, plume] < brine[2, plume])


def test_matched_hm_delta_zero_saturation_reproduces_background():
    vp, vs, density, porosity, shaliness = _local_state((5, 4))
    background = ElasticProperties(vp, vs, density)
    result = matched_hm_delta_substitution(
        background,
        shaliness,
        porosity,
        np.zeros_like(vp),
    )
    np.testing.assert_array_equal(result.vp, vp)
    np.testing.assert_array_equal(result.vs, vs)
    np.testing.assert_array_equal(result.density, density)


def test_calibrated_differential_zero_saturation_and_fixed_shear():
    vp, vs, density, porosity, shaliness = _local_state((5, 4))
    physics = FluidRockPhysics()
    calibration = _calibration()
    zero = calibrated_differential_gassmann_substitution(
        vp,
        vs,
        density,
        porosity,
        shaliness,
        np.zeros_like(vp),
        np.full_like(vp, 2800.0),
        calibration,
        physics,
    )
    np.testing.assert_allclose(zero.elastic.vp, vp, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(zero.elastic.vs, vs, rtol=0.0, atol=1e-10)
    np.testing.assert_array_equal(zero.elastic.density, density)
    substituted = calibrated_differential_gassmann_substitution(
        vp,
        vs,
        density,
        porosity,
        shaliness,
        np.full_like(vp, 0.6),
        np.full_like(vp, 2800.0),
        calibration,
        physics,
    )
    _, target_shear = elastic_moduli_gpa(
        substituted.elastic.vp,
        substituted.elastic.vs,
        substituted.elastic.density,
    )
    np.testing.assert_allclose(target_shear, substituted.rf_shear_gpa, rtol=1e-12)
    assert np.all(substituted.delta_bulk_gpa < 0.0)
    assert np.all(substituted.delta_density_g_cc < 0.0)


def test_calibrated_v003_scenario_has_no_changes_outside_plume():
    shape = (31, 41)
    vp, vs, density, porosity, _ = _local_state(shape)
    brine = np.stack((vp, vs, density)).astype(np.float32)
    scenario = apply_co2_fluid_substitution_v003(
        brine,
        porosity,
        np.full(shape, 0.9),
        np.ones(shape, dtype=bool),
        np.random.default_rng(12345),
        mode="calibrated_differential_gassmann",
        fluid_calibration=_calibration(),
        depth_m=np.full(shape, 2800.0),
        plume_count=(1, 1),
        lateral_radius_samples=(8.0, 8.0),
        vertical_radius_samples=(5.0, 5.0),
        minimum_sand_thickness_samples=1,
        co2_saturation=(0.5, 0.5),
    )
    plume = scenario.plume_mask.astype(bool)
    assert plume.any()
    np.testing.assert_array_equal(scenario.elastic[:, ~plume], brine[:, ~plume])
    assert scenario.metadata["feasibility_projection_used"] is False
    assert scenario.metadata["dry_bulk_clipping_used"] is False
    assert scenario.metadata["elastic_output_clipping_used"] is False
    assert scenario.metadata["calibration_id"] == "unit_test_calibration"
