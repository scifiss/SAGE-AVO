"""Scientific invariants and direct source parity for the independent v0 facade."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sage_avo import api
from sage_avo.config import load_config
from sage_avo.forward.pipeline import forward_avo_dense_spec
from sage_avo.forward.qc import compare_forward_outputs
from sage_avo.forward.zoeppritz import reflectivity_gather
from sage_avo.geology.fluid_calibration import calibrated_differential_gassmann_substitution
from sage_avo.geology.rock_physics import elastic_moduli_gpa

ROOT = Path(__file__).resolve().parents[1]


def elastic_fixture(shape=(30, 8)):
    vp = np.full(shape, 2800.0)
    vs = np.full(shape, 1400.0)
    density = np.full(shape, 2.2)
    vp[shape[0] // 2 :] = 3100.0
    vs[shape[0] // 2 :] = 1600.0
    density[shape[0] // 2 :] = 2.3
    return vp, vs, density


def test_catalog_is_detached_and_discovery_does_not_import_ml():
    operations = api.list_operations()
    assert api.API_VERSION == "0.1.0"
    assert api.get_operation("exact_pp_reflectivity")["eligibility"] == "STABLE_NUMERICAL"
    assert api.get_operation("experimental_rgt_gnn")["available"] is False
    operations[0]["purpose"] = "changed by consumer"
    assert api.list_operations()[0]["purpose"] != "changed by consumer"
    script = "import sys; import sage_avo.api as a; a.list_operations(); assert 'torch' not in sys.modules; assert 'torch_geometric' not in sys.modules; assert 'matplotlib' not in sys.modules"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_deformation_shared_geometry_and_seed():
    one = api.make_coherent_deformation((32, 20), 123, maximum_faults=1)
    two = api.make_coherent_deformation((32, 20), 123, maximum_faults=1)
    np.testing.assert_array_equal(one.vertical_displacement, two.vertical_displacement)
    field = np.arange(32 * 20, dtype=float).reshape(32, 20)
    warped = api.warp_geological_arrays(
        {"vp": field, "vs": 2 * field, "mask": field > 100}, one, categorical_keys=("mask",)
    )
    np.testing.assert_allclose(warped["vs"], 2 * warped["vp"])
    assert set(np.unique(warped["mask"])).issubset({0.0, 1.0})
    with pytest.raises(ValueError, match="categorical_keys"):
        api.warp_geological_arrays({"vp": field}, one, categorical_keys=("absent",))


def test_elastic_conversion_parity_and_invalid_state():
    vp, vs, density = elastic_fixture()
    bulk, shear = api.elastic_moduli(vp, vs, density)
    reference = elastic_moduli_gpa(vp, vs, density)
    np.testing.assert_array_equal(bulk, reference[0])
    np.testing.assert_array_equal(shear, reference[1])
    inverse = api.elastic_from_moduli(bulk, shear, density)
    np.testing.assert_allclose(inverse.vp, vp, rtol=1e-15)
    np.testing.assert_allclose(inverse.vs, vs, rtol=1e-15)
    with pytest.raises(ValueError, match="Vp > Vs"):
        api.elastic_moduli(vs, vp, density)
    with pytest.raises(ValueError, match="non-positive"):
        api.elastic_from_moduli(np.full_like(bulk, -1), shear, density)


def test_fluid_states_use_source_models_and_reject_invalid_support():
    brine = api.brine_fluid_state(30.0, 80.0, 0.08)
    assert brine.fluid == "NaCl brine"
    assert brine.bulk_modulus_gpa > 0
    with pytest.raises(ValueError, match="pressure"):
        api.brine_fluid_state(1.0, 80.0, 0.08)
    coolprop = pytest.importorskip("CoolProp")
    assert coolprop is not None
    co2 = api.co2_fluid_state(30.0, 80.0)
    assert co2.phase.lower().startswith("supercritical")


def test_missing_optional_co2_dependency_has_actionable_error():
    script = """
import importlib.abc
import sys
class BlockCoolProp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] == 'CoolProp':
            raise ModuleNotFoundError('blocked CoolProp')
        return None
sys.meta_path.insert(0, BlockCoolProp())
from sage_avo import api
try:
    api.co2_fluid_state(30.0, 80.0)
except ImportError as error:
    assert 'CoolProp' in str(error)
else:
    raise AssertionError('missing dependency was not rejected')
assert 'torch' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_exact_pp_and_spec_forward_parity_and_provenance():
    vp, vs, density = elastic_fixture()
    config = load_config(ROOT / "configs/forward_model_v003.yaml")
    specification = api.forward_specification_from_mapping(config)
    actual = api.exact_pp_reflectivity(vp, vs, density, specification.angles_degrees)
    expected = reflectivity_gather(vp, vs, density, np.asarray(specification.angles_degrees))
    np.testing.assert_array_equal(actual, expected)
    output = api.simulate_three_band_avo(vp, vs, density, specification)
    reference = forward_avo_dense_spec(vp, vs, density, specification)
    np.testing.assert_array_equal(output.reflectivity, reference.reflectivity)
    np.testing.assert_array_equal(output.seismic, reference.seismic)
    np.testing.assert_array_equal(output.stacks, reference.stacks)
    assert output.stacks.shape == (3, *vp.shape)
    assert output.band_names == ("near", "mid", "far")
    agreement = api.compare_avo_outputs(output.stacks, reference.stacks)
    assert agreement == compare_forward_outputs(output.stacks, reference.stacks)
    record = api.provenance(
        "simulate_three_band_avo",
        specification=specification,
        input_configuration={"sample_origin": 0},
    )
    assert record["forward_specification_sha256"] == specification.sha256
    assert record["operator_version"] != record["package_version"]
    assert record["input_configuration_sha256"] is not None


def test_public_pp_rejects_postcritical_real_component_only():
    vp, vs, density = elastic_fixture((8, 2))
    vp[:4] = 1800
    vp[4:] = 4000
    vs[:4] = 800
    vs[4:] = 1800
    with pytest.raises(ValueError, match="post-critical"):
        api.exact_pp_reflectivity(vp, vs, density, (45.0,))


def _unit_calibration_and_support():
    features = np.asarray([[-0.02, -0.02, -0.02], [0.0, 0.0, 0.0], [0.02, 0.02, 0.02]])
    calibration = api.CalibratedDryFrameModel(
        calibration_id="synthetic_unit_fixture_not_field_approved",
        feature_names=("effective_porosity", "shaliness", "depth_km"),
        feature_center=np.asarray([0.15, 0.20, 2.50]),
        feature_scale=np.ones(3),
        features_standardized=features,
        log_dry_bulk_gpa=np.log(np.full(3, 10.0)),
        log_shear_gpa=np.log(np.full(3, 8.0)),
        well_ids=np.asarray(["A", "B", "C"]),
        neighbor_count=2,
        metadata={"fixture_only": True},
    )
    support = api.support_contract_from_mapping(
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
    return calibration, support


def test_calibrated_substitution_requires_calibration_and_preserves_exterior():
    calibration, support = _unit_calibration_and_support()
    physics = api.FluidRockPhysics()
    shape = (5, 4)
    mineral_density = 0.8 * physics.quartz_density_g_cc + 0.2 * physics.clay_density_g_cc
    density = mineral_density - 0.15 * (mineral_density - physics.brine_density_g_cc)
    baseline = np.stack((np.full(shape, 3500.0), np.full(shape, 1900.0), np.full(shape, density)))
    phi = np.full(shape, 0.15)
    shale = np.full(shape, 0.20)
    depth = np.full(shape, 2500.0)
    region = np.zeros(shape, bool)
    region[1:4, 1:3] = True
    saturation = np.zeros(shape)
    with pytest.raises(ValueError, match="calibration"):
        api.substitute_calibrated_fluid(baseline, phi, shale, saturation, depth, region)
    zero = api.substitute_calibrated_fluid(
        baseline,
        phi,
        shale,
        saturation,
        depth,
        region,
        calibration=calibration,
        physics=physics,
        support=support,
    )
    np.testing.assert_array_equal(zero.elastic, baseline)
    saturation[region] = 0.5
    result = api.substitute_calibrated_fluid(
        baseline,
        phi,
        shale,
        saturation,
        depth,
        region,
        calibration=calibration,
        physics=physics,
        support=support,
    )
    np.testing.assert_array_equal(result.elastic[:, ~region], baseline[:, ~region])
    source = calibrated_differential_gassmann_substitution(
        baseline[0][region],
        baseline[1][region],
        baseline[2][region],
        phi[region],
        shale[region],
        saturation[region],
        depth[region],
        calibration,
        physics,
    )
    np.testing.assert_allclose(result.elastic[0][region], source.elastic.vp)
    np.testing.assert_allclose(result.elastic[1][region], source.elastic.vs)
    np.testing.assert_allclose(result.elastic[2][region], source.elastic.density)
    assert result.calibration_id == calibration.calibration_id
    with pytest.raises(ValueError, match="outside region_mask"):
        invalid = saturation.copy()
        invalid[0, 0] = 0.5
        api.substitute_calibrated_fluid(
            baseline,
            phi,
            shale,
            invalid,
            depth,
            region,
            calibration=calibration,
            physics=physics,
            support=support,
        )


def test_rgt_qc_and_figure_use_existing_arrays_only():
    rgt = np.broadcast_to(np.arange(8)[:, None], (8, 4)).copy()
    report = api.report_rgt_monotonicity(rgt)
    assert report["fraction_bad"] == 0.0
    vp, vs, density = elastic_fixture((8, 4))
    data = np.stack((vp, vs, density))
    figure = api.plot_elastic_comparison(data, data, data)
    assert len(figure.axes) >= 12
    import matplotlib.pyplot as plt

    plt.close(figure)
