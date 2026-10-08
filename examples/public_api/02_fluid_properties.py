"""Independent fluid states and explicitly synthetic Candidate-B substitution."""

import numpy as np

from sage_avo import api

from _synthetic_calibration_fixture import calibration_and_support


def main() -> None:
    brine = api.brine_fluid_state(30.0, 80.0, 0.08)
    co2 = api.co2_fluid_state(30.0, 80.0)
    assert brine.bulk_modulus_gpa > 0 and co2.bulk_modulus_gpa > 0
    calibration, support = calibration_and_support()
    physics = api.FluidRockPhysics()
    shape = (8, 6)
    mineral_density = 0.8 * physics.quartz_density_g_cc + 0.2 * physics.clay_density_g_cc
    density = mineral_density - 0.15 * (mineral_density - physics.brine_density_g_cc)
    baseline = np.stack((np.full(shape, 3500.0), np.full(shape, 1900.0), np.full(shape, density)))
    region = np.zeros(shape, bool)
    region[2:6, 2:5] = True
    saturation = np.zeros(shape)
    common = dict(
        brine_elastic=baseline,
        input_porosity=np.full(shape, 0.15),
        shaliness=np.full(shape, 0.20),
        depth_m=np.full(shape, 2500.0),
        region_mask=region,
        calibration=calibration,
        physics=physics,
        support=support,
    )
    zero = api.substitute_calibrated_fluid(co2_saturation=saturation, **common)
    np.testing.assert_array_equal(zero.elastic, baseline)
    saturation[region] = 0.5
    result = api.substitute_calibrated_fluid(co2_saturation=saturation, **common)
    np.testing.assert_array_equal(result.elastic[:, ~region], baseline[:, ~region])
    assert result.calibration_id == "synthetic_demonstration_not_field_approved"
    print(
        "Supported fluid states and synthetic-fixture substitution verified; no field approval implied."
    )


if __name__ == "__main__":
    main()
