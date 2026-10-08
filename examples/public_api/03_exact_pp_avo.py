"""Caller-supplied specification: exact PP coefficients and three-band seismic."""

import numpy as np

from sage_avo import api


def main() -> None:
    specification = api.ForwardModelSpecification(
        specification_id="public_synthetic_fixture_exact_pp_3_45deg",
        angles_degrees=tuple(float(value) for value in range(3, 46)),
        bands=(
            api.AngleBand("near", 3.0, 17.0),
            api.AngleBand("mid", 17.0, 31.0),
            api.AngleBand("far", 31.0, 45.0),
        ),
        dt_seconds=0.004,
        wavelets=(
            api.WaveletSpecification(wavelet_id="ricker_14hz_zero_phase_controlled_baseline"),
        ),
    )
    vp = np.full((48, 6), 2700.0)
    vs = np.full_like(vp, 1450.0)
    density = np.full_like(vp, 2.25)
    vp[24:], vs[24:], density[24:] = 3200.0, 1750.0, 2.42
    coefficients = api.exact_pp_reflectivity(vp, vs, density, specification.angles_degrees)
    result = api.simulate_three_band_avo(vp, vs, density, specification)
    np.testing.assert_array_equal(result.reflectivity, coefficients.astype(np.float32))
    assert result.stacks.shape == (3, 48, 6)
    assert result.band_names == ("near", "mid", "far")
    print(f"Exact PP and near/mid/far AVO verified; specification SHA256={specification.sha256}")


if __name__ == "__main__":
    main()
