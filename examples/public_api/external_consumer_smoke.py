"""Run OUTSIDE the source tree against an installed, commit-pinned SAGE-AVO.

Prints a small JSON result. No proprietary data, training, checkpoint or GPU.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from sage_avo import api


def main() -> None:
    assert "torch" not in sys.modules and "torch_geometric" not in sys.modules
    available = api.list_operations(include_unavailable=False)
    assert any(row["identifier"] == "elastic_moduli" for row in available)
    vp = np.full((20, 4), 2800.0)
    vs = np.full_like(vp, 1400.0)
    density = np.full_like(vp, 2.2)
    vp[10:], vs[10:], density[10:] = 3100.0, 1600.0, 2.3
    bulk, shear = api.elastic_moduli(vp, vs, density)
    recovered = api.elastic_from_moduli(bulk, shear, density)
    np.testing.assert_allclose(recovered.vp, vp, rtol=1e-14)
    np.testing.assert_allclose(recovered.vs, vs, rtol=1e-14)
    specification = api.ForwardModelSpecification(
        specification_id="external_consumer_synthetic_fixture",
        angles_degrees=(3.0, 10.0, 17.0, 24.0, 31.0, 38.0, 45.0),
        bands=(
            api.AngleBand("near", 3.0, 17.0),
            api.AngleBand("mid", 17.0, 31.0),
            api.AngleBand("far", 31.0, 45.0),
        ),
        dt_seconds=0.004,
        wavelets=(api.WaveletSpecification(wavelet_id="ricker14_synthetic_fixture"),),
    )
    coefficients = api.exact_pp_reflectivity(vp, vs, density, specification.angles_degrees)
    avo = api.simulate_three_band_avo(vp, vs, density, specification)
    assert coefficients.shape == (7, 20, 4)
    assert avo.stacks.shape == (3, 20, 4)
    np.testing.assert_allclose(avo.reflectivity, coefficients, rtol=0, atol=1e-7)
    assert api.compare_avo_outputs(avo.stacks, avo.stacks).normalized_rmse[0] < 1e-6
    try:
        api.elastic_moduli(vs, vp, density)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid elastic state was not rejected")
    try:
        api.substitute_calibrated_fluid(
            np.stack((vp, vs, density)),
            np.full_like(vp, 0.15),
            np.full_like(vp, 0.2),
            np.zeros_like(vp),
            np.full_like(vp, 2500.0),
            np.ones_like(vp, dtype=bool),
        )
    except ValueError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("missing calibration was not rejected")
    assert "torch" not in sys.modules and "torch_geometric" not in sys.modules
    print(
        json.dumps(
            {
                "external_consumer": "PASS",
                "api_version": api.API_VERSION,
                "package_version": api.provenance()["package_version"],
                "public_count": len(available),
                "forward_specification_sha256": specification.sha256,
                "no_torch_import": True,
                "no_gpu_initialization": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
