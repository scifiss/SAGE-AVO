"""Generic seeded geometry applied consistently to supplied arrays (not field geology)."""

import numpy as np

from sage_avo import api


def main() -> None:
    time, trace = np.indices((48, 24))
    vp = 2500.0 + 3.0 * time + 10.0 * trace
    vs = 0.55 * vp
    facies = (time > 24).astype(np.uint8)
    deformation = api.make_coherent_deformation(vp.shape, seed=314159, maximum_faults=1)
    repeated = api.make_coherent_deformation(vp.shape, seed=314159, maximum_faults=1)
    np.testing.assert_array_equal(deformation.vertical_displacement, repeated.vertical_displacement)
    warped = api.warp_geological_arrays(
        {"vp": vp, "vs": vs, "facies": facies}, deformation, categorical_keys=("facies",)
    )
    assert all(value.shape == vp.shape for value in warped.values())
    np.testing.assert_allclose(warped["vs"], 0.55 * warped["vp"])
    print("Shared, seeded synthetic geometry verified; not field-conditioned.")


if __name__ == "__main__":
    main()
