import numpy as np

from sage_avo.geology.conventions import delta_from_sand_probability, sand_probability_from_delta


def test_delta_is_one_minus_sand_probability():
    sand = np.array([0.0, 0.25, 1.0])
    delta = delta_from_sand_probability(sand)
    np.testing.assert_allclose(delta, [1.0, 0.75, 0.0])
    np.testing.assert_allclose(sand_probability_from_delta(delta), sand)
