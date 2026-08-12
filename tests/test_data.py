import numpy as np

from sage_avo.data.patches import extract_patch
from sage_avo.data.splits import split_realizations


def test_realization_split_is_disjoint_and_reproducible():
    first = split_realizations(100, seed=9)
    second = split_realizations(100, seed=9)
    np.testing.assert_array_equal(first.train, second.train)
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert len(first.train) == 70
    assert len(first.validation) == 20
    assert len(first.test) == 10


def test_patch_retains_raw_scale_metadata():
    volume = np.arange(3 * 20 * 30, dtype=np.float32).reshape(3, 20, 30)
    patch, metadata = extract_patch(volume, 2, 3, (10, 20), (5, 10), realization_id=4)
    assert patch.shape == (3, 5, 10)
    assert metadata.scale_factors == (2.0, 2.0)
    assert metadata.realization_id == 4
