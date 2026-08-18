import numpy as np

from sage_avo.evaluation.metrics import ssim_2d


def test_ssim_is_stable_for_smooth_large_offset_physical_fields():
    rows, columns = np.mgrid[:50, :100]
    target = 3200.0 + 0.4 * rows + 0.2 * columns
    prediction = target + 10.0 * np.sin(columns / 15.0)
    score = ssim_2d(prediction, target)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_ssim_of_identical_constant_field_is_one():
    field = np.full((24, 31), 2450.0)
    np.testing.assert_allclose(ssim_2d(field, field), 1.0, atol=1e-12)
