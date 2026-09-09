import numpy as np
import pytest

from sage_avo.evaluation.metrics import ssim_2d


torch = pytest.importorskip("torch")
inference = pytest.importorskip("sage_avo.evaluation.inference")


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


class _InferenceModeProbe(torch.nn.Module):
    def sample(self, avo, low, rgt, **_):
        assert torch.is_inference_mode_enabled()
        return low + 0.25

    def forward(self, prediction, time, avo, low, rgt):
        assert torch.is_inference_mode_enabled()
        logits = torch.stack((rgt, -rgt, torch.zeros_like(rgt)), dim=1)
        return type("Output", (), {"segmentation_logits": logits})()


def test_single_tile_inference_matches_direct_small_case():
    rng = np.random.default_rng(7)
    avo = rng.normal(size=(3, 8, 9)).astype(np.float32)
    low = rng.normal(size=(3, 8, 9)).astype(np.float32)
    rgt = rng.normal(size=(8, 9)).astype(np.float32)
    normalization = {
        "x_mean": [0.0, 0.0, 0.0],
        "x_std": [1.0, 1.0, 1.0],
        "y_mean": [10.0, 20.0, 30.0],
        "y_std": [2.0, 3.0, 4.0],
    }
    elastic, labels = inference.infer_full_realization(
        _InferenceModeProbe(),
        avo=avo,
        low=low,
        rgt=rgt,
        normalization=normalization,
        patch_shape=(8, 9),
        stride=(8, 9),
        steps=2,
        batch_size=1,
        device=torch.device("cpu"),
    )
    mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    normalized_low = (low - mean) / std
    np.testing.assert_allclose(elastic, (normalized_low + 0.25) * std + mean)
    np.testing.assert_array_equal(labels, np.where(rgt >= 0.0, 0, 1).astype(np.uint8))
