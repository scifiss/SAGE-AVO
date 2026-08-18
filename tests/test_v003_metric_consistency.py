import numpy as np

from sage_avo.evaluation.metrics import elastic_metrics


def test_normalized_and_physical_rmse_ordering_agree_for_same_samples_and_mask():
    rng = np.random.default_rng(12345)
    target = rng.normal(size=(3, 18, 11))
    prediction_a = target + rng.normal(size=target.shape) * np.array(
        [0.20, 0.15, 0.10]
    )[:, None, None]
    prediction_b = target + rng.normal(size=target.shape) * np.array(
        [0.35, 0.25, 0.18]
    )[:, None, None]
    mask = rng.random((18, 11)) > 0.15
    mean = np.array([3200.0, 1750.0, 2.45])[:, None, None]
    standard_deviation = np.array([420.0, 260.0, 0.08])[:, None, None]
    for channel in range(3):
        normalized_a = elastic_metrics(prediction_a[channel], target[channel], mask)["rmse"]
        normalized_b = elastic_metrics(prediction_b[channel], target[channel], mask)["rmse"]
        physical_target = target[channel] * standard_deviation[channel] + mean[channel]
        physical_a = prediction_a[channel] * standard_deviation[channel] + mean[channel]
        physical_b = prediction_b[channel] * standard_deviation[channel] + mean[channel]
        physical_rmse_a = elastic_metrics(physical_a, physical_target, mask)["rmse"]
        physical_rmse_b = elastic_metrics(physical_b, physical_target, mask)["rmse"]
        assert (normalized_a < normalized_b) == (physical_rmse_a < physical_rmse_b)
        np.testing.assert_allclose(
            physical_rmse_a,
            normalized_a * standard_deviation[channel].item(),
            rtol=1e-12,
        )
