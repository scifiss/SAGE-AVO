import numpy as np

from sage_avo.evaluation.metrics import elastic_metrics, segmentation_metrics


def test_elastic_metrics_exact_offset():
    target = np.arange(12, dtype=float).reshape(3, 4)
    metrics = elastic_metrics(target + 2.0, target)
    assert np.isclose(metrics["rmse"], 2.0)
    assert np.isclose(metrics["mae"], 2.0)


def test_segmentation_metrics_perfect():
    target = np.array([[0, 1, 2], [0, 1, 2]])
    metrics = segmentation_metrics(target, target)
    assert metrics["miou"] == 1.0
    assert metrics["macro_dice"] == 1.0
    assert all(metrics[f"class_{label}_iou"] == 1.0 for label in range(3))
