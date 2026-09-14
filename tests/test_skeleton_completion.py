"""No-private-data tests for frozen features and attention-sign diagnostics."""

import numpy as np
import pytest
import torch
from torch import nn

from sage_avo.diagnostics.skeleton_completion import (
    frozen_cnn_anchors,
    neighborhood_summary,
    signed_prior_softmax,
)


class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Linear(1, 2)
        self.condition_embedding = nn.Conv2d(6, 2, 1)
        self.encoder = nn.Conv2d(7, 3, 1)


def test_frozen_feature_sampler_is_repeatable_and_does_not_change_weights():
    torch.manual_seed(5)
    model = FrozenEncoder().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    avo = np.ones((3, 10, 12), np.float32)
    low = np.ones_like(avo) * 2
    points = np.array([[2.5, 3.5], [7.0, 8.0]])
    norm = {"x_mean": [0] * 3, "x_std": [1] * 3, "y_mean": [0] * 3, "y_std": [1] * 3}
    a, info = frozen_cnn_anchors(
        model, avo, low, points, norm, torch.device("cpu"), patch_shape=(6, 8), stride=(3, 4)
    )
    b, _ = frozen_cnn_anchors(
        model, avo, low, points, norm, torch.device("cpu"), patch_shape=(6, 8), stride=(3, 4)
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[0], a[1])
    assert info["channels"] == 3 and not info["graph_or_flow_executed"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_reject_training_mode_feature_extraction():
    with pytest.raises(ValueError, match="eval"):
        frozen_cnn_anchors(FrozenEncoder(), None, None, None, None, "cpu")


def test_additive_and_subtractive_priors_have_opposite_order():
    logp = np.array([0.0, -10.0, -1000.0])
    plus = signed_prior_softmax(logp, [0] * 3, ["tangent"] * 3)
    minus = signed_prior_softmax(logp, [0] * 3, ["tangent"] * 3, sign=-1)
    assert plus[0] > plus[1] > plus[2]
    assert minus[0] < minus[1] < minus[2]
    assert np.isfinite(plus).all() and np.isfinite(minus).all()
    assert plus.sum() == pytest.approx(1.0) and minus.sum() == pytest.approx(1.0)


def test_attention_normalization_keeps_relations_separate():
    a = signed_prior_softmax([0, -2, -5], [0, 0, 0], ["t", "t", "n"])
    assert a[2] == 1 and a[:2].sum() == pytest.approx(1.0)


def test_degree_and_local_support_counts():
    points = np.array([[0, 0], [0, 4], [0, 16]], float)
    edges = [{"source": 0, "target": 1, "kept": True}, {"source": 0, "target": 2, "kept": True}]
    m = neighborhood_summary(points, edges)
    assert m["degree_max"] == 2 and m["outside_encoder_9x9_fraction"] == 0.5
    assert m["outside_3x3_fraction"] == 1.0
