"""Regression tests for inference-only feature and path QC."""

from pathlib import Path
import numpy as np
import pytest
import torch

from sage_avo.config import load_config
from sage_avo.diagnostics.skeleton_feature_qc import (
    feature_distance,
    filter_decisions,
    path_observables,
    snap_candidates,
    verify_pre_gnn_tensor,
)
from sage_avo.models.sage_avo import SAGEAVO


def config():
    return load_config(
        Path(__file__).resolve().parents[1]
        / "configs/development_diagnostics_v00332u_features.yaml"
    )


def test_exact_pre_gnn_tensor_and_no_graph_execution():
    torch.manual_seed(4)
    model = SAGEAVO(hidden_channels=16).eval()
    a = torch.randn(1, 3, 8, 7)
    low = torch.randn_like(a)
    tau = torch.arange(8)[None, :, None].expand(1, 8, 7).float()
    result = verify_pre_gnn_tensor(model, a, low, tau)
    assert result["shape"] == [1, 16, 8, 7] and result["bitwise_equal_to_forward_pre_gnn"]
    assert not result["graph_called"] and all(p.grad is None for p in model.parameters())
    assert not model.encoder._forward_hooks


def test_feature_metric_scale_and_identity():
    a = np.array([[1, 2, 3], [2, 0, 1]], float)
    c, d, n = feature_distance(a, a)
    np.testing.assert_allclose(c, 1)
    np.testing.assert_allclose(d, 0)
    np.testing.assert_allclose(n, 0)
    np.testing.assert_allclose(feature_distance(a, 2 * a)[2], 2 / 3)


def test_curved_constant_tau_path_not_suppressed_as_length_grows():
    y, x = np.indices((180, 100))
    tau = (y + 0.02 * (x - 40) ** 2).astype(float)
    avo = np.stack([np.cos(tau / 5)] * 3)
    for length in [4, 16, 64]:
        xx = np.arange(10, 10 + length + 1)
        path = np.column_stack((100 - 0.02 * (xx - 40) ** 2, xx))
        result = path_observables(tau, avo, path, 1.0, config())
        assert result["path_prior"] > 0.999
        assert result["path_chord_ratio"] >= 1 - 1e-12
        assert result["tangent_gradient_cosine_rms"] < 1e-10


def test_cross_interface_path_is_not_given_high_prior():
    y, x = np.indices((100, 80))
    tau = y.astype(float)
    avo = np.stack([np.cos(tau / 3)] * 3)
    path = np.column_stack((20 + np.arange(32) * 0.5, 10 + np.arange(32)))
    assert path_observables(tau, avo, path, 1.0, config())["path_prior"] < 0.01


def test_seismic_discontinuity_changes_observable_filter():
    y, x = np.indices((100, 80))
    tau = y.astype(float)
    avo = np.stack([np.cos(tau / 3)] * 3)
    avo[:, :, 30:] *= -1
    path = np.column_stack((np.full(5, 40.0), np.arange(28, 33)))
    r = path_observables(tau, avo, path, 1.0, config())
    assert r["coherence_min"] < -0.99
    assert not filter_decisions(r, True, config())["COHERENCE"]


def test_snap_diagnostic_does_not_move_input_or_cross_interfaces():
    y, x = np.indices((60, 40))
    tau = y.astype(float)
    avo = np.stack([np.exp(-0.5 * ((tau - 21.0) / 0.5) ** 2)] * 3)
    points = np.array([[20.0, 10.0], [30.0, 10.0]])
    graph = {
        "points": points,
        "levels": np.array([20.0, 30.0]),
        "surface_ids": np.array([0, 1]),
        "curves": np.stack([np.full(40, 20.0), np.full(40, 30.0)]),
        "scale": 1.0,
    }
    old = points.copy()
    snapped, rows = snap_candidates(tau, avo, np.ones_like(tau, bool), graph, config())
    np.testing.assert_array_equal(points, old)
    # The RGT tolerance stops the proposal before the stronger integer sample.
    assert snapped[0, 0] == pytest.approx(20.75)
    assert all(abs(r["rgt_deviation"]) <= r["rgt_tolerance"] for r in rows)
    assert snapped[0, 0] < snapped[1, 0] and not any(r["adopted"] for r in rows)


def test_invalid_path_fails_closed():
    tau = np.ones((10, 10))
    avo = np.ones((3, 10, 10))
    r = path_observables(tau, avo, np.array([[1.0, 1.0], [np.nan, 2.0]]), 1.0, config())
    assert not r["valid_observable_path"] and r["path_prior"] == 0
    assert not filter_decisions(r, False, config())["COMBINED"]
