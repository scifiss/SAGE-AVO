"""Topology-only regression tests; no private data or training."""

from pathlib import Path

import numpy as np
import pytest

from sage_avo.config import load_config
from sage_avo.diagnostics.skeleton_graph import (
    attention_quantities,
    build_skeleton,
    graph_statistics,
    inverse_surfaces,
    node_features,
    path_fault_qc,
    select_interfaces,
)


def config():
    return load_config(
        Path(__file__).resolve().parents[1] / "configs/development_diagnostics_v00332u.yaml"
    )


def scene(slope=0.0):
    y, x = np.indices((100, 96))
    tau = (y + slope * x).astype(float)
    signal = sum(np.exp(-0.5 * ((tau - level) / 1.5) ** 2) for level in [20, 40, 60, 80])
    return tau, np.stack([signal, 0.8 * signal, 0.6 * signal]), np.ones_like(tau, bool)


def test_inverse_rejects_plateau_fold_and_out_of_range():
    t = np.array([[0, 0, 0], [1, 1, 2], [2, 1, 0], [3, 2, 3]], float)
    result = inverse_surfaces(t, [1, 1.5, 5])
    assert result[0, 0] == 1
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 2])
    assert np.isnan(result[2]).all()


def test_sparse_anchors_select_reflectors_not_all_samples():
    tau, avo, valid = scene()
    g = build_skeleton(tau, avo, valid, config())
    assert 0 < len(g["points"]) < 0.1 * tau.size
    assert len(g["levels"]) == 4
    np.testing.assert_allclose(g["levels"], [20, 40, 60, 80], atol=0.5)
    assert set(e["span"] for e in g["edges"] if e["relation"] == "tangential") == {4, 8, 16, 32, 64}
    assert all(e["kept"] for e in g["edges"])
    assert set(e["relation"] for e in g["edges"]) == {"tangential", "normal"}


def test_dipping_surface_long_edges_remain_aligned_and_retained():
    tau, avo, valid = scene(0.5)
    g = build_skeleton(tau, avo, valid, config())
    long = [e for e in g["edges"] if e["span"] >= 32]
    assert len(long) > 0
    assert all(e["kept"] and e["rgt_mismatch"] < 1e-10 for e in long)
    assert all(e["geodesic_length"] >= e["span"] for e in long)


def test_fault_jump_cannot_be_hidden_inside_long_edge():
    tau, avo, valid = scene()
    tau[:, 48:] += 20
    g = build_skeleton(tau, avo, valid, config())
    crossing = [
        e
        for e in g["edges"]
        if e["relation"] == "tangential"
        and g["points"][e["source"], 1] < 48 <= g["points"][e["target"], 1]
    ]
    assert crossing
    assert all(not e["kept"] for e in crossing)


def test_invalid_interior_support_cuts_edge_with_valid_endpoints():
    tau, avo, valid = scene()
    valid[:, 47] = False
    g = build_skeleton(tau, avo, valid, config())
    crossed = [
        e for e in g["edges"] if g["points"][e["source"], 1] < 47 < g["points"][e["target"], 1]
    ]
    assert crossed and all(not e["kept"] for e in crossed)


def test_node_features_match_existing_angular_definition_and_optional_cnn():
    import torch
    from sage_avo.models.sage_avo import angular_features

    tau, avo, _ = scene()
    points = np.array([[20.0, 4.0], [40.0, 8.0]])
    features = node_features(tau, avo, points)
    expected, _ = angular_features(torch.tensor(avo[None]))
    np.testing.assert_allclose(
        features[:, :6],
        expected[0, :, points[:, 0].astype(int), points[:, 1].astype(int)].T,
        atol=1e-6,
    )
    extra = node_features(tau, avo, points, cnn_features=np.ones((5, *tau.shape)))
    assert extra.shape == (2, 14)
    np.testing.assert_array_equal(extra[:, 5:], features)


def test_attention_prior_stable_normalized_and_relation_specific():
    tau, avo, valid = scene(0.5)
    g = build_skeleton(tau, avo, valid, config())
    rows = attention_quantities(tau, g["points"], g["edges"], config())
    totals = {}
    for row in rows:
        assert np.isfinite(row["logit"]) and 0 <= row["alpha"] <= 1
        key = row["source"], row["relation"]
        totals[key] = totals.get(key, 0) + row["alpha"]
        if row["relation"] == "tangential":
            assert row["prior"] == pytest.approx(1.0)
    np.testing.assert_allclose(list(totals.values()), 1.0)
    # Distinct RGT levels should not receive a same-interface prior near one.
    assert max(r["prior"] for r in rows if r["relation"] == "normal") < 0.01


def test_truth_qc_detects_crossing_and_return_along_curved_path():
    path = np.array([[0, 0], [0, 10], [0, 0]], float)
    cross, near = path_fault_qc(path, [{"column": 5, "dip": 0}])
    assert cross and near


def test_information_reach_on_known_chain():
    p = np.array([[0, 0], [0, 10], [0, 20]], float)
    result = graph_statistics(p, [[0, 1], [1, 2]])
    assert result["one_hop_reach_mean"] == 10
    assert result["two_hop_reach_max"] == 20
    assert result["edge_count_undirected"] == 2


def test_deterministic_and_no_mutation():
    tau, avo, valid = scene(0.5)
    old = tau.copy()
    a = select_interfaces(tau, avo, valid, config())
    b = select_interfaces(tau, avo, valid, config())
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(tau, old)
