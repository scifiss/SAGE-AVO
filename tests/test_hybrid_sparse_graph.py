"""Public-fixture mechanics of the optional v00332ab sparse branch."""

import numpy as np
import pytest
import torch

from sage_avo.models.hybrid_sparse import (
    HybridSparseSAGEAVO,
    SparsePatchGraph,
    SparseReflectorBranch,
    crop_accepted_graph,
    flip_sparse_graph_horizontal,
    sample_cnn_nodes,
    scatter_component_local,
)
from sage_avo.models.sage_avo import SAGEAVO
from sage_avo.evaluation.inference import infer_full_realization


def _graph(*, top=0, left=0, raw_shape=(12, 16), output_shape=(12, 16)):
    nodes = [
        {"node": 0, "component": 4, "time": 4.25, "trace": 3},
        {"node": 1, "component": 4, "time": 4.5, "trace": 10},
        {"node": 2, "component": 7, "time": 9.0, "trace": 8},
    ]
    edges = [
        {
            "source": 0,
            "target": 1,
            "component": 4,
            "delta_tau": 0.002,
            "delta_t": 0.25,
            "delta_x": 7,
            "geodesic_length": 7.1,
            "waveform_continuity": 0.98,
            "phase_continuity": 0.91,
            "shift_continuity": 0.04,
            "curvature": 0.05,
            "gap_count": 0,
        }
    ]
    return crop_accepted_graph(
        nodes, edges, top=top, left=left, raw_shape=raw_shape, output_shape=output_shape
    )


def test_crop_keeps_only_available_endpoints_and_maps_resize():
    graph = _graph(raw_shape=(12, 16), output_shape=(24, 32))
    assert graph.coordinates.shape == (3, 2)
    assert graph.edge_index.shape == (2, 2)
    assert graph.coordinates[0, 0].item() == pytest.approx(4.25 * 23 / 11)
    cropped = _graph(left=6, raw_shape=(12, 10), output_shape=(12, 16))
    assert cropped.coordinates.shape[0] == 2
    assert cropped.edge_index.shape[1] == 0


def test_fault_offset_correspondence_cannot_become_message_edge():
    graph = _graph()
    assert set(graph.components[graph.edge_index.flatten()].tolist()) == {4}
    nodes = [
        {"node": 0, "component": 0, "time": 3.0, "trace": 2},
        {"node": 1, "component": 1, "time": 3.0, "trace": 9},
    ]
    edge = {
        "source": 0, "target": 1, "component": 0, "delta_tau": 0.0,
        "delta_t": 0.0, "delta_x": 7, "geodesic_length": 7.0,
        "waveform_continuity": 0.9, "phase_continuity": 0.9,
        "shift_continuity": 0.0, "curvature": 0.0, "gap_count": 0,
    }
    with pytest.raises(ValueError, match="crosses frozen"):
        crop_accepted_graph(nodes, [edge], top=0, left=0, raw_shape=(12, 16), output_shape=(12, 16))
    excluded = crop_accepted_graph(
        nodes, [{**edge, "relation": "FAULT_OFFSET_CORRESPONDENCE"}],
        top=0, left=0, raw_shape=(12, 16), output_shape=(12, 16),
    )
    assert excluded.edge_index.shape[1] == 0


def test_bilinear_sampling_alignment_and_gradient():
    graph = _graph()
    rows, columns = torch.meshgrid(torch.arange(12), torch.arange(16), indexing="ij")
    feature = (2 * rows + 3 * columns).float().reshape(1, 1, 12, 16).requires_grad_()
    sampled = sample_cnn_nodes(feature, graph)
    assert sampled[0, 0].item() == pytest.approx(2 * 4.25 + 3 * 3, abs=1e-5)
    sampled.sum().backward()
    assert torch.isfinite(feature.grad).all()
    assert feature.grad.abs().sum() > 0
    flipped = flip_sparse_graph_horizontal(graph)
    mirrored_values = sample_cnn_nodes(feature.detach().flip(-1), flipped)
    assert torch.allclose(sampled.detach(), mirrored_values, atol=1e-5)
    assert torch.equal(flipped.edge_attr[:, 2], -graph.edge_attr[:, 2])


def test_sparse_branch_variable_graphs_and_support_locality():
    torch.manual_seed(7)
    graph = _graph()
    empty = _graph(left=11, raw_shape=(12, 5))
    cnn = torch.randn(2, 8, 12, 16, requires_grad=True)
    sparse = SparseReflectorBranch(8, heads=2)
    delta, mask = sparse(cnn, [graph, empty])
    assert delta.shape == cnn.shape
    assert mask.shape == (2, 1, 12, 16)
    assert mask[0].sum() > 0
    assert mask[1].sum() == 0
    assert torch.count_nonzero(delta[mask.expand_as(delta) == 0]) == 0
    delta.square().sum().backward()
    assert torch.isfinite(cnn.grad).all()
    assert cnn.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in sparse.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)


def test_disconnected_component_footprint_collision_has_no_support():
    graph = SparsePatchGraph(
        coordinates=torch.tensor([[4.25, 5.25], [4.25, 6.25]] * 2),
        components=torch.tensor([0, 0, 1, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        edge_attr=torch.zeros(4, 9),
        shape=(12, 16),
    )
    values = torch.ones(4, 8, requires_grad=True)
    dense, support = scatter_component_local(values, graph, torch.ones(4, dtype=torch.bool))
    assert torch.count_nonzero(support) == 0
    assert torch.count_nonzero(dense) == 0


def test_zero_scale_dense_parity_and_active_finite_forward():
    torch.manual_seed(9)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2)
    graph = _graph()
    avo = torch.randn(1, 3, 12, 16)
    low = torch.randn(1, 3, 12, 16)
    rgt = torch.from_numpy(np.broadcast_to(np.arange(12, dtype=np.float32)[:, None], (12, 16)).copy())[None]
    state = low.clone()
    for time_value in (0.0, 0.7):
        time = torch.tensor([time_value])
        reference = dense(state, time, avo, low, rgt)
        baseline = HybridSparseSAGEAVO(dense, channels=8, gamma=0.0)
        matched = baseline(state, time, avo, low, rgt, [graph])
        assert torch.equal(reference.velocity, matched.velocity)
        assert torch.equal(reference.segmentation_logits, matched.segmentation_logits)
    active = HybridSparseSAGEAVO(dense, channels=8, gamma=0.1)
    result = active(state, torch.tensor([0.4]), avo, low, rgt, [graph])
    assert torch.isfinite(result.velocity).all()
    assert torch.isfinite(result.segmentation_logits).all()
    dense_result = dense(state, torch.tensor([0.4]), avo, low, rgt)
    assert torch.equal(result.segmentation_logits, dense_result.segmentation_logits)
    assert active.last_support is not None and active.last_support.sum() > 0
    outside = active.last_support.expand_as(result.velocity) == 0
    assert torch.equal(result.velocity[outside], dense_result.velocity[outside])
    result.velocity.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in active.sparse.parameters()
    )


def test_tiled_inference_graph_provider_preserves_zero_scale_parity():
    torch.manual_seed(17)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2)
    wrapper = HybridSparseSAGEAVO(dense, channels=8, gamma=0.0)
    rng = np.random.default_rng(17)
    avo = rng.normal(size=(3, 12, 16)).astype(np.float32)
    low = rng.normal(size=(3, 12, 16)).astype(np.float32)
    rgt = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None], (12, 16)).copy()
    norm = {"x_mean": [0.0] * 3, "x_std": [1.0] * 3, "y_mean": [0.0] * 3, "y_std": [1.0] * 3}
    common = dict(
        avo=avo, low=low, rgt=rgt, normalization=norm, patch_shape=(8, 10),
        stride=(4, 6), steps=1, batch_size=2, device=torch.device("cpu"),
    )
    original, original_labels = infer_full_realization(dense, **common)

    def provider(positions):
        return [
            crop_accepted_graph([], [], top=top, left=left, raw_shape=(8, 10), output_shape=(8, 10))
            for top, left in positions
        ]

    hybrid, hybrid_labels = infer_full_realization(wrapper, graph_provider=provider, **common)
    assert np.array_equal(original, hybrid)
    assert np.array_equal(original_labels, hybrid_labels)
