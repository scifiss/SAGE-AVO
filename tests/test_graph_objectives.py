from __future__ import annotations

import torch

from sage_avo.models.graph import build_rgt_edges
from sage_avo.models.sage_avo import SAGEAVO
from sage_avo.training.losses import (
    GraphObjectiveSettings,
    edge_smoothness,
    graph_structure_loss,
    truth_edge_matching,
)


def _graph_inputs(height: int = 5, width: int = 7):
    rgt = torch.arange(height, dtype=torch.float32)[None, :, None].expand(1, -1, width)
    edges = build_rgt_edges(rgt, max_shift=1, steered=True)
    weights = [torch.linspace(0.2, 1.0, edges[0].shape[1])]
    labels = torch.zeros((1, height, width), dtype=torch.long)
    labels[:, :, width // 2 :] = 1
    return rgt, edges, weights, labels


def test_current_dispatch_is_exactly_backward_compatible() -> None:
    rgt, edges, weights, labels = _graph_inputs()
    prediction = torch.randn(1, 3, 5, 7)
    target = torch.randn_like(prediction)
    expected = edge_smoothness(prediction, edges, weights)
    actual = graph_structure_loss(
        prediction,
        target,
        rgt,
        labels,
        edges,
        weights,
        GraphObjectiveSettings(mode="current_smoothness"),
    )
    assert torch.equal(actual, expected)


def test_truth_edge_matching_preserves_signed_vector_contrast() -> None:
    rgt, edges, weights, _ = _graph_inputs()
    target = torch.stack((rgt[0], 2.0 * rgt[0], -rgt[0]))[None]
    prediction = target.clone().requires_grad_(True)
    exact = truth_edge_matching(prediction, target, edges, weights)
    assert exact.item() == 0.0

    reversed_contrast = (-target).requires_grad_(True)
    mismatch = truth_edge_matching(reversed_contrast, target, edges, weights)
    assert mismatch.item() > 0.0
    mismatch.backward()
    assert reversed_contrast.grad is not None
    assert torch.isfinite(reversed_contrast.grad).all()
    assert reversed_contrast.grad.abs().sum().item() > 0.0


def test_edge_aware_objective_has_finite_nonzero_gradient() -> None:
    rgt, edges, weights, labels = _graph_inputs()
    target = torch.randn(1, 3, 5, 7)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    objective = graph_structure_loss(
        prediction,
        target,
        rgt,
        labels,
        edges,
        weights,
        GraphObjectiveSettings(mode="edge_aware_contrast"),
    )
    objective.backward()
    assert objective.item() > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0.0


def test_graph_indices_match_property_row_major_order() -> None:
    height, width = 5, 7
    rgt, edges, _, _ = _graph_inputs(height, width)
    index_image = torch.arange(height * width).reshape(height, width)
    flattened = index_image.reshape(-1)
    source, destination = edges[0]
    assert torch.equal(flattened[source], source)
    assert torch.equal(flattened[destination], destination)
    assert bool((source >= 0).all() and (source < height * width).all())
    assert bool((destination >= 0).all() and (destination < height * width).all())
    assert len({tuple(pair) for pair in edges[0].transpose(0, 1).tolist()}) == edges[0].shape[1]
    assert rgt.shape == (1, height, width)


def test_model_inference_contract_requires_no_truth_or_facies() -> None:
    torch.manual_seed(4)
    model = SAGEAVO(
        hidden_channels=8,
        graph_layers=2,
        graph_heads=2,
        max_rgt_shift=1,
    ).eval()
    avo = torch.randn(1, 3, 6, 8)
    low = torch.randn(1, 3, 6, 8)
    rgt = torch.arange(6, dtype=torch.float32)[None, :, None].expand(1, -1, 8)
    with torch.no_grad():
        prediction = model.sample(avo, low, rgt, steps=2)
    assert prediction.shape == low.shape
    assert torch.isfinite(prediction).all()
