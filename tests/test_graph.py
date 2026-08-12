import numpy as np
import pytest

from sage_avo.structure.graph import GraphEdges, build_rgt_graph


def test_strongest_edges_are_ranked_by_weight():
    edges = GraphEdges(np.arange(10), np.arange(10), np.arange(10, dtype=float))
    strongest = edges.strongest(0.2)
    np.testing.assert_array_equal(np.sort(strongest.weight), [8.0, 9.0])


def test_rgt_graph_builds_finite_avo_similarity_weights():
    rows, columns = np.indices((8, 5))
    rgt = rows - 0.25 * columns
    gradient = np.sin(rgt)
    graph = build_rgt_graph(rgt, gradient, max_shift=2)
    assert graph.source.shape == graph.destination.shape == graph.weight.shape
    assert np.isfinite(graph.weight).all()
    assert np.all((graph.weight > 0) & (graph.weight <= 1))


def test_cartesian_torch_graph_keeps_same_row():
    torch = pytest.importorskip("torch")
    from sage_avo.models.graph import build_rgt_edges

    height, width = 5, 4
    rgt = torch.arange(height).view(1, height, 1).expand(1, -1, width).float()
    edges = build_rgt_edges(rgt, max_shift=2, steered=False)[0]
    horizontal_count = 2 * height * (width - 1)
    horizontal = edges[:, :horizontal_count]
    source_rows = torch.div(horizontal[0], width, rounding_mode="floor")
    destination_rows = torch.div(horizontal[1], width, rounding_mode="floor")
    assert torch.equal(source_rows, destination_rows)
