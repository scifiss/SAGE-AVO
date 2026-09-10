"""v00332r interventions preserve topology, weights and unrelated paths."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch_geometric.nn import TransformerConv

from sage_avo.diagnostics.attention_coupling import elastic_extension_gate, final_tie_usage, graph_intervention, headwise_uniformity
from sage_avo.models.sage_avo import InstrumentedTransformerConv, SAGEAVO, relation_preserving_edge_attr_shuffle


def inputs():
    generator = torch.Generator().manual_seed(21)
    low = torch.randn(1, 3, 8, 7, generator=generator)
    avo = torch.randn(1, 3, 8, 7, generator=generator)
    rgt = torch.arange(8, dtype=torch.float32)[None, :, None] - torch.arange(7)[None, None, :]
    return low, torch.tensor([.5]), avo, low, rgt


def test_uniform_attention_is_exact_neighbor_mean_with_same_values_and_root():
    torch.manual_seed(4)
    layer = InstrumentedTransformerConv(8, 4, heads=2, edge_dim=1).eval()
    reference = TransformerConv(8, 4, heads=2, edge_dim=1).eval()
    reference.load_state_dict(layer.state_dict())
    x = torch.randn(5, 8)
    edges = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 2, 4, 4]])
    attrs = torch.rand(5, 1)
    torch.testing.assert_close(layer(x, edges, attrs), reference(x, edges, attrs), rtol=0, atol=0)
    layer.attention_mode = "uniform"
    actual, (_, alpha) = layer(x, edges, attrs, return_attention_weights=True)
    expected_alpha = torch.tensor([1/3, 1/3, 1/3, 1/2, 1/2])[:, None].expand(-1, 2)
    torch.testing.assert_close(alpha, expected_alpha, rtol=0, atol=0)
    values = layer.lin_value(x).reshape(5, 2, 4)[edges[0]] + layer.lin_edge(attrs).reshape(5, 2, 4)
    aggregated = torch.zeros(5, 2, 4)
    aggregated.index_add_(0, edges[1], values * expected_alpha[..., None])
    expected = aggregated.reshape(5, 8) + layer.lin_skip(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert layer.lin_query.weight.grad is None
    assert layer.lin_key.weight.grad is None
    assert layer.lin_value.weight.grad.abs().sum() > 0
    assert layer.lin_edge.weight.grad.abs().sum() > 0
    assert layer.lin_skip.weight.grad.abs().sum() > 0


def test_detaching_segmentation_changes_gradients_not_forward_or_capacity():
    torch.manual_seed(12)
    shared = SAGEAVO(hidden_channels=16, rgt_topology="rgt_v3_confidence_blocked")
    detached = SAGEAVO(hidden_channels=16, rgt_topology="rgt_v3_confidence_blocked", segmentation_detach_graph=True)
    detached.load_state_dict(shared.state_dict(), strict=True)
    a, c = shared(*inputs()), detached(*inputs())
    assert list(shared.state_dict()) == list(detached.state_dict())
    torch.testing.assert_close(a.velocity, c.velocity, rtol=0, atol=0)
    torch.testing.assert_close(a.segmentation_logits, c.segmentation_logits, rtol=0, atol=0)
    c.segmentation_logits.square().mean().backward(retain_graph=True)
    assert all(p.grad is None for n, p in detached.named_parameters() if n.startswith(("graph.node_projection", "graph.layers")))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in detached.named_parameters() if n.startswith("graph.segmentation"))
    detached.zero_grad(set_to_none=True)
    c.velocity.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in detached.named_parameters() if n.startswith("graph.layers"))


def test_semantic_shuffle_preserves_relations_marginals_and_reverse_pairs():
    attrs = torch.tensor([1., 2., 3., 1., 2., 3., 10., 20., 10., 20.])[:, None]
    shuffled = relation_preserving_edge_attr_shuffle(attrs, [6, 4])
    assert torch.equal(shuffled, relation_preserving_edge_attr_shuffle(attrs, [6, 4]))
    assert not torch.equal(attrs, shuffled)
    for start, half in ((0, 3), (6, 2)):
        torch.testing.assert_close(shuffled[start:start+half], shuffled[start+half:start+2*half])
        assert torch.equal(attrs[start:start+2*half].sort(0).values, shuffled[start:start+2*half].sort(0).values)


def test_interventions_leave_topology_and_weights_and_restore_on_failure():
    model = SAGEAVO(hidden_channels=16, rgt_topology="rgt_v3_confidence_blocked").eval()
    state = deepcopy(model.state_dict())
    before = model(*inputs())
    with pytest.raises(RuntimeError):
        with graph_intervention(model, attention="uniform", edge_attr="relation_preserving_shuffled"):
            changed = model(*inputs())
            assert torch.equal(before.edge_indices[0], changed.edge_indices[0])
            raise RuntimeError("intentional")
    assert model.graph.attention_mode == "learned"
    assert model.graph.diagnostic_edge_attr_mode == "current"
    assert all(layer.attention_mode == "learned" for layer in model.graph.layers)
    after = model(*inputs())
    torch.testing.assert_close(before.velocity, after.velocity, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_headwise_statistics_do_not_hide_selective_head():
    edges = torch.tensor([[0, 1, 2, 0, 1], [2, 2, 2, 3, 3]])
    alpha = torch.tensor([[1/3, .98], [1/3, .01], [1/3, .01], [.5, .99], [.5, .01]])
    rows = headwise_uniformity(edges, alpha, 5)
    assert rows[0]["normalized_entropy"] == pytest.approx(1)
    assert rows[0]["kl_from_uniform"] == pytest.approx(0, abs=1e-12)
    assert rows[1]["normalized_entropy"] < .2
    assert rows[1]["kl_from_uniform"] > .5
    assert rows[0]["isolated_destinations"] == 3


def test_tie_audit_distinguishes_minimum_shift_from_final_parity():
    flat = final_tie_usage(np.ones((8, 3), dtype=np.float32))
    assert flat["stage1_tie"].all()
    assert flat["resolved_by_min_abs_shift"].all()
    assert not flat["parity"].any()
    tied = np.array([[0., 3.], [1., 2.], [2., 3.], [3., 2.], [4., 3.]], np.float32)
    observed = final_tie_usage(tied, max_shift=1)
    assert observed["parity"].any()


def test_all20_gate_requires_complete_paired_elastic_evidence():
    unfavorable = np.full((3, 4), -.01)
    result = elastic_extension_gate({"rgt": unfavorable}, unfavorable)
    assert not any(result.values())
    vp_vs_only = unfavorable.copy()
    vp_vs_only[:, :2] = .001
    assert elastic_extension_gate({"rgt": vp_vs_only}, unfavorable)["A_vp_and_vs_all_three_seeds"] == ["rgt"]
    density_harmed = np.full((3, 4), .02)
    density_harmed[1, 2] = -.02
    assert not elastic_extension_gate({"rgt": unfavorable}, density_harmed)["C_clear_reproducible_decoupling_without_major_density_degradation"]
    with pytest.raises(ValueError):
        elastic_extension_gate({"rgt": unfavorable[:2]}, unfavorable)


def test_condition_flags_preserve_all_four_state_layouts_and_topologies():
    from pathlib import Path

    from sage_avo.config import load_config
    from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs

    definitions = [
        {"rgt_topology": "rgt_v3_confidence_blocked"},
        {"rgt_topology": "rgt_v3_confidence_blocked", "attention_mode": "uniform"},
        {"rgt_topology": "rgt_v3_confidence_blocked", "segmentation_detach_graph": True},
        {"rgt_topology": "cartesian"},
    ]
    config = load_config(Path(__file__).resolve().parents[1] / "configs/sage_avo_s01_v0031.yaml")
    config["model"]["hidden_channels"] = 16
    models = []
    for definition in definitions:
        config["model"]["experimental_graph"] = definition
        models.append(build_sage_avo_variant("full", **sage_avo_model_kwargs(config)))
    initial = models[0].state_dict()
    for model in models:
        assert list(model.state_dict()) == list(initial)
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in models[0].parameters())
        model.load_state_dict(initial, strict=True)
    outputs = [model(*inputs()) for model in models]
    assert torch.equal(outputs[0].edge_indices[0], outputs[1].edge_indices[0])
    assert torch.equal(outputs[0].edge_indices[0], outputs[2].edge_indices[0])
    assert not torch.equal(outputs[0].edge_indices[0], outputs[3].edge_indices[0])
    torch.testing.assert_close(outputs[0].velocity, outputs[2].velocity, rtol=0, atol=0)
