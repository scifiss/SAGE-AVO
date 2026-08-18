import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
graph = pytest.importorskip("sage_avo.models.graph")
sage_avo = pytest.importorskip("sage_avo.models.sage_avo")
build_rgt_edges = graph.build_rgt_edges
SAGEAVO = sage_avo.SAGEAVO


def _inputs(height=8, width=9):
    torch.manual_seed(4)
    avo = torch.randn(1, 3, height, width)
    low = torch.randn(1, 3, height, width)
    rows = torch.arange(height).view(1, height, 1).expand(1, height, width).float()
    return avo, low, rows


def test_final_005_architecture_parameter_count_and_forward_shapes():
    model = SAGEAVO()
    assert sum(parameter.numel() for parameter in model.parameters()) == 392_646
    avo, low, rgt = _inputs()
    output = model(low, torch.zeros(1), avo, low, rgt)
    assert output.velocity.shape == low.shape
    assert output.segmentation_logits.shape == low.shape
    assert output.embeddings.shape == (1, low.shape[-2] * low.shape[-1], 64)
    assert output.edge_indices[0].shape[0] == 2
    assert output.edge_weights[0].shape[0] == output.edge_indices[0].shape[1]
    assert output.attention_edge_indices[0].shape[0] == 2
    assert output.attention_weights[0].shape[0] == output.attention_edge_indices[0].shape[1]
    assert torch.isfinite(output.attention_weights[0]).all()


def test_forward_backward_reaches_cnn_graph_and_both_decoders():
    model = SAGEAVO(hidden_channels=16, graph_layers=1, graph_heads=4)
    avo, low, rgt = _inputs()
    output = model(low, torch.full((1,), 0.4), avo, low, rgt)
    loss = output.velocity.square().mean() + output.segmentation_logits.square().mean()
    loss.backward()
    names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    assert any(name.startswith("encoder") for name in names)
    assert any(name.startswith("graph.layers") for name in names)
    assert any(name.startswith("graph.segmentation") for name in names)
    assert any(name.startswith("decoder") for name in names)


def test_rgt_steering_changes_edges_and_model_output():
    model = SAGEAVO(hidden_channels=16, graph_layers=1, graph_heads=4).eval()
    avo, low, flat_rgt = _inputs()
    columns = torch.arange(flat_rgt.shape[-1]).view(1, 1, -1)
    dipping_rgt = flat_rgt - columns.float()
    flat_edges = build_rgt_edges(flat_rgt, max_shift=3, steered=True)[0]
    dipping_edges = build_rgt_edges(dipping_rgt, max_shift=3, steered=True)[0]
    assert not torch.equal(flat_edges, dipping_edges)
    with torch.no_grad():
        flat = model(low, torch.zeros(1), avo, low, flat_rgt).velocity
        dipping = model(low, torch.zeros(1), avo, low, dipping_rgt).velocity
    assert not torch.allclose(flat, dipping)


def test_singleton_channel_rgt_is_supported():
    model = SAGEAVO(hidden_channels=16, graph_layers=1, graph_heads=4).eval()
    avo, low, rgt = _inputs()
    with torch.no_grad():
        plain = model(low, torch.zeros(1), avo, low, rgt).velocity
        channelized = model(low, torch.zeros(1), avo, low, rgt[:, None]).velocity
    torch.testing.assert_close(plain, channelized)


def test_zero_guidance_is_identical_and_guidance_is_available():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        guidance_start_fraction=0.0,
        guidance_interval_steps=1,
    ).eval()
    avo, low, rgt = _inputs(height=10, width=6)
    first = model.sample(avo, low, rgt, steps=2)
    second = model.sample(avo, low, rgt, steps=2, guidance_scale=0.0)
    torch.testing.assert_close(first, second)

    model.set_norm_stats(
        {
            "x_mean": [0.0, 0.0, 0.0],
            "x_std": [0.003, 0.002, 0.0015],
            "y_mean": [3000.0, 1600.0, 2.35],
            "y_std": [250.0, 180.0, 0.04],
        }
    )
    guided = model.sample(
        avo,
        low,
        rgt,
        steps=2,
        guidance_scale=1e-3,
        avo_mask=torch.ones(1, 1, 10, 6),
    )
    assert torch.isfinite(guided).all()
    assert not torch.allclose(first, guided)


def test_guidance_requires_training_normalization_statistics():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        guidance_start_fraction=0.0,
        guidance_interval_steps=1,
    ).eval()
    avo, low, rgt = _inputs(height=8, width=5)
    with pytest.raises(RuntimeError, match="set_norm_stats"):
        model.sample(avo, low, rgt, steps=1, guidance_scale=0.01)
