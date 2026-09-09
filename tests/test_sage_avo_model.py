import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
graph = pytest.importorskip("sage_avo.models.graph")
sage_avo = pytest.importorskip("sage_avo.models.sage_avo")
build_rgt_edges = graph.build_rgt_edges
build_horizon_edges = graph.build_horizon_edges
build_horizon_candidate_edges = graph.build_horizon_candidate_edges
build_normal_edges = graph.build_normal_edges
build_normal_candidate_edges = graph.build_normal_candidate_edges
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


def test_relational_edges_follow_tangent_and_normal_rgt_directions():
    height, width = 6, 7
    rows = torch.arange(height).view(1, height, 1).expand(1, height, width).float()
    columns = torch.arange(width).view(1, 1, width).expand(1, height, width).float()
    dipping_rgt = rows - columns
    tangential = build_horizon_edges(dipping_rgt, max_shift=2, steered=True)[0]
    normal = build_normal_edges(dipping_rgt, max_lateral_shift=1, steered=True)[0]

    tangent_source, tangent_destination = tangential[:, : height * (width - 1)]
    tangent_interior = (tangent_source // width) < height - 1
    assert torch.equal(
        dipping_rgt.reshape(-1)[tangent_source[tangent_interior]],
        dipping_rgt.reshape(-1)[tangent_destination[tangent_interior]],
    )
    normal_source, normal_destination = normal[:, : (height - 1) * width]
    interior = (normal_source % width) > 0
    assert torch.all(
        (normal_destination[interior] % width) == (normal_source[interior] % width) - 1
    )
    assert not torch.equal(tangential, normal)


def test_relational_graph_reaches_both_message_branches_and_gate():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=2,
        graph_heads=4,
        graph_mode="relational_rgt",
    )
    avo, low, rgt = _inputs()
    output = model(low, torch.full((1,), 0.4), avo, low, rgt)
    loss = output.velocity.square().mean() + output.segmentation_logits.square().mean()
    loss.backward()
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    assert any(name.startswith("graph.tangential_layers") for name in active)
    assert any(name.startswith("graph.normal_layers") for name in active)
    assert any(name.startswith("graph.relation_gate") for name in active)
    assert torch.isfinite(output.velocity).all()
    assert output.edge_indices[0].shape[1] == output.edge_weights[0].numel()


def test_candidate_relations_retain_hard_edge_and_double_relation_degree():
    height, width = 6, 7
    rows = torch.arange(height).view(1, height, 1).expand(1, height, width).float()
    columns = torch.arange(width).view(1, 1, width).expand(1, height, width).float()
    rgt = rows - columns
    builders = (
        (
            build_horizon_edges(rgt, max_shift=2, steered=True)[0],
            build_horizon_candidate_edges(rgt, max_shift=2, candidates=2, steered=True)[0],
        ),
        (
            build_normal_edges(rgt, max_lateral_shift=1, steered=True)[0],
            build_normal_candidate_edges(rgt, max_lateral_shift=1, candidates=2, steered=True)[0],
        ),
    )
    for hard, candidates in builders:
        assert candidates.shape[1] == 2 * hard.shape[1]
        hard_pairs = set(zip(hard[0].tolist(), hard[1].tolist()))
        candidate_pairs = set(zip(candidates[0].tolist(), candidates[1].tolist()))
        assert hard_pairs <= candidate_pairs


def test_candidate_relational_graph_preserves_architecture_and_expands_edges():
    torch.manual_seed(9)
    hard = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_rgt",
    ).eval()
    candidate = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_candidate_rgt",
        graph_relation_candidates=2,
    ).eval()
    candidate.load_state_dict(hard.state_dict(), strict=True)
    avo, low, rgt = _inputs()
    with torch.no_grad():
        hard_output = hard(low, torch.zeros(1), avo, low, rgt)
        candidate_output = candidate(low, torch.zeros(1), avo, low, rgt)
    assert sum(parameter.numel() for parameter in hard.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert candidate_output.edge_indices[0].shape[1] == 2 * hard_output.edge_indices[0].shape[1]
    assert torch.isfinite(candidate_output.attention_weights[0]).all()
    assert not torch.allclose(hard_output.velocity, candidate_output.velocity)


def test_structural_attention_prior_is_selective_and_trainable():
    torch.manual_seed(12)
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_structural_prior_rgt",
        graph_relation_candidates=2,
        graph_structural_prior_initial_strengths=((1.0, 1.0), (1.0, 1.0)),
    )
    avo, low, rgt = _inputs()
    tokens = torch.randn(1, rgt.shape[-2] * rgt.shape[-1], 16)
    _, detail_batches, _ = model.graph.diagnostic_forward(tokens, avo, rgt)
    for detail in detail_batches[0]:
        attention = detail["attention"]
        prior = detail["attention_prior"]
        assert prior is not None
        favorable = prior >= torch.quantile(prior, 0.5)
        assert attention[favorable].sum() / attention.sum() > 0.5

    output = model(low, torch.full((1,), 0.4), avo, low, rgt)
    output.velocity.square().mean().backward()
    gradient = model.graph.structural_attention_raw_strengths.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_task_decoupled_graph_preserves_capacity_and_j_segmentation_path():
    torch.manual_seed(13)
    shared = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_structural_prior_rgt",
        graph_relation_candidates=2,
    ).eval()
    decoupled = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_decoupled_rgt",
        graph_relation_candidates=2,
    ).eval()
    decoupled.load_state_dict(shared.state_dict(), strict=True)
    avo, low, rgt = _inputs()
    with torch.no_grad():
        shared_output = shared(low, torch.full((1,), 0.4), avo, low, rgt)
        decoupled_output = decoupled(low, torch.full((1,), 0.4), avo, low, rgt)

    assert sum(parameter.numel() for parameter in shared.parameters()) == sum(
        parameter.numel() for parameter in decoupled.parameters()
    )
    torch.testing.assert_close(
        shared_output.segmentation_logits,
        decoupled_output.segmentation_logits,
    )
    assert not torch.allclose(shared_output.velocity, decoupled_output.velocity)


def test_task_decoupled_prior_gradient_is_segmentation_only():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_decoupled_rgt",
        graph_relation_candidates=2,
    )
    avo, low, rgt = _inputs()
    output = model(low, torch.full((1,), 0.4), avo, low, rgt)
    strengths = model.graph.structural_attention_raw_strengths
    elastic_gradient = torch.autograd.grad(
        output.velocity.square().mean(), strengths, retain_graph=True, allow_unused=True
    )[0]
    segmentation_gradient = torch.autograd.grad(
        output.segmentation_logits.square().mean(), strengths, allow_unused=True
    )[0]

    assert elastic_gradient is None
    assert segmentation_gradient is not None
    assert torch.isfinite(segmentation_gradient).all()
    assert segmentation_gradient.abs().sum() > 0


def test_task_decoupled_diagnostics_identify_biased_and_unbiased_streams():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_decoupled_rgt",
        graph_relation_candidates=2,
    ).eval()
    avo, _, rgt = _inputs()
    tokens = torch.randn(1, rgt.shape[-2] * rgt.shape[-1], 16)
    _, detail_batches, gates = model.graph.diagnostic_forward(tokens, avo, rgt)
    details = detail_batches[0]

    assert set(gates) == {"segmentation", "elastic"}
    assert {detail["stream"] for detail in details} == {"segmentation", "elastic"}
    for detail in details:
        if detail["stream"] == "segmentation":
            assert detail["attention_prior"] is not None
            assert detail["structural_strengths"] is not None
        else:
            assert detail["attention_prior"] is None
            assert detail["structural_strengths"] is None


def test_task_specific_strengths_start_at_shared_structural_output():
    torch.manual_seed(14)
    shared = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_structural_prior_rgt",
        graph_relation_candidates=2,
    ).eval()
    task_specific = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_specific_rgt",
        graph_relation_candidates=2,
    ).eval()
    compatibility = task_specific.load_state_dict(shared.state_dict(), strict=False)
    assert compatibility.missing_keys == ["graph.elastic_attention_raw_strengths"]
    assert not compatibility.unexpected_keys
    torch.testing.assert_close(
        torch.nn.functional.softplus(task_specific.graph.elastic_attention_raw_strengths),
        torch.full((2, 2), 0.5),
    )
    avo, low, rgt = _inputs()
    with torch.no_grad():
        shared_output = shared(low, torch.full((1,), 0.4), avo, low, rgt)
        task_specific_output = task_specific(low, torch.full((1,), 0.4), avo, low, rgt)
    torch.testing.assert_close(shared_output.velocity, task_specific_output.velocity)
    torch.testing.assert_close(
        shared_output.segmentation_logits,
        task_specific_output.segmentation_logits,
    )


def test_task_specific_strengths_have_disjoint_objective_gradients():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_specific_rgt",
        graph_relation_candidates=2,
    )
    avo, low, rgt = _inputs()
    output = model(low, torch.full((1,), 0.4), avo, low, rgt)
    elastic_strengths = model.graph.elastic_attention_raw_strengths
    segmentation_strengths = model.graph.structural_attention_raw_strengths
    elastic_gradient = torch.autograd.grad(
        output.velocity.square().mean(),
        elastic_strengths,
        retain_graph=True,
        allow_unused=True,
    )[0]
    crossed_elastic_gradient = torch.autograd.grad(
        output.velocity.square().mean(),
        segmentation_strengths,
        retain_graph=True,
        allow_unused=True,
    )[0]
    crossed_segmentation_gradient = torch.autograd.grad(
        output.segmentation_logits.square().mean(),
        elastic_strengths,
        allow_unused=True,
    )[0]

    assert elastic_gradient is not None
    assert torch.isfinite(elastic_gradient).all()
    assert elastic_gradient.abs().sum() > 0
    assert crossed_elastic_gradient is None
    assert crossed_segmentation_gradient is None


def test_task_specific_elastic_strength_initialization_is_independent():
    model = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_specific_rgt",
        graph_relation_candidates=2,
        graph_structural_prior_initial_strengths=((0.5, 0.5), (0.5, 0.5)),
        graph_elastic_structural_prior_initial_strengths=(
            (0.25, 0.25),
            (0.25, 0.25),
        ),
    )
    torch.testing.assert_close(
        torch.nn.functional.softplus(model.graph.structural_attention_raw_strengths),
        torch.full((2, 2), 0.5),
    )
    torch.testing.assert_close(
        torch.nn.functional.softplus(model.graph.elastic_attention_raw_strengths),
        torch.full((2, 2), 0.25),
    )


def test_task_specific_elastic_component_mask_is_exact_and_capacity_matched():
    torch.manual_seed(19)
    control = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_specific_rgt",
        graph_relation_candidates=2,
        graph_elastic_structural_prior_component_mask=((0.0, 0.0), (0.0, 0.0)),
    ).eval()
    treatment = SAGEAVO(
        hidden_channels=16,
        graph_layers=1,
        graph_heads=4,
        graph_mode="relational_task_specific_rgt",
        graph_relation_candidates=2,
        graph_elastic_structural_prior_component_mask=((0.0, 0.0), (0.0, 1.0)),
    ).eval()
    treatment.load_state_dict(control.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in treatment.parameters()
    )
    torch.testing.assert_close(
        control.graph.elastic_structural_prior_component_mask,
        torch.zeros(2, 2),
    )
    torch.testing.assert_close(
        treatment.graph.elastic_structural_prior_component_mask,
        torch.tensor(((0.0, 0.0), (0.0, 1.0))),
    )

    avo, low, rgt = _inputs()
    time = torch.full((1,), 0.4)
    control_output = control(low, time, avo, low, rgt)
    treatment_output = treatment(low, time, avo, low, rgt)
    assert not torch.equal(control_output.velocity, treatment_output.velocity)

    gradient = torch.autograd.grad(
        treatment_output.velocity.square().mean(),
        treatment.graph.elastic_attention_raw_strengths,
    )[0]
    torch.testing.assert_close(gradient[:1], torch.zeros_like(gradient[:1]))
    torch.testing.assert_close(gradient[1, :1], torch.zeros_like(gradient[1, :1]))
    assert torch.isfinite(gradient[1, 1])
    assert gradient[1, 1].abs() > 0


def test_task_specific_elastic_component_mask_rejects_nonbinary_values():
    with pytest.raises(ValueError, match="must be binary"):
        SAGEAVO(
            hidden_channels=16,
            graph_layers=1,
            graph_heads=4,
            graph_mode="relational_task_specific_rgt",
            graph_elastic_structural_prior_component_mask=((0.0, 0.5), (0.0, 1.0)),
        )


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
