"""Regression tests for diagnostic-only Revision-3.3.2 instrumentation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sage_avo.diagnostics.accounting import EpochLossObserver
from sage_avo.diagnostics.checkpoint_analysis import (
    _cartesian_graph_diagnostic,
    _gradient_diagnostics,
    _graph_attention_details,
)
from sage_avo.diagnostics.contracts import build_diagnostic_sample_manifest
from sage_avo.diagnostics.elastic_strength_sweep import (
    _inverse_softplus,
    _raw_strength_matrix,
)
from sage_avo.diagnostics.live_logging import BatchProgressLogger
from sage_avo.models.variants import build_sage_avo_variant
from sage_avo.training.engine import (
    PhysicsNormalization,
    StepMetrics,
    train_epoch,
)
from sage_avo.training.losses import LossWeights


def _metrics(physics: float) -> StepMetrics:
    values = {name: 1.0 for name in StepMetrics.__dataclass_fields__}
    values["physics"] = physics
    return StepMetrics(**values)


def _small_model(variant: str = "full") -> torch.nn.Module:
    return build_sage_avo_variant(
        variant,
        hidden_channels=8,
        graph_layers=2,
        graph_heads=2,
        max_rgt_shift=1,
        classes=3,
    )


def _small_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "avo": torch.randn(2, 3, 4, 5, generator=generator),
        "low": torch.randn(2, 3, 4, 5, generator=generator),
        "target": torch.randn(2, 3, 4, 5, generator=generator),
        "rgt": torch.arange(4, dtype=torch.float32)[None, :, None].expand(2, 4, 5),
        "mask": torch.ones(2, 1, 4, 5),
        "segmentation": torch.zeros(2, 4, 5, dtype=torch.long),
        "physics_eligible": torch.zeros(2, dtype=torch.bool),
    }


def test_elastic_strength_sweep_uses_finite_inverse_softplus() -> None:
    for requested in (0.0, 0.0625, 0.25, 0.5):
        raw = _inverse_softplus(requested, dtype=torch.float32)
        assert np.isfinite(raw)
        realized = torch.nn.functional.softplus(torch.tensor(raw)).item()
        if requested == 0.0:
            assert realized <= torch.finfo(torch.float32).tiny
        else:
            assert np.isclose(realized, requested, rtol=1e-6, atol=0.0)


def test_elastic_strength_sweep_preserves_component_matrix_layout() -> None:
    requested = ((0.125, 0.0), (0.25, 0.5))
    raw = _raw_strength_matrix(requested, dtype=torch.float32, device=torch.device("cpu"))
    realized = torch.nn.functional.softplus(raw)
    torch.testing.assert_close(
        realized,
        torch.tensor(requested),
        rtol=1e-6,
        atol=torch.finfo(torch.float32).tiny,
    )


def test_physics_accounting_excludes_inactive_patches_from_conditional_mean() -> None:
    observer = EpochLossObserver(physics_weight=0.5)
    batch = _small_batch()
    batch["physics_eligible"] = torch.tensor([True, False])
    observer(batch, _metrics(2.0))
    batch["physics_eligible"] = torch.tensor([False, False])
    observer(batch, _metrics(0.0))
    summary = observer.summary()
    assert summary["conditional_raw_physics_loss"] == 2.0
    assert summary["all_step_raw_physics_loss"] == 1.0
    assert summary["physics_active_steps"] == 1
    assert summary["physics_inactive_steps"] == 1
    assert "excludes ineligible pixels" in summary["mixed_batch_reduction"]


def test_epoch_observer_does_not_change_training_trajectory(capsys, tmp_path) -> None:
    torch.manual_seed(3)
    first = _small_model("no_gnn")
    second = deepcopy(first)
    batch = _small_batch()
    normalization = PhysicsNormalization(
        x_mean=torch.zeros(1, 3, 1, 1),
        x_std=torch.ones(1, 3, 1, 1),
        y_mean=torch.zeros(1, 3, 1, 1),
        y_std=torch.ones(1, 3, 1, 1),
    )
    weights = LossWeights(physics=0.0, structure=0.0, contrastive=0.0)
    optimizer_first = torch.optim.AdamW(first.parameters(), lr=1e-4)
    optimizer_second = torch.optim.AdamW(second.parameters(), lr=1e-4)
    generator_first = torch.Generator().manual_seed(91)
    generator_second = torch.Generator().manual_seed(91)
    metrics_first = train_epoch(
        first,
        [batch],
        optimizer_first,
        normalization,
        weights,
        time_generator=generator_first,
    )
    observer = EpochLossObserver(physics_weight=0.0)
    progress = BatchProgressLogger(
        epoch=1,
        total_epochs=1,
        total_batches=1,
        physics_weight=0.0,
        interval_batches=1,
        output_path=tmp_path / "training_progress.log",
    )

    def combined_observer(batch_values, metrics) -> None:
        observer(batch_values, metrics)
        progress(batch_values, metrics)

    metrics_second = train_epoch(
        second,
        [batch],
        optimizer_second,
        normalization,
        weights,
        time_generator=generator_second,
        metrics_observer=combined_observer,
    )
    assert metrics_first == metrics_second
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
    assert (
        optimizer_first.state_dict()["state"].keys()
        == optimizer_second.state_dict()["state"].keys()
    )
    output = capsys.readouterr().out
    assert "epoch=1/1" in output
    assert "batch=1/1" in output
    assert "physics_active=false" in output
    durable_output = (tmp_path / "training_progress.log").read_text(encoding="utf-8")
    assert "epoch=1/1" in durable_output
    assert "batch=1/1" in durable_output


def test_graph_diagnostics_preserve_default_forward_and_optimizer() -> None:
    torch.manual_seed(7)
    model = _small_model()
    model.eval()
    batch = _small_batch()
    state = 0.5 * (batch["low"] + batch["target"])
    time = torch.full((2,), 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer_before = deepcopy(optimizer.state_dict())
    with torch.no_grad():
        output_before = model(state, time, batch["avo"], batch["low"], batch["rgt"])
        _, attentions, _, _, _ = _graph_attention_details(model, batch, state, time)
        output_after = model(state, time, batch["avo"], batch["low"], batch["rgt"])
    assert len(attentions) == 2
    torch.testing.assert_close(output_before.velocity, output_after.velocity, rtol=0, atol=0)
    for before, after in zip(parameters_before, model.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert optimizer.state_dict() == optimizer_before


@pytest.mark.parametrize("topology", [
    "rgt_v1_legacy", "rgt_v2_tie_fixed", "rgt_v3_confidence_blocked", "shuffled",
])
def test_cartesian_probe_changes_topology_and_restores_forward(topology: str) -> None:
    torch.manual_seed(12345)
    model = build_sage_avo_variant(
        "full", hidden_channels=8, graph_heads=2, max_rgt_shift=1,
        rgt_topology=topology,
    ).eval()
    batch = _small_batch()
    # A dipping horizon makes RGT and Cartesian neighbors observably different.
    batch["rgt"] = batch["rgt"] - torch.arange(5, dtype=torch.float32)[None, None, :]
    state = 0.5 * (batch["low"] + batch["target"])
    time = torch.full((2,), 0.5)
    parameters_before = deepcopy(model.state_dict())
    with torch.no_grad():
        before = model(state, time, batch["avo"], batch["low"], batch["rgt"])
        with _cartesian_graph_diagnostic(model.graph):
            cartesian = model(state, time, batch["avo"], batch["low"], batch["rgt"])
        after = model(state, time, batch["avo"], batch["low"], batch["rgt"])
        expected = deepcopy(model)
        expected.graph.graph_mode = "cartesian"
        expected.graph.rgt_topology = "cartesian"
        reference = expected(state, time, batch["avo"], batch["low"], batch["rgt"])
    assert not torch.equal(before.edge_indices[0], cartesian.edge_indices[0])
    assert (before.velocity - cartesian.velocity).square().mean().sqrt().item() > 0
    torch.testing.assert_close(cartesian.velocity, reference.velocity, rtol=0, atol=0)
    assert torch.equal(cartesian.edge_indices[0], reference.edge_indices[0])
    assert model.graph.graph_mode == "rgt"
    assert model.graph.rgt_topology == topology
    torch.testing.assert_close(before.velocity, after.velocity, rtol=0, atol=0)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", [
    "relational_rgt", "relational_candidate_rgt", "relational_structural_prior_rgt",
    "relational_task_decoupled_rgt", "relational_task_specific_rgt",
    "relational_task_specific_cartesian",
])
def test_cartesian_probe_preserves_relational_architecture(mode: str) -> None:
    model = build_sage_avo_variant(
        "full", hidden_channels=8, graph_heads=2, max_rgt_shift=1,
        graph_mode_override=mode.replace("_cartesian", "_rgt"), graph_relation_candidates=2,
    ).eval()
    graph = model.graph
    graph.graph_mode = mode
    modules_before = {name: id(module) for name, module in graph.named_modules()}
    expected_mode = "cartesian" if mode == "relational_rgt" else mode.replace("_rgt", "_cartesian")
    with _cartesian_graph_diagnostic(graph):
        assert graph.graph_mode == expected_mode
        assert not hasattr(graph, "rgt_topology")
        assert graph.relation_candidates == 2
        assert modules_before == {name: id(module) for name, module in graph.named_modules()}
    assert graph.graph_mode == mode


@pytest.mark.parametrize("mode", ["rgt", "relational_task_specific_rgt"])
def test_cartesian_probe_restores_selectors_after_exception(mode: str) -> None:
    model = build_sage_avo_variant(
        "full", hidden_channels=8, graph_heads=2, graph_mode_override=mode,
        rgt_topology="rgt_v2_tie_fixed", graph_neighbor_scale=0.0,
    )
    graph = model.graph
    topology_before = getattr(graph, "rgt_topology", None)
    with pytest.raises(RuntimeError, match="probe failed"):
        with _cartesian_graph_diagnostic(graph):
            raise RuntimeError("probe failed")
    assert graph.graph_mode == mode
    assert getattr(graph, "rgt_topology", None) == topology_before
    assert getattr(graph, "graph_neighbor_scale", 0.0) == 0.0


def test_relational_graph_diagnostics_separate_tangent_and_normal_attention() -> None:
    model = build_sage_avo_variant(
        "full",
        hidden_channels=8,
        graph_layers=2,
        graph_heads=2,
        max_rgt_shift=1,
        classes=3,
        graph_mode_override="relational_rgt",
    ).eval()
    batch = _small_batch()
    state = 0.5 * (batch["low"] + batch["target"])
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        _, details, _, _, _ = _graph_attention_details(model, batch, state, time)
    assert len(details) == 4
    assert {(detail["relation"], int(detail["layer"])) for detail in details} == {
        ("tangential", 1),
        ("tangential", 2),
        ("normal", 1),
        ("normal", 2),
    }
    gate = details[0]["relation_gate_mean"]
    torch.testing.assert_close(gate.sum(), torch.tensor(1.0))
    assert torch.isfinite(gate).all()


def test_task_decoupled_graph_diagnostics_separate_task_streams() -> None:
    model = build_sage_avo_variant(
        "full",
        hidden_channels=8,
        graph_layers=1,
        graph_heads=2,
        max_rgt_shift=1,
        graph_relation_candidates=2,
        classes=3,
        graph_mode_override="relational_task_decoupled_rgt",
    ).eval()
    batch = _small_batch()
    state = 0.5 * (batch["low"] + batch["target"])
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        _, details, _, _, _ = _graph_attention_details(model, batch, state, time)

    assert len(details) == 4
    assert {(detail["stream"], detail["relation"], int(detail["layer"])) for detail in details} == {
        ("segmentation", "tangential", 1),
        ("elastic", "tangential", 1),
        ("segmentation", "normal", 1),
        ("elastic", "normal", 1),
    }
    for detail in details:
        torch.testing.assert_close(detail["relation_gate_mean"].sum(), torch.tensor(1.0))


def test_gradient_diagnostics_are_finite() -> None:
    model = _small_model("no_gnn")
    scalar = sum(parameter.square().sum() for parameter in model.parameters())
    terms = {
        name: (index + 1.0) * scalar
        for index, name in enumerate(
            (
                "flow_vp",
                "flow_vs",
                "flow_density",
                "full_vp",
                "full_vs",
                "full_density",
                "ssim",
                "segmentation_ce",
                "segmentation_dice",
                "physics",
                "structure",
                "contrastive",
            )
        )
    }
    terms.update(
        {
            "inversion": terms["flow_vp"] + terms["full_vp"],
            "segmentation": terms["segmentation_ce"] + terms["segmentation_dice"],
        }
    )
    coefficients = {name: 1.0 for name in terms if name not in {"inversion", "segmentation"}}
    rows, cosines = _gradient_diagnostics(
        model=model,
        terms=terms,
        coefficients=coefficients,
        epoch=1,
    )
    assert rows and cosines
    assert all(np.isfinite(row["raw_gradient_norm"]) for row in rows)
    assert all(np.isfinite(row["cosine_similarity"]) for row in cosines)


def test_fixed_diagnostic_manifest_selection_is_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    realizations = dataset / "realizations"
    realizations.mkdir(parents=True)
    np.savez_compressed(
        realizations / "realization_0000001.npz",
        segmentation=np.full((80, 140), 2, dtype=np.int64),
    )
    categories = [
        "background",
        "high_dip",
        "reservoir",
        "high_avo_gradient_change",
        "facies_boundary",
    ]
    rows = []
    for index, category in enumerate(categories):
        rows.append(
            {
                "split": "validation",
                "realization_id": 1,
                "geology_realization_id": 1,
                "realization_file": "realization_0000001.npz",
                "top": index,
                "left": index,
                "raw_height": 50,
                "raw_width": 100,
                "output_height": 50,
                "output_width": 100,
                "candidate_category": category,
                "candidate_score": float(index + 1),
                "physics_eligible": 1,
                "absolute_t0_seconds": 2.0,
                "native_dt_seconds": 0.004,
                "convolution_halo_samples": 40,
            }
        )
    for height, width in ((40, 80), (64, 128)):
        rows.append(
            {
                **rows[0],
                "raw_height": height,
                "raw_width": width,
                "output_height": 50,
                "output_width": 100,
                "physics_eligible": 0,
            }
        )
    pd.DataFrame(rows).to_csv(dataset / "patch_index.csv", index=False)
    (dataset / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    (dataset / "normalization.json").write_text("{}\n", encoding="utf-8")
    (dataset / "split_ids.json").write_text(json.dumps({"validation": [1]}), encoding="utf-8")
    config = {
        "fixed_validation": {
            "categories": categories,
            "include_non_native_scales": [[40, 80], [64, 128]],
            "whole_realization_count": 1,
            "selection_rule": "fixed deterministic test rule",
            "require_native_physics_examples": 5,
        }
    }
    first = build_diagnostic_sample_manifest(dataset_directory=dataset, observability_config=config)
    second = build_diagnostic_sample_manifest(
        dataset_directory=dataset, observability_config=config
    )
    first.pop("created_utc")
    second.pop("created_utc")
    assert first == second
    assert first["native_physics_patch_count"] == 5
    assert first["test_data_used"] is False
