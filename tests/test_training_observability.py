"""Regression tests for diagnostic-only Revision-3.3.2 instrumentation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sage_avo.diagnostics.accounting import EpochLossObserver
from sage_avo.diagnostics.checkpoint_analysis import (
    _gradient_diagnostics,
    _graph_attention_details,
)
from sage_avo.diagnostics.contracts import build_diagnostic_sample_manifest
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


def test_epoch_observer_does_not_change_training_trajectory(capsys) -> None:
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
