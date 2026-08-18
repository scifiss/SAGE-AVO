from types import SimpleNamespace

import numpy as np

from sage_avo.training.losses import LossWeights
from sage_avo.training.selection import (
    CHECKPOINT_CRITERIA,
    CheckpointSelectionState,
    checkpoint_metadata,
    whole_realization_criterion,
    weighted_objective_contributions,
)


def _raw_metrics() -> SimpleNamespace:
    return SimpleNamespace(
        flow_vp=0.04,
        flow_vs=0.05,
        flow_density=0.02,
        full_vp=0.03,
        full_vs=0.04,
        full_density=0.01,
        ssim=0.12,
        segmentation=0.08,
        contrastive=0.0,
        physics=0.15,
        structure=0.05,
    )


def test_checkpoint_names_match_declared_formulas():
    for name in CHECKPOINT_CRITERIA:
        metadata = checkpoint_metadata(name, 0.25, 7)
        assert metadata["criterion_name"] == name
        assert metadata["criterion_formula"] == CHECKPOINT_CRITERIA[name]


def test_checkpoint_selection_state_round_trip_preserves_resume_minima_and_maxima():
    state = CheckpointSelectionState()
    assert state.update("fixed_objective", 0.4, 1)
    assert state.update("sampling", 0.3, 2)
    assert state.update("segmentation", 0.55, 3)
    restored = CheckpointSelectionState.from_mapping(state.to_dict())
    assert not restored.update("fixed_objective", 0.5, 4)
    assert not restored.update("sampling", 0.35, 4)
    assert not restored.update("segmentation", 0.50, 4)
    assert restored.best_epochs == state.best_epochs
    assert restored.best_values == state.best_values


def test_fixed_objective_is_unchanged_when_only_current_curriculum_weights_change():
    metrics = _raw_metrics()
    fixed = LossWeights(density=3.5, ssim=0.05, physics=0.35, structure=0.375)
    current_early = LossWeights(density=2.0, ssim=0.15, physics=0.5, structure=0.5)
    first = weighted_objective_contributions(metrics, fixed)
    _ = weighted_objective_contributions(metrics, current_early)
    second = weighted_objective_contributions(metrics, fixed)
    assert first == second
    assert first["total"] != weighted_objective_contributions(metrics, current_early)["total"]


def test_whole_realization_criterion_matches_declared_formula():
    assert np.isclose(
        whole_realization_criterion([0.20, 0.30, 0.40], 0.60),
        0.24,
    )
