from pathlib import Path

import pytest

from sage_avo.experiments.prediction import preferred_inference_checkpoint


def test_v003_prediction_uses_configured_whole_realization_checkpoint():
    config = {
        "schema_version": 3,
        "training": {
            "checkpointing": {"preferred_final_criterion": "whole_realization"}
        },
    }
    assert preferred_inference_checkpoint(config, Path("run")) == Path(
        "run/best_whole_realization.pt"
    )


def test_v002_prediction_preserves_historical_sampling_selection():
    assert preferred_inference_checkpoint({"schema_version": 2}, Path("run")) == Path(
        "run/best_sampling.pt"
    )


def test_unknown_v003_prediction_criterion_is_rejected():
    config = {
        "schema_version": 3,
        "training": {"checkpointing": {"preferred_final_criterion": "mystery"}},
    }
    with pytest.raises(ValueError, match="preferred_final_criterion"):
        preferred_inference_checkpoint(config, Path("run"))
