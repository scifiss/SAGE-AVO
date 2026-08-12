import json
from pathlib import Path

import numpy as np

from sage_avo.config import load_config
from sage_avo.data.prior import PriorDefinition, make_truth_derived_prior
from sage_avo.evaluation.controlled import evaluate_controlled_ablation
from sage_avo.experiments.dataset import prepare_controlled_dataset


REPOSITORY = Path(__file__).resolve().parents[1]


def test_prior_definition_is_explicit_and_truth_derived():
    definition = PriorDefinition()
    assert definition.truth_derived
    assert np.isclose(definition.sigma_time_samples, 16.625)
    assert np.isclose(definition.sigma_lateral_samples, 33.25)
    truth = np.zeros((3, 32, 40), dtype=np.float32)
    truth[:, 16:] = np.array([3000.0, 1700.0, 2.3])[:, None, None]
    prior = make_truth_derived_prior(truth, definition)
    assert prior.shape == truth.shape
    assert np.all(np.diff(prior[:, :, 20], axis=1) >= -1e-5)


def test_smoke_dataset_and_controlled_evaluator(tmp_path):
    config = load_config(REPOSITORY / "configs" / "controlled_ablation_v1.yaml")
    dataset = tmp_path / "dataset"
    manifest = prepare_controlled_dataset(config, dataset, smoke=True)
    assert manifest["status"] == "smoke"
    assert manifest["realization_count"] == 6
    assert manifest["prior"]["truth_derived"]
    split_ids = json.loads((dataset / "split_ids.json").read_text())["test"]
    assert len(split_ids) == 1

    experiment = tmp_path / "experiment"
    offsets = {
        "low_prior": None,
        "full": (4.0, 3.0, 0.001),
        "no_gnn": (8.0, 6.0, 0.002),
        "no_rgt": (10.0, 7.0, 0.003),
        "no_physics": (12.0, 9.0, 0.004),
    }
    for realization_id in split_ids:
        with np.load(dataset / "realizations" / f"realization_{realization_id:04d}.npz") as archive:
            truth = archive["elastic"]
            low = archive["low"]
            segmentation = archive["segmentation"]
        for variant, offset in offsets.items():
            destination = experiment / "predictions" / variant
            destination.mkdir(parents=True, exist_ok=True)
            if offset is None:
                np.savez_compressed(
                    destination / f"realization_{realization_id:04d}.npz", elastic=low
                )
            else:
                prediction = truth + np.asarray(offset, dtype=np.float32)[:, None, None]
                np.savez_compressed(
                    destination / f"realization_{realization_id:04d}.npz",
                    elastic=prediction,
                    segmentation=segmentation,
                )

    summary, per_realization, paired, representative = evaluate_controlled_ablation(
        experiment_directory=experiment,
        dataset_directory=dataset,
        bootstrap_repetitions=50,
        seed=3,
    )
    assert set(summary["variant"]) == set(offsets)
    assert set(per_realization["realization_id"]) == set(split_ids)
    assert set(paired["full_vs"]) == {"low_prior", "no_gnn", "no_rgt", "no_physics"}
    assert representative == split_ids[0]
    full_vp_rmse = per_realization.query(
        "variant == 'full' and domain == 'vp' and metric == 'rmse'"
    )["value"].iloc[0]
    assert np.isclose(full_vp_rmse, 4.0)
