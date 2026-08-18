import numpy as np

from sage_avo.data.candidates import (
    PatchCandidateConfig,
    diverse_patch_candidates,
)
from sage_avo.experiments.ml_dataset import _patch_rows, split_records_by_geology


def _fields() -> tuple[np.ndarray, ...]:
    shape = (160, 140)
    rows, columns = np.indices(shape)
    rgt = rows / shape[0] + 0.08 * np.sin(columns / 12.0)
    segmentation = np.zeros(shape, dtype=np.uint8)
    channel = np.abs(columns - 70 - 20 * np.sin(rows / 25.0)) < 14
    segmentation[channel] = 1
    segmentation[channel & (rows < 65)] = 2
    base = np.sin(rows / 8.0 + columns / 17.0)
    avo = np.stack((base, 0.8 * base, 0.4 * base + 0.5 * channel))
    return avo, rgt, segmentation, np.ones(shape, dtype=np.uint8)


def test_diverse_candidates_cover_categories_depths_and_have_no_duplicates():
    avo, rgt, segmentation, valid = _fields()
    config = PatchCandidateConfig(minimum_separation_samples=5.0, depth_bins=4)
    candidates = diverse_patch_candidates(
        avo=avo,
        rgt=rgt,
        segmentation=segmentation,
        valid_mask=valid,
        raw_shape=(40, 50),
        count=48,
        rng=np.random.default_rng(123),
        config=config,
    )
    coordinates = [(item.top, item.left) for item in candidates]
    assert len(coordinates) == len(set(coordinates)) == 48
    assert len({item.depth_bin for item in candidates}) == 4
    assert {item.category for item in candidates} >= {
        "facies_boundary",
        "high_dip",
        "reservoir",
        "background",
    }
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            assert np.hypot(first.top - second.top, first.left - second.left) >= 5.0


def test_uniform_candidate_mode_is_deterministic_separated_and_unique():
    avo, rgt, segmentation, valid = _fields()
    config = PatchCandidateConfig(mode="uniform", minimum_separation_samples=4.0)
    arguments = dict(
        avo=avo,
        rgt=rgt,
        segmentation=segmentation,
        valid_mask=valid,
        raw_shape=(50, 60),
        count=30,
        config=config,
    )
    first = diverse_patch_candidates(rng=np.random.default_rng(77), **arguments)
    second = diverse_patch_candidates(rng=np.random.default_rng(77), **arguments)
    assert first == second
    assert len({(item.top, item.left) for item in first}) == 30
    assert {item.category for item in first} == {"uniform"}


def test_candidates_respect_separation_from_previous_scales():
    avo, rgt, segmentation, valid = _fields()
    excluded = {(10, 12), (40, 42)}
    candidates = diverse_patch_candidates(
        avo=avo,
        rgt=rgt,
        segmentation=segmentation,
        valid_mask=valid,
        raw_shape=(40, 50),
        count=24,
        rng=np.random.default_rng(91),
        config=PatchCandidateConfig(minimum_separation_samples=7.0),
        excluded_coordinates=excluded,
    )
    for candidate in candidates:
        for top, left in excluded:
            assert np.hypot(candidate.top - top, candidate.left - left) >= 7.0


def test_observation_variants_of_one_geology_never_cross_splits():
    records = [
        {
            "realization_id": 3_000_000 + 2 * geology + variant,
            "split_group_id": 3_000_000 + geology,
        }
        for geology in range(10)
        for variant in range(2)
    ]
    split_ids, split_group_ids = split_records_by_geology(
        records, (0.6, 0.2, 0.2), seed=20260816
    )
    membership = {
        realization_id: split_name
        for split_name, ids in split_ids.items()
        for realization_id in ids
    }
    for geology in range(10):
        first = 3_000_000 + 2 * geology
        assert membership[first] == membership[first + 1]
    assert not (
        set(split_group_ids["train"]) & set(split_group_ids["validation"])
        or set(split_group_ids["train"]) & set(split_group_ids["test"])
        or set(split_group_ids["validation"]) & set(split_group_ids["test"])
    )


def test_dataset_orchestration_passes_the_configured_candidate_mode(tmp_path):
    avo, rgt, segmentation, valid = _fields()
    realization_file = "realization_3000000.npz"
    np.savez_compressed(
        tmp_path / realization_file,
        avo=avo,
        rgt=rgt,
        segmentation=segmentation,
        valid_mask=valid,
        time_ms=np.arange(rgt.shape[0], dtype=float) * 4.0,
    )
    config = {
        "prior": {"dt_seconds": 0.004},
        "physics_context": {"convolution_halo_samples": 40},
        "patches": {
            "output_shape": [40, 50],
            "scales": [{"raw_shape": [40, 50], "fraction": 1.0}],
            "per_realization": {"train": 12},
            "maximum_invalid_fraction": 0.15,
            "candidate_sampler": {
                "mode": "diverse",
                "depth_bins": 4,
                "minimum_separation_samples": 4.0,
                "top_quantile": 0.75,
                "categories": [
                    "facies_boundary",
                    "high_dip",
                    "high_rgt_change",
                    "high_avo_gradient_change",
                    "reservoir",
                    "background",
                ],
                "representative_angles_degrees": [10.0, 24.0, 38.0],
            },
        },
    }
    rows = _patch_rows(
        split_name="train",
        records=[
            {
                "realization_id": 3_000_000,
                "geology_realization_id": 3_000_000,
                "observation_variant_id": 0,
                "file": realization_file,
            }
        ],
        config=config,
        dataset_realizations=tmp_path,
        seed=123,
    )
    assert len(rows) == 12
    assert {row["candidate_category"] for row in rows} != {"uniform_random"}
