"""Stage-03 leakage-safe ML dataset construction from Stage-02 realizations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_avo.data.candidates import PatchCandidateConfig, diverse_patch_candidates
from sage_avo.data.prior import PriorDefinition, make_truth_derived_prior
from sage_avo.data.splits import split_realizations

from .manifest import file_sha256, write_json


REQUIRED_STAGE02_CHANNELS = (
    "avo",
    "elastic",
    "rgt",
    "segmentation",
    "valid_mask",
    "delta",
    "sand_probability",
    "porosity",
    "plume_mask",
)


def _configuration_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_realization(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_STAGE02_CHANNELS if name not in archive.files]
        if missing:
            raise ValueError(f"{path.name} is missing required channels: {missing}")
        shape = archive["rgt"].shape
        if archive["avo"].shape != (3, *shape) or archive["elastic"].shape != (3, *shape):
            raise ValueError(f"{path.name} has inconsistent AVO/elastic dimensions")
        for name in ("segmentation", "valid_mask", "delta", "sand_probability", "porosity", "plume_mask"):
            if archive[name].shape != shape:
                raise ValueError(f"{path.name} channel {name!r} has inconsistent dimensions")
        if not all(np.isfinite(archive[name]).all() for name in REQUIRED_STAGE02_CHANNELS):
            raise ValueError(f"{path.name} contains non-finite required channels")
        realization_id = int(archive["realization_id"])
        geology_realization_id = int(
            archive["geology_realization_id"]
            if "geology_realization_id" in archive.files
            else realization_id
        )
        observation_variant_id = int(
            archive["observation_variant_id"]
            if "observation_variant_id" in archive.files
            else 0
        )
        labels = set(np.unique(archive["segmentation"]).astype(int).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"{path.name} contains unsupported segmentation labels: {labels}")
        masks = set(np.unique(archive["valid_mask"]).astype(int).tolist())
        if not masks.issubset({0, 1}):
            raise ValueError(f"{path.name} contains a non-binary valid mask")
    return {
        "realization_id": realization_id,
        "geology_realization_id": geology_realization_id,
        "observation_variant_id": observation_variant_id,
        "split_group_id": geology_realization_id,
        "file": path.name,
        "shape": list(shape),
        "archive_sha256": file_sha256(path),
    }


def _streaming_training_statistics(paths: list[Path]) -> dict[str, list[float]]:
    sums = {"x": np.zeros(3), "y": np.zeros(3)}
    squares = {"x": np.zeros(3), "y": np.zeros(3)}
    counts = {"x": np.zeros(3), "y": np.zeros(3)}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            mask = archive["valid_mask"].astype(bool)
            for key, array_name in (("x", "avo"), ("y", "elastic")):
                array = np.asarray(archive[array_name], dtype=np.float64)
                for channel in range(3):
                    values = array[channel][mask & np.isfinite(array[channel])]
                    sums[key][channel] += values.sum()
                    squares[key][channel] += np.square(values).sum()
                    counts[key][channel] += values.size
    result: dict[str, list[float]] = {}
    for key in ("x", "y"):
        mean = sums[key] / np.maximum(counts[key], 1.0)
        variance = np.maximum(squares[key] / np.maximum(counts[key], 1.0) - mean**2, 1e-16)
        result[f"{key}_mean"] = mean.tolist()
        result[f"{key}_std"] = np.sqrt(variance).tolist()
    return result


def _copy_with_prior(source: Path, destination: Path, prior: PriorDefinition) -> None:
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in REQUIRED_STAGE02_CHANNELS}
        payload["realization_id"] = archive["realization_id"]
        for name in ("geology_realization_id", "observation_variant_id"):
            if name in archive.files:
                payload[name] = archive[name]
        for name in (
            "avo_clean",
            "elastic_brine",
            "time_ms",
            "cdp",
            "angles_degrees",
            "strat_fraction",
            "reservoir_mask",
            "co2_saturation",
            "forward_specification_sha256",
            "wavelet_id_by_angle",
        ):
            if name in archive.files:
                payload[name] = archive[name]
    payload["low"] = make_truth_derived_prior(payload["elastic"], prior)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def _scale_counts(total: int, scales: list[dict[str, Any]]) -> list[int]:
    fractions = np.asarray([float(item["fraction"]) for item in scales])
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Patch-scale fractions must sum to one")
    raw = total * fractions
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: total - counts.sum()]:
        counts[index] += 1
    return counts.tolist()


def split_records_by_geology(
    records: list[dict[str, Any]],
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Split archive IDs through their shared geology IDs without leakage."""
    group_order = list(dict.fromkeys(int(record["split_group_id"]) for record in records))
    if len(group_order) < 3:
        raise ValueError(
            "At least three distinct geology realization groups are required for splits"
        )
    split = split_realizations(len(group_order), fractions, seed)
    split_group_ids = {
        "train": [group_order[index] for index in split.train],
        "validation": [group_order[index] for index in split.validation],
        "test": [group_order[index] for index in split.test],
    }
    split_ids = {
        split_name: [
            int(record["realization_id"])
            for record in records
            if int(record["split_group_id"]) in set(group_ids)
        ]
        for split_name, group_ids in split_group_ids.items()
    }
    return split_ids, split_group_ids


def _patch_rows(
    *,
    split_name: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    dataset_realizations: Path,
    seed: int,
) -> list[dict[str, Any]]:
    patch_config = config["patches"]
    output_height, output_width = (int(value) for value in patch_config["output_shape"])
    per_realization = int(patch_config["per_realization"][split_name])
    scale_counts = _scale_counts(per_realization, patch_config["scales"])
    maximum_invalid = float(patch_config["maximum_invalid_fraction"])
    rows: list[dict[str, Any]] = []
    for record in records:
        realization_id = int(record["realization_id"])
        path = dataset_realizations / record["file"]
        with np.load(path, allow_pickle=False) as archive:
            valid_mask = archive["valid_mask"].astype(bool)
            candidate_arrays = {
                name: archive[name]
                for name in ("avo", "rgt", "segmentation", "time_ms")
                if name in archive.files
            }
            wavelet_ids = (
                archive["wavelet_id_by_angle"].astype(str).tolist()
                if "wavelet_id_by_angle" in archive.files
                else []
            )
        height, width = valid_mask.shape
        rng = np.random.default_rng(seed + 1009 * realization_id)
        candidate_mapping = patch_config.get("candidate_sampler", {"mode": "uniform_random"})
        candidate_mode = str(candidate_mapping.get("mode", "uniform_random"))
        uses_candidate_sampler = candidate_mode in {
            "diverse",
            "uniform",
        }
        used_coordinates: set[tuple[int, int]] = set()
        for scale_index, (scale, count) in enumerate(zip(patch_config["scales"], scale_counts)):
            raw_height, raw_width = (int(value) for value in scale["raw_shape"])
            if raw_height > height or raw_width > width:
                raise ValueError(f"Patch scale {(raw_height, raw_width)} exceeds realization {record['file']}")
            if uses_candidate_sampler:
                if set(candidate_arrays) < {"avo", "rgt", "segmentation"}:
                    raise ValueError(
                        f"Diverse candidates require AVO/RGT/segmentation in {record['file']}"
                    )
                candidate_config = PatchCandidateConfig(
                    mode=candidate_mode,
                    depth_bins=int(candidate_mapping["depth_bins"]),
                    minimum_separation_samples=float(
                        candidate_mapping["minimum_separation_samples"]
                    ),
                    top_quantile=float(candidate_mapping["top_quantile"]),
                    maximum_attempt_multiplier=int(
                        candidate_mapping.get("maximum_attempt_multiplier", 100)
                    ),
                    categories=tuple(candidate_mapping["categories"]),
                )
                candidates = diverse_patch_candidates(
                    avo=candidate_arrays["avo"],
                    rgt=candidate_arrays["rgt"],
                    segmentation=candidate_arrays["segmentation"],
                    valid_mask=valid_mask,
                    raw_shape=(raw_height, raw_width),
                    count=count,
                    rng=rng,
                    config=candidate_config,
                    representative_angles_degrees=tuple(
                        candidate_mapping["representative_angles_degrees"]
                    ),
                    maximum_invalid_fraction=maximum_invalid,
                    excluded_coordinates=used_coordinates,
                )
                for candidate in candidates:
                    top, left = candidate.top, candidate.left
                    used_coordinates.add((top, left))
                    rows.append(
                        {
                            "split": split_name,
                            "realization_id": realization_id,
                            "geology_realization_id": int(
                                record["geology_realization_id"]
                            ),
                            "observation_variant_id": int(
                                record["observation_variant_id"]
                            ),
                            "realization_file": record["file"],
                            "top": top,
                            "left": left,
                            "raw_height": raw_height,
                            "raw_width": raw_width,
                            "output_height": output_height,
                            "output_width": output_width,
                            "time_scale": raw_height / output_height,
                            "trace_scale": raw_width / output_width,
                            "scale_index": scale_index,
                            "candidate_category": candidate.category,
                            "depth_bin": candidate.depth_bin,
                            "candidate_score": candidate.score,
                            "physics_eligible": int(
                                (raw_height, raw_width)
                                == (output_height, output_width)
                            ),
                            "source_sample_top": top,
                            "absolute_t0_seconds": (
                                float(candidate_arrays["time_ms"][top]) / 1000.0
                                if "time_ms" in candidate_arrays
                                else np.nan
                            ),
                            "native_dt_seconds": float(
                                config.get("physics_context", {}).get(
                                    "native_dt_seconds", config["prior"]["dt_seconds"]
                                )
                            ),
                            "mute_origin_seconds": float(
                                config.get("physics_context", {}).get(
                                    "mute_origin_seconds", 0.0
                                )
                            ),
                            "convolution_halo_samples": int(
                                config.get("physics_context", {}).get(
                                    "convolution_halo_samples", 0
                                )
                            ),
                            "wavelet_ids": "|".join(sorted(set(wavelet_ids))),
                        }
                    )
                continue

            accepted = 0
            attempts = 0
            while accepted < count and attempts < max(100, 50 * count):
                attempts += 1
                top = int(rng.integers(0, height - raw_height + 1))
                left = int(rng.integers(0, width - raw_width + 1))
                mask = valid_mask[top : top + raw_height, left : left + raw_width]
                if 1.0 - float(mask.mean()) > maximum_invalid:
                    continue
                rows.append(
                    {
                        "split": split_name,
                        "realization_id": realization_id,
                        "geology_realization_id": int(record["geology_realization_id"]),
                        "observation_variant_id": int(record["observation_variant_id"]),
                        "realization_file": record["file"],
                        "top": top,
                        "left": left,
                        "raw_height": raw_height,
                        "raw_width": raw_width,
                        "output_height": output_height,
                        "output_width": output_width,
                        "time_scale": raw_height / output_height,
                        "trace_scale": raw_width / output_width,
                        "scale_index": scale_index,
                        "candidate_category": "uniform_random",
                        "depth_bin": int(
                            min(
                                3,
                                4 * (top + raw_height // 2) / max(height, 1),
                            )
                        ),
                        "candidate_score": 1.0,
                        "physics_eligible": int(
                            (raw_height, raw_width) == (output_height, output_width)
                        ),
                        "source_sample_top": top,
                        "absolute_t0_seconds": (
                            float(candidate_arrays["time_ms"][top]) / 1000.0
                            if "time_ms" in candidate_arrays
                            else np.nan
                        ),
                        "native_dt_seconds": float(config["prior"]["dt_seconds"]),
                        "mute_origin_seconds": 0.0,
                        "convolution_halo_samples": 0,
                        "wavelet_ids": "|".join(sorted(set(wavelet_ids))),
                    }
                )
                accepted += 1
            if accepted != count:
                raise RuntimeError(
                    f"Only {accepted}/{count} valid patches accepted for realization {realization_id}"
                )
    return rows


def validate_dataset_integrity(dataset_directory: str | Path) -> dict[str, Any]:
    """Check split disjointness, patch bounds, channels, priors, and normalization provenance."""
    root = Path(dataset_directory)
    manifest_path = root / "dataset_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    split_ids = json.loads((root / "split_ids.json").read_text(encoding="utf-8"))
    sets = {name: set(map(int, values)) for name, values in split_ids.items()}
    disjoint = not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    if not disjoint:
        raise ValueError("Realization leakage detected between dataset splits")
    split_group_path = root / "split_group_ids.json"
    if split_group_path.exists():
        split_group_ids = json.loads(split_group_path.read_text(encoding="utf-8"))
        group_sets = {
            name: set(map(int, values)) for name, values in split_group_ids.items()
        }
        group_disjoint = not (
            group_sets["train"] & group_sets["validation"]
            or group_sets["train"] & group_sets["test"]
            or group_sets["validation"] & group_sets["test"]
        )
        if not group_disjoint:
            raise ValueError("Geology-realization leakage detected between dataset splits")
    else:
        split_group_ids = split_ids
        group_sets = sets
        group_disjoint = disjoint
    index = pd.read_csv(root / "patch_index.csv")
    mismatched = [
        int(row.realization_id)
        for row in index.itertuples()
        if int(row.realization_id) not in sets[str(row.split)]
    ]
    if mismatched:
        raise ValueError(f"Patch rows assigned to the wrong realization split: {mismatched[:5]}")
    if "geology_realization_id" in index:
        group_mismatched = [
            int(row.geology_realization_id)
            for row in index.itertuples()
            if int(row.geology_realization_id) not in group_sets[str(row.split)]
        ]
        if group_mismatched:
            raise ValueError(
                "Patch rows assigned across geology-realization splits: "
                f"{group_mismatched[:5]}"
            )
    all_ids = sets["train"] | sets["validation"] | sets["test"]
    expected_count = int(manifest.get("realization_count", len(all_ids)))
    if len(all_ids) != expected_count:
        raise ValueError(
            f"Split files represent {len(all_ids)} realizations; expected {expected_count}"
        )
    realization_paths = sorted((root / "realizations").glob("realization_*.npz"))
    records = [_validate_realization(path) for path in realization_paths]
    record_by_id = {int(record["realization_id"]): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("Duplicate realization IDs found in the Stage-03 realization files")
    if set(record_by_id) != all_ids:
        raise ValueError("Stage-03 realization files do not match the persisted split IDs")
    realization_shapes = {
        realization_id: tuple(int(value) for value in record["shape"])
        for realization_id, record in record_by_id.items()
    }
    bounds_valid = True
    invalid_fraction_valid = True
    maximum_invalid = float(manifest.get("patch_recipe", {}).get("maximum_invalid_fraction", 1.0))
    for realization_id, group in index.groupby("realization_id", sort=False):
        realization_id = int(realization_id)
        height, width = realization_shapes[realization_id]
        tops = group["top"].to_numpy(dtype=int)
        lefts = group["left"].to_numpy(dtype=int)
        raw_heights = group["raw_height"].to_numpy(dtype=int)
        raw_widths = group["raw_width"].to_numpy(dtype=int)
        if not (
            np.all(tops >= 0)
            and np.all(lefts >= 0)
            and np.all(tops + raw_heights <= height)
            and np.all(lefts + raw_widths <= width)
        ):
            bounds_valid = False
            break
        path = root / "realizations" / record_by_id[realization_id]["file"]
        with np.load(path, allow_pickle=False) as archive:
            if "low" not in archive.files:
                raise ValueError(f"{path.name} is missing the truth-derived low-frequency prior")
            if archive["low"].shape != archive["elastic"].shape:
                raise ValueError(f"{path.name} prior and target dimensions disagree")
            if not np.isfinite(archive["low"]).all():
                raise ValueError(f"{path.name} contains a non-finite low-frequency prior")
            valid_mask = archive["valid_mask"].astype(bool)
            for row in group.itertuples():
                patch_mask = valid_mask[
                    int(row.top) : int(row.top + row.raw_height),
                    int(row.left) : int(row.left + row.raw_width),
                ]
                if 1.0 - float(patch_mask.mean()) > maximum_invalid + 1e-12:
                    invalid_fraction_valid = False
                    break
        if not invalid_fraction_valid:
            break
    if not bounds_valid:
        raise ValueError("At least one patch lies outside its source realization")
    if not invalid_fraction_valid:
        raise ValueError("At least one patch exceeds the configured invalid-sample fraction")
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    required_statistics = ("x_mean", "x_std", "y_mean", "y_std")
    finite_stats = all(np.isfinite(normalization[name]).all() for name in required_statistics)
    if not finite_stats or any(
        np.asarray(normalization[name]).shape != (3,) for name in required_statistics
    ):
        raise ValueError("Normalization statistics are invalid")
    recomputed = _streaming_training_statistics(
        [root / "realizations" / record_by_id[value]["file"] for value in split_ids["train"]]
    )
    normalization_matches_training = all(
        np.allclose(normalization[name], recomputed[name], rtol=1e-12, atol=1e-12)
        for name in required_statistics
    )
    if not normalization_matches_training:
        raise ValueError("Normalization does not match the persisted training-realization IDs")

    deterministic_validation_test = True
    patch_recipe = manifest.get("patch_recipe")
    if patch_recipe is not None and "seed" in manifest:
        regenerated: list[dict[str, Any]] = []
        for split_name in ("validation", "test"):
            regenerated.extend(
                _patch_rows(
                    split_name=split_name,
                    records=[record_by_id[value] for value in split_ids[split_name]],
                    config={
                        "patches": patch_recipe,
                        "prior": manifest["prior_definition"],
                        "physics_context": manifest.get("physics_context", {}),
                    },
                    dataset_realizations=root / "realizations",
                    seed=int(manifest["seed"]),
                )
            )
        persisted = index[index["split"].isin(("validation", "test"))].reset_index(drop=True)
        expected = pd.DataFrame(regenerated).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                persisted,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-15,
            )
            deterministic_validation_test = True
        except AssertionError:
            deterministic_validation_test = False
        if not deterministic_validation_test:
            raise ValueError("Validation/test patch coordinates are not reproducible from the manifest")
    duplicate_columns = [
        "split",
        "realization_id",
        "top",
        "left",
        "raw_height",
        "raw_width",
    ]
    duplicate_count = int(index.duplicated(duplicate_columns).sum())
    coordinate_duplicate_count = int(
        index.duplicated(["split", "realization_id", "top", "left"]).sum()
    )
    if manifest.get("patch_recipe", {}).get("candidate_sampler", {}).get("mode") in {
        "diverse",
        "uniform",
    } and coordinate_duplicate_count:
        raise ValueError("V003 candidate sampler produced duplicate coordinates")
    return {
        "split_disjoint": disjoint,
        "geology_group_split_disjoint": group_disjoint,
        "patch_rows": int(len(index)),
        "patch_rows_by_split": index.groupby("split").size().astype(int).to_dict(),
        "realizations_by_split": {name: len(values) for name, values in sets.items()},
        "all_realizations_represented": len(all_ids) == expected_count,
        "patch_bounds_valid": bounds_valid,
        "patch_invalid_fraction_valid": invalid_fraction_valid,
        "prior_finite_and_shape_matched": True,
        "normalization_finite": finite_stats,
        "normalization_matches_training_realizations": normalization_matches_training,
        "validation_test_sampling_reproducible": deterministic_validation_test,
        "duplicate_patch_count": duplicate_count,
        "duplicate_coordinate_count": coordinate_duplicate_count,
    }


def build_stage03_dataset(
    *,
    config: dict[str, Any],
    paths: dict[str, Any],
    source_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build one immutable, realization-split dataset from real Stage-02 packages."""
    inputs = config["inputs"]
    data_root = Path(paths["work_data_root"]) / inputs["dataset_id"]
    source = Path(source_directory) if source_directory else data_root / inputs["realization_directory"]
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Stage-02 manifest not found: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_count = int(inputs["expected_realization_count"])
    if source_manifest.get("status") != "complete":
        raise RuntimeError(
            "Production Stage-03 construction requires a complete Stage-02 manifest; "
            f"found {source_manifest.get('status')!r}"
        )
    if source_manifest.get("output_version") not in (None, inputs["synthetic_version"]):
        raise ValueError("Stage-02 output version does not match the Stage-03 input contract")
    realization_ids = [int(value) for value in source_manifest["realization_ids"]]
    if len(realization_ids) != len(set(realization_ids)):
        raise ValueError("Stage-02 manifest contains duplicate realization IDs")
    if (
        int(source_manifest.get("requested_realizations", -1)) != expected_count
        or int(source_manifest.get("generated_realizations", -1)) != expected_count
        or len(realization_ids) != expected_count
    ):
        raise ValueError(
            "Stage-02 manifest does not satisfy the configured production realization count "
            f"of {expected_count}"
        )
    source_paths = [source / f"realization_{value:07d}.npz" for value in realization_ids]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-02 realization: {missing[0]}")
    records = [_validate_realization(path) for path in source_paths]
    manifest_records = {
        int(record["realization_id"]): record for record in source_manifest.get("records", [])
    }
    if set(manifest_records) != set(realization_ids):
        raise ValueError("Stage-02 manifest records do not match its realization IDs")
    for record in records:
        expected_hash = manifest_records[int(record["realization_id"])].get("archive_sha256")
        if expected_hash != record["archive_sha256"]:
            raise ValueError(f"Stage-02 archive hash mismatch: {record['file']}")
    if len(records) < 3:
        raise ValueError("At least three Stage-02 realizations are required for disjoint splits")
    destination = Path(output_directory) if output_directory else data_root / config["outputs"]["directory"]
    realization_directory = destination / "realizations"
    realization_directory.mkdir(parents=True, exist_ok=True)
    prior_config = config["prior"]
    prior = PriorDefinition(
        source=str(prior_config["source"]),
        truth_derived=bool(prior_config["truth_derived"]),
        cutoff_hz=float(prior_config["cutoff_hz"]),
        dt_seconds=float(prior_config["dt_seconds"]),
        sigma_constant=float(prior_config["sigma_constant"]),
        lateral_sigma_ratio=float(prior_config["lateral_sigma_ratio"]),
        boundary_mode=str(prior_config["boundary_mode"]),
    )
    for source_path, record in zip(source_paths, records):
        _copy_with_prior(source_path, realization_directory / record["file"], prior)
    seed = int(config["stage"]["seed"])
    split_ids, split_group_ids = split_records_by_geology(
        records,
        tuple(float(value) for value in config["split"]["fractions"]),
        seed,
    )
    if any(not values for values in split_ids.values()):
        raise ValueError("The configured realization count leaves an empty split")
    record_by_id = {int(record["realization_id"]): record for record in records}
    write_json(destination / "split_ids.json", split_ids)
    write_json(destination / "split_group_ids.json", split_group_ids)
    normalization = _streaming_training_statistics(
        [realization_directory / record_by_id[value]["file"] for value in split_ids["train"]]
    )
    write_json(destination / "normalization.json", normalization)
    patch_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        patch_rows.extend(
            _patch_rows(
                split_name=split_name,
                records=[record_by_id[value] for value in split_ids[split_name]],
                config=config,
                dataset_realizations=realization_directory,
                seed=seed,
            )
        )
    patch_index = pd.DataFrame(patch_rows)
    patch_index.to_csv(destination / "patch_index.csv", index=False)
    dataset_archive_hashes = {
        record["file"]: file_sha256(realization_directory / record["file"]) for record in records
    }
    manifest = {
        "schema_version": 3 if "candidate_sampler" in config["patches"] else 2,
        "stage": "03_ml_dataset_construction",
        "status": "complete",
        "seed": seed,
        "configuration_sha256": _configuration_sha256(config),
        "source_stage02_status": source_manifest["status"],
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_generation_config_sha256": source_manifest["generation_config_sha256"],
        "source_snapshot": config.get("source_snapshot"),
        "source_archive_hashes": {
            record["file"]: record["archive_sha256"] for record in records
        },
        "dataset_archive_hashes": dataset_archive_hashes,
        "realization_count": len(records),
        "split_unit": "geology_realization",
        "split_ids": split_ids,
        "split_group_ids": split_group_ids,
        "input_channels": ["avo_near", "avo_mid", "avo_far"],
        "prior_channels": ["vp_low", "vs_low", "density_low"],
        "target_channels": ["vp", "vs", "density"],
        "structural_channels": ["rgt"],
        "categorical_target": "segmentation: 0 shale/non-reservoir, 1 sand, 2 CO2 plume",
        "delta_convention": "DELTA is shaliness; P(sand) = 1 - DELTA",
        "prior": prior.to_dict(),
        "prior_definition": config["prior"],
        "physics_context": config.get("physics_context", {}),
        "normalization_fit": "training realizations only",
        "normalization_provenance": {
            "fit_realization_ids": split_ids["train"],
            "fit_geology_realization_ids": split_group_ids["train"],
            "fit_source_archive_hashes": {
                record_by_id[value]["file"]: record_by_id[value]["archive_sha256"]
                for value in split_ids["train"]
            },
        },
        "patch_recipe": config["patches"],
        "patch_candidate_statistics": {
            "category_counts": (
                patch_index["candidate_category"].value_counts().astype(int).to_dict()
            ),
            "depth_bin_counts": patch_index["depth_bin"].value_counts().astype(int).to_dict(),
            "duplicate_patch_count": int(
                patch_index.duplicated(
                    [
                        "split",
                        "realization_id",
                        "top",
                        "left",
                        "raw_height",
                        "raw_width",
                    ]
                ).sum()
            ),
            "duplicate_coordinate_count": int(
                patch_index.duplicated(
                    ["split", "realization_id", "top", "left"]
                ).sum()
            ),
        },
        "artifact_hashes": {
            "split_ids.json": file_sha256(destination / "split_ids.json"),
            "split_group_ids.json": file_sha256(destination / "split_group_ids.json"),
            "normalization.json": file_sha256(destination / "normalization.json"),
            "patch_index.csv": file_sha256(destination / "patch_index.csv"),
        },
    }
    write_json(destination / "dataset_manifest.json", manifest)
    integrity = validate_dataset_integrity(destination)
    manifest["integrity"] = integrity
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest
