"""Stage-03 leakage-safe ML dataset construction from Stage-02 realizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    return {"realization_id": realization_id, "file": path.name, "shape": list(shape)}


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
        height, width = valid_mask.shape
        rng = np.random.default_rng(seed + 1009 * realization_id)
        for scale_index, (scale, count) in enumerate(zip(patch_config["scales"], scale_counts)):
            raw_height, raw_width = (int(value) for value in scale["raw_shape"])
            if raw_height > height or raw_width > width:
                raise ValueError(f"Patch scale {(raw_height, raw_width)} exceeds realization {record['file']}")
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
                    }
                )
                accepted += 1
            if accepted != count:
                raise RuntimeError(
                    f"Only {accepted}/{count} valid patches accepted for realization {realization_id}"
                )
    return rows


def validate_dataset_integrity(dataset_directory: str | Path) -> dict[str, Any]:
    """Check split disjointness, patch bounds, channel contracts, and normalization."""
    root = Path(dataset_directory)
    split_ids = json.loads((root / "split_ids.json").read_text(encoding="utf-8"))
    sets = {name: set(map(int, values)) for name, values in split_ids.items()}
    disjoint = not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    if not disjoint:
        raise ValueError("Realization leakage detected between dataset splits")
    index = pd.read_csv(root / "patch_index.csv")
    mismatched = [
        int(row.realization_id)
        for row in index.itertuples()
        if int(row.realization_id) not in sets[str(row.split)]
    ]
    if mismatched:
        raise ValueError(f"Patch rows assigned to the wrong realization split: {mismatched[:5]}")
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    finite_stats = all(np.isfinite(normalization[name]).all() for name in normalization)
    if not finite_stats or any(np.asarray(normalization[name]).shape != (3,) for name in normalization):
        raise ValueError("Normalization statistics are invalid")
    return {
        "split_disjoint": disjoint,
        "patch_rows": int(len(index)),
        "patch_rows_by_split": index.groupby("split").size().astype(int).to_dict(),
        "realizations_by_split": {name: len(values) for name, values in sets.items()},
        "normalization_finite": finite_stats,
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
    source_paths = [source / f"realization_{int(value):07d}.npz" for value in source_manifest["realization_ids"]]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-02 realization: {missing[0]}")
    records = [_validate_realization(path) for path in source_paths]
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
    split = split_realizations(
        len(records), tuple(float(value) for value in config["split"]["fractions"]), seed
    )
    id_order = [int(record["realization_id"]) for record in records]
    split_ids = {
        "train": [id_order[index] for index in split.train],
        "validation": [id_order[index] for index in split.validation],
        "test": [id_order[index] for index in split.test],
    }
    if any(not values for values in split_ids.values()):
        raise ValueError("The configured realization count leaves an empty split")
    record_by_id = {int(record["realization_id"]): record for record in records}
    write_json(destination / "split_ids.json", split_ids)
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
    pd.DataFrame(patch_rows).to_csv(destination / "patch_index.csv", index=False)
    integrity = validate_dataset_integrity(destination)
    manifest = {
        "schema_version": 1,
        "stage": "03_ml_dataset_construction",
        "status": "complete",
        "source_stage02_status": source_manifest["status"],
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "realization_count": len(records),
        "split_unit": "realization",
        "split_ids": split_ids,
        "input_channels": ["avo_near", "avo_mid", "avo_far"],
        "prior_channels": ["vp_low", "vs_low", "density_low"],
        "target_channels": ["vp", "vs", "density"],
        "structural_channels": ["rgt"],
        "categorical_target": "segmentation: 0 shale/non-reservoir, 1 sand, 2 CO2 plume",
        "delta_convention": "DELTA is shaliness; P(sand) = 1 - DELTA",
        "prior": prior.to_dict(),
        "normalization_fit": "training realizations only",
        "patch_recipe": config["patches"],
        "integrity": integrity,
    }
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest
