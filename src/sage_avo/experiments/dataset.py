"""Versioned public synthetic dataset for the controlled ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_avo.data.prior import PriorDefinition, make_truth_derived_prior
from sage_avo.data.splits import split_realizations
from sage_avo.forward.pipeline import ForwardConfig, forward_avo_three_band
from sage_avo.geology.synthetic import make_synthetic_geology

from .manifest import write_json


PROPERTY_NAMES = ("vp", "vs", "density")


def _base_geology(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    rows, columns = np.indices(shape, dtype=float)
    rgt = rows - 5.0 * np.sin(2.0 * np.pi * columns / max(width - 1, 1))
    layered = 0.5 + 0.32 * np.sin(rgt / 7.0) + 0.14 * np.sin(rgt / 2.7)
    sand = np.clip(layered, 0.0, 1.0)
    porosity = np.clip(0.07 + 0.19 * sand - 0.00025 * rows, 0.03, 0.34)
    return sand, porosity, rgt


def elastic_from_geology(
    delta: np.ndarray,
    porosity: np.ndarray,
    plume_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create explicit demonstration elastic truth and three-class labels.

    This deterministic public mapping is a benchmark fixture, not a calibrated
    field rock-physics model. Class 0 is shale-like, class 1 sand, and class 2
    the plume subset of sand.
    """
    vp = 2550.0 + 1050.0 * delta - 260.0 * porosity
    vs = 1320.0 + 720.0 * delta - 150.0 * porosity
    density = 2.02 + 0.36 * delta - 0.13 * porosity
    plume = plume_mask.astype(bool)
    vp = np.where(plume, 0.92 * vp, vp)
    vs = np.where(plume, 0.97 * vs, vs)
    density = np.where(plume, 0.985 * density, density)
    segmentation = (delta < 0.5).astype(np.uint8)
    segmentation[plume] = 2
    return np.stack((vp, vs, density)).astype(np.float32), segmentation


def _streaming_stats(paths: list[Path]) -> dict[str, list[float]]:
    sums = {"x": np.zeros(3), "y": np.zeros(3)}
    squares = {"x": np.zeros(3), "y": np.zeros(3)}
    counts = {"x": np.zeros(3), "y": np.zeros(3)}
    for path in paths:
        with np.load(path) as archive:
            for key, array_name in (("x", "avo"), ("y", "elastic")):
                array = np.asarray(archive[array_name], dtype=np.float64)
                for channel in range(3):
                    values = array[channel][np.isfinite(array[channel])]
                    sums[key][channel] += values.sum()
                    squares[key][channel] += np.square(values).sum()
                    counts[key][channel] += values.size
    output: dict[str, list[float]] = {}
    for key in ("x", "y"):
        mean = sums[key] / counts[key]
        variance = np.maximum(squares[key] / counts[key] - mean**2, 1e-16)
        output[f"{key}_mean"] = mean.tolist()
        output[f"{key}_std"] = np.sqrt(variance).tolist()
    return output


def _patch_rows(
    split_name: str,
    ids: list[int],
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    count_per_realization: int,
    seed: int,
) -> list[dict[str, int | str]]:
    height, width = shape
    patch_height, patch_width = patch_shape
    if patch_height > height or patch_width > width:
        raise ValueError("Patch shape exceeds realization shape")
    rows: list[dict[str, int | str]] = []
    for realization_id in ids:
        rng = np.random.default_rng(seed + 1009 * realization_id)
        for _ in range(count_per_realization):
            rows.append(
                {
                    "split": split_name,
                    "realization_id": realization_id,
                    "top": int(rng.integers(0, height - patch_height + 1)),
                    "left": int(rng.integers(0, width - patch_width + 1)),
                    "raw_height": patch_height,
                    "raw_width": patch_width,
                }
            )
    return rows


def prepare_controlled_dataset(
    config: dict[str, Any],
    output_directory: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generate one immutable dataset shared by every controlled condition."""
    destination = Path(output_directory)
    realization_directory = destination / "realizations"
    realization_directory.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["seed"])
    dataset_config = config["dataset"]
    patch_config = config["patches"]
    if smoke:
        count = 6
        shape = (32, 40)
        patch_shape = (16, 24)
        train_patches, validation_patches = 4, 2
    else:
        count = int(dataset_config["realization_count"])
        shape = tuple(int(value) for value in dataset_config["grid_shape"])
        patch_shape = tuple(int(value) for value in patch_config["shape"])
        train_patches = int(patch_config["train_per_realization"])
        validation_patches = int(patch_config["validation_per_realization"])

    fractions = tuple(float(value) for value in dataset_config["split_fractions"])
    split = split_realizations(count, fractions=fractions, seed=seed)
    split_ids = {
        "train": split.train.astype(int).tolist(),
        "validation": split.validation.astype(int).tolist(),
        "test": split.test.astype(int).tolist(),
    }
    prior_config = config["prior"]
    prior_definition = PriorDefinition(
        source=str(prior_config["source"]),
        truth_derived=bool(prior_config["truth_derived"]),
        cutoff_hz=float(prior_config["cutoff_hz"]),
        dt_seconds=float(dataset_config["dt_seconds"]),
        sigma_constant=float(prior_config["sigma_constant"]),
        lateral_sigma_ratio=float(prior_config["lateral_sigma_ratio"]),
        boundary_mode=str(prior_config["boundary_mode"]),
    )
    avo_config = config["avo"]
    angle_spec = avo_config["angles_degrees"]
    angles = tuple(
        float(value)
        for value in range(
            int(angle_spec["start"]), int(angle_spec["stop"]) + 1, int(angle_spec["step"])
        )
    )
    forward_config = ForwardConfig(
        angles_degrees=angles,
        wavelet_hz=float(avo_config["wavelet_hz"]),
        dt_seconds=float(dataset_config["dt_seconds"]),
        wavelet_samples=int(avo_config["wavelet_samples"]),
        apply_mute=bool(avo_config["front_mute"]),
    )

    sand_base, porosity_base, rgt_base = _base_geology(shape)
    noise_fraction = float(dataset_config["noise_std_fraction"])
    for realization_id in range(count):
        geology = make_synthetic_geology(
            sand_base,
            porosity_base,
            rgt_base,
            seed=seed + realization_id,
        )
        elastic, segmentation = elastic_from_geology(
            geology.delta, geology.porosity, geology.plume_mask
        )
        avo = forward_avo_three_band(*elastic, config=forward_config)
        rng = np.random.default_rng(seed + 100_000 + realization_id)
        channel_std = np.std(avo, axis=(1, 2), keepdims=True)
        avo = avo + rng.standard_normal(avo.shape) * noise_fraction * channel_std
        low = make_truth_derived_prior(elastic, prior_definition)
        np.savez_compressed(
            realization_directory / f"realization_{realization_id:04d}.npz",
            realization_id=np.int64(realization_id),
            avo=avo.astype(np.float32),
            elastic=elastic,
            low=low,
            rgt=geology.rgt.astype(np.float32),
            segmentation=segmentation,
            mask=np.ones(shape, dtype=np.uint8),
        )

    normalization = _streaming_stats(
        [realization_directory / f"realization_{item:04d}.npz" for item in split_ids["train"]]
    )
    write_json(destination / "normalization.json", normalization)
    write_json(destination / "split_ids.json", split_ids)
    patch_rows = _patch_rows(
        "train", split_ids["train"], shape, patch_shape, train_patches, seed
    ) + _patch_rows(
        "validation",
        split_ids["validation"],
        shape,
        patch_shape,
        validation_patches,
        seed,
    )
    pd.DataFrame(patch_rows).to_csv(destination / "patch_index.csv", index=False)
    manifest = {
        "schema_version": 1,
        "status": "smoke" if smoke else "controlled",
        "dataset_version": dataset_config["version"],
        "seed": seed,
        "realization_count": count,
        "grid_shape": list(shape),
        "split_ids": split_ids,
        "normalization_file": "normalization.json",
        "patch_index_file": "patch_index.csv",
        "prior": prior_definition.to_dict(),
        "forward": {
            "solver": "exact_pp_zoeppritz_numpy",
            "angles_degrees": list(angles),
            "bands_degrees": avo_config["bands_degrees"],
            "wavelet_hz": forward_config.wavelet_hz,
            "front_mute": forward_config.apply_mute,
        },
        "segmentation_classes": {"0": "shale_like", "1": "sand", "2": "plume"},
        "elastic_mapping": "explicit_public_benchmark_fixture_not_field_calibration",
    }
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
