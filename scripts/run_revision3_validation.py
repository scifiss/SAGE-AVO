#!/usr/bin/env python3
"""Run the bounded eight-realization Revision-3 validation workflow.

This driver cannot launch the 100-realization or 120-epoch production
experiment. Its bounded results validate implementation and operators but do
not establish production eligibility or scientific performance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sage_avo.config import load_config
from sage_avo.experiments import (
    build_stage03_dataset,
    generate_stage02_dataset,
    validate_dataset_integrity,
)
from sage_avo.experiments.manifest import write_json
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.forward import (
    forward_avo_three_band_spec_torch,
    forward_specification_from_mapping,
)


REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATION_REALIZATIONS = 8
VALIDATION_VERSION = "v003_validation8_stage01v003_corrected"
DATASET_VERSION = "ds_v003_validation8_stage01v003_diverse"
TRAINING_VERSION = "sage_avo_s01_v003_stage01v003_validation8"


def _load_validation_configs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    paths_path = REPOSITORY / "configs" / "paths.yaml"
    if not paths_path.exists():
        raise FileNotFoundError("Create the ignored configs/paths.yaml before local validation")
    paths = load_config(paths_path)
    synthetic = deepcopy(load_config(REPOSITORY / "configs" / "synthetic_s01_v003.yaml"))
    dataset = deepcopy(load_config(REPOSITORY / "configs" / "ml_dataset_s01_v003.yaml"))
    training = deepcopy(load_config(REPOSITORY / "configs" / "sage_avo_s01_v003.yaml"))

    synthetic["stage"].update(
        {
            "geology_realization_count": VALIDATION_REALIZATIONS,
            "observation_variants_per_geology": 1,
            "realization_count": VALIDATION_REALIZATIONS,
            "realization_id_offset": 3_100_000,
        }
    )
    synthetic["outputs"].update(
        {
            "version": VALIDATION_VERSION,
            "directory": f"synthetic/{VALIDATION_VERSION}/realizations",
        }
    )
    dataset["inputs"].update(
        {
            "synthetic_version": VALIDATION_VERSION,
            "realization_directory": f"synthetic/{VALIDATION_VERSION}/realizations",
            "expected_realization_count": VALIDATION_REALIZATIONS,
        }
    )
    dataset["split"]["fractions"] = [0.625, 0.25, 0.125]
    dataset["outputs"].update(
        {
            "version": DATASET_VERSION,
            "directory": f"datasets/{DATASET_VERSION}",
        }
    )
    training["experiment"].update(
        {
            "name": TRAINING_VERSION,
            "output_root": f"results/experiments/{TRAINING_VERSION}",
        }
    )
    training["dataset"]["directory"] = f"datasets/{DATASET_VERSION}"
    training["training"]["epochs"] = 2
    training["training"]["checkpointing"].update(
        {
            "periodic_interval_epochs": 1,
            "whole_validation_every_epochs": 1,
            "whole_validation_realization_count": 2,
        }
    )
    return paths, synthetic, dataset, training


def _locations(paths: dict[str, Any]) -> dict[str, Path]:
    private = Path(paths["private_artifact_root"])
    root = private / "revision3" / "v003_validation8_stage01v003"
    return {
        "root": root,
        "stage02": root / "stage02" / "realizations",
        "stage03": root / "stage03" / "dataset",
        "stage04": root / "stage04" / TRAINING_VERSION,
        "figures02": root / "figures" / "stage02",
        "figures03": root / "figures" / "stage03",
        "reports": root / "reports",
        "configs": root / "configs",
    }


def _snapshot_configs(
    locations: dict[str, Path],
    synthetic: dict[str, Any],
    dataset: dict[str, Any],
    training: dict[str, Any],
) -> None:
    locations["configs"].mkdir(parents=True, exist_ok=True)
    write_json(locations["configs"] / "synthetic_resolved.json", synthetic)
    write_json(locations["configs"] / "dataset_resolved.json", dataset)
    write_json(locations["configs"] / "training_resolved.json", training)


def generate_stage02(args: argparse.Namespace) -> None:
    paths, synthetic, dataset, training = _load_validation_configs()
    locations = _locations(paths)
    _snapshot_configs(locations, synthetic, dataset, training)
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=locations["stage02"],
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "generated_realizations": manifest["generated_realizations"],
                "output_version": manifest["output_version"],
                "generation_config_sha256": manifest["generation_config_sha256"],
            },
            indent=2,
        )
    )


def build_stage03(_: argparse.Namespace) -> None:
    paths, synthetic, dataset, training = _load_validation_configs()
    locations = _locations(paths)
    _snapshot_configs(locations, synthetic, dataset, training)
    manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    print(json.dumps(manifest["integrity"], indent=2))


def _round_trip_report(
    stage02: Path,
    synthetic: dict[str, Any],
) -> dict[str, Any]:
    specification = forward_specification_from_mapping(synthetic)
    path = sorted(stage02.glob("realization_*.npz"))[0]
    with np.load(path, allow_pickle=False) as archive:
        elastic = np.asarray(archive["elastic"], dtype=np.float64)
        stored = np.asarray(archive["avo_clean"], dtype=np.float64)
        reproduced = forward_avo_three_band_spec_torch(
            torch.from_numpy(elastic[0][None]),
            torch.from_numpy(elastic[1][None]),
            torch.from_numpy(elastic[2][None]),
            specification,
        )[0].cpu().numpy()
    difference = reproduced - stored
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(stored))))
    return {
        "realization_file": path.name,
        "forward_specification_sha256": specification.sha256,
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1e-15),
        "comparison": "Torch shared operator versus stored clean Stage-02 bands",
    }


def _fluid_qc(stage02: Path, destination: Path) -> dict[str, Any]:
    path = sorted(stage02.glob("realization_*.npz"))[0]
    with np.load(path, allow_pickle=False) as archive:
        brine = np.asarray(archive["elastic_brine"])
        corrected = np.asarray(archive["elastic"])
        saturation = np.asarray(archive["co2_saturation"])
        time_ms = np.asarray(archive["time_ms"])
        cdp = np.asarray(archive["cdp"])
    difference = corrected - brine
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    panels = (
        (brine[0], "RF-conditioned brine Vp", "m/s", None),
        (corrected[0], "Local inverse-Gassmann CO2 Vp", "m/s", None),
        (saturation, "CO2 saturation", "fraction", (0.0, 1.0)),
        (difference[0], "Delta Vp", "m/s", None),
        (difference[1], "Delta Vs", "m/s", None),
        (difference[2], "Delta density", "g/cc", None),
    )
    extent = [float(cdp[0]), float(cdp[-1]), float(time_ms[-1]), float(time_ms[0])]
    for axis, (values, title, unit, limits) in zip(axes.flat, panels):
        symmetric = title.startswith("Delta")
        bound = float(np.max(np.abs(values))) if symmetric else None
        image = axis.imshow(
            values,
            aspect="auto",
            extent=extent,
            cmap="RdBu_r" if symmetric else "viridis",
            vmin=(-bound if symmetric and bound > 0 else (limits[0] if limits else None)),
            vmax=(bound if symmetric and bound > 0 else (limits[1] if limits else None)),
        )
        axis.set_title(title)
        axis.set_xlabel("CDP")
        axis.set_ylabel("TWT (ms)")
        figure.colorbar(image, ax=axis, label=unit, shrink=0.82)
    output = destination / "v003_fluid_substitution_qc.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    plume = saturation > 0.0
    return {
        "source": path.name,
        "figure": output.name,
        "plume_pixels": int(np.count_nonzero(plume)),
        "delta_vp_min_max_m_s": [
            float(np.min(difference[0][plume])) if plume.any() else 0.0,
            float(np.max(difference[0][plume])) if plume.any() else 0.0,
        ],
        "delta_vs_min_max_m_s": [
            float(np.min(difference[1][plume])) if plume.any() else 0.0,
            float(np.max(difference[1][plume])) if plume.any() else 0.0,
        ],
        "delta_density_min_max_g_cc": [
            float(np.min(difference[2][plume])) if plume.any() else 0.0,
            float(np.max(difference[2][plume])) if plume.any() else 0.0,
        ],
        "outside_plume_maximum_absolute_change": float(
            np.max(np.abs(difference[:, ~plume])) if (~plume).any() else 0.0
        ),
    }


def _patch_qc(stage03: Path, destination: Path) -> dict[str, Any]:
    integrity = validate_dataset_integrity(stage03)
    index = pd.read_csv(stage03 / "patch_index.csv")
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    index["candidate_category"].value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set_title("Candidate categories")
    axes[0].set_ylabel("patches")
    index["depth_bin"].value_counts().sort_index().plot.bar(ax=axes[1])
    axes[1].set_title("Depth-bin coverage")
    axes[1].set_xlabel("depth bin")
    example_id = int(sorted(index["realization_id"].unique())[0])
    example = index[index["realization_id"] == example_id]
    axes[2].scatter(
        example["left"],
        example["top"],
        c=example["depth_bin"],
        s=18,
        cmap="viridis",
        alpha=0.75,
    )
    axes[2].invert_yaxis()
    axes[2].set_title(f"Coordinates: realization {example_id}")
    axes[2].set_xlabel("left trace")
    axes[2].set_ylabel("top sample")
    output = destination / "v003_diverse_patch_sampling_qc.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    statistics = {
        "figure": output.name,
        "patch_rows": int(len(index)),
        "category_counts": index["candidate_category"].value_counts().to_dict(),
        "depth_bin_counts": {
            str(key): int(value)
            for key, value in index["depth_bin"].value_counts().sort_index().items()
        },
        **integrity,
    }
    pd.DataFrame(
        [
            {
                "candidate_category": key,
                "count": value,
            }
            for key, value in statistics["category_counts"].items()
        ]
    ).to_csv(destination / "v003_patch_category_counts.csv", index=False)
    return statistics


def qc(_: argparse.Namespace) -> None:
    paths, synthetic, _, _ = _load_validation_configs()
    locations = _locations(paths)
    locations["reports"].mkdir(parents=True, exist_ok=True)
    round_trip = _round_trip_report(locations["stage02"], synthetic)
    fluid = _fluid_qc(locations["stage02"], locations["figures02"])
    patches = _patch_qc(locations["stage03"], locations["figures03"])
    report = {
        "validation_scope": "eight-realization v003 operator/science validation only",
        "round_trip": round_trip,
        "fluid_substitution": fluid,
        "patch_sampling": patches,
    }
    write_json(locations["reports"] / "v003_validation8_qc.json", report)
    print(json.dumps(report, indent=2))


def train(args: argparse.Namespace) -> None:
    paths, synthetic, dataset, training = _load_validation_configs()
    locations = _locations(paths)
    _snapshot_configs(locations, synthetic, dataset, training)
    if not torch.cuda.is_available():
        raise RuntimeError("The requested v003 sanity check requires CUDA")
    run = train_controlled_variant(
        repository=REPOSITORY,
        config_path=locations["configs"] / "training_resolved.json",
        config=training,
        dataset_directory=locations["stage03"],
        experiment_directory=locations["stage04"],
        variant="full",
        device_name="cuda",
        epochs_override=2,
        max_train_batches=int(args.max_train_batches),
        max_validation_batches=int(args.max_validation_batches),
        run_name="full_2epoch_cuda_sanity",
        allow_operator_validation_subset=True,
    )
    print(run)


def all_stages(args: argparse.Namespace) -> None:
    generate_stage02(args)
    build_stage03(args)
    qc(args)
    train(args)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    generation = subparsers.add_parser("stage02")
    generation.add_argument("--workers", type=int, default=1)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(function=generate_stage02)
    stage03 = subparsers.add_parser("stage03")
    stage03.set_defaults(function=build_stage03)
    qc_parser = subparsers.add_parser("qc")
    qc_parser.set_defaults(function=qc)
    training = subparsers.add_parser("train")
    training.add_argument("--max-train-batches", type=int, default=4)
    training.add_argument("--max-validation-batches", type=int, default=2)
    training.set_defaults(function=train)
    everything = subparsers.add_parser("all")
    everything.add_argument("--workers", type=int, default=1)
    everything.add_argument("--resume", action="store_true")
    everything.add_argument("--max-train-batches", type=int, default=4)
    everything.add_argument("--max-validation-batches", type=int, default=2)
    everything.set_defaults(function=all_stages)
    return result


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
