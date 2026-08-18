#!/usr/bin/env python3
"""Run and verify the versioned Revision-2 SAGE-AVO production workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.config import load_config
from sage_avo.data import IndexedRealizationPatches
from sage_avo.data.sampling import PatchSamplingConfig, build_patch_sampling_weights
from sage_avo.experiments import (
    build_stage03_dataset,
    generate_stage02_dataset,
    validate_dataset_integrity,
)
from sage_avo.experiments.manifest import file_sha256, write_json


REPOSITORY = Path(__file__).resolve().parents[1]
STAGE02_REQUIRED = (
    "avo_dense",
    "avo",
    "avo_clean",
    "elastic",
    "elastic_brine",
    "delta",
    "sand_probability",
    "porosity",
    "rgt",
    "dip_pwd",
    "structural_stack",
    "structure_oriented",
    "strat_fraction",
    "reservoir_mask",
    "segmentation",
    "plume_mask",
    "co2_saturation",
    "valid_mask",
    "angles_degrees",
    "time_ms",
    "cdp",
)


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths_path = REPOSITORY / "configs" / "paths.yaml"
    if not paths_path.exists():
        raise FileNotFoundError("Create the ignored configs/paths.yaml before production runs")
    return (
        load_config(paths_path),
        load_config(REPOSITORY / "configs" / "synthetic_s01.yaml"),
        load_config(REPOSITORY / "configs" / "ml_dataset_s01.yaml"),
        load_config(REPOSITORY / "configs" / "sage_avo_s01.yaml"),
    )


def _locations() -> dict[str, Path]:
    paths, synthetic, dataset, training = _configs()
    private = Path(paths["private_artifact_root"])
    return {
        "private": private,
        "stage02": private
        / "stage_artifacts"
        / "stage02"
        / str(synthetic["outputs"]["version"])
        / "realizations",
        "stage03": private
        / "stage_artifacts"
        / "stage03"
        / str(dataset["outputs"]["version"])
        / "dataset",
        "stage04": private
        / "stage_artifacts"
        / "stage04"
        / "sage_avo_s01_v002_production",
        "figures02": private / "figures" / "revision2" / "stage02",
        "figures03": private / "figures" / "revision2" / "stage03",
        "training_config": REPOSITORY / "configs" / "sage_avo_s01.yaml",
        "training_name": Path(str(training["experiment"]["name"])),
    }


def generate_stage02(args: argparse.Namespace) -> None:
    paths, synthetic, _, _ = _configs()
    output = Path(args.output) if args.output else _locations()["stage02"]
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=output,
        realization_limit=args.limit,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps({key: manifest[key] for key in (
        "status", "requested_realizations", "generated_realizations",
        "generation_config_sha256", "production_bands",
    )}, indent=2))


def _binary(array: np.ndarray) -> bool:
    return set(np.unique(array).astype(int).tolist()).issubset({0, 1})


def verify_stage02(args: argparse.Namespace) -> None:
    _, synthetic, _, _ = _configs()
    root = Path(args.input) if args.input else _locations()["stage02"]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(synthetic["stage"]["realization_count"])
    expected_ids = list(
        range(
            int(synthetic["stage"]["realization_id_offset"]),
            int(synthetic["stage"]["realization_id_offset"]) + expected,
        )
    )
    archive_paths = sorted(root.glob("realization_*.npz"))
    metadata_paths = sorted(root.glob("realization_*.json"))
    if manifest["status"] != "complete":
        raise ValueError(f"Stage-02 status is {manifest['status']!r}, not 'complete'")
    if manifest["realization_ids"] != expected_ids:
        raise ValueError("Stage-02 realization IDs do not match the deterministic production sequence")
    if len(archive_paths) != expected or len(metadata_paths) != expected:
        raise ValueError("Stage-02 does not contain exactly 100 archive/metadata pairs")
    records = {int(item["realization_id"]): item for item in manifest["records"]}
    if set(records) != set(expected_ids):
        raise ValueError("Stage-02 manifest records and realization IDs disagree")

    channel_contract: dict[str, tuple[tuple[int, ...], str]] | None = None
    rows: list[dict[str, Any]] = []
    archive_hashes: list[str] = []
    geological_signatures: set[str] = set()
    for realization_id, path in zip(expected_ids, archive_paths):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["realization_id"]) != realization_id or int(metadata["seed"]) != realization_id:
            raise ValueError(f"ID/seed mismatch in {path.name}")
        digest = file_sha256(path)
        if digest != metadata["archive_sha256"] or digest != records[realization_id]["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch in {path.name}")
        archive_hashes.append(digest)
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["realization_id"]) != realization_id:
                raise ValueError(f"Archive ID mismatch in {path.name}")
            missing = [name for name in STAGE02_REQUIRED if name not in archive.files]
            if missing:
                raise ValueError(f"{path.name} is missing {missing}")
            current_contract = {
                name: (tuple(archive[name].shape), str(archive[name].dtype))
                for name in STAGE02_REQUIRED
            }
            if channel_contract is None:
                channel_contract = current_contract
            elif current_contract != channel_contract:
                raise ValueError(f"Shape/dtype contract differs in {path.name}")
            if not all(np.isfinite(archive[name]).all() for name in STAGE02_REQUIRED):
                raise ValueError(f"Non-finite value in {path.name}")
            elastic = archive["elastic"]
            vp, vs, density = elastic
            if not (
                1000.0 <= float(vp.min()) <= float(vp.max()) <= 7000.0
                and 500.0 <= float(vs.min()) <= float(vs.max()) <= 4500.0
                and 1.0 <= float(density.min()) <= float(density.max()) <= 4.0
                and np.all(vp > vs)
            ):
                raise ValueError(f"Elastic properties outside conservative physical bounds in {path.name}")
            delta = archive["delta"]
            sand = archive["sand_probability"]
            porosity = archive["porosity"]
            if not (
                delta.min() >= 0.0
                and delta.max() <= 1.0
                and sand.min() >= 0.0
                and sand.max() <= 1.0
                and np.max(np.abs(delta + sand - 1.0)) < 2e-6
                and porosity.min() >= 0.0
                and porosity.max() <= 0.36
            ):
                raise ValueError(f"Geological attribute bounds/convention failed in {path.name}")
            if not all(_binary(archive[name]) for name in ("reservoir_mask", "plume_mask", "valid_mask")):
                raise ValueError(f"Non-binary mask in {path.name}")
            labels = set(np.unique(archive["segmentation"]).astype(int).tolist())
            if not labels.issubset({0, 1, 2}):
                raise ValueError(f"Invalid segmentation label in {path.name}")
            plume = archive["plume_mask"].astype(bool)
            reservoir = archive["reservoir_mask"].astype(bool)
            if not np.array_equal(plume, archive["segmentation"] == 2) or np.any(plume & ~reservoir):
                raise ValueError(f"Plume/segmentation/reservoir support mismatch in {path.name}")
            rgt_steps = np.diff(archive["rgt"], axis=0)
            if float(rgt_steps.min()) < -1e-6:
                raise ValueError(f"Non-monotonic RGT in {path.name}")
            if archive["avo_dense"].shape[0] != 43 or archive["avo"].shape[0] != 3:
                raise ValueError(f"Unexpected dense/band AVO dimensions in {path.name}")
            noise_ratios = np.std(archive["avo"] - archive["avo_clean"], axis=(1, 2)) / np.maximum(
                np.std(archive["avo_clean"], axis=(1, 2)), 1e-12
            )
            if not np.allclose(
                noise_ratios,
                float(synthetic["forward"]["noise"]["standard_deviation_fraction"]),
                atol=0.0015,
                rtol=0.0,
            ):
                raise ValueError(f"Configured observation-noise ratio not reproduced in {path.name}")
            deformation = metadata["geology"]["deformation"]
            fluid = metadata["geology"]["fluid"]
            signature = json.dumps(
                {"deformation": deformation, "fluid": fluid}, sort_keys=True, separators=(",", ":")
            )
            geological_signatures.add(signature)
            rows.append(
                {
                    "realization_id": realization_id,
                    "archive_sha256": digest,
                    "fault_count": int(deformation["fault_count"]),
                    "folds_applied": int(deformation["folds_applied"]),
                    "requested_plumes": int(fluid["requested_plumes"]),
                    "plume_pixels": int(plume.sum()),
                    "vp_mean": float(vp.mean()),
                    "vs_mean": float(vs.mean()),
                    "density_mean": float(density.mean()),
                    "sand_mean": float(sand.mean()),
                    "porosity_mean": float(porosity.mean()),
                    "noise_ratio_near": float(noise_ratios[0]),
                    "noise_ratio_mid": float(noise_ratios[1]),
                    "noise_ratio_far": float(noise_ratios[2]),
                    "rgt_minimum_step": float(rgt_steps.min()),
                }
            )
    table = pd.DataFrame(rows)
    if len(set(archive_hashes)) != expected or len(geological_signatures) != expected:
        raise ValueError("Accidental duplicate archive or geological metadata detected")
    if table["fault_count"].nunique() < 4 or table["folds_applied"].nunique() < 2:
        raise ValueError("Configured structural diversity is not expressed by the corpus")
    if table["plume_pixels"].nunique() < 4 or (table["plume_pixels"] > 0).sum() < expected // 2:
        raise ValueError("Configured plume diversity is not expressed by the corpus")
    if min(table[name].std() for name in ("vp_mean", "vs_mean", "density_mean", "sand_mean")) <= 0:
        raise ValueError("The corpus lacks numerical geological/property diversity")
    qc = {
        "status": "pass",
        "exactly_100_complete": True,
        "realization_count": expected,
        "realization_ids": expected_ids,
        "configuration_sha256": manifest["generation_config_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "channel_contract": {
            name: {"shape": list(contract[0]), "dtype": contract[1]}
            for name, contract in (channel_contract or {}).items()
        },
        "checks": {
            "archive_and_metadata_hashes": "pass",
            "finite_values": "pass",
            "elastic_physical_bounds_and_vp_gt_vs": "pass",
            "facies_masks_and_delta_convention": "pass",
            "rgt_vertical_monotonicity": "pass",
            "dense_and_three_band_avo": "pass",
            "three_percent_bandwise_noise": "pass",
            "deterministic_id_seed_mapping": "pass",
            "no_duplicate_archives_or_geological_metadata": "pass",
            "structural_property_and_plume_diversity": "pass",
        },
        "summary": table.drop(columns="archive_sha256").describe(include="all").to_dict(),
    }
    table.to_csv(root / "stage02_qc_per_realization.csv", index=False)
    write_json(root / "stage02_qc_summary.json", qc)
    print(json.dumps({"status": "pass", "realizations": expected, "qc": str(root / 'stage02_qc_summary.json')}, indent=2))


def build_stage03(_: argparse.Namespace) -> None:
    paths, _, dataset, _ = _configs()
    locations = _locations()
    manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    print(json.dumps({
        "status": manifest["status"],
        "realizations": manifest["realization_count"],
        "split_counts": {name: len(values) for name, values in manifest["split_ids"].items()},
        "patch_rows": manifest["integrity"]["patch_rows"],
    }, indent=2))


def _sampling_config(training: dict[str, Any]) -> PatchSamplingConfig:
    values = training["training"]["weighted_patch_sampling"]
    angles = tuple(float(value) for value in training["model"]["representative_angles_degrees"])
    return PatchSamplingConfig(
        foreground_boost=float(values["foreground_boost"]),
        structure_boost=float(values["structure_boost"]),
        avo_gradient_boost=float(values["avo_gradient_boost"]),
        foreground_fraction_threshold=float(values["foreground_fraction_threshold"]),
        upper_quantile=float(values["upper_quantile"]),
        representative_angles_degrees=angles,
    )


def verify_stage03(_: argparse.Namespace) -> None:
    _, _, dataset_config, training = _configs()
    root = _locations()["stage03"]
    integrity = validate_dataset_integrity(root)
    split_ids = json.loads((root / "split_ids.json").read_text(encoding="utf-8"))
    expected_counts = {"train": 70, "validation": 20, "test": 10}
    if {name: len(values) for name, values in split_ids.items()} != expected_counts:
        raise ValueError("Production split is not exactly 70/20/10")
    datasets = {name: IndexedRealizationPatches(root, name) for name in expected_counts}
    patch_counts = {
        name: len(values) for name, values in datasets.items()
    }
    expected_patch_counts = {
        name: expected_counts[name] * int(dataset_config["patches"]["per_realization"][name])
        for name in expected_counts
    }
    if patch_counts != expected_patch_counts:
        raise ValueError(f"Patch counts {patch_counts} do not match {expected_patch_counts}")
    for split_name, values in datasets.items():
        for index in range(len(values)):
            item = values[index]
            for key in ("avo", "target", "low", "rgt", "mask"):
                if not bool(item[key].isfinite().all()):
                    raise ValueError(f"Non-finite {key} tensor in {split_name} patch {index}")
            if item["avo"].shape != item["target"].shape or item["low"].shape != item["target"].shape:
                raise ValueError(f"AVO/prior/target tensor mismatch in {split_name} patch {index}")
            if not set(item["segmentation"].unique().tolist()).issubset({0, 1, 2}):
                raise ValueError(f"Invalid segmentation tensor in {split_name} patch {index}")
    weights = build_patch_sampling_weights(datasets["train"], _sampling_config(training))
    if not bool(weights.isfinite().all()) or bool((weights <= 0).any()):
        raise ValueError("Training sampler weights are not finite and positive")
    normalized = weights / weights.sum()
    if not np.isclose(float(normalized.sum()), 1.0):
        raise ValueError("Normalized training sampler weights do not sum to one")
    np.save(root / "train_patch_sampling_weights.npy", weights.numpy())
    qc = {
        "status": "pass",
        "split_ids": split_ids,
        "split_counts": expected_counts,
        "patch_counts": patch_counts,
        "integrity": integrity,
        "normalization_fit": "training realizations only, recomputed and matched",
        "all_patch_tensors_finite_and_shape_matched": True,
        "sampler": {
            "count": len(weights),
            "finite_positive": True,
            "normalized_sum": float(normalized.sum()),
            "mean": float(weights.mean()),
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "sha256": file_sha256(root / "train_patch_sampling_weights.npy"),
        },
    }
    write_json(root / "stage03_qc_summary.json", qc)
    print(json.dumps({
        "status": "pass",
        "split_counts": expected_counts,
        "patch_counts": patch_counts,
        "sampler_mean": float(weights.mean()),
    }, indent=2))


def _imshow(axis: Any, array: np.ndarray, title: str, *, cmap: str = "viridis") -> None:
    image = axis.imshow(array, aspect="auto", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("trace")
    axis.set_ylabel("sample")
    plt.colorbar(image, ax=axis, shrink=0.72)


def make_figures(_: argparse.Namespace) -> None:
    locations = _locations()
    stage02 = locations["stage02"]
    stage03 = locations["stage03"]
    figures02 = locations["figures02"]
    figures03 = locations["figures03"]
    figures02.mkdir(parents=True, exist_ok=True)
    figures03.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((stage02 / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted(int(value) for value in manifest["realization_ids"])
    selected = [ids[index] for index in (0, 24, 49, 74, 99)]
    representative = selected[2]
    with np.load(stage02 / f"realization_{representative:07d}.npz") as archive:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for axis, name, title, cmap in zip(
            axes,
            ("structural_stack", "rgt", "dip_pwd"),
            ("Field-conditioned structural image", "Coherently warped RGT", "Two-pass PWD dip"),
            ("gray", "viridis", "seismic"),
        ):
            _imshow(axis, archive[name], title, cmap=cmap)
        fig.suptitle(f"Revision-2 structure/RGT QC — realization {representative}")
        fig.savefig(figures02 / "stage02_structure_rgt_pwd.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
        for axis, array, title, cmap in zip(
            axes,
            (*archive["elastic"], archive["segmentation"]),
            ("Vp (m/s)", "Vs (m/s)", "density (g/cc)", "facies: 0/1/2"),
            ("viridis", "viridis", "viridis", "tab10"),
        ):
            _imshow(axis, array, title, cmap=cmap)
        fig.suptitle(f"Revision-2 elastic/facies QC — realization {representative}")
        fig.savefig(figures02 / "stage02_elastic_facies.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for axis, array, title in zip(axes, archive["avo"], ("near 3–17°", "mid 17–31°", "far 31–45°")):
            limit = float(np.percentile(np.abs(array), 99))
            image = axis.imshow(array, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.set_xlabel("trace")
            axis.set_ylabel("sample")
            plt.colorbar(image, ax=axis, shrink=0.72)
        fig.suptitle(f"Exact-Zoeppritz three-band AVO with 3% noise — realization {representative}")
        fig.savefig(figures02 / "stage02_three_band_avo.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, len(selected), figsize=(20, 8), constrained_layout=True)
    for column, realization_id in enumerate(selected):
        with np.load(stage02 / f"realization_{realization_id:07d}.npz") as archive:
            _imshow(axes[0, column], archive["elastic"][0], f"Vp — {realization_id}")
            _imshow(axes[1, column], archive["segmentation"], f"facies — {realization_id}", cmap="tab10")
    fig.suptitle("Systematic corpus diversity: fixed realization-ID quantiles")
    fig.savefig(figures02 / "stage02_diverse_realizations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    split_ids = json.loads((stage03 / "split_ids.json").read_text(encoding="utf-8"))
    test_id = sorted(int(value) for value in split_ids["test"])[len(split_ids["test"]) // 2]
    with np.load(stage03 / "realizations" / f"realization_{test_id:07d}.npz") as archive:
        fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        for channel, label in enumerate(("Vp", "Vs", "density")):
            _imshow(axes[channel, 0], archive["elastic"][channel], f"truth {label}")
            _imshow(axes[channel, 1], archive["low"][channel], f"2-Hz truth-derived prior {label}")
            difference = archive["elastic"][channel] - archive["low"][channel]
            limit = float(np.percentile(np.abs(difference), 99))
            image = axes[channel, 2].imshow(
                difference, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit
            )
            axes[channel, 2].set_title(f"truth − prior {label}")
            axes[channel, 2].set_xlabel("trace")
            axes[channel, 2].set_ylabel("sample")
            plt.colorbar(image, ax=axes[channel, 2], shrink=0.72)
        fig.suptitle(f"Disclosed low-frequency-prior contract — test realization {test_id}")
        fig.savefig(figures03 / "stage03_low_prior_vs_truth.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    figure_index = pd.DataFrame(
        [
            ("stage02_structure_rgt_pwd.png", "02", "fixed median-ID member", "Structure/RGT/PWD registration"),
            ("stage02_elastic_facies.png", "02", "fixed median-ID member", "Elastic/facies registration"),
            ("stage02_three_band_avo.png", "02", "fixed median-ID member", "Exact-physics three-band response"),
            ("stage02_diverse_realizations.png", "02", "fixed ID quantiles", "Corpus geological diversity"),
            ("stage03_low_prior_vs_truth.png", "03", "median sorted test ID", "Truth-derived 2-Hz prior disclosure"),
        ],
        columns=("filename", "source_notebook", "selection_rule", "scientific_message"),
    )
    figure_index["field_private_data_shown"] = False
    figure_index["private_or_field_derived"] = True
    figure_index["public_redistribution_needs_verification"] = True
    index_path = locations["private"] / "figures" / "revision2" / "figure_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    figure_index.to_csv(index_path, index=False)
    print(figure_index.to_string(index=False))


def train(args: argparse.Namespace) -> None:
    from sage_avo.experiments.training import train_controlled_variant

    _, _, _, config = _configs()
    locations = _locations()
    resume_from = locations["stage04"] / "runs" / "full" / "last.pt" if args.resume else None
    if resume_from is not None and not resume_from.exists():
        raise FileNotFoundError(f"Cannot resume because the last checkpoint is absent: {resume_from}")
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=locations["training_config"],
        config=config,
        dataset_directory=locations["stage03"],
        experiment_directory=locations["stage04"],
        variant="full",
        device_name="cuda",
        run_name="full",
        resume_from=resume_from,
    )
    print(f"Completed production training: {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate-stage02")
    generation.add_argument("--workers", type=int, default=4)
    generation.add_argument("--limit", type=int)
    generation.add_argument("--resume", action="store_true")
    generation.add_argument("--output", type=Path)
    generation.set_defaults(function=generate_stage02)
    qc02 = commands.add_parser("verify-stage02")
    qc02.add_argument("--input", type=Path)
    qc02.set_defaults(function=verify_stage02)
    build03 = commands.add_parser("build-stage03")
    build03.set_defaults(function=build_stage03)
    qc03 = commands.add_parser("verify-stage03")
    qc03.set_defaults(function=verify_stage03)
    figures = commands.add_parser("figures")
    figures.set_defaults(function=make_figures)
    training = commands.add_parser("train")
    training.add_argument("--resume", action="store_true")
    training.set_defaults(function=train)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
