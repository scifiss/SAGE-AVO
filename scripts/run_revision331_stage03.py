#!/usr/bin/env python3
"""Build, audit, visualize, and freeze Revision-3.3.1 Stage-03 data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sage_avo.config import load_config
from sage_avo.data.prior import PriorDefinition, make_truth_derived_prior
from sage_avo.experiments import build_stage03_dataset, validate_dataset_integrity
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.forward.pipeline import forward_avo_dense_spec
from sage_avo.forward.specification import forward_specification_from_mapping


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs" / "ml_dataset_s01_v00331.yaml"
FORWARD_CONFIG_PATH = REPOSITORY / "configs" / "forward_model_v003.yaml"
STAGE02_VERSION = "v00331_production100_support_aware"
STAGE03_VERSION = "ds_v00331_production100_support_aware"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _locations() -> dict[str, Path]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    return {
        "private": private,
        "stage02": private / "stage_artifacts" / "stage02" / STAGE02_VERSION / "realizations",
        "stage03": private / "stage_artifacts" / "stage03" / STAGE03_VERSION / "dataset",
        "final_stage02_qc": private
        / "revision331"
        / "support_aware_generation_gate"
        / "final_full_corpus_audit"
        / "reports"
        / "final_full_corpus_qc.json",
        "gate": private / "revision331" / "stage03_gate",
        "figures": private / "figures" / "revision331" / "stage03_production",
        "freezes": private / "dataset_freezes" / STAGE03_VERSION,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight(*, require_absent: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    locations = _locations()
    config = load_config(CONFIG_PATH)
    contract = config["source_contract"]
    stage02_manifest_path = locations["stage02"] / "manifest.json"
    final_qc_path = locations["final_stage02_qc"]
    if not stage02_manifest_path.exists() or not final_qc_path.exists():
        raise FileNotFoundError("Required Stage-02 manifest/final audit is absent")
    if require_absent and locations["stage03"].exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Stage-03 dataset: {locations['stage03']}"
        )
    stage02 = _read_json(stage02_manifest_path)
    final_qc = _read_json(final_qc_path)
    checks = {
        "stage02_decision": final_qc.get("decision") == contract["required_stage02_decision"],
        "stage02_complete": stage02.get("status") == "complete",
        "stage02_version": stage02.get("output_version") == STAGE02_VERSION,
        "stage02_count": int(stage02.get("generated_realizations", -1)) == 100,
        "stage02_manifest_sha": file_sha256(stage02_manifest_path)
        == contract["required_stage02_manifest_sha256"],
        "stage02_final_qc_sha": file_sha256(final_qc_path)
        == contract["required_stage02_final_qc_sha256"],
        "final_qc_embedded_manifest_sha": final_qc.get("corpus", {}).get("sha256")
        == contract["required_stage02_manifest_sha256"],
        "generation_config_sha": stage02.get("generation_config_sha256")
        == contract["required_generation_config_sha256"],
        "calibration_id": final_qc.get("calibration_id") == contract["calibration_id"],
        "source_snapshot_id": stage02.get("source_snapshot", {}).get("snapshot_id")
        == contract["required_source_snapshot_id"],
    }
    records = stage02.get("records", [])
    forward_hashes = {str(record.get("forward_specification_sha256")) for record in records}
    specification = forward_specification_from_mapping(load_config(FORWARD_CONFIG_PATH))
    checks["record_forward_hash"] = forward_hashes == {
        contract["required_forward_specification_sha256"]
    }
    checks["configured_forward_hash"] = (
        specification.sha256 == contract["required_forward_specification_sha256"]
    )
    ids = [int(value) for value in stage02.get("realization_ids", [])]
    checks["unique_100_ids"] = len(ids) == len(set(ids)) == 100
    checks["all_archives_present"] = all(
        (locations["stage02"] / f"realization_{value:07d}.npz").exists() for value in ids
    )
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-03 preflight failed: {failures}")
    return config, {
        "checks": checks,
        "stage02_manifest": stage02,
        "stage02_manifest_path": str(stage02_manifest_path),
        "stage02_manifest_sha256": file_sha256(stage02_manifest_path),
        "stage02_final_qc_path": str(final_qc_path),
        "stage02_final_qc_sha256": file_sha256(final_qc_path),
        "forward_specification": specification.to_dict(),
        "forward_specification_sha256": specification.sha256,
    }


def build(_: argparse.Namespace) -> None:
    config, provenance = _preflight(require_absent=True)
    locations = _locations()
    stage02 = provenance["stage02_manifest"]
    config["source_snapshot"] = stage02["source_snapshot"]
    config["calibration_id"] = config["source_contract"]["calibration_id"]
    config["resolved_source_contract"] = {
        key: value for key, value in provenance.items() if key != "stage02_manifest"
    }
    manifest = build_stage03_dataset(
        config=config,
        paths=load_config(REPOSITORY / "configs" / "paths.yaml"),
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    resolved_path = locations["stage03"] / "resolved_stage03_config.json"
    write_json(resolved_path, config)
    manifest.update(
        {
            "output_version": STAGE03_VERSION,
            "calibration_id": config["calibration_id"],
            "source_stage02_manifest_sha256": provenance["stage02_manifest_sha256"],
            "source_stage02_final_qc_sha256": provenance["stage02_final_qc_sha256"],
            "source_forward_specification": provenance["forward_specification"],
            "source_forward_specification_sha256": provenance["forward_specification_sha256"],
            "source_contract": config["source_contract"],
            "resolved_configuration": {
                "path": str(resolved_path),
                "sha256": file_sha256(resolved_path),
            },
        }
    )
    manifest["artifact_hashes"]["resolved_stage03_config.json"] = file_sha256(resolved_path)
    write_json(locations["stage03"] / "dataset_manifest.json", manifest)
    manifest["integrity"] = validate_dataset_integrity(locations["stage03"])
    write_json(locations["stage03"] / "dataset_manifest.json", manifest)
    print(json.dumps(manifest["integrity"], indent=2))


def _split_audit(root: Path, index: pd.DataFrame) -> dict[str, Any]:
    split_ids = _read_json(root / "split_ids.json")
    split_groups = _read_json(root / "split_group_ids.json")
    id_sets = {name: set(map(int, values)) for name, values in split_ids.items()}
    group_sets = {name: set(map(int, values)) for name, values in split_groups.items()}
    disjoint = not (
        id_sets["train"] & id_sets["validation"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["validation"] & id_sets["test"]
    )
    group_disjoint = not (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    )
    variant_split_counts = index.groupby("geology_realization_id")["split"].nunique()
    return {
        "realization_ids": split_ids,
        "geology_realization_ids": split_groups,
        "realization_counts": {name: len(values) for name, values in split_ids.items()},
        "exactly_100_groups": len(set().union(*group_sets.values())) == 100,
        "split_disjoint": disjoint,
        "group_split_disjoint": group_disjoint,
        "all_variants_grouped": bool((variant_split_counts == 1).all()),
        "zero_patch_split_mismatch": bool(
            all(int(row.realization_id) in id_sets[str(row.split)] for row in index.itertuples())
        ),
    }


def _patch_audit(index: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    exact_duplicates = int(index.duplicated().sum())
    coordinate_columns = [
        "split",
        "realization_id",
        "top",
        "left",
        "raw_height",
        "raw_width",
    ]
    coordinate_duplicates = int(index.duplicated(coordinate_columns).sum())
    top_left_duplicates = int(index.duplicated(["split", "realization_id", "top", "left"]).sum())
    minimum_distance = float("inf")
    for _, group in index.groupby("realization_id"):
        coordinates = group[["top", "left"]].to_numpy(dtype=float)
        if len(coordinates) > 1:
            minimum_distance = min(minimum_distance, float(np.min(pdist(coordinates))))
    required_categories = set(config["patches"]["candidate_sampler"]["categories"])
    category_counts = index["candidate_category"].value_counts().astype(int).to_dict()
    depth_counts = index["depth_bin"].value_counts().sort_index().astype(int).to_dict()
    scale_counts = index.groupby(["raw_height", "raw_width"]).size().astype(int).to_dict()
    per_realization = index.groupby("realization_id").size()
    return {
        "total_patches": int(len(index)),
        "patches_by_split": index.groupby("split").size().astype(int).to_dict(),
        "category_counts": category_counts,
        "depth_bin_counts": {str(key): value for key, value in depth_counts.items()},
        "scale_counts": {f"{h}x{w}": value for (h, w), value in scale_counts.items()},
        "patches_per_realization": {
            "minimum": int(per_realization.min()),
            "median": float(per_realization.median()),
            "maximum": int(per_realization.max()),
        },
        "exact_duplicate_patch_records": exact_duplicates,
        "duplicate_realization_top_left_scale": coordinate_duplicates,
        "duplicate_realization_top_left": top_left_duplicates,
        "minimum_top_left_separation_samples": minimum_distance,
        "required_minimum_separation_samples": float(
            config["patches"]["candidate_sampler"]["minimum_separation_samples"]
        ),
        "all_declared_categories_populated": required_categories <= set(category_counts),
        "all_depth_bins_populated": len(depth_counts)
        == int(config["patches"]["candidate_sampler"]["depth_bins"]),
        "all_scales_populated": len(scale_counts) == len(config["patches"]["scales"]),
        "uniform_fill_fraction": float((index["candidate_category"] == "uniform_fill").mean()),
    }


def _prior_normalization_audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    normalization = _read_json(root / "normalization.json")
    prior = PriorDefinition(**manifest["prior_definition"])
    maximum_prior_error = 0.0
    maximum_inverse_error = 0.0
    all_finite = True
    prior_rmse: list[np.ndarray] = []
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float64)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float64)[:, None, None]
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float64)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float64)[:, None, None]
    for path in sorted((root / "realizations").glob("realization_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            elastic = np.asarray(archive["elastic"], dtype=np.float64)
            low = np.asarray(archive["low"], dtype=np.float64)
            avo = np.asarray(archive["avo"], dtype=np.float64)
            expected = make_truth_derived_prior(archive["elastic"], prior)
            maximum_prior_error = max(maximum_prior_error, float(np.max(np.abs(low - expected))))
            normalized_y = (elastic - y_mean) / y_std
            normalized_low = (low - y_mean) / y_std
            normalized_x = (avo - x_mean) / x_std
            all_finite &= bool(
                np.isfinite(normalized_y).all()
                and np.isfinite(normalized_low).all()
                and np.isfinite(normalized_x).all()
            )
            restored = normalized_y * y_std + y_mean
            maximum_inverse_error = max(
                maximum_inverse_error, float(np.max(np.abs(restored - elastic)))
            )
            prior_rmse.append(np.sqrt(np.mean(np.square(low - elastic), axis=(1, 2))))
    return {
        "definition": prior.to_dict(),
        "normalization": normalization,
        "normalization_fit": manifest["normalization_fit"],
        "fit_realization_ids": manifest["normalization_provenance"]["fit_realization_ids"],
        "all_normalized_arrays_finite": all_finite,
        "maximum_inverse_normalization_error": maximum_inverse_error,
        "maximum_recomputed_prior_error": maximum_prior_error,
        "mean_prior_rmse_physical_units": np.mean(prior_rmse, axis=0).tolist(),
        "prior_is_nominal_2_hz_truth_derived": bool(prior.truth_derived and prior.cutoff_hz == 2.0),
    }


def _physics_audit(root: Path, index: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    specification = forward_specification_from_mapping(load_config(FORWARD_CONFIG_PATH))
    required_columns = {
        "raw_height",
        "raw_width",
        "output_height",
        "output_width",
        "time_scale",
        "trace_scale",
        "source_sample_top",
        "absolute_t0_seconds",
        "native_dt_seconds",
        "mute_origin_seconds",
        "convolution_halo_samples",
        "wavelet_ids",
        "physics_eligible",
    }
    rows: list[dict[str, Any]] = []
    for (raw_height, raw_width), scale in index.groupby(["raw_height", "raw_width"]):
        ordered = scale.sort_values("top")
        positions = sorted(set([0, len(ordered) // 2, len(ordered) - 1]))
        for position in positions:
            row = ordered.iloc[position]
            path = root / "realizations" / str(row["realization_file"])
            with np.load(path, allow_pickle=False) as archive:
                top, left = int(row["top"]), int(row["left"])
                height, width = int(raw_height), int(raw_width)
                halo = specification.maximum_wavelet_half_length
                context_top = max(0, top - halo)
                context_bottom = min(archive["elastic"].shape[1], top + height + halo)
                elastic = np.asarray(
                    archive["elastic"][:, context_top:context_bottom, left : left + width],
                    dtype=np.float64,
                )
                predicted = forward_avo_dense_spec(
                    elastic[0],
                    elastic[1],
                    elastic[2],
                    specification,
                    sample_origin=context_top,
                ).stacks
                core_start = top - context_top
                predicted = predicted[:, core_start : core_start + height]
                expected = np.asarray(
                    archive["avo_clean"][:, top : top + height, left : left + width],
                    dtype=np.float64,
                )
            difference = predicted - expected
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            reference_rms = float(np.sqrt(np.mean(np.square(expected))))
            rows.append(
                {
                    "raw_shape": f"{height}x{width}",
                    "realization_id": int(row["realization_id"]),
                    "top": top,
                    "left": left,
                    "maximum_absolute_error": float(np.max(np.abs(difference))),
                    "rmse": rmse,
                    "relative_rmse": rmse / max(reference_rms, 1e-15),
                }
            )
    round_trip = pd.DataFrame(rows)
    round_trip_path = _locations()["gate"] / "tables" / "physics_round_trip.csv"
    round_trip_path.parent.mkdir(parents=True, exist_ok=True)
    round_trip.to_csv(round_trip_path, index=False)
    forward_hashes: set[str] = set()
    absolute_time_valid = True
    for realization_id, group in index.groupby("realization_id"):
        path = root / "realizations" / str(group.iloc[0]["realization_file"])
        with np.load(path, allow_pickle=False) as archive:
            forward_hashes.add(str(archive["forward_specification_sha256"].item()))
            time_ms = archive["time_ms"]
            expected_t0 = time_ms[group["top"].to_numpy(dtype=int)] / 1000.0
            absolute_time_valid &= bool(
                np.allclose(group["absolute_t0_seconds"], expected_t0, atol=1e-12)
            )
    native = (index["raw_height"] == index["output_height"]) & (
        index["raw_width"] == index["output_width"]
    )
    return {
        "required_metadata_columns_present": required_columns <= set(index.columns),
        "absolute_time_origin_valid": absolute_time_valid,
        "native_dt_seconds": sorted(index["native_dt_seconds"].unique().tolist()),
        "mute_origin_seconds": sorted(index["mute_origin_seconds"].unique().tolist()),
        "convolution_halo_samples": sorted(
            index["convolution_halo_samples"].unique().astype(int).tolist()
        ),
        "wavelet_ids": sorted(index["wavelet_ids"].unique().tolist()),
        "angle_bands": [
            [band.name, band.minimum_degrees, band.maximum_degrees] for band in specification.bands
        ],
        "forward_specification_sha256": specification.sha256,
        "archive_forward_hashes": sorted(forward_hashes),
        "manifest_forward_hash": manifest["source_forward_specification_sha256"],
        "physics_eligible_matches_native_only": bool(
            np.array_equal(index["physics_eligible"].astype(bool), native)
        ),
        "physics_eligible_count": int(index["physics_eligible"].sum()),
        "non_native_physics_exclusion_reason": (
            "Multiscale resized patches retain supervised losses but are excluded "
            "from exact-PP physics loss; their native-grid metadata is retained."
        ),
        "round_trip_cases": rows,
        "round_trip_maximum_absolute_error": float(round_trip["maximum_absolute_error"].max()),
        "round_trip_maximum_rmse": float(round_trip["rmse"].max()),
        "round_trip_maximum_relative_rmse": float(round_trip["relative_rmse"].max()),
        "round_trip_table": str(round_trip_path),
    }


def _save_figure(fig: plt.Figure, stem: Path) -> list[dict[str, Any]]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        artifacts.append(
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        )
    plt.close(fig)
    return artifacts


def _figures(root: Path, index: pd.DataFrame) -> list[dict[str, Any]]:
    destination = _locations()["figures"]
    figure_records: list[dict[str, Any]] = []

    split_counts = (
        index.groupby("split")["realization_id"].nunique().reindex(["train", "validation", "test"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.2))
    axes[0].bar(split_counts.index, split_counts.values, color=["#2563eb", "#f59e0b", "#16a34a"])
    axes[0].set(title="Realization-level 70/20/10 split", ylabel="Realizations")
    axes[0].bar_label(axes[0].containers[0])
    for y, split in enumerate(("train", "validation", "test")):
        ids = sorted(index.loc[index["split"] == split, "realization_id"].unique())
        axes[1].scatter(ids, np.full(len(ids), y), s=22, label=split)
    axes[1].set(
        title="Every realization belongs to one split",
        xlabel="Realization ID",
        yticks=range(3),
        yticklabels=["train", "validation", "test"],
    )
    figure_records.extend(_save_figure(fig, destination / "01_realization_split"))

    counts = index["candidate_category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.barh(counts.index, counts.values, color="#0f766e")
    ax.set(title="Diverse patch-candidate composition", xlabel="Patch count")
    ax.bar_label(ax.containers[0], padding=3)
    figure_records.extend(_save_figure(fig, destination / "02_patch_categories"))

    table = pd.crosstab(
        index["depth_bin"],
        index.apply(lambda row: f"{row.raw_height}×{row.raw_width}", axis=1),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.2))
    image = axes[0].imshow(table.values, cmap="Blues", aspect="auto")
    axes[0].set(
        title="Depth bin × raw patch scale",
        xlabel="Raw scale (samples)",
        ylabel="Depth bin",
        xticks=range(len(table.columns)),
        xticklabels=table.columns,
        yticks=range(len(table.index)),
        yticklabels=table.index,
    )
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            axes[0].text(column, row, str(table.iloc[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axes[0], label="Patches")
    scale_counts = index.groupby(["raw_height", "raw_width"]).size()
    labels = [f"{height}×{width}" for height, width in scale_counts.index]
    axes[1].bar(labels, scale_counts.values, color="#7c3aed")
    axes[1].set(title="Approved multiscale mixture", xlabel="Raw scale", ylabel="Patches")
    axes[1].bar_label(axes[1].containers[0])
    figure_records.extend(_save_figure(fig, destination / "03_multiscale_depth_distribution"))

    validation_ids = sorted(index.loc[index["split"] == "validation", "realization_id"].unique())
    realization_id = int(validation_ids[len(validation_ids) // 2])
    path = root / "realizations" / f"realization_{realization_id:07d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        elastic = archive["elastic"]
        low = archive["low"]
        avo = archive["avo"]
        rgt = archive["rgt"]
        segmentation = archive["segmentation"]
    names = ("Vp (m/s)", "Vs (m/s)", "Density (g/cc)")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for channel, name in enumerate(names):
        limits = np.quantile(elastic[channel], [0.01, 0.99])
        for row, (values, label) in enumerate(((low, "2-Hz prior"), (elastic, "Truth"))):
            image = axes[row, channel].imshow(
                values[channel], cmap="viridis", aspect="auto", vmin=limits[0], vmax=limits[1]
            )
            axes[row, channel].set_title(f"{label}: {name}")
            fig.colorbar(image, ax=axes[row, channel], shrink=0.8)
    fig.suptitle(f"Disclosed truth-derived 2-Hz prior — validation realization {realization_id}")
    figure_records.extend(_save_figure(fig, destination / "04_low_prior_vs_truth"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fields = [avo[0], avo[1], avo[2], rgt, elastic[0], elastic[1], elastic[2], segmentation]
    titles = ["Near AVO", "Mid AVO", "Far AVO", "RGT", *names, "Facies / plume"]
    cmaps = ["seismic", "seismic", "seismic", "viridis", "viridis", "viridis", "viridis", "tab10"]
    for ax, values, title, cmap in zip(axes.flat, fields, titles, cmaps):
        image = ax.imshow(values, cmap=cmap, aspect="auto")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle(f"Stage-03 channels and targets — validation realization {realization_id}")
    figure_records.extend(_save_figure(fig, destination / "05_channels_and_targets"))

    requested = [
        ("background", "Simple / background"),
        ("high_dip", "High dip / faulted"),
        ("high_avo_gradient_change", "High AVO gradient"),
        ("reservoir", "Reservoir / plume"),
        ("facies_boundary", "Facies boundary"),
    ]
    selected = []
    validation = index[index["split"] == "validation"]
    for category, title in requested:
        matches = validation[validation["candidate_category"] == category]
        if matches.empty:
            raise RuntimeError(f"No validation patch found for category {category}")
        selected.append((matches.iloc[0], title))
    patches = []
    for row, title in selected:
        with np.load(
            root / "realizations" / row["realization_file"], allow_pickle=False
        ) as archive:
            patch = archive["avo"][
                0,
                int(row.top) : int(row.top + row.raw_height),
                int(row.left) : int(row.left + row.raw_width),
            ]
        patches.append((patch, title))
    maximum = max(float(np.quantile(np.abs(values), 0.99)) for values, _ in patches)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    for ax, (values, title) in zip(axes, patches):
        image = ax.imshow(values, cmap="seismic", aspect="auto", vmin=-maximum, vmax=maximum)
        ax.set_title(title)
        ax.set_xlabel("Trace")
    axes[0].set_ylabel("Native time sample")
    fig.colorbar(image, ax=axes, shrink=0.75, label="Near-stack amplitude")
    fig.suptitle("Deterministically selected validation patch diversity")
    figure_records.extend(_save_figure(fig, destination / "06_diverse_patch_examples"))

    index_path = destination / "figure_index.csv"
    pd.DataFrame(figure_records).to_csv(index_path, index=False)
    return figure_records


def audit(_: argparse.Namespace) -> None:
    config, provenance = _preflight(require_absent=False)
    locations = _locations()
    root = locations["stage03"]
    if not (root / "dataset_manifest.json").exists():
        raise FileNotFoundError("Stage-03 dataset has not been built")
    manifest = _read_json(root / "dataset_manifest.json")
    index = pd.read_csv(root / "patch_index.csv")
    integrity = validate_dataset_integrity(root)
    split = _split_audit(root, index)
    patches = _patch_audit(index, config)
    prior = _prior_normalization_audit(root, manifest)
    physics = _physics_audit(root, index, manifest)
    figures = _figures(root, index)
    gates = {
        "builder_integrity": all(
            bool(value) for value in integrity.values() if isinstance(value, bool)
        ),
        "exact_70_20_10_split": split["realization_counts"]
        == {"train": 70, "validation": 20, "test": 10},
        "exactly_100_groups": split["exactly_100_groups"],
        "no_realization_or_group_leakage": split["split_disjoint"]
        and split["group_split_disjoint"]
        and split["all_variants_grouped"]
        and split["zero_patch_split_mismatch"],
        "no_patch_duplicates": patches["exact_duplicate_patch_records"] == 0
        and patches["duplicate_realization_top_left_scale"] == 0
        and patches["duplicate_realization_top_left"] == 0,
        "spatial_separation": patches["minimum_top_left_separation_samples"]
        >= patches["required_minimum_separation_samples"],
        "candidate_strategy_populated": patches["all_declared_categories_populated"]
        and patches["all_depth_bins_populated"]
        and patches["all_scales_populated"],
        "training_only_normalization": integrity["normalization_matches_training_realizations"]
        and set(prior["fit_realization_ids"]) == set(split["realization_ids"]["train"]),
        "finite_normalized_arrays": prior["all_normalized_arrays_finite"],
        "inverse_normalization_round_trip": prior["maximum_inverse_normalization_error"] < 1e-9,
        "two_hz_prior_exact": prior["prior_is_nominal_2_hz_truth_derived"]
        and prior["maximum_recomputed_prior_error"] == 0.0,
        "physics_metadata_complete": physics["required_metadata_columns_present"]
        and physics["absolute_time_origin_valid"]
        and physics["physics_eligible_matches_native_only"],
        "physics_contract_matches_stage02": physics["archive_forward_hashes"]
        == [provenance["forward_specification_sha256"]]
        and physics["manifest_forward_hash"] == provenance["forward_specification_sha256"],
        "physics_round_trip": physics["round_trip_maximum_relative_rmse"] <= 1e-6,
        "figures_created": len(figures) == 12,
    }
    report = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "decision": "STAGE03_GO" if all(gates.values()) else "STAGE03_NO_GO",
        "dataset_path": str(root),
        "dataset_manifest_sha256": file_sha256(root / "dataset_manifest.json"),
        "source_preflight": {
            key: value for key, value in provenance.items() if key != "stage02_manifest"
        },
        "integrity": integrity,
        "split": split,
        "patches": patches,
        "prior_and_normalization": prior,
        "physics": physics,
        "figures": figures,
        "gates": gates,
    }
    report_path = locations["gate"] / "reports" / "stage03_qc.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(json.dumps({"decision": report["decision"], "gates": gates}, indent=2))
    if report["decision"] != "STAGE03_GO":
        raise RuntimeError("Stage-03 scientific/integrity audit returned NO-GO")


def freeze(args: argparse.Namespace) -> None:
    locations = _locations()
    root = locations["stage03"]
    report_path = locations["gate"] / "reports" / "stage03_qc.json"
    repository_gates_path = Path(args.repository_gates)
    report = _read_json(report_path)
    repository_gates = _read_json(repository_gates_path)
    if report.get("decision") != "STAGE03_GO":
        raise RuntimeError("Cannot freeze a Stage-03 dataset without STAGE03_GO")
    if repository_gates.get("decision") != "PASS":
        raise RuntimeError("Cannot freeze Stage-03 before all repository gates pass")
    key_files = [
        "dataset_manifest.json",
        "patch_index.csv",
        "split_ids.json",
        "split_group_ids.json",
        "normalization.json",
        "resolved_stage03_config.json",
    ]
    hashes = {name: file_sha256(root / name) for name in key_files}
    manifest = _read_json(root / "dataset_manifest.json")
    payload = {
        "schema_version": 1,
        "status": "STAGE03_GO",
        "created_utc": _utc_now(),
        "dataset_version": STAGE03_VERSION,
        "dataset_directory": str(root),
        "artifact_sha256": hashes,
        "stage02_manifest_sha256": manifest["source_stage02_manifest_sha256"],
        "stage02_final_qc_sha256": manifest["source_stage02_final_qc_sha256"],
        "stage02_generation_config_sha256": manifest["source_generation_config_sha256"],
        "forward_specification_sha256": manifest["source_forward_specification_sha256"],
        "calibration_id": manifest["calibration_id"],
        "source_snapshot": manifest["source_snapshot"],
        "dataset_configuration_sha256": manifest["configuration_sha256"],
        "stage03_qc": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "repository_gates": {
            "path": str(repository_gates_path),
            "sha256": file_sha256(repository_gates_path),
        },
    }
    freeze_id = file_sha256(root / "dataset_manifest.json")
    destination = locations["freezes"] / freeze_id
    destination.mkdir(parents=True, exist_ok=False)
    record_path = destination / "stage03_freeze_record.json"
    write_json(record_path, payload)
    pointer = {
        "schema_version": 1,
        "status": "STAGE03_GO",
        "dataset_version": STAGE03_VERSION,
        "dataset_directory": str(root),
        "freeze_id": freeze_id,
        "freeze_record": {
            "path": str(record_path),
            "sha256": file_sha256(record_path),
        },
    }
    pointer_path = locations["private"] / "dataset_freezes" / "revision331_stage03_approved.json"
    if pointer_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-03 approval pointer: {pointer_path}")
    write_json(pointer_path, pointer)
    print(json.dumps(pointer, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("build").set_defaults(function=build)
    commands.add_parser("audit").set_defaults(function=audit)
    freeze_command = commands.add_parser("freeze")
    freeze_command.add_argument("--repository-gates", required=True)
    freeze_command.set_defaults(function=freeze)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
