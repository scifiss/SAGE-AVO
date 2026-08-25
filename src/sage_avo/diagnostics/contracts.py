"""Immutable data contracts and deterministic validation-sample selection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_avo.experiments.manifest import file_sha256, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_revision331_inputs(
    *,
    dataset_directory: str | Path,
    private_artifact_root: str | Path,
    observability_config: dict[str, Any],
) -> dict[str, Any]:
    """Refuse execution when any frozen Stage-02/03 artifact has changed."""
    dataset = Path(dataset_directory)
    private = Path(private_artifact_root)
    expected = observability_config["frozen_inputs"]
    freeze_record = (
        private
        / "dataset_freezes"
        / str(expected["stage03_version"])
        / str(expected["stage03_manifest_sha256"])
        / "stage03_freeze_record.json"
    )
    stage02_manifest = (
        private
        / "stage_artifacts"
        / "stage02"
        / str(expected["stage02_version"])
        / "realizations"
        / "manifest.json"
    )
    stage02_qc = (
        private
        / "revision331"
        / "support_aware_generation_gate"
        / "final_full_corpus_audit"
        / "reports"
        / "final_full_corpus_qc.json"
    )
    paths = {
        "stage03_manifest": dataset / "dataset_manifest.json",
        "patch_index": dataset / "patch_index.csv",
        "split_ids": dataset / "split_ids.json",
        "normalization": dataset / "normalization.json",
        "stage03_freeze_record": freeze_record,
        "stage02_manifest": stage02_manifest,
        "stage02_final_qc": stage02_qc,
    }
    hashes = {
        name: file_sha256(path) if path.exists() else "MISSING" for name, path in paths.items()
    }
    required = {
        "stage03_manifest": str(expected["stage03_manifest_sha256"]),
        "patch_index": str(expected["patch_index_sha256"]),
        "split_ids": str(expected["split_ids_sha256"]),
        "normalization": str(expected["normalization_sha256"]),
        "stage03_freeze_record": str(expected["stage03_freeze_record_sha256"]),
        "stage02_manifest": str(expected["stage02_manifest_sha256"]),
        "stage02_final_qc": str(expected["stage02_final_qc_sha256"]),
    }
    checks = {name: hashes[name] == value for name, value in required.items()}
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Frozen Revision-3.3.1 input verification failed: {failures}")
    record = _read_json(freeze_record)
    manifest = _read_json(dataset / "dataset_manifest.json")
    checks.update(
        {
            "freeze_status": record.get("status") == "STAGE03_GO",
            "source_snapshot": manifest.get("source_snapshot", {}).get("snapshot_id")
            == expected["source_snapshot_id"],
            "calibration": manifest.get("calibration_id") == expected["calibration_id"],
        }
    )
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Frozen Revision-3.3.1 provenance failed: {failures}")
    return {
        "status": "verified",
        "checks": checks,
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "source_snapshot_id": expected["source_snapshot_id"],
        "calibration_id": expected["calibration_id"],
    }


def _patch_signal(dataset: Path, row: pd.Series, category: str) -> float:
    path = dataset / "realizations" / str(row["realization_file"])
    top, left = int(row["top"]), int(row["left"])
    height, width = int(row["raw_height"]), int(row["raw_width"])
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["segmentation"][top : top + height, left : left + width]
    if category == "reservoir":
        return float(np.mean(labels == 2))
    if category == "background":
        return float(np.mean(labels == 0))
    return float(row["candidate_score"])


def _select_category_patch(dataset: Path, validation: pd.DataFrame, category: str) -> pd.Series:
    candidates = validation[
        (validation["candidate_category"] == category) & (validation["physics_eligible"] == 1)
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No native validation patch for category {category!r}")
    candidates["diagnostic_signal"] = [
        _patch_signal(dataset, row, category) for _, row in candidates.iterrows()
    ]
    return candidates.sort_values(
        ["diagnostic_signal", "realization_id", "top", "left"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _record(row: pd.Series, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "patch_index_row": int(row.name),
        "realization_id": int(row["realization_id"]),
        "geology_realization_id": int(row["geology_realization_id"]),
        "top": int(row["top"]),
        "left": int(row["left"]),
        "raw_scale": [int(row["raw_height"]), int(row["raw_width"])],
        "resized_scale": [int(row["output_height"]), int(row["output_width"])],
        "category": str(row["candidate_category"]),
        "physics_eligible": bool(row["physics_eligible"]),
        "absolute_t0_seconds": float(row["absolute_t0_seconds"]),
        "native_dt_seconds": float(row["native_dt_seconds"]),
        "convolution_halo_samples": int(row["convolution_halo_samples"]),
    }


def build_diagnostic_sample_manifest(
    *,
    dataset_directory: str | Path,
    observability_config: dict[str, Any],
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Select fixed validation-only examples without observing model outputs."""
    dataset = Path(dataset_directory)
    index = pd.read_csv(dataset / "patch_index.csv")
    validation = index[index["split"] == "validation"].copy()
    requested = observability_config["fixed_validation"]
    patches = [
        _record(_select_category_patch(dataset, validation, category), category)
        for category in requested["categories"]
    ]
    for raw_shape in requested["include_non_native_scales"]:
        height, width = (int(value) for value in raw_shape)
        candidates = validation[
            (validation["raw_height"] == height) & (validation["raw_width"] == width)
        ].sort_values(["realization_id", "top", "left"])
        if candidates.empty:
            raise RuntimeError(f"No validation patch for raw scale {height}x{width}")
        patches.append(_record(candidates.iloc[0], f"multiscale_{height}x{width}"))
    split_ids = _read_json(dataset / "split_ids.json")
    whole_count = int(requested["whole_realization_count"])
    whole_ids = sorted(map(int, split_ids["validation"]))[:whole_count]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_training": True,
        "selection_uses_model_outputs": False,
        "selection_split": "validation",
        "test_data_used": False,
        "selection_rule": requested["selection_rule"],
        "dataset_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
        "patch_index_sha256": file_sha256(dataset / "patch_index.csv"),
        "normalization_sha256": file_sha256(dataset / "normalization.json"),
        "patches": patches,
        "whole_realization_ids": whole_ids,
        "native_physics_patch_count": sum(int(record["physics_eligible"]) for record in patches),
    }
    required_native = int(requested["require_native_physics_examples"])
    if manifest["native_physics_patch_count"] < required_native:
        raise RuntimeError("Fixed diagnostic selection lacks native physics patches")
    if destination is not None:
        path = Path(destination)
        if path.exists():
            raise FileExistsError(f"Refusing to replace frozen diagnostic samples: {path}")
        write_json(path, manifest)
    return manifest
