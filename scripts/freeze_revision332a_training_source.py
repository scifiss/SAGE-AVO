#!/usr/bin/env python3
"""Create the immutable code-only Revision-3.3.2a corrected-training snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any

from sage_avo.experiments.manifest import file_sha256, write_json


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
PREVIOUS_POINTER = PRIVATE / "source_freezes" / "revision332_training_instrumentation_approved.json"
PREVIOUS_POINTER_SHA256 = "35ee47b9586acc15b5394ed51fa48bbdc5c0f9a19c5a1984bd2e07a6f52faeda"
PREVIOUS_SNAPSHOT_ID = "0db8aa4908f998f37649f95f0fa7d27c8f3bbaadd4042ba0ed1aa588084a93a8"
STAGE02_MANIFEST = (
    PRIVATE
    / "stage_artifacts"
    / "stage02"
    / "v00331_production100_support_aware"
    / "realizations"
    / "manifest.json"
)
STAGE02_MANIFEST_SHA256 = "4943684922380b0d82c56d7b5595ad8a1be8f0770be3f41bb0c6dae6216aa46c"
STAGE03_MANIFEST = (
    PRIVATE
    / "stage_artifacts"
    / "stage03"
    / "ds_v00331_production100_support_aware"
    / "dataset"
    / "dataset_manifest.json"
)
STAGE03_MANIFEST_SHA256 = "1afe64debc9b0901afde88b327a1c04088c1a2a8e51efafadd38bc5e6ba845ee"
CORRECTION_ROOT = PRIVATE / "revision332a" / "masked_physics_correction"
CLEAN_PRODUCTION = (
    PRIVATE
    / "stage_artifacts"
    / "stage04"
    / "sage_avo_s01_v00332a_masked_physics_corrected_production"
)
ALLOWED_DIRECTORIES = {".github", "configs", "docs", "notebooks", "scripts", "src", "tests"}
EXCLUDED_PATHS = {Path("configs/paths.yaml")}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024
REVISION332A_CHANGED_FILES = (
    "scripts/compare_revision332a_epoch1.py",
    "scripts/freeze_revision332a_training_source.py",
    "scripts/run_revision332a_masked_physics_gate.py",
    "scripts/run_revision332a_production.py",
    "scripts/run_revision332a_repository_gates.py",
    "src/sage_avo/training/losses.py",
    "tests/test_masked_physics_loss.py",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _listed_files(*arguments: str) -> list[Path]:
    result = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return [Path(value.decode()) for value in result.stdout.split(b"\0") if value]


def _is_source(relative: Path) -> bool:
    if relative in EXCLUDED_PATHS or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return len(relative.parts) == 1 or relative.parts[0] in ALLOWED_DIRECTORIES


def _source_paths() -> list[Path]:
    paths = {
        path
        for path in (
            *_listed_files("ls-files"),
            *_listed_files("ls-files", "--others", "--exclude-standard"),
        )
        if _is_source(path) and (REPOSITORY / path).is_file()
    }
    selected = sorted(paths, key=lambda path: path.as_posix())
    oversized = [
        path for path in selected if (REPOSITORY / path).stat().st_size > MAX_SOURCE_FILE_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Oversized source file(s): {oversized}")
    if Path("configs/paths.yaml") in selected:
        raise RuntimeError("Private paths configuration entered source freeze")
    return selected


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_interrupted_run(provenance: dict[str, Any]) -> dict[str, Any]:
    if (
        provenance.get("epoch_1_completed") is not True
        or provenance.get("durable_optimizer_steps") != 1750
        or provenance.get("interrupted_during_epoch") != 2
        or provenance["epoch_1_logging_validity"] != {
            "reason": "masked residual squared before zero eligibility mask caused Inf * 0 -> NaN",
            "train_physics": False,
            "train_total": False,
        }
    ):
        raise RuntimeError("Interrupted-run provenance contract is invalid")
    run = Path(provenance["run_directory"])
    for row in provenance["files"]:
        path = run / row["relative_path"]
        if (
            not path.is_file()
            or file_sha256(path) != row["sha256"]
            or path.stat().st_mtime_ns != row["mtime_ns"]
        ):
            raise RuntimeError(f"Interrupted-run artifact changed: {path}")
    return {
        "path": str(CORRECTION_ROOT / "interrupted_revision332_run_provenance.json"),
        "sha256": file_sha256(
            CORRECTION_ROOT / "interrupted_revision332_run_provenance.json"
        ),
        "verified_file_count": len(provenance["files"]),
        "changed_file_count": 0,
    }


def _verify_gates() -> dict[str, Any]:
    if file_sha256(PREVIOUS_POINTER) != PREVIOUS_POINTER_SHA256:
        raise RuntimeError("Superseded Revision-3.3.2 pointer changed")
    if _read(PREVIOUS_POINTER)["source_snapshot"]["snapshot_id"] != PREVIOUS_SNAPSHOT_ID:
        raise RuntimeError("Unexpected superseded Revision-3.3.2 snapshot")
    if file_sha256(STAGE02_MANIFEST) != STAGE02_MANIFEST_SHA256:
        raise RuntimeError("Immutable Stage-02 manifest changed")
    if file_sha256(STAGE03_MANIFEST) != STAGE03_MANIFEST_SHA256:
        raise RuntimeError("Immutable Stage-03 manifest changed")

    provenance_path = CORRECTION_ROOT / "interrupted_revision332_run_provenance.json"
    step88_path = CORRECTION_ROOT / "step88_before_after.json"
    comparison_path = CORRECTION_ROOT / "epoch1_checkpoint_comparison.json"
    repository_path = CORRECTION_ROOT / "repository_gates.json"
    provenance = _read(provenance_path)
    step88 = _read(step88_path)
    comparison = _read(comparison_path)
    repository = _read(repository_path)
    if step88.get("status") != "passed" or not step88.get("batch_exactly_reproduced"):
        raise RuntimeError("Step-88 before/after gate did not pass")
    after = step88["after_correction"]
    inactive = step88["all_inactive_batch"]
    if (
        step88["before_correction"].get("legacy_loss_nonfinite") is not True
        or after.get("loss_absolute_difference") != 0.0
        or after.get("gradient_max_absolute_difference") != 0.0
        or after.get("inactive_gradient_nonzero_count") != 0
        or after.get("all_components_finite") is not True
        or after.get("all_model_gradients_finite") is not True
        or inactive.get("raw_physics_loss") != 0.0
        or inactive.get("weighted_physics_contribution") != 0.0
        or inactive.get("all_gradients_finite") is not True
    ):
        raise RuntimeError("Corrected masked-physics contract did not pass")
    if comparison.get("status") not in {
        "OLD_EPOCH1_CHECKPOINT_EQUIVALENT",
        "OLD_EPOCH1_CHECKPOINT_NOT_EQUIVALENT",
    }:
        raise RuntimeError("Epoch-1 comparison classification is missing")
    if repository.get("status") != "passed" or not all(
        record["passed"] for record in repository["commands"]
    ):
        raise RuntimeError("Repository gates did not pass")

    replay = CORRECTION_ROOT / "corrected_epoch1_replay" / "runs" / "full"
    rows = list(csv.DictReader((replay / "training_log.csv").open(encoding="utf-8")))
    if len(rows) != 1 or rows[0].get("epoch") != "1":
        raise RuntimeError("Corrected epoch-1 log is incomplete")
    numeric = [
        float(value)
        for name, value in rows[0].items()
        if name.startswith(("train_", "validation_")) and value not in {"", "nan", "NaN"}
    ]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("Corrected epoch-1 log contains nonfinite values")
    diagnostics_path = replay / "diagnostics" / "checkpoint_diagnostics_epoch_0001.json"
    diagnostics = _read(diagnostics_path)
    if (
        diagnostics.get("model_state_unchanged") is not True
        or diagnostics.get("optimizer_loaded_or_modified") is not False
        or diagnostics.get("fixed_validation_only") is not True
        or diagnostics.get("test_data_used") is not False
    ):
        raise RuntimeError("Corrected checkpoint diagnostics violated isolation")
    required = (
        "training_statistics.csv",
        "raw_loss_components.csv",
        "weighted_loss_components.csv",
        "gradient_contributions.csv",
        "gradient_cosines.csv",
        "physics_floor_diagnostics.csv",
        "graph_floor_diagnostics.csv",
        "graph_learning_summary.csv",
        "fixed_patch_metrics.csv",
        "whole_realization_metrics.csv",
        "checkpoint_comparison.csv",
        "training_diagnostics_summary.md",
    )
    missing = [name for name in required if not (replay / "diagnostics" / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing corrected diagnostic products: {missing}")
    if CLEAN_PRODUCTION.exists():
        raise RuntimeError(f"Clean production path must not exist before approval: {CLEAN_PRODUCTION}")
    return {
        "superseded_revision332_pointer": {
            "path": str(PREVIOUS_POINTER),
            "sha256": PREVIOUS_POINTER_SHA256,
            "snapshot_id": PREVIOUS_SNAPSHOT_ID,
        },
        "immutable_stage02_manifest_sha256": STAGE02_MANIFEST_SHA256,
        "immutable_stage03_manifest_sha256": STAGE03_MANIFEST_SHA256,
        "interrupted_run": _verify_interrupted_run(provenance),
        "step88_before_after": {"path": str(step88_path), "sha256": file_sha256(step88_path)},
        "epoch1_comparison": {
            "path": str(comparison_path),
            "sha256": file_sha256(comparison_path),
            "status": comparison["status"],
            "difference_count": comparison["total_difference_count"],
        },
        "corrected_training_log": {
            "path": str(replay / "training_log.csv"),
            "sha256": file_sha256(replay / "training_log.csv"),
            "train_total": float(rows[0]["train_total"]),
            "train_physics": float(rows[0]["train_physics"]),
        },
        "checkpoint_diagnostics": {
            "path": str(diagnostics_path),
            "sha256": file_sha256(diagnostics_path),
            "operator_floor": diagnostics["physics"]["truth_noiseless_operator_floor"],
            "noisy_observation_floor": diagnostics["physics"][
                "truth_noisy_observation_floor"
            ],
        },
        "repository_gates": {
            "path": str(repository_path),
            "sha256": file_sha256(repository_path),
        },
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def main() -> None:
    missing = [name for name in REVISION332A_CHANGED_FILES if not (REPOSITORY / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing Revision-3.3.2a source files: {missing}")
    gates = _verify_gates()
    selected = _source_paths()
    source_records = [
        {
            "path": relative.as_posix(),
            "bytes": (REPOSITORY / relative).stat().st_size,
            "sha256": file_sha256(REPOSITORY / relative),
        }
        for relative in selected
    ]
    git_head = _git("rev-parse", "HEAD").strip()
    git_status = _git("status", "--short", "--branch")
    git_diff = _git("diff", "--binary", "HEAD")
    identity = {
        "revision": "3.3.2a",
        "scope": "masked-physics numerical correction only",
        "scientific_objective_changed": False,
        "exact_bug": "inactive residual squared to Inf before multiplication by zero mask",
        "exact_correction": "select eligible residuals before squaring; denominator unchanged",
        "superseded_revision332_snapshot_id": PREVIOUS_SNAPSHOT_ID,
        "immutable_stage02_manifest_sha256": STAGE02_MANIFEST_SHA256,
        "immutable_stage03_manifest_sha256": STAGE03_MANIFEST_SHA256,
        "git_head": git_head,
        "source_files": source_records,
        "revision332a_changed_files": list(REVISION332A_CHANGED_FILES),
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        "gate_artifacts": gates,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    freeze_root = PRIVATE / "source_freezes"
    destination = freeze_root / "v00332a_masked_physics" / snapshot_id
    pointer_path = freeze_root / "revision332a_masked_physics_approved.json"
    if destination.exists() or pointer_path.exists():
        raise FileExistsError("Revision-3.3.2a corrected source freeze already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v00332a_source_freeze_", dir=freeze_root) as temporary:
        staging = Path(temporary)
        source_root = staging / "source"
        metadata_root = staging / "metadata"
        for relative in selected:
            source = REPOSITORY / relative
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        metadata_root.mkdir(parents=True)
        (metadata_root / "git_status.txt").write_text(git_status, encoding="utf-8")
        (metadata_root / "git_diff.patch").write_text(git_diff, encoding="utf-8")
        (metadata_root / "git_diff_stat.txt").write_text(
            _git("diff", "--stat", "HEAD"), encoding="utf-8"
        )
        (metadata_root / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "immutable code-only Revision-3.3.2a corrected-training source",
            "snapshot_id": snapshot_id,
            **identity,
            "source_file_count": len(source_records),
            "exclusions": [
                "private field data",
                "Stage artifacts",
                "checkpoints and logs",
                "configs/paths.yaml",
                "generated arrays",
                "private figures",
            ],
        }
        manifest_path = staging / "source_file_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination.mkdir()
        shutil.move(str(source_root), destination / "source")
        shutil.move(str(metadata_root), destination / "metadata")
        shutil.move(str(manifest_path), destination / "source_file_manifest.json")
        archive = destination / "source_snapshot.tar"
        with tarfile.open(archive, mode="w") as bundle:
            bundle.add(destination / "source", arcname="source", recursive=True)
            bundle.add(destination / "metadata", arcname="metadata", recursive=True)
            bundle.add(
                destination / "source_file_manifest.json", arcname="source_file_manifest.json"
            )
        snapshot = {
            "snapshot_id": snapshot_id,
            "git_head": git_head,
            "source_manifest_path": str(destination / "source_file_manifest.json"),
            "source_manifest_sha256": file_sha256(destination / "source_file_manifest.json"),
            "archive_path": str(archive),
            "archive_sha256": file_sha256(archive),
        }
        _make_read_only(destination)
    pointer = {
        "schema_version": 1,
        "status": "TRAINING_BUGFIX_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_objective_changed": False,
        "superseded_revision332_snapshot_id": PREVIOUS_SNAPSHOT_ID,
        "immutable_inputs": {
            "stage02_manifest_sha256": STAGE02_MANIFEST_SHA256,
            "stage03_manifest_sha256": STAGE03_MANIFEST_SHA256,
        },
        "snapshot_id": snapshot_id,
        "archive_sha256": snapshot["archive_sha256"],
        "source_snapshot": snapshot,
        "gate_artifacts": gates,
        "clean_production_run_path": str(CLEAN_PRODUCTION / "runs" / "full"),
        "clean_production_started": False,
        "exact_start_command": "python -u scripts/run_revision332a_production.py train --device cuda",
    }
    write_json(pointer_path, pointer)
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    main()
