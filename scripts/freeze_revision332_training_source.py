#!/usr/bin/env python3
"""Create the immutable code-only Revision-3.3.2 training source snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
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
PREVIOUS_SOURCE_SNAPSHOT = "3cea3dff45296b97dfe8374695fb09dd09ac50a12f2d80599806e6af64b00456"
STAGE03_MANIFEST_SHA256 = "1afe64debc9b0901afde88b327a1c04088c1a2a8e51efafadd38bc5e6ba845ee"
STAGE03_FREEZE_SHA256 = "2a84c9984460e16a76c5a3266a7a19076170d1788d08e15c0b4930c2e9174dfe"
ALLOWED_DIRECTORIES = {
    ".github",
    "configs",
    "docs",
    "notebooks",
    "scripts",
    "src",
    "tests",
}
EXCLUDED_PATHS = {Path("configs/paths.yaml")}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024
REVISION332_CHANGED_FILES = (
    "configs/training_observability_v00332.yaml",
    "scripts/freeze_revision332_training_source.py",
    "scripts/run_revision332_production.py",
    "scripts/run_revision332_repository_gates.py",
    "scripts/run_revision332_trajectory_probe.py",
    "src/sage_avo/diagnostics/__init__.py",
    "src/sage_avo/diagnostics/accounting.py",
    "src/sage_avo/diagnostics/checkpoint_analysis.py",
    "src/sage_avo/diagnostics/contracts.py",
    "src/sage_avo/diagnostics/live_logging.py",
    "src/sage_avo/diagnostics/reporting.py",
    "src/sage_avo/experiments/training.py",
    "src/sage_avo/training/engine.py",
    "tests/test_training_observability.py",
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
    selected = sorted(
        {
            path
            for path in (
                *_listed_files("ls-files"),
                *_listed_files("ls-files", "--others", "--exclude-standard"),
            )
            if _is_source(path) and (REPOSITORY / path).is_file()
        },
        key=lambda path: path.as_posix(),
    )
    if Path("configs/paths.yaml") in selected:
        raise RuntimeError("Private paths configuration entered source freeze")
    oversized = [
        path for path in selected if (REPOSITORY / path).stat().st_size > MAX_SOURCE_FILE_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Oversized source file(s): {oversized}")
    return selected


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_gates() -> dict[str, Any]:
    root = PRIVATE / "revision332" / "training_instrumentation"
    trajectory_path = root / "trajectory_equivalence_report.json"
    repository_path = root / "repository_gates.json"
    sample_path = root / "diagnostic_sample_manifest.json"
    contract_path = root / "instrumentation_contract.json"
    short_summary = (
        root
        / "short_execution_gate_final"
        / "runs"
        / "full"
        / "diagnostics"
        / "training_diagnostics_summary.md"
    )
    trajectory = _read(trajectory_path)
    repository = _read(repository_path)
    sample = _read(sample_path)
    contract = _read(contract_path)
    if trajectory.get("status") != "passed" or trajectory.get("difference_count") != 0:
        raise RuntimeError("Training trajectory equivalence did not pass exactly")
    if repository.get("status") != "passed":
        raise RuntimeError("Repository gates did not pass")
    if (
        not short_summary.exists()
        or "short instrumentation execution passed" not in short_summary.read_text(encoding="utf-8")
    ):
        raise RuntimeError("Short checkpoint-reload instrumentation gate did not pass")
    if sample.get("test_data_used") is not False or sample.get("native_physics_patch_count", 0) < 4:
        raise RuntimeError("Frozen validation-only diagnostic set is invalid")
    if contract.get("scientific_methodology_changed") is not False:
        raise RuntimeError("Instrumentation contract unexpectedly changes methodology")
    stage03_pointer = PRIVATE / "dataset_freezes" / "revision331_stage03_approved.json"
    if file_sha256(stage03_pointer) == "":
        raise RuntimeError("Missing Stage-03 freeze pointer")
    stage03_manifest = (
        PRIVATE
        / "stage_artifacts"
        / "stage03"
        / "ds_v00331_production100_support_aware"
        / "dataset"
        / "dataset_manifest.json"
    )
    if file_sha256(stage03_manifest) != STAGE03_MANIFEST_SHA256:
        raise RuntimeError("Frozen Stage-03 manifest changed")
    stage03_record = (
        PRIVATE
        / "dataset_freezes"
        / "ds_v00331_production100_support_aware"
        / STAGE03_MANIFEST_SHA256
        / "stage03_freeze_record.json"
    )
    if file_sha256(stage03_record) != STAGE03_FREEZE_SHA256:
        raise RuntimeError("Frozen Stage-03 record changed")
    return {
        "trajectory_equivalence": {
            "path": str(trajectory_path),
            "sha256": file_sha256(trajectory_path),
            "maximum_absolute_numeric_difference": trajectory[
                "maximum_absolute_numeric_difference"
            ],
        },
        "repository_gates": {
            "path": str(repository_path),
            "sha256": file_sha256(repository_path),
        },
        "diagnostic_sample_manifest": {
            "path": str(sample_path),
            "sha256": file_sha256(sample_path),
        },
        "instrumentation_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "short_execution_summary": {
            "path": str(short_summary),
            "sha256": file_sha256(short_summary),
        },
        "stage03_manifest_sha256": STAGE03_MANIFEST_SHA256,
        "stage03_freeze_record_sha256": STAGE03_FREEZE_SHA256,
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def main() -> None:
    missing = [name for name in REVISION332_CHANGED_FILES if not (REPOSITORY / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing Revision-3.3.2 source files: {missing}")
    gate_artifacts = _verify_gates()
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
    observability_sha = file_sha256(REPOSITORY / "configs" / "training_observability_v00332.yaml")
    identity = {
        "revision": "3.3.2",
        "scope": "training observability only",
        "scientific_model_data_training_methodology_changed": False,
        "previous_revision331_source_snapshot": PREVIOUS_SOURCE_SNAPSHOT,
        "immutable_stage03_manifest_sha256": STAGE03_MANIFEST_SHA256,
        "immutable_stage03_freeze_record_sha256": STAGE03_FREEZE_SHA256,
        "diagnostics_configuration_sha256": observability_sha,
        "git_head": git_head,
        "source_files": source_records,
        "revision332_changed_files": list(REVISION332_CHANGED_FILES),
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        "gate_artifacts": gate_artifacts,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    freeze_root = PRIVATE / "source_freezes"
    destination = freeze_root / "v00332_training_observability" / snapshot_id
    pointer_path = freeze_root / "revision332_training_instrumentation_approved.json"
    if destination.exists() or pointer_path.exists():
        raise FileExistsError("Revision-3.3.2 source freeze already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v00332_source_freeze_", dir=freeze_root) as temporary:
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
            "purpose": "immutable code-only Revision-3.3.2 observable-training source",
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
            "diagnostics_configuration_sha256": observability_sha,
        }
        _make_read_only(destination)
    pointer = {
        "schema_version": 1,
        "status": "TRAINING_INSTRUMENTATION_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_model_data_training_methodology_changed": False,
        "previous_revision331_source_snapshot": PREVIOUS_SOURCE_SNAPSHOT,
        "immutable_stage03": {
            "manifest_sha256": STAGE03_MANIFEST_SHA256,
            "freeze_record_sha256": STAGE03_FREEZE_SHA256,
        },
        "source_snapshot": snapshot,
        "gate_artifacts": gate_artifacts,
        "full_120_epoch_training_started": False,
    }
    write_json(pointer_path, pointer)
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    main()
