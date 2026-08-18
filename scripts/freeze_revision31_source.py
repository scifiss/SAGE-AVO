#!/usr/bin/env python3
"""Create the validated, code-only Revision-3.1 fluid-production snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any

from sage_avo.config import load_config
from sage_avo.experiments.manifest import file_sha256, write_json


REPOSITORY = Path(__file__).resolve().parents[1]
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
VALIDATION_NAME = "v0031_validation8_fluid_corrected"
OLD_NO_GO_SNAPSHOT = "5d9f9726845d9496d3de6b14af63c7bc9a737feda60a7ae7d2a52a78b1001d56"


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _listed_files(*arguments: str) -> list[Path]:
    result = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return [Path(value.decode()) for value in result.stdout.split(b"\0") if value]


def _is_code_snapshot_path(relative: Path) -> bool:
    if relative in EXCLUDED_PATHS or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name not in {"configs/paths.yaml"}
    return relative.parts[0] in ALLOWED_DIRECTORIES


def _source_paths() -> list[Path]:
    tracked = _listed_files("ls-files")
    untracked = _listed_files("ls-files", "--others", "--exclude-standard")
    selected = sorted(
        {
            path
            for path in (*tracked, *untracked)
            if _is_code_snapshot_path(path) and (REPOSITORY / path).is_file()
        },
        key=lambda path: path.as_posix(),
    )
    if not selected:
        raise RuntimeError("No source files were selected for the v003 freeze")
    oversized = [
        path for path in selected if (REPOSITORY / path).stat().st_size > MAX_SOURCE_FILE_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Unexpectedly large source file: {oversized[0]}")
    if Path("configs/paths.yaml") in selected:
        raise RuntimeError("The private paths.yaml file must never enter a source freeze")
    return selected


def _copy_sources(paths: list[Path], destination: Path) -> list[dict[str, Any]]:
    records = []
    for relative in paths:
        source = REPOSITORY / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        )
    return records


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Revision-3.1 freeze gate requires {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_revision31_gate(private_root: Path) -> tuple[Path, dict[str, Any]]:
    root = private_root / "revision31" / VALIDATION_NAME
    candidate_path = root / "fluid_gate" / "candidates" / "candidate_evaluation_report.json"
    execution_qc_path = root / "reports" / "bounded_execution_qc.json"
    notebooks_path = root / "executed_notebooks" / "execution_report.json"
    training_path = (
        root
        / "stage04"
        / "sage_avo_s01_v0031_validation8_fluid_corrected"
        / "runs"
        / "full_2epoch_cuda_sanity"
        / "manifest.json"
    )
    candidate = _read_json(candidate_path)
    execution = _read_json(execution_qc_path)
    notebooks = _read_json(notebooks_path)
    training = _read_json(training_path)
    paths_config = load_config(REPOSITORY / "configs" / "paths.yaml")
    fluid_validation_path = (
        Path(paths_config["work_data_root"])
        / "s01data"
        / "derived"
        / "fluid_models_v0031"
        / "fluid_property_validation.json"
    )
    fluid_validation = _read_json(fluid_validation_path)
    if fluid_validation.get("status") != "passed":
        raise RuntimeError("Independent fluid-property validation has not passed")
    if fluid_validation.get("pressure_temperature_source") in {None, "generic", "unavailable"}:
        raise RuntimeError("Fluid-property validation lacks a measured or independently validated P/T source")
    if candidate.get("selected_production_mode") != "calibrated_differential_gassmann":
        raise RuntimeError("Candidate B was not selected for v0031 production")
    if not candidate.get("candidate_B", {}).get("passes_physical_gate", False):
        raise RuntimeError("Candidate B did not pass the physical gate")
    if candidate.get("candidate_A", {}).get("passes_physical_gate", True):
        raise RuntimeError("Candidate A was not rejected by the calibrated envelope gate")
    round_trip = execution.get("round_trip", {})
    if round_trip.get("bands_in_order") != ["near", "mid", "far"]:
        raise RuntimeError("Round-trip band order is not canonical near/mid/far")
    if int(round_trip.get("realizations", 0)) != 8:
        raise RuntimeError("Round-trip QC does not cover all eight realizations")
    if float(round_trip.get("maximum_relative_rmse", float("inf"))) > 1e-6:
        raise RuntimeError("Stage-02/Stage-04 round-trip relative RMSE exceeds 1e-6")
    fluid = execution.get("fluid", {})
    if float(fluid.get("maximum_nonfluid_channel_difference_from_immutable_v003", -1.0)) != 0.0:
        raise RuntimeError("A non-fluid channel changed relative to the immutable v003 gate")
    if float(fluid.get("outside_plume_maximum_absolute_change", -1.0)) != 0.0:
        raise RuntimeError("Regenerated v0031 data changed elastic properties outside the plume")
    integrity = execution.get("dataset_integrity", {})
    required_integrity = (
        "split_disjoint",
        "geology_group_split_disjoint",
        "all_realizations_represented",
        "patch_bounds_valid",
        "patch_invalid_fraction_valid",
        "prior_finite_and_shape_matched",
        "normalization_finite",
        "normalization_matches_training_realizations",
        "validation_test_sampling_reproducible",
    )
    if any(not integrity.get(name, False) for name in required_integrity):
        raise RuntimeError("Bounded Stage-03 integrity gate is incomplete")
    if notebooks.get("status") != "complete" or not all(
        record.get("error_outputs") == 0 for record in notebooks.get("records", [])
    ):
        raise RuntimeError("Local notebook execution validation is incomplete")
    if len(notebooks.get("records", [])) != 5:
        raise RuntimeError("Local notebook execution did not cover notebooks 01--05")
    if training.get("status") != "complete" or int(training.get("last_completed_epoch", 0)) != 2:
        raise RuntimeError("The two-epoch CUDA sanity gate is incomplete")
    summary = {
        "status": "all_revision31_physical_and_execution_gates_passed",
        "selected_mode": candidate["selected_production_mode"],
        "candidate_B_outside_local_response_envelope_fraction": candidate["candidate_B"]["outside_local_well_response_envelope_fraction"],
        "round_trip_maximum_absolute_error": round_trip["maximum_absolute_error"],
        "round_trip_maximum_relative_rmse": round_trip["maximum_relative_rmse"],
        "dataset_patch_rows": integrity["patch_rows"],
        "cuda_sanity_last_completed_epoch": training["last_completed_epoch"],
        "private_notebooks_executed": len(notebooks["records"]),
        "gate_artifacts": {
            str(path): file_sha256(path)
            for path in (
                candidate_path,
                execution_qc_path,
                notebooks_path,
                training_path,
                fluid_validation_path,
            )
        },
    }
    return root, summary


def _attach_validated_snapshot(
    validation_root: Path,
    pointer: dict[str, Any],
) -> None:
    reference = {
        "snapshot_id": pointer["snapshot_id"],
        "source_manifest_sha256": pointer["source_manifest_sha256"],
        "archive_sha256": pointer["archive_sha256"],
        "git_head": pointer["git_head"],
    }
    targets = [
        validation_root / "stage02" / "realizations" / "manifest.json",
        validation_root / "stage03" / "dataset" / "dataset_manifest.json",
        validation_root
        / "stage04"
        / "sage_avo_s01_v0031_validation8_fluid_corrected"
        / "runs"
        / "full_2epoch_cuda_sanity"
        / "manifest.json",
        validation_root / "configs" / "synthetic_resolved.json",
        validation_root / "configs" / "dataset_resolved.json",
        validation_root / "configs" / "training_resolved.json",
    ]
    for path in targets:
        payload = _read_json(path)
        payload["validated_source_snapshot"] = reference
        write_json(path, payload)
    write_json(validation_root / "validated_source_snapshot.json", reference)


def main() -> None:
    paths_config = load_config(REPOSITORY / "configs" / "paths.yaml")
    freeze_root = Path(paths_config["private_artifact_root"]) / "source_freezes"
    validation_root, gate_summary = _verify_revision31_gate(
        Path(paths_config["private_artifact_root"])
    )
    production_root = freeze_root / "v0031_production"
    production_root.mkdir(parents=True, exist_ok=True)
    selected = _source_paths()
    git_head = _git("rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory(prefix="v0031_source_freeze_", dir=freeze_root) as temp:
        staging = Path(temp)
        source_directory = staging / "source"
        metadata_directory = staging / "metadata"
        records = _copy_sources(selected, source_directory)
        metadata_directory.mkdir(parents=True, exist_ok=True)
        (metadata_directory / "git_status.txt").write_text(
            _git("status", "--short", "--branch"), encoding="utf-8"
        )
        (metadata_directory / "git_diff.patch").write_text(
            _git("diff", "--binary", "HEAD"), encoding="utf-8"
        )
        (metadata_directory / "git_diff_stat.txt").write_text(
            _git("diff", "--stat", "HEAD"), encoding="utf-8"
        )
        (metadata_directory / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "purpose": "immutable code-only source snapshot for SAGE-AVO v0031 fluid-corrected production",
            "git_head": git_head,
            "source_file_count": len(records),
            "source_files": records,
            "revision31_gate": gate_summary,
            "supersedes_for_fluid_production": OLD_NO_GO_SNAPSHOT,
            "exclusions": [
                "private field data",
                "stage artifacts",
                "checkpoints and logs",
                "configs/paths.yaml",
                "generated arrays",
                "private figures",
            ],
        }
        manifest_path = staging / "source_file_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot_id = file_sha256(manifest_path)
        destination = production_root / snapshot_id
        if destination.exists():
            raise FileExistsError(f"Frozen source snapshot already exists: {destination}")
        destination.mkdir()
        shutil.move(str(source_directory), destination / "source")
        shutil.move(str(metadata_directory), destination / "metadata")
        shutil.move(str(manifest_path), destination / "source_file_manifest.json")
        archive = destination / "source_snapshot.tar"
        with tarfile.open(archive, mode="w") as bundle:
            bundle.add(destination / "source", arcname="source", recursive=True)
            bundle.add(destination / "metadata", arcname="metadata", recursive=True)
            bundle.add(
                destination / "source_file_manifest.json",
                arcname="source_file_manifest.json",
            )
        archive_sha256 = file_sha256(archive)
        source_manifest = destination / "source_file_manifest.json"
        pointer = {
            "schema_version": 1,
            "status": "frozen",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": snapshot_id,
            "git_head": git_head,
            "source_file_count": len(records),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "archive_path": str(archive),
            "archive_sha256": archive_sha256,
        }
        write_json(freeze_root / "v0031_production_current.json", pointer)
        _make_read_only(destination)
    _attach_validated_snapshot(validation_root, pointer)
    write_json(
        freeze_root / "v003_no_go_fluid_supersession.json",
        {
            "status": "superseded_for_fluid_production_retained_for_provenance",
            "old_snapshot_id": OLD_NO_GO_SNAPSHOT,
            "new_snapshot_id": pointer["snapshot_id"],
            "reason": "The superseded snapshot used a projected local inverse-Gassmann state that fails the physical acceptance criteria.",
            "old_snapshot_deleted": False,
        },
    )
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    main()
