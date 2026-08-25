#!/usr/bin/env python3
"""Freeze and approve the validated Revision-3.3 production source."""

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
VALIDATION_NAME = "v0033_validation8_dry_frame_supported"
ROUTE = "GO_SCENARIO_CO2"
ALLOWED_DIRECTORIES = {".github", "configs", "docs", "notebooks", "scripts", "src", "tests"}
EXCLUDED_PATHS = {Path("configs/paths.yaml")}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024


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
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def _is_source(relative: Path) -> bool:
    if relative in EXCLUDED_PATHS or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name != "configs/paths.yaml"
    return relative.parts[0] in ALLOWED_DIRECTORIES


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
    if not selected:
        raise RuntimeError("No source files selected")
    if any((REPOSITORY / path).stat().st_size > MAX_SOURCE_FILE_BYTES for path in selected):
        raise RuntimeError("A selected source file exceeds the 10-MiB source limit")
    if Path("configs/paths.yaml") in selected:
        raise RuntimeError("Private configs/paths.yaml entered source selection")
    return selected


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(private: Path, work_data: Path) -> tuple[Path, dict[str, Any], Path]:
    root = private / "revision33" / VALIDATION_NAME
    analysis = _read(root / "dry_frame_gate" / "analysis" / "support_gap_report.json")
    calibration = _read(root / "dry_frame_gate" / "calibration" / "dry_frame_calibration_report.json")
    candidate = _read(root / "dry_frame_gate" / "evaluation" / "candidate_b_evaluation_report.json")
    bounded = _read(root / "reports" / "bounded_execution_qc.json")
    cuda = _read(root / "reports" / "cuda_sanity.json")
    notebooks = _read(root / "executed_notebooks" / "execution_report.json")
    repository = _read(root / "reports" / "repository_gates.json")
    provisional = _read(root / "reports" / "provisional_route_decision.json")
    if analysis.get("status") != "analysis_complete_no_model_changed":
        raise RuntimeError("Support-gap analysis is incomplete")
    if calibration.get("status") != "passed" or not all(calibration.get("gates", {}).values()):
        raise RuntimeError("Dry-frame calibration did not pass every gate")
    if candidate.get("status") != "passed" or not all(candidate.get("gates", {}).values()):
        raise RuntimeError("Candidate B did not pass every gate")
    if provisional.get("status") != ROUTE:
        raise RuntimeError("Provisional route is not GO_SCENARIO_CO2")
    if bounded.get("stage02_realizations") != 8:
        raise RuntimeError("Bounded execution does not contain exactly eight realizations")
    if not bounded.get("matched_qc", {}).get("all_unaffected_channels_bitwise_equal", False):
        raise RuntimeError("An unaffected matched channel changed")
    if not bounded.get("exactly_zero_outside_plume", False):
        raise RuntimeError("Bounded elastic changes are not zero outside plume")
    if not bounded.get("fixed_shear_within_tolerance", False):
        raise RuntimeError("Bounded fixed-shear gate failed")
    round_trip = bounded.get("round_trip", {})
    if round_trip.get("bands_in_order") != ["near", "mid", "far"]:
        raise RuntimeError("Round-trip bands are not canonical near/mid/far")
    if float(round_trip.get("maximum_relative_rmse", float("inf"))) > 1e-6:
        raise RuntimeError("Round-trip relative RMSE exceeds 1e-6")
    integrity = bounded.get("stage03_integrity", {})
    required = (
        "split_disjoint", "geology_group_split_disjoint", "all_realizations_represented",
        "patch_bounds_valid", "patch_invalid_fraction_valid", "prior_finite_and_shape_matched",
        "normalization_finite", "normalization_matches_training_realizations",
        "validation_test_sampling_reproducible",
    )
    if any(not integrity.get(name, False) for name in required):
        raise RuntimeError("Stage-03 integrity gate is incomplete")
    if integrity.get("duplicate_patch_count") != 0 or integrity.get("duplicate_coordinate_count") != 0:
        raise RuntimeError("Stage-03 contains duplicate patches or coordinates")
    if cuda.get("status") != "complete" or cuda.get("epochs") != 2:
        raise RuntimeError("Capped CUDA sanity gate is incomplete")
    if notebooks.get("status") != "complete" or len(notebooks.get("records", [])) != 5:
        raise RuntimeError("Five-notebook private execution gate is incomplete")
    if any(row.get("error_outputs") != 0 for row in notebooks["records"]):
        raise RuntimeError("A private executed notebook contains an error output")
    if repository.get("status") != "passed":
        raise RuntimeError("Repository gates did not pass")
    validation_path = work_data / "s01data" / "derived" / "fluid_models_v0033" / "fluid_property_validation.json"
    validation = _read(validation_path)
    if validation.get("status") != "scenario_validated":
        raise RuntimeError("Fluid scenario is not validated")
    summary = {
        "route": ROUTE,
        "calibration_id": calibration["calibration_id"],
        "all_well_support_coverage": calibration["all_well_support"]["overall_coverage"],
        "candidate_b_envelope_coverage": candidate["predictive_envelope_coverage"],
        "round_trip_maximum_relative_rmse": round_trip["maximum_relative_rmse"],
        "stage03_patch_rows": integrity["patch_rows"],
        "cuda_epochs": cuda["epochs"],
        "private_notebooks": len(notebooks["records"]),
        "gate_artifacts": {
            str(path): file_sha256(path)
            for path in (
                root / "dry_frame_gate" / "analysis" / "support_gap_report.json",
                root / "dry_frame_gate" / "calibration" / "dry_frame_calibration_report.json",
                root / "dry_frame_gate" / "evaluation" / "candidate_b_evaluation_report.json",
                root / "reports" / "bounded_execution_qc.json",
                root / "reports" / "cuda_sanity.json",
                root / "executed_notebooks" / "execution_report.json",
                root / "reports" / "repository_gates.json",
            )
        },
    }
    return root, summary, validation_path


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _attach(root: Path, reference: dict[str, Any]) -> None:
    targets = [
        root / "stage02" / "realizations" / "manifest.json",
        root / "stage03" / "dataset" / "dataset_manifest.json",
        root / "stage04" / "sage_avo_s01_v0033_validation8" / "runs" / "full_2epoch_cuda_sanity" / "manifest.json",
        root / "configs" / "synthetic_resolved.json",
        root / "configs" / "dataset_resolved.json",
        root / "configs" / "training_resolved.json",
    ]
    for path in targets:
        payload = _read(path)
        payload["validated_source_snapshot"] = reference
        write_json(path, payload)
    write_json(root / "validated_source_snapshot.json", reference)


def main() -> None:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    work_data = Path(paths["work_data_root"])
    validation_root, gate_summary, validation_path = _verify(private, work_data)
    freeze_root = private / "source_freezes"
    destination_root = freeze_root / "v0033_production"
    destination_root.mkdir(parents=True, exist_ok=True)
    selected = _source_paths()
    git_head = _git("rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory(prefix="v0033_source_freeze_", dir=freeze_root) as temp:
        staging = Path(temp)
        source_directory = staging / "source"
        metadata_directory = staging / "metadata"
        records = []
        for relative in selected:
            source = REPOSITORY / relative
            target = source_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append({"path": relative.as_posix(), "bytes": source.stat().st_size, "sha256": file_sha256(source)})
        metadata_directory.mkdir(parents=True)
        (metadata_directory / "git_status.txt").write_text(_git("status", "--short", "--branch"), encoding="utf-8")
        (metadata_directory / "git_diff.patch").write_text(_git("diff", "--binary", "HEAD"), encoding="utf-8")
        (metadata_directory / "git_diff_stat.txt").write_text(_git("diff", "--stat", "HEAD"), encoding="utf-8")
        (metadata_directory / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "immutable code-only source snapshot for SAGE-AVO Revision-3.3 production",
            "approved_route": ROUTE,
            "git_head": git_head,
            "source_file_count": len(records),
            "source_files": records,
            "revision33_gate": gate_summary,
            "prior_snapshots_retained": True,
            "exclusions": [
                "private field data", "stage artifacts", "checkpoints and logs",
                "configs/paths.yaml", "generated arrays", "private figures",
            ],
        }
        manifest_path = staging / "source_file_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snapshot_id = file_sha256(manifest_path)
        destination = destination_root / snapshot_id
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir()
        shutil.move(str(source_directory), destination / "source")
        shutil.move(str(metadata_directory), destination / "metadata")
        shutil.move(str(manifest_path), destination / "source_file_manifest.json")
        archive = destination / "source_snapshot.tar"
        with tarfile.open(archive, mode="w") as bundle:
            bundle.add(destination / "source", arcname="source", recursive=True)
            bundle.add(destination / "metadata", arcname="metadata", recursive=True)
            bundle.add(destination / "source_file_manifest.json", arcname="source_file_manifest.json")
        source_snapshot = {
            "snapshot_id": snapshot_id,
            "git_head": git_head,
            "source_manifest_path": str(destination / "source_file_manifest.json"),
            "source_manifest_sha256": file_sha256(destination / "source_file_manifest.json"),
            "archive_path": str(archive),
            "archive_sha256": file_sha256(archive),
        }
        _make_read_only(destination)
    validation = _read(validation_path)
    validation["bounded_input_validation_sha256"] = file_sha256(validation_path)
    validation["production_approval"] = True
    validation["approved_route"] = ROUTE
    validation["approved_source_snapshot"] = source_snapshot
    write_json(validation_path, validation)
    pointer = {
        "schema_version": 1,
        "status": ROUTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": source_snapshot,
        "fluid_validation": _source(validation_path),
        "calibration_id": gate_summary["calibration_id"],
        "allowed_claim": "CO2-related elastic perturbations are scenario-conditioned over reviewed pressure, temperature, salinity, saturation and dry-frame ranges.",
        "prohibited_claim": "The modeled changes are quantitative field-specific S01 CO2 responses.",
        "full_production_has_not_run": True,
    }
    pointer_path = freeze_root / "revision33_approved_production.json"
    if pointer_path.exists():
        raise FileExistsError(f"Revision-3.3 approval pointer already exists: {pointer_path}")
    write_json(pointer_path, pointer)
    reference = {**source_snapshot, "route": ROUTE, "approval_pointer_sha256": file_sha256(pointer_path)}
    _attach(validation_root, reference)
    write_json(
        validation_root / "reports" / "final_route_decision.json",
        {
            "status": ROUTE,
            "all_scientific_and_execution_gates_passed": True,
            "source_snapshot": source_snapshot,
            "approval_pointer": _source(pointer_path),
            "full_100_realization_generation_executed": False,
            "full_training_executed": False,
        },
    )
    print(json.dumps(pointer, indent=2))


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


if __name__ == "__main__":
    main()
