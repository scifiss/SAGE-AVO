#!/usr/bin/env python3
"""Freeze the passed Revision-3.3.1 support-aware production source."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
ROUTE = "GO_SCENARIO_CO2"
CALIBRATION_ID = "v0033_58a5fe39a11c4fe66431"
PRIOR_SNAPSHOT_ID = (
    "7dceaac7300313cd5390f55e7620baa5cdc563f1f5b9b2f2f3b3614118290507"
)
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
    if relative in EXCLUDED_PATHS or any(
        part in EXCLUDED_PARTS for part in relative.parts
    ):
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
    if not selected:
        raise RuntimeError("No source files selected")
    if Path("configs/paths.yaml") in selected:
        raise RuntimeError("Private paths configuration entered source freeze")
    oversized = [
        path
        for path in selected
        if (REPOSITORY / path).stat().st_size > MAX_SOURCE_FILE_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Oversized source file(s): {oversized}")
    return selected


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(private: Path) -> dict[str, Any]:
    root = private / "revision331" / "support_aware_generation_gate"
    audit_path = root / "audit" / "reports" / "full_corpus_support_failure_audit.json"
    stress_path = root / "reports" / "stress_gate.json"
    bounded_path = root / "reports" / "bounded_support_gate.json"
    repository_path = root / "reports" / "repository_gates.json"
    audit = _read(audit_path)
    stress = _read(stress_path)
    bounded = _read(bounded_path)
    repository = _read(repository_path)
    if audit.get("status") != "audit_complete_no_source_changed":
        raise RuntimeError("Failed-corpus audit is incomplete")
    if not all(audit.get("integrity", {}).get(name, False) for name in (
        "all_configuration_hashes_match",
        "all_metadata_hashes_match",
        "manifest_ids_match",
    )):
        raise RuntimeError("Failed-corpus provenance integrity gate failed")
    failed_manifest = Path(
        audit["immutable_inputs"]["failed_corpus_manifest"]["path"]
    )
    if file_sha256(failed_manifest) != audit["immutable_inputs"][
        "failed_corpus_manifest"
    ]["sha256"]:
        raise RuntimeError("Failed Revision-3.3 corpus changed after audit")
    if stress.get("status") != "passed" or not all(stress["gates"].values()):
        raise RuntimeError("Stress/diversity gate did not pass")
    if bounded.get("status") != "passed" or not all(bounded["gates"].values()):
        raise RuntimeError("Bounded support-aware gate did not pass")
    if repository.get("status") != "passed":
        raise RuntimeError("Repository gates did not pass")
    old_pointer_path = private / "source_freezes" / "revision33_approved_production.json"
    old_pointer = _read(old_pointer_path)
    if old_pointer["source_snapshot"]["snapshot_id"] != PRIOR_SNAPSHOT_ID:
        raise RuntimeError("Unexpected prior Revision-3.3 snapshot")
    old_archive = Path(old_pointer["source_snapshot"]["archive_path"])
    if file_sha256(old_archive) != old_pointer["source_snapshot"]["archive_sha256"]:
        raise RuntimeError("Prior Revision-3.3 source archive hash mismatch")
    return {
        "audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
        "stress": {"path": str(stress_path), "sha256": file_sha256(stress_path)},
        "bounded": {"path": str(bounded_path), "sha256": file_sha256(bounded_path)},
        "repository": {
            "path": str(repository_path),
            "sha256": file_sha256(repository_path),
        },
        "failed_corpus_manifest": {
            "path": str(failed_manifest),
            "sha256": file_sha256(failed_manifest),
        },
        "prior_pointer": {
            "path": str(old_pointer_path),
            "sha256": file_sha256(old_pointer_path),
        },
        "minimum_bounded_overall_support": bounded["minimum_overall_support"],
        "minimum_bounded_class_support": bounded["minimum_class_support"],
        "round_trip_maximum_relative_rmse": bounded["round_trip"][
            "maximum_relative_rmse"
        ],
        "stress_initial_acceptance_rate": stress["initial_acceptance_rate"],
        "stress_final_acceptance_rate": stress["final_acceptance_rate"],
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def main() -> None:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    gate_summary = _verify(private)
    support_path = REPOSITORY / "configs" / "revision331_support_acceptance.yaml"
    support_sha256 = file_sha256(support_path)
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
        "revision": "3.3.1",
        "route": ROUTE,
        "calibration_id": CALIBRATION_ID,
        "git_head": git_head,
        "source_files": source_records,
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        "gate_artifacts": gate_summary,
        "support_contract_sha256": support_sha256,
        "prior_snapshot_id": PRIOR_SNAPSHOT_ID,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    freeze_root = private / "source_freezes"
    destination = freeze_root / "v00331_support_aware_production" / snapshot_id
    pointer_path = freeze_root / "revision331_approved_production.json"
    if destination.exists() or pointer_path.exists():
        raise FileExistsError(
            "Revision-3.3.1 source freeze or approval pointer already exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="v00331_source_freeze_", dir=freeze_root
    ) as temporary:
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
        (metadata_root / "git_head.txt").write_text(
            git_head + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "immutable code-only source snapshot for Revision-3.3.1 "
                "support-aware Stage-02 production"
            ),
            "snapshot_id": snapshot_id,
            **identity,
            "source_file_count": len(source_records),
            "exclusions": [
                "private field data",
                "stage artifacts",
                "checkpoints and logs",
                "configs/paths.yaml",
                "generated arrays",
                "private figures",
            ],
            "prior_revision33_snapshot_retained": True,
            "prior_revision33_snapshot_superseded_scope": (
                "full-corpus geological generation only"
            ),
            "revision33_scientific_calibration_preserved": True,
        }
        manifest_path = staging / "source_file_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
                destination / "source_file_manifest.json",
                arcname="source_file_manifest.json",
            )
        snapshot = {
            "snapshot_id": snapshot_id,
            "git_head": git_head,
            "source_manifest_path": str(
                destination / "source_file_manifest.json"
            ),
            "source_manifest_sha256": file_sha256(
                destination / "source_file_manifest.json"
            ),
            "archive_path": str(archive),
            "archive_sha256": file_sha256(archive),
        }
        _make_read_only(destination)
    pointer = {
        "schema_version": 1,
        "status": ROUTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": CALIBRATION_ID,
        "support_contract_sha256": support_sha256,
        "source_snapshot": snapshot,
        "prior_revision33_snapshot": {
            "snapshot_id": PRIOR_SNAPSHOT_ID,
            "status": "retained",
            "superseded_scope": "full-corpus geological generation only",
            "scientific_calibration_status": "preserved",
        },
        "gate_artifacts": gate_summary,
        "allowed_claim": (
            "CO2-related elastic perturbations are scenario-conditioned over "
            "reviewed pressure, temperature, salinity, saturation and dry-frame ranges."
        ),
        "prohibited_claim": (
            "The modeled changes are quantitative field-specific S01 CO2 responses."
        ),
        "full_support_aware_100_realization_generation_executed": False,
        "stage03_executed": False,
        "training_executed": False,
    }
    write_json(pointer_path, pointer)
    decision_path = (
        private
        / "revision331"
        / "support_aware_generation_gate"
        / "reports"
        / "final_go_decision.json"
    )
    write_json(
        decision_path,
        {
            "status": ROUTE,
            "all_support_and_execution_gates_passed": True,
            "source_snapshot": snapshot,
            "approval_pointer": {
                "path": str(pointer_path),
                "sha256": file_sha256(pointer_path),
            },
            "full_100_realization_generation_executed": False,
            "stage03_executed": False,
            "training_executed": False,
        },
    )
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    main()
