#!/usr/bin/env python3
"""Create a local, content-addressed code-only snapshot for v003 production."""

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


def main() -> None:
    paths_config = load_config(REPOSITORY / "configs" / "paths.yaml")
    freeze_root = Path(paths_config["private_artifact_root"]) / "source_freezes"
    production_root = freeze_root / "v003_production"
    production_root.mkdir(parents=True, exist_ok=True)
    selected = _source_paths()
    git_head = _git("rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory(prefix="v003_source_freeze_", dir=freeze_root) as temp:
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
            "purpose": "immutable code-only source snapshot for SAGE-AVO v003 production",
            "git_head": git_head,
            "source_file_count": len(records),
            "source_files": records,
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
        write_json(freeze_root / "v003_production_current.json", pointer)
        _make_read_only(destination)
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    main()
