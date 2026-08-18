import json
from pathlib import Path

import pytest

from sage_avo.experiments.manifest import (
    file_sha256,
    load_frozen_source_reference,
)


def _pointer(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "snapshot" / "source_file_manifest.json"
    archive = tmp_path / "snapshot" / "source_snapshot.tar"
    manifest.parent.mkdir()
    manifest.write_text('{"source_files": []}\n', encoding="utf-8")
    archive.write_bytes(b"immutable source archive")
    manifest_hash = file_sha256(manifest)
    pointer = tmp_path / "source_freezes" / "v003_production_current.json"
    pointer.parent.mkdir()
    pointer.write_text(
        json.dumps(
            {
                "status": "frozen",
                "snapshot_id": manifest_hash,
                "source_manifest_path": str(manifest),
                "source_manifest_sha256": manifest_hash,
                "archive_path": str(archive),
                "archive_sha256": file_sha256(archive),
                "git_head": "0123456789abcdef",
            }
        ),
        encoding="utf-8",
    )
    return pointer, manifest, archive


def test_frozen_source_reference_verifies_both_hashes(tmp_path: Path):
    _pointer(tmp_path)
    reference = load_frozen_source_reference(tmp_path)
    assert reference["snapshot_id"] == reference["source_manifest_sha256"]
    assert reference["git_head"] == "0123456789abcdef"


def test_frozen_source_reference_rejects_archive_tampering(tmp_path: Path):
    _, _, archive = _pointer(tmp_path)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive SHA-256"):
        load_frozen_source_reference(tmp_path)


def test_frozen_source_reference_accepts_explicit_revision_pointer(tmp_path: Path):
    pointer, _, _ = _pointer(tmp_path)
    revision31 = pointer.with_name("v0031_production_current.json")
    pointer.rename(revision31)
    reference = load_frozen_source_reference(
        tmp_path, pointer_name="v0031_production_current.json"
    )
    assert reference["snapshot_id"] == reference["source_manifest_sha256"]


def test_frozen_source_reference_rejects_pointer_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="filename"):
        load_frozen_source_reference(tmp_path, pointer_name="../pointer.json")


def test_frozen_source_reference_rejects_no_go_pointer(tmp_path: Path):
    pointer, _, _ = _pointer(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["status"] = "frozen_no_go"
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not production-approved"):
        load_frozen_source_reference(tmp_path)
