"""Machine-readable manifests for expensive SAGE-AVO runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_source_reference(
    private_artifact_root: str | Path,
    *,
    pointer_name: str = "v003_production_current.json",
) -> dict[str, str]:
    """Load and verify a named local code snapshot required by production."""
    if Path(pointer_name).name != pointer_name:
        raise ValueError("pointer_name must be a filename, not a path")
    pointer = Path(private_artifact_root) / "source_freezes" / pointer_name
    if not pointer.exists():
        raise FileNotFoundError(
            "Production requires a frozen source pointer at "
            f"{pointer}. The production source-freeze utility creates this pointer."
        )
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError(
            f"Frozen source pointer is not production-approved: status={payload.get('status')!r}"
        )
    required = {
        "snapshot_id",
        "source_manifest_path",
        "source_manifest_sha256",
        "archive_path",
        "archive_sha256",
        "git_head",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Frozen source pointer is missing keys: {missing}")
    manifest_path = Path(payload["source_manifest_path"])
    archive_path = Path(payload["archive_path"])
    if file_sha256(manifest_path) != payload["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest SHA-256 verification failed")
    if file_sha256(archive_path) != payload["archive_sha256"]:
        raise ValueError("Frozen source archive SHA-256 verification failed")
    if payload["snapshot_id"] != payload["source_manifest_sha256"]:
        raise ValueError("Frozen snapshot ID must equal the source-manifest SHA-256")
    return {
        "snapshot_id": str(payload["snapshot_id"]),
        "source_manifest_sha256": str(payload["source_manifest_sha256"]),
        "archive_sha256": str(payload["archive_sha256"]),
        "git_head": str(payload["git_head"]),
    }


def git_commit(repository: str | Path) -> str | None:
    """Return the local commit without requiring the repository to be initialized."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or None,
    }
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        summary["torch"] = None
    return summary


def build_run_manifest(
    *,
    repository: str | Path,
    config_path: str | Path,
    seed: int,
    split_ids: dict[str, list[int]],
    model_variant: str,
    checkpoint: str | None,
    training_epochs: int,
    normalization: dict[str, Any],
    prior_settings: dict[str, Any],
    metric_definitions: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    """Construct the required immutable experiment record."""
    config = Path(config_path)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repository),
        "config_file": str(config.name),
        "config_sha256": file_sha256(config),
        "seed": int(seed),
        "dataset_split_ids": split_ids,
        "model_variant": model_variant,
        "checkpoint": checkpoint,
        "training_epochs": int(training_epochs),
        "normalization": normalization,
        "prior": prior_settings,
        "metrics": metric_definitions,
        "hardware": hardware_summary(),
        "status": status,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
