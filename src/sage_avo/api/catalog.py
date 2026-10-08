"""Read-only informational capability catalog and versioned provenance."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
from typing import Any

API_VERSION = "0.1.0"


def list_operations(*, include_unavailable: bool = True) -> tuple[dict[str, Any], ...]:
    """Return detached catalog entries without importing or executing operators."""
    payload = json.loads(resources.files("sage_avo.api").joinpath("catalog.json").read_text())
    rows = payload["operations"]
    return tuple(deepcopy(row) for row in rows if include_unavailable or row["available"])


def get_operation(identifier: str) -> dict[str, Any]:
    """Describe one declared operation, including conditional requirements."""
    for row in list_operations():
        if row["identifier"] == identifier:
            return row
    raise KeyError(f"Unknown SAGE-AVO public operator: {identifier}")


def _source_revision() -> str | None:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(
    identifier: str | None = None,
    *,
    specification: Any | None = None,
    calibration_id: str | None = None,
    input_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return distinct API/package/source/operator identities; no operator run."""
    try:
        package_version = metadata.version("sage-avo")
    except metadata.PackageNotFoundError:
        from sage_avo import __version__

        package_version = __version__
    operation = get_operation(identifier) if identifier is not None else None
    config_hash = None
    if input_configuration is not None:
        encoded = json.dumps(input_configuration, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(encoded.encode()).hexdigest()
    return {
        "api_schema_version": API_VERSION,
        "package_version": package_version,
        "git_source_revision": _source_revision(),
        "operator_identifier": identifier,
        "operator_version": operation["operator_version"] if operation else None,
        "scientific_status": operation["eligibility"] if operation else None,
        "forward_specification_id": getattr(specification, "specification_id", None),
        "forward_specification_sha256": getattr(specification, "sha256", None),
        "calibration_id": calibration_id,
        "input_configuration_sha256": config_hash,
        "validation_status": "bounded_unit_and_parity_tests_not_field_validation",
    }
