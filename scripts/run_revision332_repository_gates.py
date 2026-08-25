#!/usr/bin/env python3
"""Run and persist the Revision-3.3.2 observability repository gates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from sage_avo.experiments.manifest import write_json


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
OUTPUT = PRIVATE / "revision332" / "training_instrumentation" / "repository_gates.json"
COMMANDS = (
    ("ruff", ["ruff", "check", "--no-cache", "src", "tests", "scripts"]),
    ("pytest", ["pytest", "-p", "no:cacheprovider"]),
    ("smoke_test", ["python", "scripts/smoke_test.py"]),
    ("public_repository_scan", ["python", "scripts/check_public_repo.py"]),
    (
        "instrumentation_tests",
        ["pytest", "-q", "-p", "no:cacheprovider", "tests/test_training_observability.py"],
    ),
)


def _run(name: str, command: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "return_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    records = [_run(name, command) for name, command in COMMANDS]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": "3.3.2",
        "scope": "training observability only",
        "commands": records,
        "status": "passed" if all(row["passed"] for row in records) else "failed",
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
