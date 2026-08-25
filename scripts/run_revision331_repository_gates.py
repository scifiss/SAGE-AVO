#!/usr/bin/env python3
"""Run and record the Revision-3.3.1 source and public-repository gates."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import nbformat

from sage_avo.config import load_config
from sage_avo.experiments.manifest import write_json


REPOSITORY = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("ruff", ["ruff", "check", "--no-cache", "src", "tests", "scripts"]),
    ("pytest", ["pytest", "-p", "no:cacheprovider"]),
    ("smoke_test", ["python", "scripts/smoke_test.py"]),
    ("public_repository_scan", ["python", "scripts/check_public_repo.py"]),
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


def _notebook_gate() -> dict[str, Any]:
    records = []
    for path in sorted((REPOSITORY / "notebooks").glob("0[1-5]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        errors = []
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                ast.parse(cell.source)
            except SyntaxError as error:
                errors.append({"cell": index, "error": str(error)})
        records.append(
            {
                "notebook": path.name,
                "parse_errors": errors,
                "non_null_execution_counts": sum(
                    cell.cell_type == "code" and cell.execution_count is not None
                    for cell in notebook.cells
                ),
                "output_blocks": sum(
                    len(cell.get("outputs", []))
                    for cell in notebook.cells
                    if cell.cell_type == "code"
                ),
            }
        )
    passed = len(records) == 5 and all(
        not row["parse_errors"]
        and row["non_null_execution_counts"] == 0
        and row["output_blocks"] == 0
        for row in records
    )
    return {"passed": passed, "records": records}


def main() -> None:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    destination = (
        Path(paths["private_artifact_root"])
        / "revision331"
        / "support_aware_generation_gate"
        / "reports"
        / "repository_gates.json"
    )
    commands = [_run(name, command) for name, command in COMMANDS]
    notebook = _notebook_gate()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commands": commands,
        "public_notebooks": notebook,
        "status": (
            "passed"
            if all(row["passed"] for row in commands) and notebook["passed"]
            else "failed"
        ),
    }
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
