#!/usr/bin/env python3
"""Execute private Revision-3.2 validation copies of notebooks 01--05.

The repository notebooks remain untouched.  Notebook 01 writes to an isolated
Stage-01 clone; notebooks 02--05 consume the bounded v0032 artifacts and capped
CUDA-sanity checkpoint through the existing validation-root contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from nbclient import NotebookClient
import nbformat

from sage_avo.config import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATION_NAME = "v0032_validation8_fluid_provenance"


def _locations() -> tuple[Path, Path]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    validation = private / "revision32" / VALIDATION_NAME
    return Path(paths["work_data_root"]), validation


def _prepare_notebook01_clone(work_data_root: Path, validation_root: Path) -> Path:
    clone_root = validation_root / "notebook01_private_work_data"
    shutil.copytree(
        work_data_root / "s01data",
        clone_root / "s01data",
        dirs_exist_ok=True,
    )
    return clone_root


def _patch_private_copy(
    notebook: nbformat.NotebookNode,
    name: str,
    notebook01_work_root: Path,
) -> nbformat.NotebookNode:
    if name.startswith("01_"):
        original = "paths = load_config(paths_path)"
        replacement = (
            "paths = load_config(paths_path)\n"
            f"paths['work_data_root'] = {str(notebook01_work_root)!r}"
        )
        for cell in notebook.cells:
            if cell.cell_type == "code" and original in cell.source:
                cell.source = cell.source.replace(original, replacement, 1)
                break
        else:
            raise RuntimeError("Could not redirect Notebook 01 work_data_root")
        project_bundle_old = (
            'project_bundle = ROOT / "data" / "avo" / "s01" / "bundles" / layout.version'
        )
        project_bundle_new = (
            'project_bundle = layout.dataset_root / "private_project_bundle" / layout.version'
        )
        manifest_old = (
            '"dataset_manifest": str((layout.bundles / "manifest.json").relative_to(ROOT / "data")),'
        )
        manifest_new = '"dataset_manifest": str(layout.bundles / "manifest.json"),'
        for cell in notebook.cells:
            if cell.cell_type == "code" and project_bundle_old in cell.source:
                cell.source = cell.source.replace(project_bundle_old, project_bundle_new, 1)
                cell.source = cell.source.replace(manifest_old, manifest_new, 1)
                break
        else:
            raise RuntimeError("Could not redirect Notebook 01 project bundle")
    if name.startswith(("04_", "05_")):
        old = (
            'experiment_dir = validation_root / "stage04" / '
            '"sage_avo_s01_v003_stage01v003_validation8"'
        )
        new = 'experiment_dir = validation_root / "stage04" / workflow["experiment"]["name"]'
        for cell in notebook.cells:
            if cell.cell_type == "code" and old in cell.source:
                cell.source = cell.source.replace(old, new)
                break
        else:
            raise RuntimeError(f"Could not generalize validation experiment path in {name}")
    notebook.metadata["revision32_validation"] = {
        "validation_name": VALIDATION_NAME,
        "public_source_notebook": str(REPOSITORY / "notebooks" / name),
        "private_execution_only": True,
        "public_notebook_modified": False,
    }
    return notebook


def main() -> None:
    work_data_root, validation_root = _locations()
    destination = validation_root / "executed_notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    notebook01_work_root = _prepare_notebook01_clone(work_data_root, validation_root)
    environment = {
        "SAGE_AVO_REVISION3_VALIDATION_ROOT": str(validation_root),
        "SAGE_AVO_REUSE_STAGE02": "1",
        "SAGE_AVO_STAGE02_LIMIT": "8",
        "SAGE_AVO_STAGE02_WORKERS": "1",
        "SAGE_AVO_RUN_PRODUCTION_TRAINING": "0",
        "SAGE_AVO_RUN_EVALUATION": "0",
        "SAGE_AVO_RUN_FIELD_INFERENCE": "0",
        "SAGE_AVO_RUN_FIELD_SENSITIVITY": "0",
    }
    prior = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    records = []
    try:
        for path in sorted((REPOSITORY / "notebooks").glob("0[1-5]_*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            notebook = _patch_private_copy(notebook, path.name, notebook01_work_root)
            client = NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(REPOSITORY)}},
                allow_errors=False,
            )
            executed = client.execute()
            output = destination / path.name.replace(
                ".ipynb", ".v0032_validation.executed.ipynb"
            )
            nbformat.write(executed, output)
            record = {
                "source": path.name,
                "executed": output.name,
                "code_cells": sum(cell.cell_type == "code" for cell in executed.cells),
                "executed_code_cells": sum(
                    cell.cell_type == "code" and cell.execution_count is not None
                    for cell in executed.cells
                ),
                "error_outputs": sum(
                    item.output_type == "error"
                    for cell in executed.cells
                    if cell.cell_type == "code"
                    for item in cell.get("outputs", [])
                ),
            }
            records.append(record)
            print(json.dumps(record))
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    report = {
        "status": "complete" if len(records) == 5 and all(row["error_outputs"] == 0 for row in records) else "failed",
        "validation_root": str(validation_root),
        "notebook01_work_data_clone": str(notebook01_work_root),
        "public_notebooks_modified": False,
        "records": records,
    }
    (destination / "execution_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
