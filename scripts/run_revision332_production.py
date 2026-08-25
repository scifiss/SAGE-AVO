#!/usr/bin/env python3
"""Prepare, train, and diagnose Revision-3.3.2 without changing its science."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from sage_avo.config import load_config
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.contracts import (
    build_diagnostic_sample_manifest,
    verify_frozen_revision331_inputs,
)
from sage_avo.diagnostics.reporting import (
    generate_observability_figures,
    health_gate,
    initialize_machine_outputs,
    update_checkpoint_tables,
    write_summary,
)
from sage_avo.experiments.manifest import file_sha256
from sage_avo.experiments.training import train_controlled_variant


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(
    os.environ.get(
        "SAGE_AVO_PRIVATE_ARTIFACT_ROOT",
        REPOSITORY.parent / "SAGE_AVO_private_artifacts",
    )
)
DATASET = (
    PRIVATE / "stage_artifacts" / "stage03" / "ds_v00331_production100_support_aware" / "dataset"
)
CONFIG_PATH = REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"
OBSERVABILITY_PATH = REPOSITORY / "configs" / "training_observability_v00332.yaml"
INSTRUMENTATION = PRIVATE / "revision332" / "training_instrumentation"
SAMPLE_MANIFEST = INSTRUMENTATION / "diagnostic_sample_manifest.json"
PRODUCTION_EXPERIMENT = (
    PRIVATE / "stage_artifacts" / "stage04" / "sage_avo_s01_v00332_observable_production"
)


def _configuration() -> tuple[dict, dict]:
    config = load_config(CONFIG_PATH)
    observability = load_config(OBSERVABILITY_PATH)
    observability["diagnostic_sample_manifest"] = {
        "path": str(SAMPLE_MANIFEST),
        "sha256": file_sha256(SAMPLE_MANIFEST) if SAMPLE_MANIFEST.exists() else None,
    }
    config["observability"] = observability
    return config, observability


def _verify(observability: dict) -> dict:
    return verify_frozen_revision331_inputs(
        dataset_directory=DATASET,
        private_artifact_root=PRIVATE,
        observability_config=observability,
    )


def prepare(_: argparse.Namespace) -> None:
    _, observability = _configuration()
    verification = _verify(observability)
    if SAMPLE_MANIFEST.exists():
        existing = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
        candidate = build_diagnostic_sample_manifest(
            dataset_directory=DATASET,
            observability_config=observability,
        )
        for value in (existing, candidate):
            value.pop("created_utc", None)
        if existing != candidate:
            raise RuntimeError("Frozen diagnostic sample selection no longer reproduces")
        status = "existing immutable diagnostic sample manifest verified"
    else:
        SAMPLE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        build_diagnostic_sample_manifest(
            dataset_directory=DATASET,
            observability_config=observability,
            destination=SAMPLE_MANIFEST,
        )
        status = "diagnostic sample manifest created"
    report = {
        "status": status,
        "frozen_inputs": verification,
        "diagnostic_sample_manifest": {
            "path": str(SAMPLE_MANIFEST),
            "sha256": file_sha256(SAMPLE_MANIFEST),
        },
    }
    patch_index = pd.read_csv(DATASET / "patch_index.csv")
    eligibility = {}
    for split in ("train", "validation", "test"):
        rows = patch_index[patch_index["split"] == split]
        eligible = int(rows["physics_eligible"].sum())
        eligibility[split] = {
            "patches": len(rows),
            "native_50x100_physics_eligible": eligible,
            "static_eligible_fraction": eligible / len(rows),
        }
    contract = {
        "schema_version": 1,
        "revision": "3.3.2",
        "scientific_methodology_changed": False,
        "frozen_inputs": verification,
        "diagnostic_sample_manifest": report["diagnostic_sample_manifest"],
        "physics_eligibility": eligibility,
        "mixed_batch_reduction": (
            "physics_mask excludes ineligible pixels inside a mixed batch; "
            "fully inactive optimizer/evaluation steps contribute zero to the "
            "ordinary epoch mean and are therefore reported separately"
        ),
        "diagnostic_process_contract": (
            "checkpoint reload only; no live optimizer, scheduler, RNG, model gradient, "
            "or AMP-scaler state is accessed"
        ),
        "scheduled_checkpoints": observability["diagnostic_checkpoints"]["epochs"],
        "health_gate_epochs": observability["diagnostic_checkpoints"]["health_gate_epochs"],
    }
    contract_path = INSTRUMENTATION / "instrumentation_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report["instrumentation_contract"] = {
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
    }
    print(json.dumps(report, indent=2))


def train(arguments: argparse.Namespace) -> None:
    config, observability = _configuration()
    _verify(observability)
    if not SAMPLE_MANIFEST.exists():
        raise RuntimeError("Run the prepare command before training")
    resume = None
    run_directory = PRODUCTION_EXPERIMENT / "runs" / "full"
    if arguments.resume:
        resume = run_directory / "last.pt"
        if not resume.exists():
            raise FileNotFoundError(resume)
    stop_after_epoch = arguments.stop_after_epoch
    if stop_after_epoch is None:
        if resume is None:
            stop_after_epoch = 5
        else:
            import torch

            completed = int(torch.load(resume, map_location="cpu", weights_only=False)["epoch"])
            if completed in {5, 10}:
                health_path = (
                    run_directory / "diagnostics" / f"multi_objective_health_epoch{completed}.json"
                )
                if not health_path.exists():
                    raise RuntimeError(
                        f"Run separate diagnostics for epoch {completed} before resuming"
                    )
                health = json.loads(health_path.read_text(encoding="utf-8"))
                if health["stop_before_continuing"]:
                    raise RuntimeError(
                        f"Epoch-{completed} health gate requires STOP; inspect {health_path}"
                    )
            stop_after_epoch = 10 if completed < 10 else int(config["training"]["epochs"])
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=CONFIG_PATH,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=PRODUCTION_EXPERIMENT,
        variant="full",
        device_name=arguments.device,
        max_train_batches=arguments.max_train_batches,
        max_validation_batches=arguments.max_validation_batches,
        run_name="full",
        resume_from=resume,
        stop_after_epoch=stop_after_epoch,
    )
    print(output)


def diagnose(arguments: argparse.Namespace) -> None:
    _, observability = _configuration()
    _verify(observability)
    checkpoint = Path(arguments.checkpoint)
    run_directory = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "diagnostic_checkpoints"
        else checkpoint.parent
    )
    output = run_directory / "diagnostics"
    initialize_machine_outputs(output)
    report = analyze_checkpoint(
        checkpoint_path=checkpoint,
        dataset_directory=DATASET,
        run_directory=run_directory,
        sample_manifest_path=SAMPLE_MANIFEST,
        output_directory=output,
        device=arguments.device,
        maximum_patches=arguments.maximum_patches,
        flow_steps=arguments.flow_steps,
        include_whole_realizations=not arguments.skip_whole_realizations,
    )
    update_checkpoint_tables(run_directory, output)
    figures = generate_observability_figures(output, run_directory / "diagnostic_figures")
    epoch = int(report["epoch"])
    gate = health_gate(output, observability, epoch) if epoch in {5, 10} else None
    write_summary(
        output,
        "checkpoint diagnostics complete",
        {
            "checkpoint epoch": epoch,
            "model state unchanged": report["model_state_unchanged"],
            "optimizer loaded or modified": report["optimizer_loaded_or_modified"],
            "figure products": len(figures),
            "health gate": gate if gate is not None else "not scheduled at this epoch",
        },
    )
    print(json.dumps({"report": report, "figures": figures, "health_gate": gate}, indent=2))


def sanity(arguments: argparse.Namespace) -> None:
    config, observability = _configuration()
    _verify(observability)
    if not SAMPLE_MANIFEST.exists():
        raise RuntimeError("Run the prepare command before the sanity gate")
    root = INSTRUMENTATION / "short_execution_gate_final"
    if root.exists():
        if not arguments.replace:
            raise FileExistsError(f"Refusing to replace prior short gate: {root}")
        shutil.rmtree(root)
    config["training"]["sample_steps_validation"] = 2
    config["instrumentation_gate_override"] = {
        "scope": "short execution test only",
        "sample_steps_validation": 2,
        "production_configuration_unchanged": True,
        "scientific_performance_interpretation": False,
    }
    run_directory = train_controlled_variant(
        repository=REPOSITORY,
        config_path=CONFIG_PATH,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=root,
        variant="full",
        device_name=arguments.device,
        max_train_batches=1,
        max_validation_batches=1,
        run_name="full",
        stop_after_epoch=1,
    )
    output = run_directory / "diagnostics"
    initialize_machine_outputs(output)
    checkpoint = run_directory / "diagnostic_checkpoints" / "epoch_0001.pt"
    report = analyze_checkpoint(
        checkpoint_path=checkpoint,
        dataset_directory=DATASET,
        run_directory=run_directory,
        sample_manifest_path=SAMPLE_MANIFEST,
        output_directory=output,
        device=arguments.device,
        maximum_patches=1,
        flow_steps=2,
        include_whole_realizations=False,
    )
    update_checkpoint_tables(run_directory, output)
    figures = generate_observability_figures(output, root / "diagnostic_figures")
    write_summary(
        output,
        "short instrumentation execution passed",
        {
            "optimizer steps": 1,
            "validation batches": 1,
            "diagnostic patches": 1,
            "diagnostic flow steps": 2,
            "model state unchanged by diagnostics": report["model_state_unchanged"],
            "figures generated": len(figures),
            "not a scientific performance result": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "short instrumentation execution passed",
                "run_directory": str(run_directory),
                "diagnostic_report": report,
                "figures": figures,
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.set_defaults(function=prepare)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--stop-after-epoch", type=int)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-validation-batches", type=int)
    train_parser.set_defaults(function=train)

    diagnostic_parser = commands.add_parser("diagnose")
    diagnostic_parser.add_argument("--checkpoint", required=True)
    diagnostic_parser.add_argument("--device", default="cpu")
    diagnostic_parser.add_argument("--maximum-patches", type=int)
    diagnostic_parser.add_argument("--flow-steps", type=int)
    diagnostic_parser.add_argument("--skip-whole-realizations", action="store_true")
    diagnostic_parser.set_defaults(function=diagnose)

    sanity_parser = commands.add_parser("sanity")
    sanity_parser.add_argument("--device", default="cpu")
    sanity_parser.add_argument("--replace", action="store_true")
    sanity_parser.set_defaults(function=sanity)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
