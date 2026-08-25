#!/usr/bin/env python3
"""Run corrected Revision-3.3.2a training without touching the interrupted run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import run_revision332_production as revision332


PRIVATE = revision332.PRIVATE
CORRECTION = PRIVATE / "revision332a" / "masked_physics_correction"
CLEAN_PRODUCTION_EXPERIMENT = (
    PRIVATE
    / "stage_artifacts"
    / "stage04"
    / "sage_avo_s01_v00332a_masked_physics_corrected_production"
)
EPOCH1_REPLAY_EXPERIMENT = CORRECTION / "corrected_epoch1_replay"
APPROVED_POINTER = PRIVATE / "source_freezes" / "revision332a_masked_physics_approved.json"
_REVISION332_CONFIGURATION = revision332._configuration


def _configuration() -> tuple[dict, dict]:
    config, observability = _REVISION332_CONFIGURATION()
    if APPROVED_POINTER.exists():
        pointer = json.loads(APPROVED_POINTER.read_text(encoding="utf-8"))
        if pointer.get("status") != "TRAINING_BUGFIX_GO":
            raise RuntimeError(f"Invalid Revision-3.3.2a source pointer: {APPROVED_POINTER}")
        config["source_snapshot"] = {
            "revision": "3.3.2a",
            "snapshot_id": pointer["snapshot_id"],
            "archive_sha256": pointer["archive_sha256"],
            "pointer": str(APPROVED_POINTER),
        }
        config["numerical_corrections"] = {
            "masked_physics_reduction": (
                "physics-ineligible residuals are selected out before squaring; "
                "the declared objective, coefficients, and denominator are unchanged"
            )
        }
    return config, observability


def _run_train(arguments: argparse.Namespace, experiment: Path, *, clean_only: bool) -> None:
    run_directory = experiment / "runs" / "full"
    if clean_only and not arguments.resume and run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite corrected run: {run_directory}")
    original_experiment = revision332.PRODUCTION_EXPERIMENT
    original_configuration: Callable[[], tuple[dict, dict]] = revision332._configuration
    try:
        revision332.PRODUCTION_EXPERIMENT = experiment
        revision332._configuration = _configuration
        revision332.train(arguments)
    finally:
        revision332.PRODUCTION_EXPERIMENT = original_experiment
        revision332._configuration = original_configuration


def train(arguments: argparse.Namespace) -> None:
    _run_train(arguments, CLEAN_PRODUCTION_EXPERIMENT, clean_only=True)


def replay_epoch1(arguments: argparse.Namespace) -> None:
    arguments.resume = False
    arguments.stop_after_epoch = 1
    arguments.max_train_batches = None
    arguments.max_validation_batches = None
    _run_train(arguments, EPOCH1_REPLAY_EXPERIMENT, clean_only=True)


def diagnose(arguments: argparse.Namespace) -> None:
    revision332.diagnose(arguments)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--stop-after-epoch", type=int)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-validation-batches", type=int)
    train_parser.set_defaults(function=train)

    replay_parser = commands.add_parser("replay-epoch1")
    replay_parser.add_argument("--device", default="cuda")
    replay_parser.set_defaults(function=replay_epoch1)

    diagnostic_parser = commands.add_parser("diagnose")
    diagnostic_parser.add_argument("--checkpoint", required=True)
    diagnostic_parser.add_argument("--device", default="cpu")
    diagnostic_parser.add_argument("--maximum-patches", type=int)
    diagnostic_parser.add_argument("--flow-steps", type=int)
    diagnostic_parser.add_argument("--skip-whole-realizations", action="store_true")
    diagnostic_parser.set_defaults(function=diagnose)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
