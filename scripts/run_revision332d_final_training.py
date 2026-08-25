#!/usr/bin/env python3
"""Run the gated Revision-3.3.2d stable full-model training workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from sage_avo.config import load_config
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.reporting import (
    generate_observability_figures,
    initialize_machine_outputs,
    update_checkpoint_tables,
    write_summary,
)
from sage_avo.experiments.manifest import file_sha256
from sage_avo.experiments.training import train_controlled_variant
import sage_avo.forward.torch_forward as torch_forward
from sage_avo.forward.torch_forward import exact_zoeppritz_pp_closed_form

import run_graph_objective_selection as graph_selection
import run_revision332_production as revision332
import run_revision332a_production as revision332a


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = revision332.PRIVATE
DATASET = revision332.DATASET
BASE_CONFIG = REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"
FINAL_CONTRACT_CONFIG = REPOSITORY / "configs" / "final_training_v00332d.yaml"
OBSERVABILITY_CONFIG = REPOSITORY / "configs" / "training_observability_v00332d.yaml"
ROOT = PRIVATE / "revision332d"
CONTRACT = ROOT / "final_training_contract.json"
FAILURE_STATES = ROOT / "former_failure_states"
OPERATOR_REPORT = ROOT / "exact_zoeppritz_validation" / "operator_validation_report.json"
TRUST_REGION_REPORT = ROOT / "residual_trust_region.json"
TRUST_REGION_SELECTION = ROOT / "residual_trust_region_selection.json"
REPLAY_EXPERIMENT = (
    PRIVATE
    / "stage_artifacts"
    / "stage04"
    / "sage_avo_s01_v00332d_eligible_only_former_failure_replay"
)
REPLAY_REPORT = ROOT / "former_failure_replay.json"
GATE_EXPERIMENT = (
    PRIVATE / "stage_artifacts" / "stage04" / "sage_avo_s01_v00332d_final_bounded_gate"
)
GATE_RUN = GATE_EXPERIMENT / "runs" / "full"
GATE_DIAGNOSTICS = ROOT / "final_bounded_gate" / "diagnostics"
GATE_FIGURES = ROOT / "final_bounded_gate" / "figures"
GATE_DECISION = ROOT / "final_bounded_gate" / "final_training_gate.json"
PRODUCTION_EXPERIMENT = (
    PRIVATE / "stage_artifacts" / "stage04" / "sage_avo_s01_v00332d_final_production"
)
PRODUCTION_RUN = PRODUCTION_EXPERIMENT / "runs" / "full"
FAILURE_TARGETS = {
    "edge_aware_contrast": 235,
    "truth_edge_matching": 537,
    "no_aux_graph_loss": 690,
}
EXPECTED_INITIALIZATION = "43cb4de81066fb6737f13d0ee87ec00b135e5790fa1629589794d62b31eff577"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _configuration() -> tuple[dict[str, Any], dict[str, Any]]:
    config, _ = revision332a._configuration()
    observability = load_config(OBSERVABILITY_CONFIG)
    observability["diagnostic_sample_manifest"] = {
        "path": str(revision332.SAMPLE_MANIFEST),
        "sha256": file_sha256(revision332.SAMPLE_MANIFEST),
    }
    config["observability"] = observability
    config["experiment"] = {
        **config["experiment"],
        "name": "sage_avo_s01_v00332d_final_stable_training",
    }
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["checkpointing"]["periodic_interval_epochs"] = 5
    config["training"]["checkpointing"]["whole_validation_every_epochs"] = 5
    trust_region_selection = (
        json.loads(TRUST_REGION_SELECTION.read_text(encoding="utf-8"))
        if TRUST_REGION_SELECTION.exists()
        else {"status": "NOT_SELECTED"}
    )
    trust_region_enabled = trust_region_selection["status"] == "SELECTED"
    if trust_region_enabled:
        trust_region = json.loads(TRUST_REGION_REPORT.read_text(encoding="utf-8"))
        if trust_region["status"] != "RESIDUAL_TRUST_REGION_GO":
            raise RuntimeError(f"Invalid residual trust-region report: {TRUST_REGION_REPORT}")
        config["training"]["residual_trust_region"] = {
            "enabled": True,
            "normalized_scales": trust_region["normalized_scales"],
            "physical_scales": trust_region["physical_scales"],
            "formula": trust_region["formula"],
            "source_report": str(TRUST_REGION_REPORT),
            "source_report_sha256": file_sha256(TRUST_REGION_REPORT),
        }
    config["capabilities"]["auxiliary_graph_smoothness"] = {
        "implemented": True,
        "enabled": False,
        "role": "diagnostic_only",
    }
    config["capabilities"]["matrix_exact_zoeppritz_training_loss"] = {
        "implemented": True,
        "enabled": True,
        "solver": "torch.linalg.solve",
    }
    config["final_training_contract"] = {
        "revision": "3.3.2d",
        "path": str(FINAL_CONTRACT_CONFIG),
        "sha256": file_sha256(FINAL_CONTRACT_CONFIG),
        "graph_smoothness_optimized": False,
        "graph_architecture_enabled": True,
        "exact_pp_operator": "complex_boundary_condition_matrix_solve",
        "trust_region_fallback": trust_region_enabled,
    }
    return config, observability


def _verify_prerequisites() -> dict[str, Any]:
    config, observability = _configuration()
    frozen = revision332._verify(observability)
    if config["training"]["loss_weights"]["structure"] != 0.0:
        raise RuntimeError("Final auxiliary graph coefficient is not zero")
    if config["training"]["graph_objective"]["mode"] != "no_aux_graph_loss":
        raise RuntimeError("Final graph objective is not disabled")
    return {
        "frozen_inputs": frozen,
        "resolved_config_sha256": _canonical_sha256(config),
        "final_contract_config_sha256": file_sha256(FINAL_CONTRACT_CONFIG),
        "observability_config_sha256": file_sha256(OBSERVABILITY_CONFIG),
    }


def prepare(_: argparse.Namespace) -> None:
    verification = _verify_prerequisites()
    contract = {
        "schema_version": 1,
        "revision": "3.3.2d",
        "status": "PREDECLARED_BEFORE_NUMERICAL_VALIDATION",
        "immutable_inputs": verification,
        "objective": {
            "graph_architecture": "full RGT-guided TransformerConv and reinjection",
            "graph_smoothness_coefficient": 0.0,
            "graph_smoothness_role": "diagnostic_only",
            "replacement_graph_loss": None,
            "non_graph_losses_and_weights": "unchanged Revision-3.3.2a schedule",
        },
        "operator": {
            "type": "exact complex 4x4 boundary-condition matrix",
            "solver": "torch.linalg.solve",
            "row_equilibration": "same nonzero multiplier on each matrix row and RHS",
            "denominator_epsilon": None,
            "approximation": None,
        },
        "former_failure_replay_batches": sorted(FAILURE_TARGETS.values()),
        "trust_region_fallback": {
            "enabled": False,
            "may_be_considered_only_if_matrix_operator_or_replay_fails": True,
        },
        "long_training_allowed_only_after": "FINAL_TRAINING_GATE_GO",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract": str(CONTRACT), "sha256": file_sha256(CONTRACT)}, indent=2))


def capture_failure_state(arguments: argparse.Namespace) -> None:
    variant = arguments.variant
    target = FAILURE_TARGETS[variant]
    capture_batch = int(arguments.capture_batch or target)
    if capture_batch < 1 or capture_batch > target:
        raise ValueError(f"capture-batch must lie in [1, {target}]")
    config, _ = graph_selection._resolved(variant)
    role = "first_nonfinite" if capture_batch == target else "trigger_input"
    destination = FAILURE_STATES / f"{variant}_{role}_batch_{capture_batch:04d}.npz"
    metadata_path = destination.with_suffix(".json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite failure-state capture: {destination}")
    run_name = f"capture_{variant}_{role}_batch_{capture_batch:04d}"
    experiment = ROOT / "legacy_failure_state_capture"
    run_directory = experiment / "runs" / run_name
    if run_directory.exists():
        raise FileExistsError(run_directory)
    FAILURE_STATES.mkdir(parents=True, exist_ok=True)
    call_count = 0
    captured = False

    def capture_operator(
        vp: torch.Tensor,
        vs: torch.Tensor,
        density: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal call_count, captured
        call_count += 1
        if call_count == capture_batch:
            np.savez_compressed(
                destination,
                vp=vp.detach().cpu().numpy(),
                vs=vs.detach().cpu().numpy(),
                density=density.detach().cpu().numpy(),
                angles_degrees=angles.detach().cpu().numpy(),
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "former_first_nonfinite_batch": target,
                        "captured_operator_input_batch": capture_batch,
                        "capture_role": role,
                        "operator_call": call_count,
                        "trajectory_operator": "legacy_closed_form",
                        "trajectory_graph_objective": config["training"][
                            "graph_objective"
                        ],
                        "trajectory_structure_coefficient": config["training"][
                            "loss_weights"
                        ]["structure"],
                        "initialization_expected_sha256": EXPECTED_INITIALIZATION,
                        "state_path": str(destination),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            captured = True
        return exact_zoeppritz_pp_closed_form(vp, vs, density, angles)

    production_operator = torch_forward.exact_zoeppritz_pp
    torch_forward.exact_zoeppritz_pp = capture_operator
    try:
        train_controlled_variant(
            repository=REPOSITORY,
            config_path=BASE_CONFIG,
            config=config,
            dataset_directory=DATASET,
            experiment_directory=experiment,
            variant="full",
            device_name=arguments.device,
            max_train_batches=capture_batch,
            max_validation_batches=1,
            run_name=run_name,
            stop_after_epoch=1,
        )
    finally:
        torch_forward.exact_zoeppritz_pp = production_operator
    if not captured:
        raise RuntimeError(
            f"Expected operator call {capture_batch}, observed only {call_count}"
        )
    print(json.dumps({"state": str(destination), "calls": call_count}, indent=2))


def derive_trust_region(_: argparse.Namespace) -> None:
    operator = json.loads(OPERATOR_REPORT.read_text(encoding="utf-8"))
    trigger_states = [
        record
        for record in operator["former_failure_states"].values()
        if record.get("capture_role") == "trigger_input"
    ]
    if not any(not record.get("finite_forward", False) for record in trigger_states):
        raise RuntimeError("Trust-region fallback is not justified by the operator audit")
    split_ids = json.loads((DATASET / "split_ids.json").read_text(encoding="utf-8"))
    normalization = json.loads((DATASET / "normalization.json").read_text(encoding="utf-8"))
    training_ids = [int(value) for value in split_ids["train"]]
    residuals: list[list[np.ndarray]] = [[], [], []]
    low_minimum = np.full(3, np.inf)
    low_maximum = np.full(3, -np.inf)
    for realization_id in training_ids:
        archive = DATASET / "realizations" / f"realization_{realization_id:07d}.npz"
        with np.load(archive) as data:
            truth = np.asarray(data["elastic"], dtype=np.float64)
            prior = np.asarray(data["low"], dtype=np.float64)
            valid = np.asarray(data["valid_mask"]) > 0
        for channel in range(3):
            residuals[channel].append((truth[channel] - prior[channel])[valid])
            low_minimum[channel] = min(low_minimum[channel], prior[channel][valid].min())
            low_maximum[channel] = max(low_maximum[channel], prior[channel][valid].max())
    values = [np.concatenate(channel) for channel in residuals]
    safety_margin = 0.10
    physical_scales = np.asarray(
        [(1.0 + safety_margin) * np.max(np.abs(channel)) for channel in values]
    )
    normalized_scales = physical_scales / np.asarray(normalization["y_std"], dtype=float)
    quantiles = (0.0, 0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99, 100.0)
    property_names = ("vp", "vs", "density")
    statistics = {}
    for name, channel, scale in zip(property_names, values, physical_scales):
        statistics[name] = {
            "samples": int(channel.size),
            "residual_percentiles": {
                str(percentile): float(value)
                for percentile, value in zip(quantiles, np.percentile(channel, quantiles))
            },
            "maximum_absolute_residual": float(np.max(np.abs(channel))),
            "physical_scale": float(scale),
            "truth_coverage_fraction": float(np.mean(np.abs(channel) < scale)),
            "maximum_fraction_of_scale": float(np.max(np.abs(channel)) / scale),
        }
    lower_bounds = low_minimum - physical_scales
    upper_bounds = low_maximum + physical_scales
    passed = (
        all(record["truth_coverage_fraction"] == 1.0 for record in statistics.values())
        and bool(np.all(lower_bounds > 0.0))
        and len(training_ids) == 70
    )
    report = {
        "schema_version": 1,
        "status": "RESIDUAL_TRUST_REGION_GO" if passed else "RESIDUAL_TRUST_REGION_NO_GO",
        "activation_reason": (
            "finite legacy trigger predictions left the supported positive elastic domain; "
            "the exact matrix boundary system is undefined for zero/negative properties"
        ),
        "formula": (
            "m_normalized = m0_normalized + s_normalized * "
            "tanh(raw_velocity / s_normalized)"
        ),
        "scale_source": (
            "maximum absolute truth-minus-2Hz-prior residual across all valid pixels "
            "of the 70 training realizations"
        ),
        "safety_margin_fraction": safety_margin,
        "training_realization_ids": training_ids,
        "training_split_ids_sha256": file_sha256(DATASET / "split_ids.json"),
        "validation_or_test_arrays_accessed": False,
        "physical_scales": physical_scales.tolist(),
        "normalized_scales": normalized_scales.tolist(),
        "prior_minimum": low_minimum.tolist(),
        "prior_maximum": low_maximum.tolist(),
        "reachable_minimum": lower_bounds.tolist(),
        "reachable_maximum": upper_bounds.tolist(),
        "truth_target_statistics": statistics,
        "direct_clipping": False,
        "truth_targets_truncated": False,
    }
    TRUST_REGION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TRUST_REGION_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


def capture_final_trigger(arguments: argparse.Namespace) -> None:
    capture_batch = int(arguments.capture_batch)
    config, _ = _configuration()
    destination = FAILURE_STATES / f"final_matrix_trigger_input_batch_{capture_batch:04d}.npz"
    metadata_path = destination.with_suffix(".json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite final trigger capture: {destination}")
    run_name = f"capture_final_matrix_trigger_batch_{capture_batch:04d}"
    experiment = ROOT / "final_failure_state_capture"
    run_directory = experiment / "runs" / run_name
    if run_directory.exists():
        raise FileExistsError(run_directory)
    call_count = 0
    captured = False
    production_operator = torch_forward.exact_zoeppritz_pp

    def capture_operator(
        vp: torch.Tensor,
        vs: torch.Tensor,
        density: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal call_count, captured
        call_count += 1
        if call_count == capture_batch:
            np.savez_compressed(
                destination,
                vp=vp.detach().cpu().numpy(),
                vs=vs.detach().cpu().numpy(),
                density=density.detach().cpu().numpy(),
                angles_degrees=angles.detach().cpu().numpy(),
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "revision": "3.3.2d",
                        "captured_operator_input_batch": capture_batch,
                        "capture_role": "final_matrix_trigger_input",
                        "graph_auxiliary_coefficient": 0.0,
                        "trust_region_report": str(TRUST_REGION_REPORT),
                        "trust_region_report_sha256": file_sha256(TRUST_REGION_REPORT),
                        "state_path": str(destination),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            captured = True
        return production_operator(vp, vs, density, angles)

    torch_forward.exact_zoeppritz_pp = capture_operator
    try:
        train_controlled_variant(
            repository=REPOSITORY,
            config_path=BASE_CONFIG,
            config=config,
            dataset_directory=DATASET,
            experiment_directory=experiment,
            variant="full",
            device_name=arguments.device,
            max_train_batches=capture_batch,
            max_validation_batches=1,
            run_name=run_name,
            stop_after_epoch=1,
        )
    finally:
        torch_forward.exact_zoeppritz_pp = production_operator
    if not captured:
        raise RuntimeError(
            f"Expected operator call {capture_batch}, observed only {call_count}"
        )
    print(json.dumps({"state": str(destination), "calls": call_count}, indent=2))


def record_trust_not_selected(_: argparse.Namespace) -> None:
    report = {
        "schema_version": 1,
        "status": "NOT_SELECTED",
        "derived_candidate": str(TRUST_REGION_REPORT),
        "derived_candidate_sha256": file_sha256(TRUST_REGION_REPORT),
        "reason": (
            "The apparent domain escape was localized to physics-ineligible mixed-batch "
            "placeholder contexts. Subsetting eligible samples before exact forward evaluation "
            "preserves the masked objective and removes the invalid operator inputs."
        ),
        "final_training_parameterization_changed": False,
        "direct_clipping": False,
    }
    TRUST_REGION_SELECTION.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def replay_failures(arguments: argparse.Namespace) -> None:
    operator = json.loads(OPERATOR_REPORT.read_text(encoding="utf-8"))
    accepted_operator_statuses = {
        "ZOEPPRITZ_MATRIX_VALIDATION_GO",
        "ZOEPPRITZ_MATRIX_VALIDATION_TRUST_REGION_REQUIRED",
    }
    if operator["status"] not in accepted_operator_statuses:
        raise RuntimeError(f"Matrix operator has not passed validation: {OPERATOR_REPORT}")
    config, _ = _configuration()
    run_directory = REPLAY_EXPERIMENT / "runs" / "final_no_aux_replay"
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite replay: {run_directory}")
    started = time.perf_counter()
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=BASE_CONFIG,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=REPLAY_EXPERIMENT,
        variant="full",
        device_name=arguments.device,
        max_train_batches=max(FAILURE_TARGETS.values()),
        max_validation_batches=1,
        run_name="final_no_aux_replay",
        stop_after_epoch=1,
        finite_state_check_batches=tuple(sorted(FAILURE_TARGETS.values())),
        abort_on_nonfinite=True,
    )
    elapsed = time.perf_counter() - started
    checks = json.loads((output / "finite_state_checks.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    rows = checks["checks"]
    passed = (
        [int(row["batch"]) for row in rows] == sorted(FAILURE_TARGETS.values())
        and all(
            bool(row[name])
            for row in rows
            for name in (
                "metrics_finite",
                "parameters_finite",
                "gradients_finite",
                "optimizer_state_finite",
            )
        )
        and checkpoint["config"]["training"]["loss_weights"]["structure"] == 0.0
    )
    report = {
        "schema_version": 1,
        "status": "FORMER_FAILURE_REPLAY_GO" if passed else "FORMER_FAILURE_REPLAY_NO_GO",
        "final_objective": "no_aux_graph_loss",
        "matrix_exact_pp": True,
        "run_directory": str(output),
        "elapsed_seconds": elapsed,
        "initialization_sha256": json.loads((output / "manifest.json").read_text())[
            "model_initialization_sha256"
        ],
        "former_failure_thresholds": FAILURE_TARGETS,
        "checks": rows,
    }
    REPLAY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPLAY_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


def train_gate(arguments: argparse.Namespace) -> None:
    replay = json.loads(REPLAY_REPORT.read_text(encoding="utf-8"))
    if replay["status"] != "FORMER_FAILURE_REPLAY_GO":
        raise RuntimeError(f"Former failure replay has not passed: {REPLAY_REPORT}")
    if GATE_RUN.exists():
        raise FileExistsError(f"Refusing to overwrite bounded gate: {GATE_RUN}")
    config, _ = _configuration()
    started = time.perf_counter()
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=BASE_CONFIG,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=GATE_EXPERIMENT,
        variant="full",
        device_name=arguments.device,
        run_name="full",
        stop_after_epoch=5,
        abort_on_nonfinite=True,
    )
    elapsed = time.perf_counter() - started
    runtime = {
        "run_directory": str(output),
        "epochs": 5,
        "wall_seconds": elapsed,
        "mean_seconds_per_epoch": elapsed / 5.0,
    }
    (ROOT / "final_bounded_gate").mkdir(parents=True, exist_ok=True)
    (ROOT / "final_bounded_gate" / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(runtime, indent=2))


def _finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return not numeric.empty and bool(np.isfinite(numeric.to_numpy()).all())


def _segmentation_values_finite(frame: pd.DataFrame) -> bool:
    """Accept undefined class scores only when that class is absent on both sides."""
    always_defined = [
        "epoch",
        "miou",
        "macro_dice",
        *(
            f"{source}_class_{class_index}_fraction"
            for class_index in range(3)
            for source in ("predicted", "true")
        ),
    ]
    if frame.empty or not _finite_frame(frame[always_defined]):
        return False
    for class_index in range(3):
        present = (
            frame[f"predicted_class_{class_index}_fraction"]
            + frame[f"true_class_{class_index}_fraction"]
        ) > 0.0
        scores = frame.loc[present, [f"class_{class_index}_iou", f"class_{class_index}_dice"]]
        if not scores.empty and not np.isfinite(scores.to_numpy()).all():
            return False
    return True


def _graph_values_finite(frame: pd.DataFrame) -> bool:
    """Validate per-layer attention and layer-1-only global mechanism summaries."""
    attention_columns = [
        "epoch",
        "layer",
        "attention_mean",
        "attention_entropy_normalized",
        "attention_concentration",
        "top_decile_attention_mass",
        "lateral_attention_fraction",
        "vertical_attention_fraction",
        "low_rgt_mismatch_attention_fraction",
        "high_avo_similarity_attention_fraction",
    ]
    if frame.empty or not _finite_frame(frame[attention_columns]):
        return False
    if set(frame.groupby("epoch").layer.apply(lambda values: tuple(sorted(values)))) != {
        (1, 2)
    }:
        return False
    global_rows = frame[frame.layer == 1]
    return _finite_frame(
        global_rows[
            [
                "epoch",
                "graph_embedding_rms",
                "graph_reinjection_velocity_rms",
                "rgt_vs_cartesian_velocity_rms",
            ]
        ]
    )


def _raw_loss_values_finite(frame: pd.DataFrame) -> bool:
    """Treat 0/0 normalization as undefined only for identically disabled terms."""
    if frame.empty or not _finite_frame(frame[["epoch", "raw_loss", "epoch_1_raw_loss"]]):
        return False
    active = frame.epoch_1_raw_loss != 0.0
    if not np.isfinite(frame.loc[active, "normalized_to_epoch_1"]).all():
        return False
    disabled = ~active
    return bool(
        (frame.loc[disabled, "raw_loss"] == 0.0).all()
        and frame.loc[disabled, "normalized_to_epoch_1"].isna().all()
    )


def _copy_live_epoch_diagnostics() -> None:
    """Copy append-only live logs into the separate immutable gate report."""
    live = GATE_RUN / "diagnostics"
    filenames = (
        "training_statistics.csv",
        "raw_loss_components.csv",
        "weighted_loss_components.csv",
        "physics_eligibility_statistics.csv",
    )
    for filename in filenames:
        source = live / filename
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing live gate diagnostic: {source}")
        shutil.copy2(source, GATE_DIAGNOSTICS / filename)


def _gate_evidence() -> dict[str, Any]:
    training = pd.read_csv(GATE_RUN / "training_log.csv")
    raw = pd.read_csv(GATE_DIAGNOSTICS / "raw_loss_components.csv")
    gradients = pd.read_csv(GATE_DIAGNOSTICS / "gradient_contributions.csv")
    cosines = pd.read_csv(GATE_DIAGNOSTICS / "gradient_cosines.csv")
    physics = pd.read_csv(GATE_DIAGNOSTICS / "physics_floor_diagnostics.csv")
    graph = pd.read_csv(GATE_DIAGNOSTICS / "graph_learning_summary.csv")
    whole = pd.read_csv(GATE_DIAGNOSTICS / "whole_realization_metrics.csv")
    segmentation = pd.read_csv(GATE_DIAGNOSTICS / "segmentation_metrics.csv")
    elastic_physics = cosines[
        (cosines.objective_a == "elastic_flow") & (cosines.objective_b == "physics")
    ].sort_values("epoch")
    graph_gradients = gradients[
        (gradients.parameter_group == "graph_branch")
        & (gradients.objective != "structure")
        & (gradients.epoch.isin((1, 5)))
    ]
    graph_activity = graph[graph.epoch.isin((1, 5))]
    whole_summary = (
        whole.groupby(["epoch", "property"])
        .agg(rmse=("rmse", "mean"), prior_rmse=("prior_rmse", "mean"))
        .reset_index()
    )
    physics_sorted = physics.sort_values("epoch")
    layer_one_graph_activity = graph_activity[graph_activity.layer == 1]
    reinjection = layer_one_graph_activity.graph_reinjection_velocity_rms.abs()
    graph_gradient_by_epoch = graph_gradients.groupby("epoch").raw_gradient_norm.sum()
    epoch_training = training.drop(
        columns=[
            column for column in training if column.startswith("whole_validation_")
        ]
    )
    essential_values_finite = all(
        _finite_frame(frame)
        for frame in (
            epoch_training,
            gradients[
                ["epoch", "raw_gradient_norm", "weighted_gradient_norm"]
            ],
            physics[
                [
                    "epoch",
                    "prior_noiseless",
                    "prediction_noiseless",
                    "truth_noiseless_operator_floor",
                    "truth_noisy_observation_floor",
                ]
            ],
            whole[["epoch", "rmse", "mae", "r2", "correlation", "ssim", "prior_rmse"]],
        )
    ) and all(
        (
            _raw_loss_values_finite(raw),
            _graph_values_finite(graph_activity),
            _segmentation_values_finite(segmentation),
        )
    )
    conditions = {
        "five_epochs_completed": training.epoch.tolist() == [1, 2, 3, 4, 5],
        "all_training_and_diagnostic_values_finite": essential_values_finite,
        "elastic_physics_not_severely_conflicting": (
            len(elastic_physics) >= 2
            and bool((elastic_physics.cosine_similarity > -0.30).all())
        ),
        "physics_residual_improved": (
            len(physics_sorted) >= 2
            and float(physics_sorted.prediction_noiseless.iloc[-1])
            < float(physics_sorted.prediction_noiseless.iloc[0])
        ),
        "segmentation_finite": _segmentation_values_finite(segmentation),
        "graph_branch_nonzero_aligned_gradient": (
            set(graph_gradient_by_epoch.index.astype(int)) == {1, 5}
            and bool((graph_gradient_by_epoch > 0.0).all())
        ),
        "attention_finite": _graph_values_finite(graph_activity),
        "graph_reinjection_active": bool(
            len(reinjection) >= 2 and (reinjection > np.finfo(float).eps).all()
        ),
        "auxiliary_graph_weight_zero": bool((training.structure_weight == 0.0).all()),
    }
    return {
        "conditions": conditions,
        "elastic_physics_cosines": elastic_physics.to_dict(orient="records"),
        "whole_realization_vs_prior": whole_summary.to_dict(orient="records"),
        "physics_evolution": physics_sorted.to_dict(orient="records"),
        "graph_aligned_gradient_rows": graph_gradients.to_dict(orient="records"),
        "graph_mechanism": graph_activity.to_dict(orient="records"),
    }


def diagnose_gate(arguments: argparse.Namespace) -> None:
    initialize_machine_outputs(GATE_DIAGNOSTICS)
    _copy_live_epoch_diagnostics()
    for epoch in (1, 5):
        checkpoint = GATE_RUN / "diagnostic_checkpoints" / f"epoch_{epoch:04d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        analyze_checkpoint(
            checkpoint_path=checkpoint,
            dataset_directory=DATASET,
            run_directory=GATE_RUN,
            sample_manifest_path=revision332.SAMPLE_MANIFEST,
            output_directory=GATE_DIAGNOSTICS,
            device=arguments.device,
            include_whole_realizations=True,
        )
    update_checkpoint_tables(GATE_RUN, GATE_DIAGNOSTICS)
    figures = generate_observability_figures(GATE_DIAGNOSTICS, GATE_FIGURES)
    evidence = _gate_evidence()
    passed = all(evidence["conditions"].values())
    decision = {
        "schema_version": 1,
        "decision": "FINAL_TRAINING_GATE_GO" if passed else "FINAL_TRAINING_GATE_NO_GO",
        "trust_region_fallback_used": False,
        "optimized_graph_auxiliary_loss": False,
        "full_rgt_transformerconv_architecture": True,
        "evidence": evidence,
        "runtime": json.loads(
            (ROOT / "final_bounded_gate" / "runtime.json").read_text(encoding="utf-8")
        ),
        "figures": figures,
    }
    GATE_DECISION.parent.mkdir(parents=True, exist_ok=True)
    GATE_DECISION.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_summary(GATE_DIAGNOSTICS, decision["decision"], decision)
    print(json.dumps(decision, indent=2))
    if not passed:
        raise SystemExit(1)


def production(arguments: argparse.Namespace) -> None:
    decision = json.loads(GATE_DECISION.read_text(encoding="utf-8"))
    if decision["decision"] != "FINAL_TRAINING_GATE_GO":
        raise RuntimeError(f"Final bounded gate has not passed: {GATE_DECISION}")
    config, _ = _configuration()
    run_directory = PRODUCTION_RUN
    resume = None
    if arguments.resume:
        resume = run_directory / "last.pt"
        if not resume.exists():
            raise FileNotFoundError(resume)
    elif run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite clean production: {run_directory}")
    if arguments.stop_after_epoch > 100:
        raise ValueError("Epoch 100 is a mandatory review pause")
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=BASE_CONFIG,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=PRODUCTION_EXPERIMENT,
        variant="full",
        device_name=arguments.device,
        run_name="full",
        resume_from=resume,
        stop_after_epoch=arguments.stop_after_epoch,
        abort_on_nonfinite=True,
    )
    print(output)


def diagnose_production(arguments: argparse.Namespace) -> None:
    """Update the cumulative presentation package without modifying training state."""
    decision = json.loads(GATE_DECISION.read_text(encoding="utf-8"))
    if decision["decision"] != "FINAL_TRAINING_GATE_GO":
        raise RuntimeError(f"Final bounded gate has not passed: {GATE_DECISION}")
    epoch = int(arguments.epoch)
    if epoch < 5 or epoch > 100 or epoch % 5:
        raise ValueError("Production diagnostic epoch must be a multiple of 5 in [5, 100]")
    checkpoint = PRODUCTION_RUN / "diagnostic_checkpoints" / f"epoch_{epoch:04d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output = PRODUCTION_RUN / "diagnostics"
    figures_directory = PRODUCTION_RUN / "diagnostic_figures"
    initialize_machine_outputs(output)
    report = analyze_checkpoint(
        checkpoint_path=checkpoint,
        dataset_directory=DATASET,
        run_directory=PRODUCTION_RUN,
        sample_manifest_path=revision332.SAMPLE_MANIFEST,
        output_directory=output,
        device=arguments.device,
        include_whole_realizations=True,
    )
    update_checkpoint_tables(PRODUCTION_RUN, output)
    figures = generate_observability_figures(output, figures_directory)
    summary = {
        "schema_version": 1,
        "status": "PRODUCTION_DIAGNOSTIC_COMPLETE",
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "model_state_unchanged": report["model_state_unchanged"],
        "optimizer_loaded_or_modified": report["optimizer_loaded_or_modified"],
        "cumulative_figure_count": len(figures),
        "figures_directory": str(figures_directory),
        "epoch_100_review_required": epoch == 100,
    }
    write_summary(output, "production diagnostic checkpoint complete", summary)
    (output / f"production_diagnostic_epoch_{epoch:04d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.set_defaults(function=prepare)

    capture = commands.add_parser("capture-failure-state")
    capture.add_argument("--variant", required=True, choices=tuple(FAILURE_TARGETS))
    capture.add_argument("--capture-batch", type=int)
    capture.add_argument("--device", default="cuda")
    capture.set_defaults(function=capture_failure_state)

    trust = commands.add_parser("derive-trust-region")
    trust.set_defaults(function=derive_trust_region)

    final_capture = commands.add_parser("capture-final-trigger")
    final_capture.add_argument("--capture-batch", type=int, required=True)
    final_capture.add_argument("--device", default="cuda")
    final_capture.set_defaults(function=capture_final_trigger)

    trust_selection = commands.add_parser("record-trust-not-selected")
    trust_selection.set_defaults(function=record_trust_not_selected)

    replay = commands.add_parser("replay-former-failures")
    replay.add_argument("--device", default="cuda")
    replay.set_defaults(function=replay_failures)

    gate = commands.add_parser("train-gate")
    gate.add_argument("--device", default="cuda")
    gate.set_defaults(function=train_gate)

    diagnose = commands.add_parser("diagnose-gate")
    diagnose.add_argument("--device", default="cuda")
    diagnose.set_defaults(function=diagnose_gate)

    production_parser = commands.add_parser("production")
    production_parser.add_argument("--device", default="cuda")
    production_parser.add_argument("--resume", action="store_true")
    production_parser.add_argument("--stop-after-epoch", type=int, default=5)
    production_parser.set_defaults(function=production)

    production_diagnostics = commands.add_parser("diagnose-production")
    production_diagnostics.add_argument("--epoch", type=int, required=True)
    production_diagnostics.add_argument("--device", default="cuda")
    production_diagnostics.set_defaults(function=diagnose_production)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
