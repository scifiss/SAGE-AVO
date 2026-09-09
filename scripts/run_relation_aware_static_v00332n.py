#!/usr/bin/env python3
"""Run bounded relation-aware static-GNN reproducibility experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from run_development_diagnostics_v00332j import (
    _completed_epoch,
    canonical_hash,
    file_sha256,
    state_sha256,
)
from sage_avo.config import load_config, seed_everything
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint, load_fixed_batch
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.runtime import print_torch_runtime, select_torch_device


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332n.yaml"
CONDITIONS = ("unbiased_elastic_gnn", "normal_avo_only")


def _roots() -> tuple[dict[str, Any], Path, Path, Path]:
    contract = load_config(CONTRACT_PATH)
    private = Path(load_config(REPOSITORY / "configs" / "paths.yaml")["private_artifact_root"])
    dataset = (
        private
        / "stage_artifacts"
        / "stage03"
        / contract["immutable_dataset"]
        / "dataset"
    )
    experiment = private / "stage_artifacts" / "stage04" / contract["experiment_name"]
    source = (
        private
        / "stage_artifacts"
        / "stage04"
        / contract["frozen_source"]["experiment_name"]
    )
    return contract, dataset, experiment, source


def _resolved_config(
    contract: dict[str, Any], *, seed: int, condition: str
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    budget = contract["fixed_patch_budget"]
    graph = contract["experimental_graph"]
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = str(contract["experiment_name"])
    config["experiment"]["seed"] = int(seed)
    config["training"]["epochs"] = int(budget["epochs"])
    config["training"]["batch_size"] = int(budget["batch_size"])
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"].update(enabled=False, weight=0.0)
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"].update(enabled=False, guidance_scale=0.0)
    config["model"]["experimental_graph"] = {
        "mode": str(graph["mode"]),
        "normal_lateral_shift_samples": int(graph["normal_lateral_shift_samples"]),
        "relation_candidates": int(graph["relation_candidates"]),
        "structural_prior_initial_strengths": graph[
            "segmentation_structural_prior_initial_strengths"
        ],
        "elastic_structural_prior_initial_strengths": graph[
            "elastic_structural_prior_initial_strengths"
        ],
        "elastic_structural_prior_component_mask": graph["conditions"][condition][
            "elastic_structural_prior_component_mask"
        ],
    }
    config["capabilities"]["normal_avo_component_mask"] = {
        "implemented": True,
        "enabled": condition == "normal_avo_only",
        "diagnostic_only": True,
        "mask": graph["conditions"][condition]["elastic_structural_prior_component_mask"],
    }
    config["observability"] = load_config(
        REPOSITORY / "configs" / contract["observability_config"]
    )
    config["observability"]["revision"] = f"{contract['revision']}-{condition}-seed{seed}"
    config["observability"]["scientific_methodology_changed"] = True
    diagnostics = config["observability"]["diagnostics"]
    diagnostics["deterministic_time"] = float(
        contract["diagnostic_probes"]["deterministic_time"]
    )
    diagnostics["flow_integration_steps"] = int(
        contract["diagnostic_probes"]["flow_integration_steps"]
    )
    return config


def _index_hash(values: list[int]) -> str:
    return canonical_hash([int(value) for value in values])


def _verify_source(contract: dict[str, Any], source: Path) -> dict[str, Any]:
    expected = contract["frozen_source"]
    selection = source / "fixed_patch_selection.json"
    probes = source / "fixed_graph_probe_samples.json"
    initial = source / "initial_model_state.pt"
    for path, field in (
        (selection, "expected_selection_file_sha256"),
        (probes, "expected_probe_file_sha256"),
        (initial, "expected_seed12345_initial_state_file_sha256"),
    ):
        if not path.exists() or file_sha256(path) != expected[field]:
            raise RuntimeError(f"Frozen source mismatch: {path}")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if (
        len(selected["train_indices"]) != int(expected["expected_train_patch_count"])
        or len(selected["validation_indices"])
        != int(expected["expected_validation_patch_count"])
        or _index_hash(selected["train_indices"]) != expected["expected_train_indices_sha256"]
        or _index_hash(selected["validation_indices"])
        != expected["expected_validation_indices_sha256"]
    ):
        raise RuntimeError("Frozen patch selection does not match the declared contract")
    source_state = torch.load(initial, map_location="cpu", weights_only=True)
    if state_sha256(source_state) != expected["expected_seed12345_initial_model_state_sha256"]:
        raise RuntimeError("Frozen seed-12345 initial model state mismatch")
    return {
        "selection_sha256": file_sha256(selection),
        "probe_sha256": file_sha256(probes),
        "seed12345_initial_state_file_sha256": file_sha256(initial),
        "seed12345_initial_model_state_sha256": state_sha256(source_state),
    }


def _phase0_audit() -> dict[str, Any]:
    return {
        "tangential_edges": (
            "Bidirectional adjacent-trace candidate edges to the two smallest absolute-RGT "
            "mismatches within +/-3 rows."
        ),
        "normal_edges": (
            "Bidirectional one-row-deeper candidate edges to the two largest absolute-RGT-change "
            "per-distance candidates within +/-1 columns."
        ),
        "rgt_topology_role": "RGT directly selects both relation neighborhoods before attention.",
        "edge_directionality": "Every selected relation edge is explicitly made bidirectional.",
        "node_features": [
            "CNN conditional-flow token",
            "near",
            "mid",
            "far",
            "Shuey intercept",
            "Shuey gradient",
            "three-band curvature",
        ],
        "edge_attributes_shared_by_relations": [
            "AVO-gradient affinity",
            "RGT affinity",
            "signed row offset",
            "signed column offset",
        ],
        "relation_parameters": "Different tangential and normal TransformerConv stacks.",
        "attention_normalization": (
            "Same destination-softmax algorithm but normalized independently inside each relation."
        ),
        "attention_bias": (
            "Optional relation-specific additive logit bias; segmentation and elastic streams can "
            "receive different biases while sharing relation operators."
        ),
        "elastic_reinjection": (
            "Node-wise local/tangential/normal fusion is reshaped and added to the CNN feature map "
            "before the Vp/Vs/density velocity decoder."
        ),
        "segmentation_path": (
            "A separately fused graph activation is reshaped and consumed by the segmentation head."
        ),
        "tangential_rgt_redundancy_answer": (
            "Yes, largely redundant in the tested architecture: RGT already defines tangential "
            "connectivity, and the completed scalar/component sweeps found negligible added value."
        ),
    }


def _initial_state_path(experiment: Path, seed: int) -> Path:
    return experiment / "initial_states" / f"seed_{seed}.pt"


def _run_path(experiment: Path, seed: int, condition: str) -> Path:
    return experiment / "runs" / f"seed_{seed}_{condition}"


def prepare(_: argparse.Namespace) -> None:
    contract, dataset, experiment, source = _roots()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    contract_output = experiment / "relation_aware_experiment_contract.json"
    if contract_output.exists():
        raise FileExistsError("v00332n is already prepared; refusing to replace its contract")
    verification = _verify_source(contract, source)
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "initial_states").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        source / "fixed_patch_selection.json", experiment / "fixed_patch_selection.json"
    )
    shutil.copyfile(
        source / "fixed_graph_probe_samples.json",
        experiment / "fixed_graph_probe_samples.json",
    )
    source_seed_state = torch.load(
        source / "initial_model_state.pt", map_location="cpu", weights_only=True
    )
    initializations: list[dict[str, Any]] = []
    for seed in map(int, contract["seeds"]):
        seed_everything(seed, deterministic_torch=True)
        control_config = _resolved_config(contract, seed=seed, condition=CONDITIONS[0])
        control = build_sage_avo_variant("full", **sage_avo_model_kwargs(control_config))
        treatment_config = _resolved_config(contract, seed=seed, condition=CONDITIONS[1])
        treatment = build_sage_avo_variant("full", **sage_avo_model_kwargs(treatment_config))
        treatment.load_state_dict(control.state_dict(), strict=True)
        control_count = sum(parameter.numel() for parameter in control.parameters())
        treatment_count = sum(parameter.numel() for parameter in treatment.parameters())
        if control_count != treatment_count:
            raise RuntimeError("Phase-1 condition parameter counts differ")
        state = dict(control.state_dict())
        historical_common_match: bool | None = None
        historical_mismatch_count: int | None = None
        if seed == 12345:
            common = {
                name: value
                for name, value in state.items()
                if name != "graph.elastic_attention_raw_strengths"
            }
            historical_mismatch_count = sum(
                name not in source_seed_state
                or not torch.equal(value, source_seed_state[name])
                for name, value in common.items()
            ) + sum(name not in common for name in source_seed_state)
            historical_common_match = historical_mismatch_count == 0
        path = _initial_state_path(experiment, seed)
        torch.save(state, path)
        initializations.append(
            {
                "seed": seed,
                "state_path": str(path),
                "state_file_sha256": file_sha256(path),
                "model_state_sha256": state_sha256(state),
                "parameter_count_each_condition": control_count,
                "condition_states_identical": True,
                "seed12345_common_state_matches_v00332k": historical_common_match,
                "seed12345_historical_mismatch_tensor_count": historical_mismatch_count,
                "historical_state_reused_for_training": False,
            }
        )
    payload = {
        "status": "PREPARED_NOT_TRAINED",
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "source_verification": verification,
        "phase0_architecture_audit": _phase0_audit(),
        "initializations": initializations,
        "condition_parameter_counts_equal": True,
        "initialization_policy_note": (
            "All seeds use today's fresh seeded constructor and are exactly matched between "
            "conditions. The historical seed-12345 state is hash-verified but not reused because "
            "its inherited relation weights and installed normalization buffers are not reproduced "
            "by a fresh constructor."
        ),
        "time_dependent_gating_implemented": False,
        "production_files_or_runs_modified": False,
    }
    contract_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def smoke(args: argparse.Namespace) -> None:
    contract, dataset, experiment, _ = _roots()
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332n phase-1 smoke test",
    )
    sample_manifest = json.loads(
        (experiment / "fixed_graph_probe_samples.json").read_text(encoding="utf-8")
    )
    batch, _ = load_fixed_batch(dataset, sample_manifest, physics_only=True, maximum_patches=2)
    values = {
        name: value.to(device) for name, value in batch.items() if isinstance(value, torch.Tensor)
    }
    rows: list[dict[str, Any]] = []
    for seed in map(int, contract["seeds"]):
        state = torch.load(_initial_state_path(experiment, seed), map_location=device, weights_only=True)
        outputs: dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            config = _resolved_config(contract, seed=seed, condition=condition)
            model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = model(
                values["low"],
                torch.full((values["low"].shape[0],), 0.5, device=device),
                values["avo"],
                values["low"],
                values["rgt"],
            )
            forward_seconds = time.perf_counter() - started
            gradient = torch.autograd.grad(
                output.velocity.square().mean(),
                model.graph.elastic_attention_raw_strengths,
            )[0]
            expected_nonzero = torch.tensor(
                config["model"]["experimental_graph"][
                    "elastic_structural_prior_component_mask"
                ],
                dtype=torch.bool,
                device=device,
            )
            observed_nonzero = gradient.abs() > 0
            if not torch.equal(observed_nonzero, expected_nonzero):
                raise RuntimeError(
                    f"Gradient mask mismatch for seed={seed}, condition={condition}: {gradient}"
                )
            if not torch.isfinite(output.velocity).all() or not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite v00332n smoke-test output or gradient")
            outputs[condition] = output.velocity.detach()
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "forward_seconds": forward_seconds,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "gradient_mask_verified": True,
                    "output_finite": True,
                    "peak_allocated_mib": (
                        float(torch.cuda.max_memory_allocated(device) / 2**20)
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
            )
            del model, output, gradient
            _release_cuda()
        if torch.equal(outputs[CONDITIONS[0]], outputs[CONDITIONS[1]]):
            raise RuntimeError("Normal-AVO treatment did not change the elastic output")
    report = {
        "status": "PASS",
        "rows": rows,
        "same_parameter_count": len({row["parameter_count"] for row in rows}) == 1,
        "time_dependent_gating_present": False,
    }
    (experiment / "phase1_smoke_test.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def train_phase1(args: argparse.Namespace) -> None:
    contract, dataset, experiment, _ = _roots()
    smoke_path = experiment / "phase1_smoke_test.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text())["status"] != "PASS":
        raise RuntimeError("Run and pass the v00332n smoke test before training")
    selection = json.loads(
        (experiment / "fixed_patch_selection.json").read_text(encoding="utf-8")
    )
    probes = experiment / "fixed_graph_probe_samples.json"
    maximum = int(contract["fixed_patch_budget"]["epochs"])
    if args.defer_whole_realizations:
        amendment = {
            "status": "ACTIVE",
            "scope": "phase1_screening_only",
            "reason": (
                "A 20-step 301x160 whole-realization reconstruction did not finish within the "
                "practical bounded-screen window while sustaining approximately 97% GPU use and "
                "3923 MiB of 4096 MiB VRAM."
            ),
            "completed_training_or_checkpoint_modified": False,
            "phase1_fixed_patch_protocol_changed": False,
            "whole_realization_evaluation": "deferred_to_phase3_finalists",
            "production_claim_allowed": False,
        }
        (experiment / "phase1_runtime_amendment.json").write_text(
            json.dumps(amendment, indent=2) + "\n", encoding="utf-8"
        )
    for seed in map(int, contract["seeds"]):
        for condition in CONDITIONS:
            config = _resolved_config(contract, seed=seed, condition=condition)
            run = _run_path(experiment, seed, condition)
            for epoch in range(_completed_epoch(run) + 1, maximum + 1):
                _release_cuda()
                train_controlled_variant(
                    repository=REPOSITORY,
                    config_path=CONTRACT_PATH,
                    config=config,
                    dataset_directory=dataset,
                    experiment_directory=experiment,
                    variant="full",
                    device_name=args.device,
                    epochs_override=maximum,
                    max_train_batches=int(
                        contract["fixed_patch_budget"]["train_batches_per_epoch"]
                    ),
                    max_validation_batches=int(
                        contract["fixed_patch_budget"]["validation_batches_per_epoch"]
                    ),
                    run_name=run.name,
                    resume_from=(run / "last.pt") if (run / "last.pt").exists() else None,
                    stop_after_epoch=epoch,
                    fixed_train_indices=selection["train_indices"],
                    fixed_validation_indices=selection["validation_indices"],
                    initial_model_state=_initial_state_path(experiment, seed),
                    finite_state_check_batches=(1, 16, 32),
                    abort_on_nonfinite=True,
                )
                _release_cuda()
                analyze_checkpoint(
                    checkpoint_path=run / "last.pt",
                    dataset_directory=dataset,
                    run_directory=run,
                    sample_manifest_path=probes,
                    output_directory=run / "gradient_graph_probe",
                    device=args.device,
                    maximum_patches=int(contract["diagnostic_probes"]["probe_patch_count"]),
                    flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
                    include_whole_realizations=(
                        epoch == maximum
                        and not args.defer_whole_realizations
                        and bool(
                            contract["diagnostic_probes"][
                                "include_whole_realizations_at_final_epoch"
                            ]
                        )
                    ),
                )
                _release_cuda()
                print(
                    f"[v00332n] completed seed={seed} condition={condition} epoch={epoch}",
                    flush=True,
                )
            final_report = (
                run
                / "gradient_graph_probe"
                / f"checkpoint_diagnostics_epoch_{maximum:04d}.json"
            )
            if _completed_epoch(run) == maximum and not final_report.exists():
                _release_cuda()
                analyze_checkpoint(
                    checkpoint_path=run / "last.pt",
                    dataset_directory=dataset,
                    run_directory=run,
                    sample_manifest_path=probes,
                    output_directory=run / "gradient_graph_probe",
                    device=args.device,
                    maximum_patches=int(contract["diagnostic_probes"]["probe_patch_count"]),
                    flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
                    include_whole_realizations=not args.defer_whole_realizations,
                )
                _release_cuda()


def _whole_metrics(path: Path, y_std: np.ndarray) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_property: dict[str, list[float]] = {name: [] for name in ("vp", "vs", "density")}
    realization_segmentation: dict[int, tuple[float, float, float, float]] = {}
    physics: list[float] = []
    for row in rows:
        by_property[row["property"]].append(float(row["rmse"]))
        realization_segmentation[int(row["realization_id"])] = (
            float(row["miou"]),
            float(row["class_1_iou"]),
            float(row["class_2_iou"]),
            float(row["class_0_iou"]),
        )
        physics.append(float(row["forward_avo_rmse_clean_normalized"]))
    normalized = {
        name: float(np.mean(values) / y_std[index])
        for index, (name, values) in enumerate(by_property.items())
    }
    segmentation = np.asarray(list(realization_segmentation.values()), dtype=float)
    miou = float(np.mean(segmentation[:, 0]))
    return {
        "whole_rmse_vp_normalized": normalized["vp"],
        "whole_rmse_vs_normalized": normalized["vs"],
        "whole_rmse_density_normalized": normalized["density"],
        "whole_miou": miou,
        "whole_class_1_iou": float(np.mean(segmentation[:, 1])),
        "whole_class_2_iou": float(np.mean(segmentation[:, 2])),
        "whole_class_0_iou": float(np.mean(segmentation[:, 3])),
        "whole_exact_pp_rmse_clean_normalized": float(np.mean(physics)),
        "whole_checkpoint_criterion": float(np.mean(list(normalized.values())) - 0.1 * miou),
    }


def analyze_phase1(_: argparse.Namespace) -> None:
    contract, dataset, experiment, _ = _roots()
    normalization = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    y_std = np.asarray(normalization["y_std"], dtype=float)
    records: list[dict[str, Any]] = []
    paired: dict[int, dict[str, dict[str, float]]] = {}
    metric_names = (
        "sample_rmse_vp_normalized",
        "sample_rmse_vs_normalized",
        "sample_rmse_density_normalized",
        "validation_full_property",
        "validation_physics",
        "probe_exact_pp_prediction_noiseless",
        "probe_exact_pp_progress_from_prior_to_floor",
        "sample_miou",
        "sample_class_1_iou",
        "sample_class_2_iou",
        "validation_fixed_objective",
        "sample_criterion",
        "whole_rmse_vp_normalized",
        "whole_rmse_vs_normalized",
        "whole_rmse_density_normalized",
        "whole_miou",
        "whole_class_0_iou",
        "whole_class_1_iou",
        "whole_class_2_iou",
        "whole_exact_pp_rmse_clean_normalized",
        "whole_checkpoint_criterion",
    )
    for seed in map(int, contract["seeds"]):
        paired[seed] = {}
        for condition in CONDITIONS:
            run = _run_path(experiment, seed, condition)
            if _completed_epoch(run) != int(contract["fixed_patch_budget"]["epochs"]):
                raise RuntimeError(f"Incomplete Phase-1 run: {run}")
            with (run / "training_log.csv").open(newline="", encoding="utf-8") as stream:
                final = list(csv.DictReader(stream))[-1]
            values = {name: float(final[name]) for name in metric_names if name in final}
            values.update(
                _whole_metrics(
                    run / "gradient_graph_probe" / "whole_realization_metrics.csv", y_std
                )
            )
            with (
                run / "gradient_graph_probe" / "physics_floor_diagnostics.csv"
            ).open(newline="", encoding="utf-8") as stream:
                physics_probe = list(csv.DictReader(stream))[-1]
            values.update(
                {
                    "probe_exact_pp_prediction_noiseless": float(
                        physics_probe["prediction_noiseless"]
                    ),
                    "probe_exact_pp_progress_from_prior_to_floor": float(
                        physics_probe["normalized_progress_from_prior_to_operator_floor"]
                    ),
                }
            )
            paired[seed][condition] = values
            records.append({"record_type": "run", "seed": seed, "condition": condition, **values})

    deltas: dict[str, list[float]] = {name: [] for name in metric_names}
    for seed, conditions in paired.items():
        control = conditions[CONDITIONS[0]]
        treatment = conditions[CONDITIONS[1]]
        row: dict[str, Any] = {"record_type": "paired_delta", "seed": seed, "condition": "normal_avo_minus_unbiased"}
        for name in metric_names:
            if name in control and name in treatment:
                value = treatment[name] - control[name]
                row[name] = value
                deltas[name].append(value)
        records.append(row)
    for statistic in ("mean", "std"):
        row = {"record_type": statistic, "seed": "all", "condition": "normal_avo_minus_unbiased"}
        for name, values in deltas.items():
            if values:
                row[name] = float(np.mean(values) if statistic == "mean" else np.std(values))
        records.append(row)

    fields = ["record_type", "seed", "condition", *metric_names]
    output = experiment / "normal_avo_multiseed_results.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    expected_pattern = {
        "sample_rmse_vp_normalized": "improve",
        "sample_rmse_vs_normalized": "improve",
        "sample_rmse_density_normalized": "degrade",
        "validation_physics": "improve",
        "sample_class_1_iou": "degrade",
        "sample_criterion": "degrade",
    }
    sign_consistency: dict[str, dict[str, Any]] = {}
    for metric, expectation in expected_pattern.items():
        values = deltas[metric]
        favorable = [value < 0 if expectation == "improve" else value > 0 for value in values]
        sign_consistency[metric] = {
            "expected": expectation,
            "paired_deltas": values,
            "consistent_seed_count": int(sum(favorable)),
            "seed_count": len(favorable),
            "all_seeds_consistent": bool(favorable and all(favorable)),
        }
    repeated = all(item["all_seeds_consistent"] for item in sign_consistency.values())
    classification = (
        contract["reproducibility_rule"]["repeated_tradeoff_classification"]
        if repeated
        else "NORMAL_AVO_EFFECT_NOT_REPRODUCIBLE"
    )
    summary = {
        "status": "COMPLETE",
        "classification": classification,
        "normal_avo_useful_claim_allowed": False,
        "all_declared_tradeoff_signs_repeated": repeated,
        "sign_consistency": sign_consistency,
        "mean_paired_deltas": {
            name: float(np.mean(values)) for name, values in deltas.items() if values
        },
        "std_paired_deltas": {
            name: float(np.std(values)) for name, values in deltas.items() if values
        },
        "phase2_motivated_by_repeated_tradeoff": repeated,
        "results_csv": str(output),
    }
    (experiment / "phase1_reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not repeated:
        gated_reason = (
            "Phase 1 did not reproduce the predeclared normal-AVO tradeoff across all "
            "three matched seeds, so Phases 2-7 were not authorized."
        )
        gated_outputs = {
            "relation_aware_variant_results.csv": [
                "status,phase,reason",
                f'NOT_RUN,3,"{gated_reason}"',
            ],
            "complexity_stratified_results.csv": [
                "status,phase,reason",
                f'NOT_RUN,5,"{gated_reason}"',
            ],
            "relation_gate_statistics.csv": [
                "status,phase,reason",
                f'NOT_RUN,3,"{gated_reason}"',
            ],
            "relation_message_statistics.csv": [
                "status,phase,reason",
                f'NOT_RUN,3,"{gated_reason}"',
            ],
            "relation_gradient_statistics.csv": [
                "status,phase,reason",
                f'NOT_RUN,3,"{gated_reason}"',
            ],
        }
        for name, lines in gated_outputs.items():
            (experiment / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        prepared_contract = json.loads(
            (experiment / "relation_aware_experiment_contract.json").read_text(
                encoding="utf-8"
            )
        )
        final_summary = {
            "status": "BOUNDED_EXPERIMENT_COMPLETE",
            "decision": "RELATION_AWARE_GNN_NO_GAIN",
            "decision_scope": (
                "No reproducible gain from the Phase-1 normal-AVO prerequisite; the "
                "proposed relation-aware contrast architecture was gated out and was not tested."
            ),
            "phase0": prepared_contract["phase0_architecture_audit"],
            "phase1": summary,
            "phase2_through_phase7": {
                "status": "NOT_RUN_PREDECLARED_GATE_FAILED",
                "reason": gated_reason,
            },
            "whole_realization_evaluation": {
                "status": "DEFERRED_NOT_RUN",
                "reason": (
                    "The first 20-step 301x160 reconstruction exceeded the bounded-screen "
                    "runtime and used approximately 3923 MiB of 4096 MiB VRAM. It was "
                    "deferred to Phase-3 finalists, but Phase 3 was subsequently gated out."
                ),
            },
            "cuda_reproducibility_limitation": (
                "PyTorch warned that CUBLAS_WORKSPACE_CONFIG was not set. Seeds, initial "
                "states, patch order, and configuration were matched, but bitwise CUDA "
                "determinism is not claimed."
            ),
            "required_scientific_answers": {
                "1_tangential_rgt_bias_redundant": (
                    "Yes, largely: RGT already defines tangential topology, and prior "
                    "component sweeps found negligible incremental scalar-bias value."
                ),
                "2_normal_avo_reproducibly_useful": "No under this three-seed screen.",
                "3_positive_normal_aggregation_harmful": (
                    "Not established causally; metrics were mixed and the effect was not reproducible."
                ),
                "4_contrast_message_better": "Not tested because the Phase-1 gate failed.",
                "5_relation_gating_better_than_global_scalar": (
                    "Not tested because the Phase-1 gate failed."
                ),
                "6_larger_complex_geology_benefit": (
                    "Not tested because no relation-aware finalist was authorized."
                ),
                "7_primary_graph_contribution": (
                    "Undetermined; current topology represents both relations, but this screen "
                    "did not establish a robust normal-AVO contribution."
                ),
                "8_density_differs_from_vp_vs": (
                    "No reproducible directional conclusion: density changed sign across seeds."
                ),
                "9_segmentation_needs_different_graph_behavior": (
                    "Not established; class-1 IoU worsened in two seeds and improved in one."
                ),
                "10_time_dependent_gating_justified": "No.",
            },
            "presentation_figures": {
                "status": "NOT_GENERATED",
                "reason": (
                    "The requested figures compare relation-aware variants and complex-geology "
                    "outcomes that were not run; generating them would fabricate evidence."
                ),
            },
            "production_training_started": False,
            "commit_or_push_performed": False,
        }
        (experiment / "relation_aware_summary.json").write_text(
            json.dumps(final_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


def status(_: argparse.Namespace) -> None:
    contract, dataset, experiment, _ = _roots()
    runs = []
    for seed in map(int, contract["seeds"]):
        for condition in CONDITIONS:
            run = _run_path(experiment, seed, condition)
            runs.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "completed_epoch": _completed_epoch(run),
                    "resumable": (run / "last.pt").exists(),
                }
            )
    print(
        json.dumps(
            {
                "dataset_ready": (dataset / "dataset_manifest.json").exists(),
                "prepared": (experiment / "relation_aware_experiment_contract.json").exists(),
                "smoke_passed": (experiment / "phase1_smoke_test.json").exists(),
                "runs": runs,
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(function=prepare)
    commands.add_parser("status").set_defaults(function=status)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--device", required=True)
    smoke_parser.set_defaults(function=smoke)
    training = commands.add_parser("train-phase1")
    training.add_argument("--device", required=True)
    training.add_argument(
        "--defer-whole-realizations",
        action="store_true",
        help="Defer prohibitively slow whole sections to Phase-3 finalists",
    )
    training.set_defaults(function=train_phase1)
    commands.add_parser("analyze-phase1").set_defaults(function=analyze_phase1)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
