#!/usr/bin/env python3
"""Run the clean, resumable v00332q 20-epoch corrected-RGT experiment.

This deliberately reuses the already tested v00332p training/evaluation code
while redirecting every mutable output to a new v00332q artifact directory.
The v00332p checkpoints and results are inputs to neither training nor outcome
evaluation; only its frozen topology/confidence/validation-mask contract is
inherited.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_rgt_topology_repair_v00332p as v00332p
from sage_avo.config import load_config
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.checkpoints import load_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(load_config(REPOSITORY / "configs/paths.yaml")["private_artifact_root"])
P_EXPERIMENT = PRIVATE / "stage_artifacts/stage04/sage_avo_s01_v00332p_rgt_topology_repair"
EXPERIMENT = PRIVATE / "stage_artifacts/stage04/sage_avo_s01_v00332q_clean_20epoch_corrected_rgt"
CONFIG = REPOSITORY / "configs/development_diagnostics_v00332q.yaml"
INTERNAL_CONTRACT = EXPERIMENT / "rgt_topology_repair_contract.json"
PUBLIC_CONTRACT = EXPERIMENT / "v00332q_contract.json"
MILESTONES = (1, 3, 5, 10, 15, 20)
CONTROLS = ("cartesian_gnn", "shuffled_graph", "root_only_gnn")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_P_TRAINING_CONFIG = v00332p._training_config


def _q_training_config(
    training_contract: dict[str, Any],
    topology_contract: dict[str, Any],
    seed: int,
    architecture: str,
) -> dict[str, Any]:
    config = _P_TRAINING_CONFIG(training_contract, topology_contract, seed, architecture)
    config["capabilities"]["rgt_topology_repair"]["diagnostic_revision"] = "v00332q"
    config["experiment"]["name"] = training_contract["experiment_name"]
    return config


def _activate_q() -> None:
    v00332p.EXPERIMENT = EXPERIMENT
    v00332p.TRAINING_CONFIG = CONFIG
    v00332p._training_config = _q_training_config


def _lr_rows() -> list[dict[str, Any]]:
    base = load_config(REPOSITORY / "configs/sage_avo_s01_v0031.yaml")["training"]
    lr = float(base["learning_rate"])
    eta_min = float(base["scheduler_eta_min"])
    values = [
        eta_min + (lr - eta_min) * (1.0 + math.cos(math.pi * step / 20)) / 2.0 for step in range(21)
    ]
    if any(right > left + 1e-15 for left, right in zip(values, values[1:])):
        raise RuntimeError("The declared LR schedule is not monotonically non-increasing")
    return [
        {
            "epoch_boundary": step,
            "scheduler_step_count": step,
            "learning_rate": values[step],
            "monotonic_non_increasing_0_to_20": True,
            "training_log_epoch_using_this_lr": step + 1 if step < 20 else "none_hard_stop",
        }
        for step in (0, 1, 3, 5, 10, 15, 20)
    ]


def prepare(_: argparse.Namespace) -> None:
    _activate_q()
    source_path = P_EXPERIMENT / "rgt_topology_repair_contract.json"
    source = _json(source_path)
    if source.get("status") != "BOUNDED_V00332P_EPOCH10_COMPLETE_EXTENSION_BLOCKED":
        raise RuntimeError("v00332p is not in its expected frozen epoch-10 state")
    contract = deepcopy(source)
    contract.update(
        revision="v00332q-clean-20epoch-corrected-rgt-causal",
        status="PREPARED_CONTRACT_PENDING_FRESH_INITIALIZATION",
        source_topology_contract={
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "inherited_fields": [
                "split_ids",
                "adaptive_search",
                "confidence_rule",
                "diverse_validation_subset",
            ],
            "v00332p_checkpoints_used": False,
        },
        scientific_question=(
            "Does corrected RGT topology provide reproducible benefit over Cartesian, "
            "shuffled, and root-only controls under a coherent 20-epoch schedule?"
        ),
        hard_stop_epoch=20,
        warm_restarts=False,
        mechanism_changes_from_v00332p=[],
    )
    for key in (
        "bounded_training",
        "outcome_decision",
        "epoch20_extension_status",
        "epoch20_scheduler_issue",
    ):
        contract.pop(key, None)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if (
        INTERNAL_CONTRACT.exists()
        and _json(INTERNAL_CONTRACT).get("revision") != contract["revision"]
    ):
        raise RuntimeError("Refusing to replace a different existing q contract")
    _write_json(INTERNAL_CONTRACT, contract)
    pd.DataFrame(_lr_rows()).to_csv(EXPERIMENT / "v00332q_lr_schedule.csv", index=False)
    v00332p.prepare_training(SimpleNamespace())
    prepared = _json(INTERNAL_CONTRACT)
    q_config = load_config(CONFIG)
    p_schedule = _json(P_EXPERIMENT / "fixed_patch_schedule.json")
    q_schedule = _json(EXPERIMENT / "fixed_patch_schedule.json")
    reproducibility = {
        "config_sha256": _sha256(CONFIG),
        "dataset_manifest_sha256": _sha256(v00332p.DATASET / "dataset_manifest.json"),
        "split_ids_sha256": _canonical_sha(prepared["split_ids"]),
        "diverse_validation_subset_sha256": _canonical_sha(prepared["diverse_validation_subset"]),
        "confidence_rule_sha256": _canonical_sha(prepared["confidence_rule"]),
        "train_schedule_20epoch_sha256": q_schedule["train_schedule_sha256"],
        "train_schedule_first10_matches_v00332p": (
            q_schedule["train_schedule"][:10] == p_schedule["train_schedule"][:10]
        ),
        "validation_indices_sha256": q_schedule["validation_indices_sha256"],
        "initializations": prepared["bounded_training"]["initializations"],
        "fresh_q_artifact_root": str(EXPERIMENT),
        "v00332p_training_state_loaded": False,
    }
    prepared.update(
        status="PREPARED_FRESH_MATCHED_TRAINING_NOT_STARTED",
        lr_contract={
            "scheduler": "CosineAnnealingLR",
            "T_max": 20,
            "eta_min": float(
                load_config(REPOSITORY / "configs/sage_avo_s01_v0031.yaml")["training"][
                    "scheduler_eta_min"
                ]
            ),
            "base_learning_rate": float(
                load_config(REPOSITORY / "configs/sage_avo_s01_v0031.yaml")["training"][
                    "learning_rate"
                ]
            ),
            "scheduler_step_convention": "train/validate/log LR, then scheduler.step(), then checkpoint",
            "monotonic_non_increasing": True,
            "hard_stop": 20,
            "warm_restarts": False,
        },
        conditions=list(q_config["architectures"]),
        milestones=list(MILESTONES),
        reproducibility=reproducibility,
        forbidden_changes_performed=False,
        production_training=False,
    )
    _write_json(INTERNAL_CONTRACT, prepared)
    _write_json(PUBLIC_CONTRACT, prepared)
    print(pd.DataFrame(_lr_rows()).to_string(index=False))
    print(json.dumps(reproducibility, indent=2))


def smoke(args: argparse.Namespace) -> None:
    _activate_q()
    v00332p.training_smoke(args)


def train(args: argparse.Namespace) -> None:
    _activate_q()
    v00332p.train_bounded(args)
    contract = _json(INTERNAL_CONTRACT)
    contract["status"] = "CLEAN_20EPOCH_TRAINING_COMPLETE_PENDING_EVALUATION"
    _write_json(INTERNAL_CONTRACT, contract)
    _write_json(PUBLIC_CONTRACT, contract)


def evaluate(args: argparse.Namespace) -> None:
    _activate_q()
    v00332p.evaluate_diverse(args)


def _parameter_gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    squared = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return float(squared.sqrt().cpu())


def mechanism(args: argparse.Namespace) -> None:
    _activate_q()
    contract = _json(INTERNAL_CONTRACT)
    source = PRIVATE / "stage_artifacts/stage04/sage_avo_s01_v00332o_causal_gnn_multiseed"
    probe_manifest = _json(source / "fixed_graph_probe_samples.json")
    probes = [
        row
        for row in probe_manifest["patches"]
        if tuple(row["raw_scale"]) == tuple(row["resized_scale"])
    ][:5]
    normalization = _json(v00332p.DATASET / "normalization.json")
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332q milestone message diagnostics",
    )
    rows: list[dict[str, Any]] = []
    for seed in map(int, load_config(CONFIG)["seeds"]):
        run = EXPERIMENT / "runs" / f"seed_{seed}_corrected_rgt_gnn"
        raw = torch.load(
            run / "milestone_checkpoints/epoch_0001.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build_sage_avo_variant("full", **sage_avo_model_kwargs(raw["config"])).to(device)
        model.set_norm_stats(normalization)
        for epoch in MILESTONES:
            load_checkpoint(
                run / "milestone_checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                restore_rng=False,
                map_location=device,
            )
            model.eval()
            for probe_index, record in enumerate(probes):
                arrays = v00332p._probe_arrays(record)
                avo = torch.from_numpy(((arrays["avo"] - x_mean) / x_std)[None]).to(device)
                low = torch.from_numpy(((arrays["low"] - y_mean) / y_std)[None]).to(device)
                rgt = torch.from_numpy(arrays["rgt"][None]).to(device)
                time = torch.full((1,), 0.5, dtype=low.dtype, device=device)
                v00332p._reset_diagnostic_controls(model, contract["confidence_rule"])
                model.graph.diagnostic_capture_contributions = True
                model.diagnostic_capture_fusion = True
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    baseline = model(low, time, avo, low, rgt)
                    diagnostic_scalar = baseline.velocity.square().mean()
                    diagnostic_scalar.backward()
                contributions = [dict(value) for value in model.graph.last_contribution_diagnostics]
                fusion = dict(model.last_fusion_diagnostics)
                attention = v00332p._attention_statistics(baseline.attention_weights[0])
                baseline_velocity = baseline.velocity.detach().clone()
                baseline_rms = baseline_velocity.square().mean().sqrt().clamp_min(1e-12)
                grad_1 = _parameter_gradient_norm(model, "graph.layers.0.")
                grad_2 = _parameter_gradient_norm(model, "graph.layers.1.")
                v00332p._reset_diagnostic_controls(model, contract["confidence_rule"])
                model.graph.diagnostic_neighbor_scale = 0.0
                with torch.inference_mode():
                    no_neighbor = model(low, time, avo, low, rgt)
                neighbor_change = float(
                    (
                        (no_neighbor.velocity - baseline_velocity).square().mean().sqrt()
                        / baseline_rms
                    ).cpu()
                )
                v00332p._reset_diagnostic_controls(model, contract["confidence_rule"])
                model.graph.diagnostic_root_scale = 0.0
                with torch.inference_mode():
                    no_root = model(low, time, avo, low, rgt)
                root_change = float(
                    (
                        (no_root.velocity - baseline_velocity).square().mean().sqrt() / baseline_rms
                    ).cpu()
                )
                layer = {int(row["layer"]): row for row in contributions}
                rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "probe_index": probe_index,
                        "realization_id": int(record["realization_id"]),
                        "role": record["role"],
                        "diagnostic_scalar": "mean_squared_flow_velocity_at_t_0.5",
                        "transformerconv_layer_1_gradient_l2": grad_1,
                        "transformerconv_layer_2_gradient_l2": grad_2,
                        "neighbor_root_ratio_layer_1": layer[1]["neighbor_root_ratio"],
                        "neighbor_root_ratio_layer_2": layer[2]["neighbor_root_ratio"],
                        "velocity_relative_change_neighbor_zeroed": neighbor_change,
                        "velocity_relative_change_root_zeroed": root_change,
                        "graph_reinjection_rms": fusion["decoder_graph_effect_rms"],
                        **attention,
                    }
                )
            print(f"[v00332q] mechanism seed={seed} epoch={epoch}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(EXPERIMENT / "v00332q_message_diagnostics.csv", index=False)


def _paired(
    frame: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    group = frame.groupby(["epoch", "seed", "architecture"], sort=True)[metric].mean().unstack()
    rows = []
    for (epoch, seed), values in group.iterrows():
        candidate = float(values["corrected_rgt_gnn"])
        for control in CONTROLS:
            baseline = float(values[control])
            signed = candidate - baseline
            gain = signed if higher_is_better else -signed
            rows.append(
                {
                    "epoch": int(epoch),
                    "seed": int(seed),
                    "control": control,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "corrected_rgt": candidate,
                    "control_value": baseline,
                    "delta_corrected_minus_control": signed,
                    "relative_gain": gain / max(abs(baseline), 1e-12),
                    "favorable": gain > 0,
                }
            )
    return pd.DataFrame(rows)


def _stratum_effects(strata: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "normalized_rmse",
        "exact_pp_rmse_clean_normalized",
        "miou",
        "class_1_iou",
        "class_2_iou",
    ]
    rows = []
    for metric in metrics:
        keys = ["epoch", "seed", "stratum", "property", "architecture"]
        grouped = strata.groupby(keys, sort=True)[metric].mean().unstack()
        for index, values in grouped.iterrows():
            epoch, seed, stratum, prop = index
            candidate = float(values["corrected_rgt_gnn"])
            baseline = float(values["cartesian_gnn"])
            rows.append(
                {
                    "epoch": int(epoch),
                    "seed": int(seed),
                    "stratum": stratum,
                    "property": prop,
                    "metric": metric,
                    "corrected_rgt": candidate,
                    "cartesian": baseline,
                    "delta_corrected_minus_cartesian": candidate - baseline,
                    "favorable": (candidate > baseline)
                    if metric in {"miou", "class_1_iou", "class_2_iou"}
                    else (candidate < baseline),
                }
            )
    return pd.DataFrame(rows)


def _criterion_at(temporal: pd.DataFrame, control: str, epoch: int = 20) -> pd.DataFrame:
    return temporal[
        (temporal["metric"] == "checkpoint_criterion")
        & (temporal["control"] == control)
        & (temporal["epoch"] == epoch)
    ].copy()


def _plot_outputs(temporal: pd.DataFrame, geology: pd.DataFrame, messages: pd.DataFrame) -> None:
    figures = EXPERIMENT / "figures"
    figures.mkdir(exist_ok=True)

    criterion = temporal[temporal["metric"] == "checkpoint_criterion"]
    summary = (
        criterion.groupby(["epoch", "control"])["delta_corrected_minus_control"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for control, group in summary.groupby("control"):
        ax.plot(group["epoch"], group["mean"], marker="o", label=control)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Epoch", ylabel="Criterion delta (RGT - control)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "01_rgt_vs_controls_criterion.png", dpi=160)
    plt.close(fig)

    final = temporal[
        (temporal["epoch"] == 20)
        & temporal["metric"].isin(
            ["vp_normalized_rmse", "vs_normalized_rmse", "density_normalized_rmse"]
        )
    ]
    pivot = final.pivot_table(
        index="seed", columns=["control", "metric"], values="delta_corrected_minus_control"
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("RGT - control normalized RMSE")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "02_property_effects_by_seed.png", dpi=160)
    plt.close(fig)

    rc = criterion[criterion["control"] == "cartesian_gnn"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, group in rc.groupby("seed"):
        ax.plot(group["epoch"], group["delta_corrected_minus_control"], marker="o", label=str(seed))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Epoch", ylabel="Criterion delta (RGT - Cartesian)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_rgt_cartesian_effect_vs_epoch.png", dpi=160)
    plt.close(fig)

    geo = geology[(geology["epoch"] == 20) & (geology["metric"] == "normalized_rmse")]
    geo = geo.groupby(["seed", "stratum"])["delta_corrected_minus_cartesian"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    geo.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("3-property normalized RMSE delta")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "04_performance_by_geology.png", dpi=160)
    plt.close(fig)

    msg = (
        messages.groupby(["epoch", "seed"])[
            ["neighbor_root_ratio_layer_1", "neighbor_root_ratio_layer_2"]
        ]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("neighbor_root_ratio_layer_1", "neighbor_root_ratio_layer_2"):
        mean = msg.groupby("epoch")[column].mean()
        ax.plot(mean.index, mean.values, marker="o", label=column)
    ax.set(xlabel="Epoch", ylabel="Neighbor/root norm ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "05_neighbor_root_utilization.png", dpi=160)
    plt.close(fig)

    selectivity = messages.groupby("epoch")[["attention_entropy", "top_decile_mass"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    selectivity.plot(marker="o", ax=ax)
    ax.set(xlabel="Epoch", ylabel="Normalized statistic")
    fig.tight_layout()
    fig.savefig(figures / "06_attention_selectivity.png", dpi=160)
    plt.close(fig)

    fault = geo[[column for column in ("fault_corridor", "away_from_fault") if column in geo]]
    fig, ax = plt.subplots(figsize=(7, 4))
    fault.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("3-property normalized RMSE delta")
    fig.tight_layout()
    fig.savefig(figures / "07_fault_vs_nonfault_effect.png", dpi=160)
    plt.close(fig)


def analyze(_: argparse.Namespace) -> None:
    result_path = EXPERIMENT / "rgt_causal_multiseed_results.csv"
    strata_path = EXPERIMENT / "rgt_complexity_stratified_results.csv"
    message_path = EXPERIMENT / "v00332q_message_diagnostics.csv"
    if not all(path.exists() for path in (result_path, strata_path, message_path)):
        raise FileNotFoundError(
            "Training, diverse evaluation, and mechanism diagnostics must complete first"
        )
    results = pd.read_csv(result_path)
    strata = pd.read_csv(strata_path)
    messages = pd.read_csv(message_path)
    expected = 3 * 4 * 6 * 6
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} whole-realization rows, found {len(results)}")
    results.to_csv(EXPERIMENT / "v00332q_multiseed_results.csv", index=False)
    paired_frames = []
    for metric in (
        "vp_normalized_rmse",
        "vs_normalized_rmse",
        "density_normalized_rmse",
        "checkpoint_criterion",
        "miou",
        "class_1_iou",
        "class_2_iou",
        "exact_pp_rmse_clean_normalized",
        "vp_ssim",
        "vs_ssim",
        "density_ssim",
    ):
        paired_frames.append(
            _paired(
                results,
                metric=metric,
                higher_is_better=metric
                in {"miou", "class_1_iou", "class_2_iou", "vp_ssim", "vs_ssim", "density_ssim"},
            )
        )
    temporal = pd.concat(paired_frames, ignore_index=True)
    temporal.to_csv(EXPERIMENT / "v00332q_temporal_effects.csv", index=False)
    geology = _stratum_effects(strata)
    geology.to_csv(EXPERIMENT / "v00332q_geology_stratified_results.csv", index=False)

    comparisons: dict[str, Any] = {}
    for control in CONTROLS:
        frame = _criterion_at(temporal, control)
        comparisons[control] = {
            "per_seed_delta_corrected_minus_control": {
                str(int(row.seed)): float(row.delta_corrected_minus_control)
                for row in frame.itertuples()
            },
            "mean_delta": float(frame["delta_corrected_minus_control"].mean()),
            "std_delta": float(frame["delta_corrected_minus_control"].std(ddof=0)),
            "mean_relative_gain": float(frame["relative_gain"].mean()),
            "favorable_seed_count": int(frame["favorable"].sum()),
            "sign_consistent": bool(frame["favorable"].nunique() == 1),
        }
    rc = _criterion_at(temporal, "cartesian_gnn")
    cart_favorable = int(rc["favorable"].sum())
    all_controls_reproducible = all(
        value["favorable_seed_count"] == 3 for value in comparisons.values()
    )

    def geo_effect(name: str) -> dict[str, Any]:
        frame = (
            geology[
                (geology["epoch"] == 20)
                & (geology["metric"] == "normalized_rmse")
                & (geology["stratum"] == name)
            ]
            .groupby("seed")["delta_corrected_minus_cartesian"]
            .mean()
        )
        return {
            "mean_delta": float(frame.mean()),
            "std_delta": float(frame.std(ddof=0)),
            "favorable_seed_count": int((frame < 0).sum()),
            "per_seed_delta": {str(int(k)): float(v) for k, v in frame.items()},
        }

    geo_summary = {
        name: geo_effect(name)
        for name in (
            "low_dip",
            "high_dip_continuous",
            "facies_boundary",
            "fault_corridor",
            "away_from_fault",
            "reservoir",
            "plume",
        )
    }
    away_only = (
        geo_summary["away_from_fault"]["favorable_seed_count"] == 3
        and geo_summary["fault_corridor"]["favorable_seed_count"] <= 1
    )
    if all_controls_reproducible:
        decision = "RGT_GAIN_REPRODUCED_AT_20"
    elif away_only:
        decision = "RGT_GAIN_AWAY_FROM_FAULTS_ONLY"
    elif cart_favorable == 3 and comparisons["cartesian_gnn"]["mean_delta"] < 0:
        decision = "RGT_SMALL_BUT_REPRODUCIBLE_GAIN"
    elif 0 < cart_favorable < 3:
        decision = "RGT_SEED_DEPENDENT"
    elif comparisons["cartesian_gnn"]["mean_delta"] > 0 and cart_favorable <= 1:
        decision = "RGT_HARMFUL"
    else:
        decision = "RGT_NO_REPRODUCIBLE_GAIN"

    msg_epoch = messages.groupby("epoch").mean(numeric_only=True)
    neighbor_change = float(msg_epoch.loc[20, "velocity_relative_change_neighbor_zeroed"])
    neighbor_status = (
        "MATERIAL"
        if neighbor_change >= 0.01
        else ("WEAK" if neighbor_change >= 0.001 else "ROOT_DOMINATED")
    )
    fault_status = (
        "LIMITING"
        if away_only or geo_summary["fault_corridor"]["favorable_seed_count"] == 0
        else (
            "ACCEPTABLE"
            if geo_summary["fault_corridor"]["favorable_seed_count"] == 3
            else "UNRESOLVED"
        )
    )
    rc_history = (
        temporal[
            (temporal["metric"] == "checkpoint_criterion")
            & (temporal["control"] == "cartesian_gnn")
        ]
        .groupby("epoch")["delta_corrected_minus_control"]
        .mean()
    )
    changes = np.diff(rc_history.loc[[3, 5, 10, 15, 20]].to_numpy())
    if np.all(changes <= 0):
        gap_trend = "grows_favorable"
    elif np.all(changes >= 0):
        gap_trend = "shrinks_or_grows_harmful"
    else:
        gap_trend = "oscillates"
    attention_more_selective = bool(
        msg_epoch.loc[20, "attention_entropy"] < msg_epoch.loc[3, "attention_entropy"]
        and msg_epoch.loc[20, "top_decile_mass"] > msg_epoch.loc[3, "top_decile_mass"]
    )
    neighbor_material_change = float(
        msg_epoch.loc[20, "velocity_relative_change_neighbor_zeroed"]
        - msg_epoch.loc[3, "velocity_relative_change_neighbor_zeroed"]
    )
    proposal = (
        "Propose v00332r focused only on fault-aware topology/confidence blocking; do not execute automatically."
        if decision
        in {
            "RGT_GAIN_REPRODUCED_AT_20",
            "RGT_GAIN_AWAY_FROM_FAULTS_ONLY",
            "RGT_SMALL_BUT_REPRODUCIBLE_GAIN",
        }
        else "Audit topology/data differences before relation-aware message passing; do not add relation layers automatically."
    )
    questions = {
        "1_corrected_beats_cartesian": comparisons["cartesian_gnn"]["favorable_seed_count"] == 3,
        "2_corrected_beats_shuffled": comparisons["shuffled_graph"]["favorable_seed_count"] == 3,
        "3_corrected_beats_root_only": comparisons["root_only_gnn"]["favorable_seed_count"] == 3,
        "4_rgt_cartesian_gap_epoch3_to20": gap_trend,
        "5_attention_more_selective": attention_more_selective,
        "6_neighbor_root_utilization_change": neighbor_material_change,
        "7_benefit_concentrated_away_from_faults": away_only,
        "8_high_dip_continuous_benefits": geo_summary["high_dip_continuous"]["favorable_seed_count"]
        == 3,
        "9_fault_corridors_dominant_failure": fault_status == "LIMITING",
        "10_fault_aware_experiment_justified": decision
        in {
            "RGT_GAIN_REPRODUCED_AT_20",
            "RGT_GAIN_AWAY_FROM_FAULTS_ONLY",
            "RGT_SMALL_BUT_REPRODUCIBLE_GAIN",
        },
    }
    run_hashes: dict[str, Any] = {}
    base_training = load_config(REPOSITORY / "configs/sage_avo_s01_v0031.yaml")["training"]
    base_lr = float(base_training["learning_rate"])
    eta_min = float(base_training["scheduler_eta_min"])
    actual_lr_matches_contract = True
    for seed in map(int, load_config(CONFIG)["seeds"]):
        for architecture in load_config(CONFIG)["architectures"]:
            run = EXPERIMENT / "runs" / f"seed_{seed}_{architecture}"
            log_path = run / "training_log.csv"
            log = pd.read_csv(log_path)
            for row in log.itertuples():
                # Training epoch E logs the LR at scheduler boundary E - 1.
                step = int(row.epoch) - 1
                expected_lr = (
                    eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * step / 20)) / 2.0
                )
                actual_lr_matches_contract &= math.isclose(
                    float(row.learning_rate), expected_lr, rel_tol=1e-12, abs_tol=1e-15
                )
            checkpoint_path = run / "milestone_checkpoints" / "epoch_0020.pt"
            run_hashes[run.name] = {
                "epoch20_checkpoint_sha256": _sha256(checkpoint_path),
                "training_log_sha256": _sha256(log_path),
                "completed_epochs": int(log["epoch"].max()),
            }
    if not actual_lr_matches_contract:
        raise RuntimeError("Observed training LR values violate the v00332q contract")
    summary = {
        "status": "V00332Q_COMPLETE",
        "decision": decision,
        "NEIGHBOR_MESSAGE_STATUS": neighbor_status,
        "FAULT_STATUS": fault_status,
        "AVO_EDGE_ATTR_STATUS": "UNRESOLVED",
        "epoch20_paired_criterion": comparisons,
        "geology_stratified_corrected_minus_cartesian": geo_summary,
        "temporal_rgt_cartesian_mean_delta": {str(int(k)): float(v) for k, v in rc_history.items()},
        "message_diagnostics_epoch_means": {
            str(int(k)): {name: float(value) for name, value in row.items()}
            for k, row in msg_epoch.iterrows()
        },
        "required_questions": questions,
        "observed_learning_rates_match_contract": actual_lr_matches_contract,
        "run_reproducibility_hashes": run_hashes,
        "next_experiment_proposal_only": proposal,
        "v00332p_modified": False,
        "warm_restarts_used": False,
        "training_stopped_at_epoch20": True,
        "architecture_or_objective_changes": False,
        "production_training_started": False,
        "commit_or_push_performed": False,
    }
    _write_json(EXPERIMENT / "v00332q_summary.json", summary)
    _plot_outputs(temporal, geology, messages)
    report = f"""# v00332q: clean 20-epoch corrected-RGT causal experiment

## Decision

**{decision}**

- NEIGHBOR_MESSAGE_STATUS: **{neighbor_status}**
- FAULT_STATUS: **{fault_status}**
- AVO_EDGE_ATTR_STATUS: **UNRESOLVED**

## Epoch-20 paired criterion results

{pd.DataFrame(comparisons).T.to_markdown()}

Lower criterion is better; every delta is corrected RGT minus the named control.

## Geological effects

{pd.DataFrame(geo_summary).T.to_markdown()}

Negative delta is favorable and is the mean of Vp, Vs, and density normalized RMSE.

## Required questions

{chr(10).join(f"{index}. **{name}**: {value}" for index, (name, value) in enumerate(questions.items(), 1))}

## Mechanism interpretation

At epoch 20, zeroing neighbor messages changes flow velocity by {neighbor_change:.6g} relative RMS.
Attention becomes more selective from epoch 3 to 20: {attention_more_selective}.
The mean RGT-vs-Cartesian criterion gap from epochs 3–20: {gap_trend}.

## Next experiment (proposal only)

{proposal}

No warm restart, production run, graph-mechanism change, automatic follow-on architecture, commit, or push was performed.
"""
    (EXPERIMENT / "v00332q_report.md").write_text(report, encoding="utf-8")
    contract = _json(INTERNAL_CONTRACT)
    contract.update(status="V00332Q_COMPLETE", outcome_decision=decision)
    _write_json(INTERNAL_CONTRACT, contract)
    _write_json(PUBLIC_CONTRACT, contract)
    print(json.dumps(summary, indent=2))


def run_all(args: argparse.Namespace) -> None:
    prepare(args)
    smoke(args)
    train(args)
    evaluate(args)
    mechanism(args)
    analyze(args)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(required=True)
    for name, function, needs_device in (
        ("prepare", prepare, False),
        ("smoke", smoke, True),
        ("train", train, True),
        ("evaluate", evaluate, True),
        ("mechanism", mechanism, True),
        ("analyze", analyze, False),
        ("run-all", run_all, True),
    ):
        command = commands.add_parser(name)
        if needs_device:
            command.add_argument("--device", default="cuda")
        command.set_defaults(function=function)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
