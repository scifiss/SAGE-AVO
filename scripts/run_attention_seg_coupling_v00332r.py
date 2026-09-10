#!/usr/bin/env python3
"""Resumable, fixed-topology v00332r audits; frozen q inputs are read-only.

Use PYTHONPATH=src and CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts.
Every mutable artifact is confined to the new r directory. No q runner is
activated, no optimizer is constructed by the frozen-checkpoint audits, and
training is gated on complete audited outputs.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

import run_rgt_topology_repair_v00332p as frozen_helpers
from sage_avo.config import load_config, seed_everything
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.diagnostics.attention_coupling import (
    elastic_extension_gate, final_tie_usage, graph_intervention, headwise_uniformity, task_gradient_rows,
)
from sage_avo.diagnostics.rgt_topology_repair import load_faults, load_realization
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.experiments.training import (
    _class_weights, _normalization_tensors, curriculum_from_config,
    graph_objective_from_config, loss_weights_from_config, physics_settings_from_config,
    train_controlled_variant,
)
from sage_avo.forward.specification import forward_specification_from_mapping
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.models.variants import (
    build_sage_avo_variant, sage_avo_model_kwargs, variant_definition,
)
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.checkpoints import load_checkpoint
from sage_avo.training.engine import ContrastiveSettings, _forward_objective, _move_batch


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(load_config(REPOSITORY / "configs/paths.yaml")["private_artifact_root"])
STAGE04 = PRIVATE / "stage_artifacts/stage04"
Q = STAGE04 / "sage_avo_s01_v00332q_clean_20epoch_corrected_rgt"
P = STAGE04 / "sage_avo_s01_v00332p_rgt_topology_repair"
CONFIG_PATH = REPOSITORY / "configs/development_diagnostics_v00332r.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = STAGE04 / CONFIG["experiment_name"]
DATASET = PRIVATE / "stage_artifacts/stage03" / CONFIG["immutable_dataset"] / "dataset"
STAGE02 = PRIVATE / "stage_artifacts/stage02/v00331_production100_support_aware/realizations"
A, B, C, D = tuple(CONFIG["conditions"])
PROPERTIES = ("vp", "vs", "density")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def execution_sources():
    paths = list((REPOSITORY / "src").rglob("*.py")) + list((REPOSITORY / "configs").glob("*.yaml"))
    paths.extend([Path(__file__), Path(frozen_helpers.__file__)])
    return {str(path.relative_to(REPOSITORY)): file_sha(path) for path in sorted(set(paths))}


@contextmanager
def atomic_output(path):
    """Only r-owned destinations; interrupted temporary files are not completion."""
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError(f"Refusing output outside the v00332r directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    with atomic_output(path) as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def write_csv(path, values):
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    with atomic_output(path) as stream:
        stream.write(frame.to_csv(index=False).encode())


def progress(message):
    print(f"[v00332r {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def q_contract():
    return read_json(Q / "v00332q_contract.json")


def checkpoint(seed, epoch):
    return Q / "runs" / f"seed_{seed}_corrected_rgt_gnn/milestone_checkpoints/epoch_{epoch:04d}.pt"


def training_config(seed, condition):
    config = frozen_helpers._training_config(
        load_config(REPOSITORY / "configs/development_diagnostics_v00332q.yaml"),
        q_contract(), seed, "corrected_rgt_gnn",
    )
    config["experiment"]["name"] = CONFIG["experiment_name"]
    config["model"]["experimental_graph"].update(CONFIG["conditions"][condition])
    config["capabilities"]["rgt_topology_repair"]["diagnostic_revision"] = "v00332r"
    return config


def protected_inputs():
    paths = [
        Q / "v00332q_contract.json", Q / "fixed_patch_schedule.json",
        P / "rgt_topology_repair_contract.json", DATASET / "normalization.json",
        DATASET / "patch_index.csv", DATASET / "dataset_manifest.json",
        Q / "v00332q_multiseed_results.csv", Q / "rgt_complexity_stratified_results.csv",
        REPOSITORY / "src/sage_avo/models/graph.py",
        REPOSITORY / "src/sage_avo/training/flow.py",
        REPOSITORY / "src/sage_avo/training/losses.py",
        REPOSITORY / "src/sage_avo/training/engine.py",
        REPOSITORY / "configs/development_diagnostics_v00332q.yaml",
    ]
    paths += sorted((REPOSITORY / "src/sage_avo/forward").glob("*.py"))
    paths += [checkpoint(seed, epoch) for seed in CONFIG["seeds"] for epoch in CONFIG["milestones"]]
    return {str(path): file_sha(path) for path in paths}


def prepare(_):
    review = subprocess.check_output(
        ["git", "rev-parse", "review/rgt-gnn-reconciliation"], cwd=REPOSITORY, text=True
    ).strip()
    if review != CONFIG["review_commit"]:
        raise RuntimeError("The frozen review branch no longer points to its declared commit")
    contract_path = OUT / "v00332r_baseline_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        if contract["config_sha256"] != file_sha(CONFIG_PATH):
            raise RuntimeError("The frozen r configuration changed; refusing to mix protocols")
        for path, expected in contract["protected_input_sha256"].items():
            if file_sha(path) != expected:
                raise RuntimeError(f"Frozen input changed: {path}")
        progress("Existing baseline contract and protected inputs verified")
        return
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY, text=True
    ).strip()
    if head != CONFIG["review_commit"] or branch != "experiment/v00332r-attention-seg-coupling":
        raise RuntimeError("Expected the local r branch based exactly on the frozen review commit")
    q = q_contract()
    p = read_json(P / "rgt_topology_repair_contract.json")
    if q["confidence_rule"] != p["confidence_rule"]:
        raise RuntimeError("q/p confidence-rule mismatch")
    schedule = read_json(Q / "fixed_patch_schedule.json")
    if len(schedule["train_schedule"]) != 20 or any(len(x) != 70 for x in schedule["train_schedule"]):
        raise RuntimeError("Expected the exact q 20-by-70 training schedule")
    if len(schedule["validation_indices"]) != 40:
        raise RuntimeError("Expected the exact q 40-patch validation schedule")
    protected = protected_inputs()
    write_json(contract_path, {
        "revision": CONFIG["revision"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "review_commit": head, "local_branch": branch, "config": CONFIG,
        "config_sha256": file_sha(CONFIG_PATH), "protected_input_sha256": protected,
        "confidence_rule": q["confidence_rule"],
        "confidence_rule_sha256": canonical_sha(q["confidence_rule"]),
        "p_q_confidence_equal": True, "split_ids": q["split_ids"],
        "validation_subset": q["diverse_validation_subset"],
        "train_schedule_sha256": schedule["train_schedule_sha256"],
        "validation_indices_sha256": schedule["validation_indices_sha256"],
        "normalization_sha256": file_sha(DATASET / "normalization.json"),
        "decision_thresholds_note": "Predeclared operational effect-size margins, not statistical significance tests",
        "q_training_state_used_for_r_training": False,
        "precision": "float32, unchanged from q; no AMP change to exact-PP",
        "segmentation_detach_scope": "blocks segmentation gradients into graph and its upstream CNN inputs; head still trains",
        "audit_whole_scope": "epoch 20, all three q seeds, frozen six-realization validation subset",
        "audit_patch_scope": "all six milestones, all three q seeds, six fixed native validation patches",
        "topology_source_unchanged": True,
    })
    write_json(OUT / "fixed_patch_schedule.json", schedule)
    progress("Frozen baseline/confidence/schedule contract created")


def tie_audit(_):
    destination = OUT / "final_sign_tiebreak_audit.csv"
    if destination.exists() and "stage1_exact_tie_count" in pd.read_csv(destination, nrows=0).columns:
        return
    q = q_contract()
    rows = []
    examples = []
    for split in ("train", "validation"):
        for realization_id in q["split_ids"][split]:
            arrays = load_realization(DATASET, realization_id)
            usage = final_tie_usage(arrays["rgt"])
            masks = frozen_helpers._pixel_strata(
                arrays, load_faults(STAGE02, realization_id), q["adaptive_search"]
            )
            masks["entire_lattice"] = np.ones_like(arrays["rgt"], dtype=bool)
            for stratum, mask in masks.items():
                mask = mask[:, :-1]
                count = int(mask.sum())
                if not count:
                    continue
                stage1 = int((usage["stage1_tie"] & mask).sum())
                parity = usage["parity"] & mask
                horizontal_pairs = parity[:, 1:] & parity[:, :-1]
                vertical_pairs = parity[1:] & parity[:-1]
                flip_h = usage["shift"][:, 1:] * usage["shift"][:, :-1] < 0
                flip_v = usage["shift"][1:] * usage["shift"][:-1] < 0
                adjacent = int(horizontal_pairs.sum() + vertical_pairs.sum())
                flips = int((horizontal_pairs & flip_h).sum() + (vertical_pairs & flip_v).sum())
                rows.append({
                    "split": split, "realization_id": realization_id, "stratum": stratum,
                    "directed_forward_candidate_count": count,
                    "stage1_near_or_exact_tie_count": stage1,
                    "stage1_exact_tie_count": int((usage["stage1_exact_tie"] & mask).sum()),
                    "stage1_tie_fraction": stage1 / count,
                    "minimum_absolute_shift_resolved_count": int((usage["resolved_by_min_abs_shift"] & mask).sum()),
                    "final_parity_count": int(parity.sum()), "final_parity_fraction": float(parity.sum() / count),
                    "parity_fraction_among_stage1_ties": float(parity.sum() / stage1) if stage1 else 0.,
                    "adjacent_parity_pairs": adjacent, "alternating_sign_pairs": flips,
                    "alternating_sign_fraction_of_adjacent_parity_pairs": flips / adjacent if adjacent else float("nan"),
                    "topology_modified": False,
                })
            with atomic_output(OUT / "tie_maps" / f"realization_{realization_id}.npz") as stream:
                np.savez_compressed(stream, **usage)
            if realization_id in q["diverse_validation_subset"]["all_ids"]:
                examples.append((realization_id, usage["parity"]))
            progress(f"Final-sign audit {split} realization={realization_id}")
    write_csv(destination, rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (realization_id, parity) in zip(axes.flat, examples):
        axis.imshow(parity, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        axis.set_title(f"{realization_id}: parity {parity.mean():.4%}")
    fig.suptitle("Frozen final-sign use; topology unchanged")
    save_figure(fig, "final_sign_spatial_distribution.png")


def decompose(_):
    destination = OUT / "v00332q_criterion_decomposition.csv"
    if destination.exists():
        return
    whole = pd.read_csv(Q / "v00332q_multiseed_results.csv")
    whole["metric_scope"] = "whole_realization_milestone"
    whole["mean_elastic_normalized_rmse"] = whole[[f"{p}_normalized_rmse" for p in PROPERTIES]].mean(axis=1)
    whole["segmentation_contribution"] = -0.1 * whole["miou"]
    for prop in PROPERTIES:
        whole[prop + "_criterion_contribution"] = whole[prop + "_normalized_rmse"] / 3
    recomposed = whole["mean_elastic_normalized_rmse"] + whole["segmentation_contribution"]
    whole["criterion_recomposition_error"] = recomposed - whole["checkpoint_criterion"]
    if whole["criterion_recomposition_error"].abs().max() > 1e-10:
        raise RuntimeError("q whole criterion did not match its documented decomposition")
    samples = []
    for run in sorted((Q / "runs").glob("seed_*")):
        path = run / "training_log.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        _, seed, architecture = run.name.split("_", 2)
        for _, row in frame.iterrows():
            values = [row["sample_rmse_" + name + "_normalized"] for name in PROPERTIES]
            samples.append({
                "seed": int(seed), "architecture": architecture, "epoch": int(row["epoch"]),
                "metric_scope": "fixed_patch_validation_every_epoch_not_whole",
                "mean_elastic_normalized_rmse": float(np.mean(values)),
                **{f"{p}_normalized_rmse": x for p, x in zip(PROPERTIES, values)},
                **{f"{p}_criterion_contribution": x / 3 for p, x in zip(PROPERTIES, values)},
                "miou": row["sample_miou"], "segmentation_contribution": -0.1 * row["sample_miou"],
                "checkpoint_criterion": float(np.mean(values) - 0.1 * row["sample_miou"]),
                "validation_physics_raw_loss": row["validation_physics"],
                "validation_velocity_ssim_loss": row["validation_ssim"],
                **{f"class_{k}_iou": row.get(f"sample_class_{k}_iou", float("nan")) for k in range(3)},
            })
    write_csv(destination, pd.concat([whole, pd.DataFrame(samples)], ignore_index=True))
    strata = pd.read_csv(Q / "rgt_complexity_stratified_results.csv")
    write_csv(OUT / "v00332q_highdip_elastic_decomposition.csv", strata[
        strata["stratum"].isin(["all", "high_dip_continuous"])
    ])
    progress("q criterion decomposed: every logged epoch; whole metrics only at saved milestones")


def save_figure(fig, name):
    fig.tight_layout()
    with atomic_output(OUT / "figures" / name) as stream:
        fig.savefig(stream, format="png", dpi=140)
    plt.close(fig)


def select_probes():
    destination = OUT / "fixed_probe_patches.json"
    if destination.exists():
        return read_json(destination)
    dataset = IndexedRealizationPatches(DATASET, "validation")
    index = dataset.index
    eligible = index[(index["physics_eligible"] == 1)]
    # Select using input geometry/truth labels only, never checkpoint outputs.
    # Two patches per category, within the same six q validation realizations.
    q = q_contract()
    candidates = []
    for rid in q["diverse_validation_subset"]["all_ids"]:
        arrays = load_realization(DATASET, rid)
        masks = frozen_helpers._pixel_strata(arrays, load_faults(STAGE02, rid), q["adaptive_search"])
        for idx, row in eligible[eligible["realization_id"] == rid].iterrows():
            top, left, height, width = (int(row[k]) for k in ("top", "left", "raw_height", "raw_width"))
            spatial = np.s_[top:top + height, left:left + width]
            if not arrays["valid_mask"][spatial].any():
                continue
            candidates.append({
                "index": int(idx), "realization_id": int(rid),
                **{role: float(masks[role][spatial].mean()) for role in ("low_dip", "high_dip_continuous", "facies_boundary")},
            })
    selected = []
    for role in ("low_dip", "high_dip_continuous", "facies_boundary"):
        available = [x for x in candidates if x["index"] not in [p["index"] for p in selected]]
        ordered = sorted(available, key=lambda x: (-x[role], x["index"]))
        for item in ordered[:2]:
            selected.append({**item, "role": role})
    if len(selected) != 6:
        raise RuntimeError("Could not select six distinct native physics-eligible probes")
    write_json(destination, selected)
    return selected


def runtime(args):
    torch.set_num_threads(args.threads)
    print_torch_runtime()
    seed_everything(CONFIG["seeds"][0], deterministic_torch=True)
    return select_torch_device(args.device, require_cuda=True, context="v00332r bounded causal experiment")


def load_frozen_model(seed, epoch, device):
    config = training_config(seed, A)
    raw = torch.load(checkpoint(seed, epoch), map_location="cpu", weights_only=False)
    observed_graph = raw["config"]["model"]["experimental_graph"]
    expected_graph = config["model"]["experimental_graph"]
    for key, value in observed_graph.items():
        if expected_graph.get(key) != value:
            raise RuntimeError(f"Frozen q graph config differs from inherited r baseline: {key}")
    if observed_graph["rgt_topology"] != "rgt_v3_confidence_blocked":
        raise RuntimeError("Frozen q checkpoint is not the confidence-blocked corrected-RGT baseline")
    del raw
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
    model.set_norm_stats(read_json(DATASET / "normalization.json"))
    load_checkpoint(checkpoint(seed, epoch), model, restore_rng=False, map_location=device)
    model.eval()
    return model, config


def collect_jobs(directory, key):
    rows = []
    for path in sorted(directory.glob("*.json")):
        rows.extend(read_json(path)[key])
    return rows


def frozen_patch_audits(args):
    device = runtime(args)
    probes = select_probes()
    dataset = IndexedRealizationPatches(DATASET, "validation")
    normalization = _normalization_tensors(read_json(DATASET / "normalization.json"))
    weights_path = OUT / "frozen_class_weights.json"
    config = training_config(CONFIG["seeds"][0], A)
    if not weights_path.exists():
        progress("Computing unchanged training-only class weights")
        weights = _class_weights(
            IndexedRealizationPatches(DATASET, "train"),
            classes=int(config["model"]["classes"]),
            foreground_boost=float(config["training"]["class_weight_foreground_boost"]),
        )
        write_json(weights_path, {"values": weights.tolist(), "source": "all immutable training patches, q rule"})
    class_weights = torch.tensor(read_json(weights_path)["values"], device=device)
    directory = OUT / "audit_jobs/patches"
    for seed in CONFIG["seeds"]:
        for epoch in CONFIG["milestones"]:
            destination = directory / f"seed_{seed}_epoch_{epoch:04d}.json"
            if destination.exists():
                continue
            model, config = load_frozen_model(seed, epoch, device)
            state_sha = frozen_helpers._state_sha(model.state_dict())
            physics = physics_settings_from_config(config)
            weights = curriculum_from_config(config).weights_for_epoch(
                loss_weights_from_config(config, variant_definition(
                    "full", physics_weight=float(config["training"]["loss_weights"]["physics"])
                ).physics_weight), epoch - 1, 20
            )
            head_rows, gradient_rows, intervention_rows = [], [], []
            for probe in probes:
                values = _move_batch(default_collate([dataset[probe["index"]]]), device)
                time_tensor = torch.full((1,), CONFIG["diagnostics"]["time"], device=device)
                common = {"seed": seed, "epoch": epoch, **probe}
                handles = []
                for layer_index, layer in enumerate(model.graph.layers):
                    def capture(module, inputs, output, layer_index=layer_index):
                        edges, alpha = output[1]
                        for row in headwise_uniformity(edges, alpha, inputs[0].shape[0]):
                            head_rows.append({**common, "layer": layer_index + 1, **row})
                    handles.append(layer.register_forward_hook(capture))
                try:
                    with torch.inference_mode():
                        state = values["low"] + time_tensor[:, None, None, None] * (values["target"] - values["low"])
                        reference = model(state, time_tensor, values["avo"], values["low"], values["rgt"])
                        reference_velocity = reference.velocity.clone()
                finally:
                    for handle in handles:
                        handle.remove()
                for mode, attention, edge_attr in (
                    ("learned_current", "learned", "current"),
                    ("uniform_current", "uniform", "current"),
                    ("learned_zero", "learned", "zero"),
                    ("learned_relation_shuffle", "learned", "relation_preserving_shuffled"),
                    ("learned_legacy_roll", "learned", "shuffled"),
                ):
                    with graph_intervention(model, attention=attention, edge_attr=edge_attr), torch.inference_mode():
                        output = model(state, time_tensor, values["avo"], values["low"], values["rgt"])
                        change = (output.velocity - reference_velocity).square().mean().sqrt()
                        intervention_rows.append({
                            **common, "mode": mode, "velocity_rms_difference": float(change),
                            "relative_velocity_rms_difference": float(change / reference_velocity.square().mean().sqrt().clamp_min(1e-12)),
                            "scope": "fixed_teacher_forced_state_t0.5_not_endpoint",
                        })
                # Recreate a gradient-enabled, unmodified forward; no optimizer step.
                _, terms = _forward_objective(
                    model, values, time_tensor, normalization, weights, class_weights, physics,
                    ContrastiveSettings(), deterministic_contrastive=True,
                    contrastive_generator=None, adaptive_weighter=None,
                    graph_objective=graph_objective_from_config(config),
                )
                objectives = {
                    "elastic": weights.inversion * terms["inversion"],
                    "physics": weights.physics * terms["physics"],
                    "segmentation": weights.segmentation * terms["segmentation"],
                }
                if not all(torch.isfinite(value).all() for value in objectives.values()):
                    raise RuntimeError("Nonfinite frozen task objective")
                for row in task_gradient_rows(model, objectives):
                    gradient_rows.append({**common, **row, "effective_weights": json.dumps(asdict(weights), sort_keys=True)})
                del terms, objectives, output, reference, reference_velocity
            if frozen_helpers._state_sha(model.state_dict()) != state_sha:
                raise RuntimeError("An inference-only diagnostic mutated checkpoint state")
            write_json(destination, {"headwise": head_rows, "gradients": gradient_rows, "interventions": intervention_rows})
            del model
            torch.cuda.empty_cache()
            progress(f"Frozen patch/head/task-gradient audits seed={seed}, epoch={epoch}")
    write_csv(OUT / "attention_headwise_uniformity.csv", collect_jobs(directory, "headwise"))
    write_csv(OUT / "graph_task_gradient_coupling.csv", collect_jobs(directory, "gradients"))
    write_csv(OUT / "frozen_patch_interventions.csv", collect_jobs(directory, "interventions"))


def whole_metrics(prediction, labels, arrays, realization_id, config):
    normalization = read_json(DATASET / "normalization.json")
    y_std = np.asarray(normalization["y_std"])
    x_std = np.asarray(normalization["x_std"], np.float32)[:, None, None]
    with torch.inference_mode():
        tensor = torch.from_numpy(prediction[None])
        modeled = forward_avo_three_band_spec_torch(
            tensor[:, 0], tensor[:, 1], tensor[:, 2],
            forward_specification_from_mapping(config), sample_origin=0,
        )[0].numpy()
    squared_physics = ((modeled - arrays["avo_clean"]) / x_std) ** 2
    masks = frozen_helpers._pixel_strata(
        arrays, load_faults(STAGE02, realization_id), q_contract()["adaptive_search"]
    )
    rows = []
    for stratum, mask in masks.items():
        if not mask.any():
            continue
        row = {"stratum": stratum, "pixel_count": int(mask.sum()), "realization_id": realization_id}
        row.update(frozen_helpers._segmentation_metrics(labels, arrays["segmentation"], mask))
        for channel, prop in enumerate(PROPERTIES):
            rmse = float(np.sqrt(np.mean((prediction[channel][mask] - arrays["elastic"][channel][mask]) ** 2)))
            row[prop + "_rmse"] = rmse
            row[prop + "_normalized_rmse"] = rmse / float(y_std[channel])
            row[prop + "_ssim"] = frozen_helpers._global_ssim(prediction[channel][mask], arrays["elastic"][channel][mask])
        row["mean_elastic_normalized_rmse"] = float(np.mean([row[p + "_normalized_rmse"] for p in PROPERTIES]))
        row["exact_pp_rmse_clean_normalized"] = float(np.sqrt(squared_physics[:, mask].mean()))
        row["checkpoint_criterion"] = row["mean_elastic_normalized_rmse"] - .1 * row["miou"]
        if not all(np.isfinite(row[p + "_normalized_rmse"]) for p in PROPERTIES):
            raise RuntimeError("Nonfinite whole-realization elastic metric")
        rows.append(row)
    return rows


def infer_one(model, config, destination, realization_id, device, frozen_prediction=None):
    arrays = load_realization(DATASET, realization_id)
    if frozen_prediction is not None:
        with np.load(frozen_prediction, allow_pickle=False) as saved:
            prediction, labels = saved["prediction"], saved["segmentation_prediction"]
    elif destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            prediction, labels = saved["prediction"], saved["segmentation_prediction"]
    else:
        budget = CONFIG["bounded_training"]
        prediction, labels = infer_full_realization(
            model, avo=arrays["avo"], low=arrays["low"], rgt=arrays["rgt"],
            normalization=read_json(DATASET / "normalization.json"),
            patch_shape=tuple(budget["whole_patch_shape"]), stride=tuple(budget["whole_stride"]),
            steps=budget["whole_flow_steps"], batch_size=budget["whole_batch_size"],
            device=device, valid_mask=arrays["valid_mask"],
        )
        with atomic_output(destination) as stream:
            np.savez_compressed(stream, prediction=prediction, segmentation_prediction=labels)
    return whole_metrics(prediction, labels, arrays, realization_id, config)


def frozen_whole_audits(args):
    device = runtime(args)
    directory = OUT / "audit_jobs/whole"
    for seed in CONFIG["seeds"]:
        model, config = load_frozen_model(seed, 20, device)
        for mode, attention, edge_attr in (
            ("learned_current", "learned", "current"),
            ("uniform_current", "uniform", "current"),
            ("learned_zero", "learned", "zero"),
            ("learned_relation_shuffle", "learned", "relation_preserving_shuffled"),
            ("learned_legacy_roll", "learned", "shuffled"),
        ):
            for rid in q_contract()["diverse_validation_subset"]["all_ids"]:
                name = f"seed_{seed}_{mode}_{rid}"
                destination = directory / (name + ".json")
                if destination.exists():
                    continue
                frozen = Q / "runs" / f"seed_{seed}_corrected_rgt_gnn/whole_evaluation/epoch_0020/realization_{rid:07d}.npz"
                # Existing q baseline predictions are reused; no q experiment rerun.
                with graph_intervention(model, attention=attention, edge_attr=edge_attr):
                    rows = infer_one(model, config, OUT / "frozen_inference" / (name + ".npz"), rid, device,
                                     frozen_prediction=frozen if mode == "learned_current" else None)
                rows = [{"seed": seed, "epoch": 20, "mode": mode, **row} for row in rows]
                write_json(destination, {"metrics": rows})
                progress(f"Frozen whole inference {name}")
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(collect_jobs(directory, "metrics"))
    write_csv(OUT / "uniform_attention_inference_ablation.csv", frame[frame["mode"].isin(["learned_current", "uniform_current"])])
    write_csv(OUT / "edge_attr_relation_preserving_ablation.csv", frame[frame["mode"] != "uniform_current"])


def audit_all(args):
    prepare(args)
    tie_audit(args)
    decompose(args)
    frozen_patch_audits(args)
    frozen_whole_audits(args)
    audit_summary(args)


def paired_effects(frame, candidate, control, *, key="condition", epoch=20):
    """Paired seed effects, retaining properties rather than hiding them in mIoU."""
    frame = frame[frame["epoch"] == epoch]
    metrics = [f"{p}_normalized_rmse" for p in PROPERTIES] + [
        "mean_elastic_normalized_rmse", "checkpoint_criterion", "miou",
        "class_1_iou", "exact_pp_rmse_clean_normalized",
    ]
    grouped = frame.groupby(["seed", key])[metrics].mean()
    rows = []
    for seed in CONFIG["seeds"]:
        for metric in metrics:
            value = float(grouped.loc[(seed, candidate), metric])
            reference = float(grouped.loc[(seed, control), metric])
            sign = 1 if metric in {"miou", "class_1_iou"} else -1
            rows.append({
                "seed": seed, "candidate": candidate, "control": control, "metric": metric,
                "candidate_value": value, "control_value": reference,
                "delta_candidate_minus_control": value - reference,
                "relative_gain": sign * (value - reference) / max(abs(reference), 1e-12),
            })
    return pd.DataFrame(rows)


def effect_values(frame, metric):
    return frame[frame["metric"] == metric]["relative_gain"].to_numpy()


def all_favorable(frame, metric, margin=0.):
    values = effect_values(frame, metric)
    return bool(len(values) == len(CONFIG["seeds"]) and np.all(values > margin))


def coupling_class(cosine):
    thresholds = CONFIG["decision_thresholds"]
    if cosine > thresholds["gradient_orthogonal_cosine"]:
        return "COOPERATIVE"
    if abs(cosine) <= thresholds["gradient_orthogonal_cosine"]:
        return "ORTHOGONAL"
    if cosine > thresholds["gradient_severe_conflict_cosine"]:
        return "MILDLY_CONFLICTING"
    return "SEVERELY_CONFLICTING"


def audit_summary(_):
    """Completeness and interpretation gate for phases 1–6; no training here."""
    files = (
        "final_sign_tiebreak_audit.csv", "v00332q_criterion_decomposition.csv",
        "attention_headwise_uniformity.csv", "uniform_attention_inference_ablation.csv",
        "edge_attr_relation_preserving_ablation.csv", "graph_task_gradient_coupling.csv",
    )
    if any(not (OUT / name).exists() for name in files):
        raise RuntimeError("Phases 1–6 are incomplete; fresh training remains blocked")
    ties, q, attention, uniform, edge, gradients = [pd.read_csv(OUT / name) for name in files]
    n = len(CONFIG["seeds"]) * len(CONFIG["milestones"]) * 6
    if len(attention) != n * 2 * 4 or len(gradients) != n * 9:
        raise RuntimeError("Incomplete per-seed/milestone/head/task-gradient audit")
    if len(ties[ties.stratum == "all"]) != 90:
        raise RuntimeError("Tie audit must cover all 70 training and 20 validation realizations")
    for frame, count in ((uniform, 36), (edge, 72)):
        if len(frame[frame.stratum == "all"]) != count:
            raise RuntimeError("Incomplete six-realization frozen inference ablation")
    if not np.isfinite(gradients[["gradient_norm_a", "gradient_norm_b", "cosine_similarity"]]).all().all():
        raise RuntimeError("Gradient probe has nonfinite or undefined comparisons; inspect before training")
    uniform_effect = paired_effects(uniform[uniform.stratum == "all"], "uniform_current", "learned_current", key="mode")
    edge_effects = pd.concat([
        paired_effects(edge[edge.stratum == "all"], candidate, "learned_current", key="mode")
        for candidate in ("learned_zero", "learned_relation_shuffle", "learned_legacy_roll")
    ], ignore_index=True)
    write_csv(OUT / "frozen_uniform_paired_effects.csv", uniform_effect)
    write_csv(OUT / "frozen_edge_attr_paired_effects.csv", edge_effects)
    epoch20 = gradients[gradients.epoch == 20]
    grouped = epoch20.groupby(["parameter_group", "objective_a", "objective_b"]).cosine_similarity.agg(["mean", "min", "max"])
    coupling = [{
        "parameter_group": group, "objective_a": first, "objective_b": second,
        "cosine_mean": float(row["mean"]), "cosine_min": float(row["min"]),
        "cosine_max": float(row["max"]), "classification": coupling_class(float(row["mean"])),
    } for (group, first, second), row in grouped.iterrows()]
    # Independent additive contributions to the q whole validation criterion.
    q_whole = q[q.metric_scope == "whole_realization_milestone"]
    columns = ["mean_elastic_normalized_rmse", "segmentation_contribution", "checkpoint_criterion"] + [p + "_criterion_contribution" for p in PROPERTIES]
    q_means = q_whole.groupby(["seed", "epoch", "architecture"])[columns].mean()
    differences = []
    for seed in CONFIG["seeds"]:
        for epoch in CONFIG["milestones"]:
            delta = q_means.loc[(seed, epoch, "corrected_rgt_gnn")] - q_means.loc[(seed, epoch, "cartesian_gnn")]
            differences.append({"seed": seed, "epoch": epoch, **{key: float(value) for key, value in delta.items()},
                                "segmentation_flips_elastic_ranking": bool(delta["checkpoint_criterion"] * delta["mean_elastic_normalized_rmse"] < 0)})
    write_csv(OUT / "q_corrected_cartesian_criterion_contributions.csv", differences)
    q20 = pd.DataFrame(differences)
    q20 = q20[q20.epoch == 20]
    elastic_magnitude = float(q20.mean_elastic_normalized_rmse.abs().sum())
    segmentation_magnitude = float(q20.segmentation_contribution.abs().sum())
    q_high = pd.read_csv(OUT / "v00332q_highdip_elastic_decomposition.csv")
    q_high = q_high[(q_high.epoch == 20) & (q_high.stratum == "high_dip_continuous")]
    q_high_mean = q_high.groupby(["seed", "architecture"]).normalized_rmse.mean()
    q_high_delta = q_high_mean.xs("corrected_rgt_gnn", level="architecture") - q_high_mean.xs("cartesian_gnn", level="architecture")
    margin = CONFIG["decision_thresholds"]["attention_equivalence_relative_margin"]
    equivalent = all(np.all(np.abs(effect_values(uniform_effect, prop + "_normalized_rmse")) <= margin) for prop in PROPERTIES)
    velocity = pd.read_csv(OUT / "frozen_patch_interventions.csv")
    velocity_max = float(velocity[velocity["mode"] == "uniform_current"].relative_velocity_rms_difference.max())
    parity_fraction = float(ties[ties.stratum == "all"].final_parity_fraction.max())
    parity_status = (
        "NEGLIGIBLE" if parity_fraction < CONFIG["decision_thresholds"]["parity_negligible_fraction"]
        else "NEEDS_FUTURE_REDESIGN" if parity_fraction >= CONFIG["decision_thresholds"]["parity_material_fraction"]
        else "MATERIAL_BUT_UNTESTED"
    )
    material = CONFIG["decision_thresholds"]["material_relative_elastic_effect"]
    shuffled = edge_effects[edge_effects.candidate == "learned_relation_shuffle"]
    zeroed = edge_effects[edge_effects.candidate == "learned_zero"]
    if all_favorable(shuffled, "mean_elastic_normalized_rmse", material):
        edge_status = "MISALIGNED"
    elif all_favorable(shuffled, "mean_elastic_normalized_rmse", -np.inf) and np.all(effect_values(shuffled, "mean_elastic_normalized_rmse") < -material) and np.all(effect_values(zeroed, "mean_elastic_normalized_rmse") < -material):
        edge_status = "USEFUL"
    elif all(np.all(np.abs(effect_values(frame, "mean_elastic_normalized_rmse")) <= margin) for frame in (shuffled, zeroed)):
        edge_status = "NEUTRAL"
    else:
        edge_status = "UNRESOLVED"
    summary = {
        "status": "PHASES_1_TO_6_COMPLETE", "audit_file_sha256": {name: file_sha(OUT / name) for name in files},
        "FINAL_SIGN_TIEBREAK_STATUS": parity_status, "maximum_valid_realization_parity_fraction": parity_fraction,
        "uniform_frozen_inference_status": "ATTENTION_NOT_ADDING_MATERIAL_VALUE" if equivalent and velocity_max < material else "ATTENTION_EFFECT_REQUIRES_TRAINED_CONTROL",
        "uniform_max_relative_velocity_change": velocity_max,
        "uniform_three_property_seed_effects_within_operational_margin": equivalent,
        "headwise_entropy_minimum": float(attention.normalized_entropy.min()),
        "headwise_kl_maximum": float(attention.kl_from_uniform.max()),
        "epoch20_coupling": coupling, "AVO_EDGE_ATTR_STATUS": edge_status,
        "q_epoch20_decomposition": {
            "segmentation_flips_elastic_ranking_seed_count": int(q20.segmentation_flips_elastic_ranking.sum()),
            "segmentation_absolute_share_of_elastic_plus_segmentation_deltas": segmentation_magnitude / max(elastic_magnitude + segmentation_magnitude, 1e-12),
            "vp_favorable_seed_count": int((q20.vp_criterion_contribution < 0).sum()),
            "vs_favorable_seed_count": int((q20.vs_criterion_contribution < 0).sum()),
            "density_favorable_seed_count": int((q20.density_criterion_contribution < 0).sum()),
            "highdip_mean_elastic_delta_by_seed": {str(seed): float(value) for seed, value in q_high_delta.items()},
            "highdip_elastic_favorable_all_seeds_without_segmentation": bool((q_high_delta < 0).all()),
        },
        "clean_shuffle_improves_mean_elastic_all_seeds": all_favorable(shuffled, "mean_elastic_normalized_rmse"),
        "legacy_roll_improves_mean_elastic_all_seeds": all_favorable(edge_effects[edge_effects.candidate == "learned_legacy_roll"], "mean_elastic_normalized_rmse"),
        "clean_shuffle_improves_combined_criterion_all_seeds": all_favorable(shuffled, "checkpoint_criterion"),
        "legacy_roll_improves_combined_criterion_all_seeds": all_favorable(edge_effects[edge_effects.candidate == "learned_legacy_roll"], "checkpoint_criterion"),
        "elastic_gradient_definition": "effective inversion-weighted flow MSE + full-property MSE + velocity SSIM, with frozen epoch curriculum",
        "physics_gradient_definition": "effective exact-PP context loss, native-grid physics-eligible patches",
        "segmentation_gradient_definition": "effective segmentation-weighted class-weighted CE + Dice",
        "frozen_checkpoint_inference_only": True, "topology_modified": False,
    }
    write_json(OUT / "pretraining_audit_summary.json", summary)
    progress(json.dumps(summary, indent=2))


def require_audits():
    summary_path = OUT / "pretraining_audit_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Frozen phases 1–6 must complete before training")
    summary = read_json(summary_path)
    if summary["status"] != "PHASES_1_TO_6_COMPLETE":
        raise RuntimeError("Frozen phases 1–6 did not pass")
    for name, expected in summary["audit_file_sha256"].items():
        if file_sha(OUT / name) != expected:
            raise RuntimeError(f"Frozen audit output changed: {name}")


def validate(_):
    """Unit/regression validation, never training or data regeneration."""
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "pytest", "-ra"],
        ["git", "diff", "--check"],
    ]
    results = []
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH="src")
    for command in commands:
        progress("Validation: " + " ".join(command))
        result = subprocess.run(command, cwd=REPOSITORY, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        results.append({"command": command, "exit_code": result.returncode, "output": result.stdout})
        write_json(OUT / "validation.json", {
            "commands": results,
            "all_passed": len(results) == len(commands) and all(x["exit_code"] == 0 for x in results),
            "source_sha256": execution_sources(),
        })
        if result.returncode:
            raise RuntimeError("Validation failed; training will not start")


def fresh_initializations():
    rows = []
    for seed in CONFIG["seeds"]:
        seed_everything(seed, deterministic_torch=True)
        reference = build_sage_avo_variant("full", **sage_avo_model_kwargs(training_config(seed, A)))
        state = reference.state_dict()
        path = OUT / "initial_states" / f"seed_{seed}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if frozen_helpers._state_sha(saved) != frozen_helpers._state_sha(state):
                raise RuntimeError("Existing r initialization does not match fresh seeded state")
        else:
            with atomic_output(path) as stream:
                torch.save(state, stream)
        for condition in CONFIG["conditions"]:
            candidate = build_sage_avo_variant("full", **sage_avo_model_kwargs(training_config(seed, condition)))
            if list(candidate.state_dict()) != list(state):
                raise RuntimeError(f"State-key mismatch: {condition}")
            candidate.load_state_dict(state, strict=True)
            if frozen_helpers._state_sha(candidate.state_dict()) != frozen_helpers._state_sha(state):
                raise RuntimeError(f"Initial-state mismatch: {condition}")
            count = sum(p.numel() for p in candidate.parameters())
            if count != sum(p.numel() for p in reference.parameters()):
                raise RuntimeError(f"Parameter-count mismatch: {condition}")
            rows.append({"seed": seed, "condition": condition, "parameter_count": count,
                         "state_key_equality": True, "initial_state_sha256": frozen_helpers._state_sha(state),
                         "initial_state_file_sha256": file_sha(path),
                         "train_schedule_sha256": read_json(OUT / "fixed_patch_schedule.json")["train_schedule_sha256"],
                         "query_key_instantiated_but_attention_inactive": condition == B,
                         "fresh_initialization_no_q_checkpoint": True})
    write_csv(OUT / "initialization_matching.csv", rows)


def train(args):
    require_audits()
    prepare(args)
    validation_path = OUT / "validation.json"
    if (not validation_path.exists() or not read_json(validation_path)["all_passed"]
            or len(read_json(validation_path)["commands"]) != 3):
        raise RuntimeError("Run passing validation before matched training")
    sources = execution_sources()
    if read_json(validation_path)["source_sha256"] != sources:
        raise RuntimeError("Code/config changed after validation; revalidate before training")
    if args.loaded_source_sha256 != sources:
        raise RuntimeError("Code/config changed while this runner was alive; restart it before training")
    frozen_source_path = OUT / "matched_training_source_contract.json"
    if frozen_source_path.exists():
        if read_json(frozen_source_path)["source_sha256"] != sources:
            raise RuntimeError("Refusing to mix changed source/config into an existing matched experiment")
    else:
        write_json(frozen_source_path, {"source_sha256": sources, "validation_sha256": file_sha(validation_path)})
    runtime(args)
    fresh_initializations()
    schedule = read_json(OUT / "fixed_patch_schedule.json")
    for seed in CONFIG["seeds"]:
        for condition in CONFIG["conditions"]:
            if execution_sources() != sources:
                raise RuntimeError("Source/config changed between matched conditions; stopped before next condition")
            run = OUT / "runs" / f"seed_{seed}_{condition}"
            if (run / "manifest.json").exists() and read_json(run / "manifest.json").get("last_completed_epoch", 0) >= 20:
                continue
            config = training_config(seed, condition)
            progress(f"Fresh/resumable bounded training seed={seed} condition={condition}; hard stop 20")
            train_controlled_variant(
                repository=REPOSITORY, config_path=CONFIG_PATH, config=config,
                dataset_directory=DATASET, experiment_directory=OUT, variant="full", device_name=args.device,
                epochs_override=20, max_train_batches=35, max_validation_batches=20,
                run_name=run.name, resume_from=run / "last.pt" if (run / "last.pt").exists() else None,
                stop_after_epoch=20, fixed_train_indices_by_epoch=schedule["train_schedule"],
                fixed_validation_indices=schedule["validation_indices"],
                initial_model_state=OUT / "initial_states" / f"seed_{seed}.pt",
                finite_state_check_batches=(1, 18, 35), abort_on_nonfinite=True,
            )
            log = pd.read_csv(run / "training_log.csv")
            if log.epoch.tolist() != list(range(1, 21)):
                raise RuntimeError("Training did not produce exactly epochs 1–20")
            base_lr = float(config["training"]["learning_rate"])
            expected_lr = 1e-6 + (base_lr - 1e-6) * (1 + np.cos(np.pi * np.arange(20) / 20)) / 2
            if not np.allclose(log.learning_rate, expected_lr, rtol=1e-12, atol=1e-15):
                raise RuntimeError("Observed LR schedule differs from the frozen 20-epoch cosine")
            IndexedRealizationPatches._load.cache_clear()
            torch.cuda.empty_cache()
            progress(f"Completed seed={seed} condition={condition}")


def evaluate(args, *, all_validation=False):
    device = runtime(args)
    directory = OUT / ("evaluation_jobs/all20" if all_validation else "evaluation_jobs/diverse6")
    ids = q_contract()["split_ids"]["validation"] if all_validation else q_contract()["diverse_validation_subset"]["all_ids"]
    epochs = [20] if all_validation else CONFIG["milestones"]
    for seed in CONFIG["seeds"]:
        for condition in CONFIG["conditions"]:
            config = training_config(seed, condition)
            model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
            model.set_norm_stats(read_json(DATASET / "normalization.json"))
            run = OUT / "runs" / f"seed_{seed}_{condition}"
            for epoch in epochs:
                path = run / f"checkpoint_epoch_{epoch:04d}.pt"
                if not path.exists():
                    raise RuntimeError(f"Training checkpoint incomplete: {path}")
                load_checkpoint(path, model, restore_rng=False, map_location=device)
                model.eval()
                for rid in ids:
                    name = f"seed_{seed}_{condition}_epoch_{epoch:04d}_{rid}"
                    destination = directory / (name + ".json")
                    if destination.exists():
                        continue
                    # Same destination allows all20 to reuse the six completed predictions.
                    rows = infer_one(model, config, OUT / "predictions" / (name + ".npz"), rid, device)
                    rows = [{"seed": seed, "epoch": epoch, "condition": condition,
                             "evaluation_scope": "all20" if all_validation else "diverse6", **row} for row in rows]
                    write_json(destination, {"metrics": rows})
                    progress(f"Whole evaluation {name}")
                if not all_validation:
                    evaluate_heads(model, seed, condition, epoch, device)
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(collect_jobs(directory, "metrics"))
    if all_validation:
        write_csv(OUT / "v00332r_all20_validation_results.csv", frame)
    else:
        write_csv(OUT / "v00332r_multiseed_results.csv", frame[frame.stratum == "all"])
        write_csv(OUT / "v00332r_highdip_results.csv", frame[frame.stratum == "high_dip_continuous"])
        write_csv(OUT / "v00332r_all_strata_results.csv", frame)
        write_csv(OUT / "v00332r_attention_results.csv", collect_jobs(OUT / "evaluation_jobs/attention", "headwise"))
        write_csv(OUT / "v00332r_segmentation_decoupling_results.csv", frame[frame.condition.isin([A, C])])


def evaluate_heads(model, seed, condition, epoch, device):
    destination = OUT / "evaluation_jobs/attention" / f"seed_{seed}_{condition}_{epoch:04d}.json"
    if destination.exists():
        return
    rows = []
    dataset = IndexedRealizationPatches(DATASET, "validation")
    for probe in select_probes():
        values = _move_batch(default_collate([dataset[probe["index"]]]), device)
        handles = []
        for layer_index, layer in enumerate(model.graph.layers):
            def capture(module, inputs, output, layer_index=layer_index):
                edges, alpha = output[1]
                rows.extend({"seed": seed, "condition": condition, "epoch": epoch, **probe,
                             "layer": layer_index + 1, **row}
                            for row in headwise_uniformity(edges, alpha, inputs[0].shape[0]))
            handles.append(layer.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                t = torch.full((1,), .5, device=device)
                state = .5 * values["low"] + .5 * values["target"]
                model(state, t, values["avo"], values["low"], values["rgt"])
        finally:
            for handle in handles:
                handle.remove()
    write_json(destination, {"headwise": rows})
    IndexedRealizationPatches._load.cache_clear()


def all20_gate(_):
    frame = pd.read_csv(OUT / "v00332r_multiseed_results.csv")
    if len(frame) != 432 or len(frame[frame.epoch == 20]) != 72:
        raise RuntimeError("All three seeds/four conditions/six milestones/six sections must complete")
    comparisons = {condition: paired_effects(frame, condition, D) for condition in (A, B, C)}
    decoupled = paired_effects(frame, C, A)
    material = CONFIG["decision_thresholds"]["material_relative_elastic_effect"]
    density_margin = CONFIG["decision_thresholds"]["major_relative_density_degradation"]
    metrics = [p + "_normalized_rmse" for p in PROPERTIES] + ["mean_elastic_normalized_rmse"]
    gates = elastic_extension_gate(
        {condition: np.column_stack([effect_values(result, metric) for metric in metrics])
         for condition, result in comparisons.items()},
        np.column_stack([effect_values(decoupled, metric) for metric in metrics]),
        material_margin=material, density_degradation_margin=density_margin,
    )
    result = {"passed": any(bool(value) for value in gates.values()), "gates": gates,
              "mIoU_used_for_gate": False, "primary_epoch": 20,
              "bounded_results_sha256": file_sha(OUT / "v00332r_multiseed_results.csv")}
    write_json(OUT / "all20_gate.json", result)
    progress(f"All-20 elastic gate: {json.dumps(result)}")
    return result["passed"]


def figures(whole, high, comparisons):
    labels = {A: "A: RGT learned/shared", B: "B: RGT uniform/shared",
              C: "C: RGT learned/detached", D: "D: Cartesian learned/shared"}
    last = whole[whole.epoch == 20].groupby(["seed", "condition"]).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, prop in zip(axes, PROPERTIES):
        for condition in CONFIG["conditions"]:
            values = last.xs(condition, level="condition")[prop + "_normalized_rmse"]
            axis.plot(values.index.astype(str), values, marker="o", label=labels[condition])
        axis.set_title(prop + " epoch-20 normalized RMSE")
    axes[0].legend(fontsize=7)
    save_figure(fig, "01_elastic_by_seed.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    high_effect = paired_effects(high, A, D)
    for axis, prop in zip(axes, PROPERTIES):
        for name, frame in (("Global", comparisons[A]), ("High dip", high_effect)):
            axis.plot([str(s) for s in CONFIG["seeds"]], 100 * effect_values(frame, prop + "_normalized_rmse"), marker="o", label=name)
        axis.axhline(0, color="grey", linewidth=.7)
        axis.set_title(prop + " RGT vs Cartesian gain (%)")
        axis.legend()
    save_figure(fig, "02_global_vs_highdip.png")
    heads = pd.read_csv(OUT / "attention_headwise_uniformity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for layer in (1, 2):
        for head in range(4):
            frame = heads[(heads.layer == layer) & (heads["head"] == head)].groupby("epoch").mean(numeric_only=True)
            for column, metric in enumerate(("normalized_entropy", "kl_from_uniform")):
                axes[layer - 1, column].plot(frame.index, frame[metric], marker="o", label=f"head {head}")
                axes[layer - 1, column].set_title(f"Frozen q layer {layer}: {metric}")
        axes[layer - 1, 0].legend()
    save_figure(fig, "03_headwise_entropy_kl.png")
    uniform = pd.read_csv(OUT / "frozen_uniform_paired_effects.csv")
    trained_uniform = paired_effects(whole, B, A)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, prop in zip(axes, PROPERTIES):
        axis.plot([str(s) for s in CONFIG["seeds"]], 100 * effect_values(uniform, prop + "_normalized_rmse"), marker="o", label="Frozen q inference only")
        axis.plot([str(s) for s in CONFIG["seeds"]], 100 * effect_values(trained_uniform, prop + "_normalized_rmse"), marker="o", label="Fresh r matched training")
        axis.axhline(0, color="grey", linewidth=.7)
        axis.set_title(prop + " uniform vs learned gain (%)")
    axes[0].legend(fontsize=8)
    save_figure(fig, "04_learned_vs_uniform.png")
    gradients = pd.read_csv(OUT / "graph_task_gradient_coupling.csv")
    gradients = gradients[gradients.epoch == 20]
    tasks = ("elastic", "physics", "segmentation")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, group in zip(axes, ("graph.node_projection", "graph.layers.0", "graph.layers.1")):
        matrix = np.eye(3)
        means = gradients[gradients.parameter_group == group].groupby(["objective_a", "objective_b"]).cosine_similarity.mean()
        for (first, second), value in means.items():
            i, j = tasks.index(first), tasks.index(second)
            matrix[i, j] = matrix[j, i] = value
        axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(3), tasks, rotation=30)
        axis.set_yticks(range(3), tasks)
        axis.set_title(group)
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.3f}", ha="center", va="center")
    save_figure(fig, "05_task_gradient_cosines.png")
    detached = paired_effects(whole, C, A)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, prop in zip(axes, PROPERTIES):
        axis.bar([str(s) for s in CONFIG["seeds"]], 100 * effect_values(detached, prop + "_normalized_rmse"))
        axis.axhline(0, color="grey", linewidth=.7)
        axis.set_title(prop + " detached vs shared gain (%)")
    save_figure(fig, "06_segmentation_detachment.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, seed in zip(axes, CONFIG["seeds"]):
        for condition in CONFIG["conditions"]:
            frame = whole[(whole.seed == seed) & (whole.condition == condition)].groupby("epoch").class_1_iou.mean()
            axis.plot(frame.index, frame, marker="o", label=labels[condition])
        axis.set_title(f"Seed {seed}: whole class-1 IoU")
    axes[0].legend(fontsize=7)
    save_figure(fig, "07_class1_trajectory.png")
    edges = pd.read_csv(OUT / "frozen_edge_attr_paired_effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, prop in zip(axes, PROPERTIES):
        for mode in ("learned_zero", "learned_legacy_roll", "learned_relation_shuffle"):
            values = effect_values(edges[edges.candidate == mode], prop + "_normalized_rmse")
            axis.plot([str(s) for s in CONFIG["seeds"]], 100 * values, marker="o", label=mode)
        axis.axhline(0, color="grey", linewidth=.7)
        axis.set_title(prop + " frozen q intervention gain (%)")
    axes[0].legend(fontsize=7)
    save_figure(fig, "08_old_vs_semantic_shuffle.png")


def markdown_table(frame):
    """Small report tables without introducing an optional tabulate dependency."""
    frame = frame.copy()
    for column in frame.select_dtypes(include="number"):
        frame[column] = frame[column].map(lambda value: f"{value:.6g}")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def finalize(args):
    require_audits()
    prepare(args)
    gate = read_json(OUT / "all20_gate.json")
    whole = pd.read_csv(OUT / "v00332r_multiseed_results.csv")
    high = pd.read_csv(OUT / "v00332r_highdip_results.csv")
    all20 = OUT / "v00332r_all20_validation_results.csv"
    decision_whole, decision_high = whole, high
    decision_scope = "diverse6"
    if gate["passed"]:
        if not all20.exists():
            raise RuntimeError("All-20 gate passed, but expanded validation is incomplete")
        expanded = pd.read_csv(all20)
        if len(expanded[expanded.stratum == "all"]) != 240:
            raise RuntimeError("All-20 validation lacks complete 3x4x20 coverage")
        write_csv(OUT / "all20_epoch20_paired_elastic_effects.csv", pd.concat([
            paired_effects(expanded[expanded.stratum == "all"], condition, D)
            for condition in (A, B, C)
        ], ignore_index=True))
        decision_whole = expanded[expanded.stratum == "all"]
        decision_high = expanded[expanded.stratum == "high_dip_continuous"]
        decision_scope = "all20_after_prespecified_elastic_gate"
    comparisons = {condition: paired_effects(decision_whole, condition, D) for condition in (A, B, C)}
    uniform = paired_effects(decision_whole, B, A)
    detached = paired_effects(decision_whole, C, A)
    high_effect = paired_effects(decision_high, A, D)
    write_csv(OUT / "epoch20_paired_elastic_effects.csv", pd.concat([*comparisons.values(), uniform, detached], ignore_index=True))
    margin = CONFIG["decision_thresholds"]["material_relative_elastic_effect"]
    equivalence = CONFIG["decision_thresholds"]["attention_equivalence_relative_margin"]
    audit = read_json(OUT / "pretraining_audit_summary.json")
    if np.all(effect_values(uniform, "mean_elastic_normalized_rmse") < -margin):
        attention_status = "USEFUL"
    elif all_favorable(uniform, "mean_elastic_normalized_rmse", margin):
        attention_status = "HARMFUL"
    elif (all(np.all(np.abs(effect_values(uniform, p + "_normalized_rmse")) <= equivalence) for p in PROPERTIES)
          and audit["uniform_frozen_inference_status"] == "ATTENTION_NOT_ADDING_MATERIAL_VALUE"):
        attention_status = "NEAR_UNIFORM_REDUNDANT"
    else:
        attention_status = "UNRESOLVED"
    elastic_conflict = [row for row in audit["epoch20_coupling"]
                        if row["objective_a"] == "elastic" and row["objective_b"] == "segmentation"]
    conflict = any(row["cosine_mean"] < -CONFIG["decision_thresholds"]["gradient_orthogonal_cosine"] for row in elastic_conflict)
    detached_better = all_favorable(detached, "mean_elastic_normalized_rmse", margin)
    density_safe = bool(np.all(effect_values(detached, "density_normalized_rmse") >= -CONFIG["decision_thresholds"]["major_relative_density_degradation"]))
    if conflict and detached_better and density_safe:
        segmentation_status = "ELASTIC_CONFLICT"
    elif np.all(effect_values(detached, "mean_elastic_normalized_rmse") < -margin):
        segmentation_status = "HELPFUL"
    elif np.all(np.abs(effect_values(detached, "mean_elastic_normalized_rmse")) <= equivalence):
        segmentation_status = "NEUTRAL"
    else:
        segmentation_status = "UNRESOLVED"
    baseline = comparisons[A]
    reproduced_conditions = [condition for condition, result in comparisons.items()
                             if (all_favorable(result, "vp_normalized_rmse") and all_favorable(result, "vs_normalized_rmse"))
                             or all_favorable(result, "mean_elastic_normalized_rmse")]
    reproducible = bool(reproduced_conditions)
    highdip_reproduced_conditions = [condition for condition in (A, B, C)
                                    if all_favorable(paired_effects(decision_high, condition, D), "mean_elastic_normalized_rmse")]
    high_gain = bool(highdip_reproduced_conditions)
    if reproducible:
        rgt_status = "REPRODUCIBLE"
    elif high_gain:
        rgt_status = "HIGH_DIP_ONLY"
    elif any((effect_values(result, "mean_elastic_normalized_rmse") > 0).any() for result in comparisons.values()):
        rgt_status = "SEED_DEPENDENT"
    else:
        rgt_status = "NO_GAIN"
    density = effect_values(baseline, "density_normalized_rmse")
    density_status = ("NEUTRAL" if np.all(np.abs(density) <= equivalence)
                      else "BENEFITS" if np.all(density > 0)
                      else "DEGRADES" if np.all(density < 0) else "SEED_DEPENDENT")
    # Prespecified interpretation order: identified coupling mechanism, then
    # learned-vs-uniform mechanism, then spatial/global reproducibility.
    if segmentation_status == "ELASTIC_CONFLICT":
        classification = "SEGMENTATION_COUPLING_LIMITS_RGT"
    elif attention_status == "USEFUL":
        classification = "TRANSFORMER_ATTENTION_ADDS_VALUE"
    elif any(condition in reproduced_conditions for condition in (A, B)) and attention_status == "NEAR_UNIFORM_REDUNDANT":
        classification = "RGT_TOPOLOGY_GAIN_ATTENTION_REDUNDANT"
    elif reproducible:
        classification = "RGT_GLOBAL_GAIN_REPRODUCED"
    elif high_gain:
        classification = "RGT_ELASTIC_GAIN_HIGH_DIP_ONLY"
    elif rgt_status == "SEED_DEPENDENT":
        classification = "RGT_STILL_SEED_DEPENDENT"
    else:
        classification = "RGT_NO_REPRODUCIBLE_ELASTIC_GAIN"
    # Secondary selection reports the six whole-evaluation milestone candidates,
    # and separately the existing fixed-patch-selected checkpoint epoch.
    grouped = whole.groupby(["seed", "condition", "epoch"]).mean(numeric_only=True).reset_index()
    selected = grouped.loc[grouped.groupby(["seed", "condition"]).checkpoint_criterion.idxmin()].copy()
    selected["selection_scope"] = "minimum mean whole criterion among six validation milestones; secondary only"
    patch_selections = []
    for seed in CONFIG["seeds"]:
        for condition in CONFIG["conditions"]:
            run = OUT / "runs" / f"seed_{seed}_{condition}"
            log = pd.read_csv(run / "training_log.csv")
            best = log.loc[log.sample_criterion.idxmin()]
            patch_selections.append({"seed": seed, "condition": condition,
                                     "validation_patch_selected_epoch": int(best.epoch),
                                     "validation_patch_selected_criterion": float(best.sample_criterion)})
    selected = selected.merge(pd.DataFrame(patch_selections), on=["seed", "condition"])
    write_csv(OUT / "validation_selected_checkpoints_secondary.csv", selected)
    epoch20 = decision_whole[decision_whole.epoch == 20].groupby(["seed", "condition"]).mean(numeric_only=True)
    variability = {condition: float(epoch20.xs(condition, level="condition").mean_elastic_normalized_rmse.std(ddof=1)) for condition in CONFIG["conditions"]}
    next_proposal = {
        "SEGMENTATION_COUPLING_LIMITS_RGT": "Propose explicit task-specific graph streams, with matched parameter and budget controls.",
        "TRANSFORMER_ATTENTION_ADDS_VALUE": "Retain TransformerConv; propose a separate fault-aware topology experiment.",
        "RGT_TOPOLOGY_GAIN_ATTENTION_REDUNDANT": "Propose a simpler topology-driven learned message operator with matched inference/training comparisons.",
        "RGT_ELASTIC_GAIN_HIGH_DIP_ONLY": "Propose confidence/adaptive use of RGT only where steering carries informative structure.",
        "RGT_GLOBAL_GAIN_REPRODUCED": "Propose an independently seeded confirmation before production-scale claims.",
        "RGT_STILL_SEED_DEPENDENT": "Audit representation and data information before adding architecture.",
        "RGT_NO_REPRODUCIBLE_ELASTIC_GAIN": "Audit representation and data information; do not infer theoretical impossibility from this bounded test.",
    }[classification]
    summary = {
        "status": "V00332R_COMPLETE", "primary_classification": classification,
        "classification_scope": f"fixed epoch 20, {decision_scope}; bounded comparison retained separately; no best-epoch substitution",
        "ATTENTION_STATUS": attention_status, "SEGMENTATION_GRAPH_STATUS": segmentation_status,
        "RGT_ELASTIC_STATUS": rgt_status, "DENSITY_STATUS": density_status,
        "reproducible_RGT_conditions_vs_Cartesian": reproduced_conditions,
        "highdip_reproducible_RGT_conditions_vs_Cartesian": highdip_reproduced_conditions,
        "density_status_scope": "A corrected-RGT learned/shared versus D Cartesian; all other paired density effects retained",
        "AVO_EDGE_ATTR_STATUS": audit["AVO_EDGE_ATTR_STATUS"],
        "FINAL_SIGN_TIEBREAK_STATUS": audit["FINAL_SIGN_TIEBREAK_STATUS"],
        "all20_gate": gate, "all20_executed": gate["passed"],
        "hypotheses": {
            "H1_RGT_vp_vs_benefit_all_seeds": all_favorable(baseline, "vp_normalized_rmse") and all_favorable(baseline, "vs_normalized_rmse"),
            "H2_uniform_learned_operationally_similar": attention_status == "NEAR_UNIFORM_REDUNDANT",
            "H3_measured_elastic_segmentation_graph_conflict": conflict,
            "H4_detachment_reproducible_material_elastic_improvement": detached_better,
            "H4_cross_seed_standard_deviation": variability,
            "H5_highdip_gain_greater_than_global_all_seeds": bool(np.all(effect_values(high_effect, "mean_elastic_normalized_rmse") > effect_values(baseline, "mean_elastic_normalized_rmse"))),
            "H6_clean_shuffle_changes_all_seed_elastic_conclusion": audit["clean_shuffle_improves_mean_elastic_all_seeds"] != audit["legacy_roll_improves_mean_elastic_all_seeds"],
            "H6_clean_shuffle_changes_all_seed_combined_conclusion": audit["clean_shuffle_improves_combined_criterion_all_seeds"] != audit["legacy_roll_improves_combined_criterion_all_seeds"],
        },
        "epoch20_paired_effects": pd.concat([*comparisons.values(), uniform, detached], ignore_index=True).to_dict("records"),
        "epoch20_highdip_paired_effects": high_effect.to_dict("records"),
        "next_experiment_proposal_only": next_proposal,
        "scientific_limits": ["Three seeds and fixed short patch budget are not production evidence",
                              "Operational margins are not equivalence tests or statistical significance",
                              "Physics/SSIM in whole sections are not identical tensor metrics to teacher-forced training losses",
                              "No test-set or field validation used; no new topology, losses, flow or exact-PP operator"],
        "production_training": False, "commit_or_push_performed": False,
        "review_branch_unchanged": True, "q_inputs_verified_unchanged": True,
    }
    write_json(OUT / "v00332r_summary.json", summary)
    figures(whole, high, {condition: paired_effects(whole, condition, D) for condition in (A, B, C)})
    simple = epoch20.reset_index()[["seed", "condition"] + [p + "_normalized_rmse" for p in PROPERTIES] + ["miou", "class_1_iou", "checkpoint_criterion"]]
    q_contributions = pd.read_csv(OUT / "q_corrected_cartesian_criterion_contributions.csv")
    q_contributions = q_contributions[q_contributions.epoch == 20]
    init = pd.read_csv(OUT / "initialization_matching.csv")
    report = f"""# v00332r — topology, attention and segmentation coupling

Primary classification: **{classification}**.

This is a bounded causal diagnostic, not a release or production result. All
four conditions retain the same operator, data split, prior, normalization,
patch schedule, 20-epoch cosine schedule and non-graph objectives. Fresh
initializations were used; no q training state was loaded into r training.

## Conditions and matching

- A: corrected confidence-blocked RGT, learned attention, shared segmentation gradients.
- B: same graph/values/roots, uniform incoming attention; query/key parameters remain instantiated but unused by attention.
- C: same predictions and parameters as A; segmentation head reads graph_spatial.detach(). This also blocks segmentation gradients into upstream CNN inputs through the graph.
- D: Cartesian topology; otherwise identical to A.

Each condition has {int(init.parameter_count.iloc[0])} parameters, with verified
state-key and initial-state equality within each seed. Exact hashes are in
initialization_matching.csv. The original graph builder, confidence rule,
conditional flow, loss definitions and exact-PP code are protected by hashes.

## Frozen q audits (before r training)

Final sign rule: {audit['FINAL_SIGN_TIEBREAK_STATUS']}; maximum parity-use
fraction on valid pixels was {audit['maximum_valid_realization_parity_fraction']:.8g}.
The census covers all 70 training and 20 validation realizations. Maps and
stratum counts are retained; the tie-break was not changed.

Below are additive corrected-minus-Cartesian contributions to q's epoch-20
criterion. Negative favors RGT. The three property contributions sum to the
elastic difference; segmentation contributes -0.1 times the mIoU difference.

{markdown_table(q_contributions)}

Whole metrics exist at the six saved milestones. Every intervening epoch is
reported as fixed-patch validation, not invented whole-section evaluation.
High-dip elastic errors are independently recorded without mIoU in the score.

Quantitative q decomposition (epoch 20):

```json
{json.dumps(audit['q_epoch20_decomposition'], indent=2)}
```

The segmentation share above uses the sum of absolute seed-wise contributions,
so cancellation across seeds cannot make segmentation look artificially large.
Density's separate contribution is shown in the preceding table; it is not
hidden in the combined score.

Frozen uniform inference: {audit['uniform_frozen_inference_status']}.
Minimum per-head mean normalized entropy: {audit['headwise_entropy_minimum']:.6g};
maximum per-head mean KL from uniform: {audit['headwise_kl_maximum']:.6g}.
The full table retains every layer/head/seed/epoch/probe, including CV, max/mean,
min/mean, and degree-adjusted top-decile mass. Singleton nodes are excluded from
selectivity averages, not counted as evidence of uniform attention.

Frozen edge-attribute status: {audit['AVO_EDGE_ATTR_STATUS']}. The clean shuffle
improves mean elastic error in all seeds: {audit['clean_shuffle_improves_mean_elastic_all_seeds']};
the old concatenated roll does: {audit['legacy_roll_improves_mean_elastic_all_seeds']}.
For the old combined criterion the corresponding all-seed improvement flags
are clean={audit['clean_shuffle_improves_combined_criterion_all_seeds']} and
legacy={audit['legacy_roll_improves_combined_criterion_all_seeds']}.
Both now have directly comparable inference outputs. AVO is a key/value edge
feature, not an enforced multiplicative attention weight.

Actual effective task-gradient comparisons at epoch 20:

{markdown_table(pd.DataFrame(audit['epoch20_coupling']))}

Elastic includes flow/property MSE and velocity SSIM. Physics uses native-grid
context exact-PP; segmentation uses the same training-only class-weighted CE
and Dice. Six fixed native patches cover low dip, high dip and facies boundaries;
this is a diagnostic sample, not a population-level gradient claim.

## Primary fixed epoch-20 whole-section results

Classification scope: {decision_scope}. If the elastic gate triggered expanded
validation, the final interpretation uses all 20 sections, including any
contradictions of the preliminary six-section pattern. Bounded six-section
results remain intact in the primary CSV and figures; the expanded scope is
not mixed into intermediate-epoch trajectories.

{markdown_table(simple)}

Separate high-dip Vp/Vs/density, exact-PP residual, three property SSIM values,
mIoU and all class IoUs are in the CSVs. Figures 01–08 show paired effects and
mechanism diagnostics. The original combined score is retained but never used
alone to claim elastic improvement.

## Interpretation

```json
{json.dumps({key: value for key, value in summary.items() if key.endswith('_STATUS')}, indent=2)}
```

Hypotheses:

```json
{json.dumps(summary['hypotheses'], indent=2)}
```

All-20 gate passed: {gate['passed']}. Its exact elastic-only conditions are in
all20_gate.json. When passed, the additional validation results and paired
effects are in v00332r_all20_validation_results.csv and
all20_epoch20_paired_elastic_effects.csv. These are confirmation on expanded
validation, not a new test set. Both bounded and expanded causal comparisons use
fixed epoch 20; validation-selected milestones and patch-selected epochs are
secondary and explicitly separated.

The classifications use predeclared practical effect margins, not formal
equivalence or significance tests. Three seeds and this bounded patch budget do
not establish general scientific superiority or failure of RGT/GNNs.

## Next experiment — proposal only

{next_proposal}

No next experiment, production training, commit or push was performed. Frozen
review and q inputs remain unchanged. Validation details are in validation.json.
"""
    with atomic_output(OUT / "v00332r_report.md") as stream:
        stream.write(report.encode())
    progress(f"Complete: {classification}; report={OUT / 'v00332r_report.md'}")


def run_all(args):
    audit_all(args)
    validate(args)
    train(args)
    evaluate(args)
    if all20_gate(args):
        evaluate(args, all_validation=True)
    finalize(args)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("prepare", "tie-audit", "decompose", "audit-patches", "audit-whole", "audit-all",
                                             "audit-summary", "validate", "train", "evaluate", "all20-gate", "finalize", "run-all"))
    result.add_argument("--device", default="cuda", choices=("cuda",))
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--wait-for-lock", action="store_true",
                        help="Queue behind an existing r audit; resume completed jobs once it exits")
    return result


def main():
    args = parser().parse_args()
    args.loaded_source_sha256 = execution_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("a+") as lock:
        if args.wait_for_lock:
            progress("Waiting for any existing v00332r runner to release its lock")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError("Another v00332r runner holds the lock; refusing duplicate work") from error
        functions = {
            "prepare": prepare, "tie-audit": tie_audit, "decompose": decompose,
            "audit-patches": frozen_patch_audits, "audit-whole": frozen_whole_audits,
            "audit-all": audit_all,
            "audit-summary": audit_summary, "train": train, "evaluate": evaluate,
            "all20-gate": all20_gate, "finalize": finalize, "run-all": run_all, "validate": validate,
        }
        write_json(OUT / "runner_status.json", {"status": "running", "command": args.command,
                                                "pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            functions[args.command](args)
        except BaseException as error:
            write_json(OUT / "runner_status.json", {"status": "stopped_on_error", "command": args.command,
                                                    "error_type": type(error).__name__, "error": str(error)})
            raise
        write_json(OUT / "runner_status.json", {"status": "command_complete", "command": args.command,
                                                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
