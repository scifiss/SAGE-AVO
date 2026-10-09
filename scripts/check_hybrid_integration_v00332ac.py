#!/usr/bin/env python3
"""Pushed-commit-gated loader/cache/flow verification; never create an optimizer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from sage_avo.config import load_config, seed_everything
from sage_avo.data.augmentation import AugmentationConfig
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import build_patch_sampling_weights
from sage_avo.data.sparse_topology import TopologyCache, canonical_hash, collate_sparse_patches
from sage_avo.experiments.hybrid_integration import build_hybrid_condition, resolve_hybrid_config
from sage_avo.experiments.training import (
    _normalization_tensors,
    loss_weights_from_config,
    physics_settings_from_config,
)
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.models import hybrid_sparse as hs
from sage_avo.models.sage_avo import SAGEAVO
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.engine import (
    ContrastiveSettings,
    _forward_objective,
    _move_batch,
)
from sage_avo.training.flow import heun_integrate


REPO = Path(__file__).resolve().parents[1]
BRANCH = "experiment/v00332ac-hybrid-training-integration"
PARENT = "a5434c2c73e6db8b39dc4d5b6d1db54abb602a73"
CONFIG_PATH = REPO / "configs/development_diagnostics_v00332ac.yaml"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_provenance(expected):
    if (
        git("rev-parse", "HEAD"),
        git("branch", "--show-current"),
        git("rev-parse", f"origin/{BRANCH}"),
    ) != (expected, BRANCH, expected) or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Require exact pushed commit, intended parent and clean tracked source")
    git("merge-base", "--is-ancestor", PARENT, "HEAD")
    paths = git("ls-files", "src", "configs", "scripts", "tests", "pyproject.toml").splitlines()
    hashes = {name: sha(REPO / name) for name in paths}
    for name, digest in hashes.items():
        blob = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=REPO)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError(f"Protected source hash mismatch: {name}")
    return {
        "repository": str(REPO),
        "branch": BRANCH,
        "commit_sha": expected,
        "parent_commit_sha": git("rev-parse", "HEAD^"),
        "reviewed_parent_commit_sha": PARENT,
        "source_config_test_sha256": hashes,
        "tracked_worktree_clean_at_start": True,
        "remote_push_verified": True,
    }


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def markdown_property_table(rows):
    """Render the small report table without an optional pandas dependency."""
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = [
            format(row[key], ".6g") if isinstance(row[key], float) else str(row[key])
            for key in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def historical_test_exposure(private, test_ids):
    """Inspect prediction filenames/metadata only; never open test arrays."""
    baseline = private / "stage_artifacts/stage05/v00332d_epoch40_baseline/predictions"
    files = sorted(baseline.glob("*/realization_*.npz"))
    exposed = {int(path.stem.split("_")[-1]) for path in files}
    exposed &= set(map(int, test_ids))
    return {
        "immutable_test_ids": list(map(int, test_ids)),
        "historically_evaluated_test_ids": sorted(exposed),
        "prediction_metadata_paths": [str(p.relative_to(private)) for p in files],
        "test_arrays_opened": 0,
        "independent_confirmation_available": False,
        "status": "ALL_IMMUTABLE_TEST_CASES_PREVIOUSLY_EVALUATED"
        if exposed == set(test_ids)
        else "HISTORICAL_EXPOSURE_NOT_FULLY_ESTABLISHED",
        "required_resolution": "new disjoint untouched confirmation cohort; do not relabel existing test cases",
    }


def graph_equal(a, b):
    return all(
        torch.equal(getattr(a, k), getattr(b, k))
        for k in ("coordinates", "components", "edge_index", "edge_attr")
    )


def component_isolation(device):
    """Perturb one sparse component while holding the dense state/input fixed."""
    torch.manual_seed(91)
    shape = (12, 24)
    nodes = [
        {"node": i, "component": i // 2, "time": t, "trace": x}
        for i, (t, x) in enumerate([(3.2, 2), (3.4, 5), (8.2, 18), (8.4, 21)])
    ]
    edges = [
        {
            "source": i,
            "target": i + 1,
            "component": i // 2,
            "delta_tau": 0.001,
            "delta_t": 0.2,
            "delta_x": 3.0,
            "geodesic_length": 3.1,
            "waveform_continuity": 0.9,
            "phase_continuity": 0.9,
            "shift_continuity": 0.01,
            "curvature": 0.01,
            "gap_count": 0,
        }
        for i in (0, 2)
    ]
    graph = hs.crop_accepted_graph(
        nodes, edges, top=0, left=0, raw_shape=shape, output_shape=shape
    ).to(device)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2).to(device).eval()
    avo, low = torch.randn(1, 3, *shape, device=device), torch.randn(1, 3, *shape, device=device)
    rgt = torch.arange(12, device=device).float()[None, :, None].expand(1, *shape)
    original = hs.sample_cnn_nodes

    def perturb(cnn, item):
        features = original(cnn, item).clone()
        features[item.components == 0, 0] += 5
        return features

    result = {}
    try:
        for name, cls in (
            ("legacy_pre_decoder", hs.LegacyDecoderHybridSAGEAVO),
            ("pointwise_post_decoder", hs.HybridSparseSAGEAVO),
        ):
            model = cls(dense, 8, gamma=0.1).to(device).eval()
            hs.sample_cnn_nodes = original
            with torch.no_grad():
                before = model(
                    low, torch.tensor([0.4], device=device), avo, low, rgt, [graph]
                ).velocity
            hs.sample_cnn_nodes = perturb
            with torch.no_grad():
                after = model(
                    low, torch.tensor([0.4], device=device), avo, low, rgt, [graph]
                ).velocity
            result[name + "_component_B_max_change"] = float(
                (after - before)[:, :, 8:10, 18:23].abs().max()
            )
        branch = hs.SparseReflectorBranch(8).to(device).eval()
        features = torch.randn(4, 8, device=device)
        changed = features.clone()
        changed[:2, 0] += 5
        with torch.no_grad():
            result["message_only_component_B_max_change"] = float(
                (branch.messages(features, graph)[2:] - branch.messages(changed, graph)[2:])
                .abs()
                .max()
            )
        overlapping = deepcopy(graph)
        overlapping.coordinates[2:] = overlapping.coordinates[:2]
        _, support = hs.scatter_component_local(
            torch.ones(4, 8, device=device),
            overlapping,
            torch.ones(4, dtype=torch.bool, device=device),
        )
        result["conflicting_component_footprint_support_count"] = int(support.sum())
    finally:
        hs.sample_cnn_nodes = original
    return result


def flow_qc(config, batch, normalization, device, steps, output):
    """Initialized models on synthetic Stage-03 observations; no performance claim."""
    a, b, c = [build_hybrid_condition(config, name).to(device).eval() for name in ("A", "B", "C")]
    zero = (
        hs.HybridSparseSAGEAVO(a, int(config["model"]["hidden_channels"]), gamma=0)
        .to(device)
        .eval()
    )
    for model in (a, b, c, zero):
        model.set_norm_stats(normalization)
    values = _move_batch(batch, device)
    args = (
        values["low"],
        torch.tensor([0.4], device=device),
        values["avo"],
        values["low"],
        values["rgt"],
    )
    graphs = values["sparse_graphs"]
    with torch.no_grad():
        dense_velocity = a(*args)
        zero_velocity = zero(*args, graphs)
        active_velocity = c(*args, graphs)
        control_velocity = b(*args, graphs)
        mask = c.last_support.bool().expand_as(active_velocity.velocity)
        parity = float((dense_velocity.velocity - zero_velocity.velocity).abs().max())
        direct_outside = float(
            (active_velocity.velocity - dense_velocity.velocity)[~mask].abs().max()
        )
        prediction_a = a.sample(values["avo"], values["low"], values["rgt"], steps=steps)
        prediction_zero = zero.sample(
            values["avo"], values["low"], values["rgt"], graphs, steps=steps
        )
        finite_steps = []

        def velocity(state, time):
            output_value = c(state, time, values["avo"], values["low"], values["rgt"], graphs)
            finite_steps.append(bool(torch.isfinite(output_value.velocity).all()))
            return output_value.velocity

        prediction_c = heun_integrate(values["low"].clone(), velocity, steps=steps)
        endpoint_a = a(
            prediction_a, torch.ones(1, device=device), values["avo"], values["low"], values["rgt"]
        )
        endpoint_c = c(
            prediction_c,
            torch.ones(1, device=device),
            values["avo"],
            values["low"],
            values["rgt"],
            graphs,
        )
        delta = prediction_c - prediction_a
        y_std = torch.tensor(normalization["y_std"], device=device)[None, :, None, None]
        y_mean = torch.tensor(normalization["y_mean"], device=device)[None, :, None, None]
        physical_a, physical_c = prediction_a * y_std + y_mean, prediction_c * y_std + y_mean
        specification = physics_settings_from_config(config).specification
        origin = int(values["top"][0])
        pp_a = forward_avo_three_band_spec_torch(
            *physical_a.split(1, dim=1), specification, sample_origin=origin
        )
        pp_c = forward_avo_three_band_spec_torch(
            *physical_c.split(1, dim=1), specification, sample_origin=origin
        )
    physical_delta = delta * y_std
    rows = []
    for channel, name in enumerate(("Vp", "Vs", "density")):
        outside = ~mask[:, channel]
        difference = physical_delta[:, channel]
        rows.append(
            dict(
                property=name,
                physical_rms=float(difference.square().mean().sqrt()),
                physical_max=float(difference.abs().max()),
                outside_support_rms=float(difference[outside].square().mean().sqrt()),
                outside_support_max=float(difference[outside].abs().max()),
                outside_changed_fraction=float(
                    (delta[:, channel][outside].abs() > 1e-6).float().mean()
                ),
            )
        )
    pd.DataFrame(rows).to_csv(output / "whole_flow_response.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].imshow(mask[0, 0].cpu(), aspect="auto")
    axes[0].set_title("Instantaneous sparse support")
    axes[1].imshow(physical_delta[0, 0].cpu(), aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Endpoint hybrid - dense Vp")
    axes[2].imshow((delta[0, 0].abs() > 1e-6).cpu(), aspect="auto")
    axes[2].set_title("Endpoint change > 1e-6 normalized")
    fig.tight_layout()
    fig.savefig(output / "whole_flow_support_response.png", dpi=140)
    plt.close(fig)
    # Single isolated backward through the actual complete objective; no optimizer.
    c.zero_grad(set_to_none=True)
    state_hash_before = canonical_hash({k: sha_tensor(v) for k, v in c.state_dict().items()})
    total, terms = _forward_objective(
        c,
        values,
        torch.tensor([0.4], device=device),
        _normalization_tensors(normalization),
        loss_weights_from_config(config, 0.5),
        None,
        physics_settings_from_config(config),
        ContrastiveSettings(),
        deterministic_contrastive=True,
        contrastive_generator=None,
        adaptive_weighter=None,
    )
    total.backward()
    grads = [p.grad for p in c.parameters() if p.grad is not None]
    sparse_grads = [
        p.grad
        for name, p in c.named_parameters()
        if not name.startswith("dense.") and p.grad is not None
    ]
    permutation = hs.component_source_permutation(graphs[0], b.sparse.control_seed)
    active_nodes = torch.unique(graphs[0].edge_index)
    result = {
        "zero_scale_velocity_max_error": parity,
        "zero_scale_whole_flow_max_error": float((prediction_a - prediction_zero).abs().max()),
        "active_direct_outside_support_max_error": direct_outside,
        "direct_segmentation_max_error": float(
            (dense_velocity.segmentation_logits - active_velocity.segmentation_logits).abs().max()
        ),
        "endpoint_segmentation_logits_rms": float(
            (endpoint_a.segmentation_logits - endpoint_c.segmentation_logits).square().mean().sqrt()
        ),
        "endpoint_segmentation_label_change_fraction": float(
            (endpoint_a.segmentation_logits.argmax(1) != endpoint_c.segmentation_logits.argmax(1))
            .float()
            .mean()
        ),
        "all_40_velocity_evaluations_finite": all(finite_steps),
        "velocity_evaluation_count": len(finite_steps),
        "exact_pp_endpoints_finite": bool(
            torch.isfinite(pp_a).all() and torch.isfinite(pp_c).all()
        ),
        "exact_pp_change_rms": float((pp_c - pp_a).square().mean().sqrt()),
        "complete_objective_finite": bool(
            torch.isfinite(total) and all(torch.isfinite(v) for v in terms.values())
        ),
        "all_gradients_finite": all(bool(torch.isfinite(g).all()) for g in grads),
        "sparse_gradient_l1": sum(float(g.abs().sum()) for g in sparse_grads),
        "parameter_state_unchanged": state_hash_before
        == canonical_hash({k: sha_tensor(v) for k, v in c.state_dict().items()}),
        "B_C_parameter_count_equal": sum(p.numel() for p in b.parameters())
        == sum(p.numel() for p in c.parameters()),
        "dense_parameter_count": sum(p.numel() for p in a.parameters()),
        "B_C_parameter_count": sum(p.numel() for p in c.parameters()),
        "active_source_node_count": len(active_nodes),
        "reassigned_source_node_count": int((permutation[active_nodes] != active_nodes).sum()),
        "dense_initial_weights_sha256": canonical_hash(
            {k: sha_tensor(v) for k, v in a.state_dict().items()}
        ),
        "hybrid_initial_weights_sha256": state_hash_before,
        "B_C_initial_state_identical": all(
            torch.equal(v, c.state_dict()[k]) for k, v in b.state_dict().items()
        ),
        "A_C_dense_initial_state_identical": all(
            torch.equal(v, c.dense.state_dict()[k]) for k, v in a.state_dict().items()
        ),
        "B_C_velocity_max_difference": float(
            (control_velocity.velocity - active_velocity.velocity).abs().max()
        ),
        "full_flow_spread_mechanism": "dense CNN/GroupNorm/state feedback at subsequent Heun evaluations",
        "properties": rows,
        "optimizer_steps": 0,
        "checkpoint_loads": 0,
    }
    return result


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    provenance = check_provenance(args.expected_commit)
    contract = load_config(CONFIG_PATH)
    config = resolve_hybrid_config(CONFIG_PATH)
    runtime = print_torch_runtime()
    device = select_torch_device(
        args.device, require_cuda=args.device.startswith("cuda"), context="v00332ac no-update QC"
    )
    seed_everything(config["experiment"]["seed"])
    torch.set_num_threads(1)
    private = args.private_root.resolve()
    if not private.is_dir() or private.is_relative_to(REPO):
        raise ValueError("Private artifacts must be outside this Git worktree")
    dataset_root = private / "stage_artifacts/stage03/ds_v00331_production100_support_aware/dataset"
    z_root = private / "stage_artifacts/stage04/sage_avo_s01_v00332z_gap_tolerant_rgt_tracking"
    frozen_path = z_root / "v00332z_frozen_contract.json"
    z_contract_path = z_root / "v00332z_experiment_contract.json"
    frozen = json.loads(frozen_path.read_text())
    z_contract = json.loads(z_contract_path.read_text())
    if z_contract["commit_sha"] != "30e0b012467b3553b6aef44089175dad24a2019a":
        raise ValueError("Require exact frozen v00332z provenance")
    if z_contract["config"] != load_config(REPO / "configs/development_diagnostics_v00332z.yaml"):
        raise ValueError("Frozen topology parameters differ from reviewed source")
    splits = json.loads((dataset_root / "split_ids.json").read_text())
    train_ids, val_ids = (
        contract["diagnostic_train_ids"],
        contract["diagnostic_reused_validation_ids"],
    )
    if not set(train_ids) <= set(splits["train"]) or not set(val_ids) <= set(splits["validation"]):
        raise ValueError("Diagnostic IDs violate the immutable split")
    final = private / "scientific_reconciliation/v00332ac_hybrid_training_integration"
    output = final.with_name(final.name + ".incomplete")
    if output.exists() or final.exists():
        raise FileExistsError("Refuse overwriting an existing private scientific audit")
    output.mkdir(parents=True)
    cache = TopologyCache(output / "topology_cache", frozen, z_contract["config"])
    inputs = [
        frozen_path,
        z_contract_path,
        dataset_root / "patch_index.csv",
        dataset_root / "normalization.json",
        dataset_root / "split_ids.json",
    ]
    inputs += [
        dataset_root / "realizations" / f"realization_{rid:07d}.npz" for rid in train_ids + val_ids
    ]
    exposure = historical_test_exposure(private, splits["test"])
    baseline_manifest = (
        private / "stage_artifacts/stage05/v00332d_epoch40_baseline/predictions/full/manifest.json"
    )
    if baseline_manifest.exists():
        inputs.append(baseline_manifest)
    run_contract = {
        **provenance,
        "frozen_config": contract,
        "resolved_training_config": config,
        "resolved_training_config_sha256": canonical_hash(config),
        "split_ids": splits,
        "input_sha256": {str(p.relative_to(private)): sha(p) for p in inputs},
        "runtime": runtime,
        "training": False,
        "optimizer_steps": 0,
        "checkpoint_loads": 0,
        "test_realizations_loaded": [],
        "historical_test_exposure": exposure,
        "truth_use": "training objective only; never topology",
    }
    write_json(output / "v00332ac_experiment_contract.json", run_contract)
    # Verify the saved contract before any scientific detection/flow computation.
    for name, digest in run_contract["source_config_test_sha256"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError("Source changed since contract creation")
    topology_rows = []
    for rid in train_ids + val_ids:
        print(f"[v00332ac] full observable cache: {rid}", flush=True)
        with np.load(dataset_root / "realizations" / f"realization_{rid:07d}.npz") as archive:
            record = cache.prepare(
                rid, avo=archive["avo"], rgt=archive["rgt"], support=archive["valid_mask"]
            )
        topology_rows.append(
            dict(
                realization_id=rid,
                nodes=len(record["nodes"]),
                edges=len(record["edges"]),
                topology_sha256=record["topology_sha256"],
            )
        )
    pd.DataFrame(topology_rows).to_csv(output / "topology_cache_identity.csv", index=False)
    qc = []
    augmentation = []
    selected_batch = None
    for split, ids in (("train", train_ids), ("validation", val_ids)):
        source = IndexedRealizationPatches(dataset_root, split, topology_cache=cache)
        forced = IndexedRealizationPatches(
            dataset_root,
            split,
            topology_cache=cache,
            augment=True,
            matched_augmentation=True,
            augmentation_config=AugmentationConfig(
                horizontal_flip_probability=1,
                avo_gain_probability=0,
                avo_noise_probability=0,
            ),
        )
        for rid in ids:
            available = source.index[source.index.realization_id == rid]
            indices = []
            for _, group in available.groupby(["raw_height", "raw_width"], sort=True):
                indices.extend(group.index[: contract["diagnostic_indexed_patches_per_raw_scale"]])
            for index in indices:
                item = source[int(index)]
                graph = item["sparse_graph"]
                raw = tuple(item["raw_shape"].tolist())
                shape = tuple(item["output_shape"].tolist())
                native = source._load(rid)
                # Same cache and transform, including multiscale raw crops.
                expected = cache.patch(
                    rid,
                    top=int(item["top"]),
                    left=int(item["left"]),
                    raw_shape=raw,
                    output_shape=shape,
                )
                provider_same = True
                if raw == shape:
                    provider = cache.tile_provider(
                        rid,
                        shape,
                        avo=native["avo"],
                        rgt=native["rgt"],
                        support=native["valid_mask"],
                    )
                    provider_same = graph_equal(
                        graph, provider([(int(item["top"]), int(item["left"]))])[0]
                    )
                qc.append(
                    dict(
                        split=split,
                        realization_id=rid,
                        patch_index=int(index),
                        raw_shape=str(raw),
                        nodes=len(graph.coordinates),
                        edges=graph.edge_index.shape[1] // 2,
                        cache_crop_equal=graph_equal(graph, expected),
                        tile_provider_equal=provider_same,
                    )
                )
                flipped = forced[int(index)]
                augmentation.append(
                    dict(
                        split=split,
                        realization_id=rid,
                        patch_index=int(index),
                        graph_equal=graph_equal(
                            flipped["sparse_graph"], hs.flip_sparse_graph_horizontal(graph)
                        ),
                        avo_equal=torch.equal(flipped["avo"], item["avo"].flip(-1)),
                        rgt_equal=torch.equal(flipped["rgt"], item["rgt"].flip(-1)),
                        physics_context_equal=torch.equal(
                            flipped["physics_context"], item["physics_context"].flip(-1)
                        ),
                        physics_avo_equal=torch.equal(
                            flipped["physics_avo"], item["physics_avo"].flip(-1)
                        ),
                    )
                )
                if (
                    split == "train"
                    and raw == shape
                    and graph.edge_index.numel()
                    and selected_batch is None
                ):
                    selected_batch = next(
                        iter(
                            DataLoader(
                                Subset(source, [int(index)]),
                                batch_size=1,
                                collate_fn=collate_sparse_patches,
                            )
                        )
                    )
    pd.DataFrame(qc).to_csv(output / "training_inference_topology_qc.csv", index=False)
    pd.DataFrame(augmentation).to_csv(output / "augmentation_qc.csv", index=False)
    if selected_batch is None:
        raise RuntimeError("No native training patch with usable graph was available")
    # Compare the original weights and order on the explicitly bounded train cohort.
    datasets = [
        IndexedRealizationPatches(dataset_root, "train", topology_cache=c) for c in (None, cache)
    ]
    for d in datasets:
        d.index = d.index[d.index.realization_id.isin(train_ids)].reset_index(drop=True)
    weights = [build_patch_sampling_weights(d) for d in datasets]
    orders = [
        list(
            WeightedRandomSampler(
                w, 40, replacement=True, generator=torch.Generator().manual_seed(12358)
            )
        )
        for w in weights
    ]
    sampling = {
        "cohort_candidate_count": len(weights[0]),
        "weights_equal": torch.equal(*weights),
        "40_draw_order_equal": orders[0] == orders[1],
        "scope": "conditional diagnostic train cohort; production sampler code and distribution unchanged",
    }
    write_json(output / "sampler_qc.json", sampling)
    print("[v00332ac] component isolation and full 20-step conditional flow", flush=True)
    isolation = component_isolation(device)
    normalization = datasets[0].normalization
    flow = flow_qc(
        config, selected_batch, normalization, device, contract["diagnostic_flow_steps"], output
    )
    write_json(output / "component_isolation.json", isolation)
    write_json(output / "whole_flow_qc.json", flow)
    topology_pass = all(row["cache_crop_equal"] and row["tile_provider_equal"] for row in qc)
    augmentation_pass = all(
        all(
            row[k]
            for k in (
                "graph_equal",
                "avo_equal",
                "rgt_equal",
                "physics_context_equal",
                "physics_avo_equal",
            )
        )
        for row in augmentation
    )
    isolation_pass = (
        isolation["pointwise_post_decoder_component_B_max_change"] <= 1e-6
        and isolation["message_only_component_B_max_change"] <= 1e-6
        and isolation["conflicting_component_footprint_support_count"] == 0
    )
    flow_pass = (
        flow["zero_scale_velocity_max_error"] == 0
        and flow["zero_scale_whole_flow_max_error"] == 0
        and flow["all_40_velocity_evaluations_finite"]
        and flow["all_gradients_finite"]
        and flow["exact_pp_endpoints_finite"]
        and flow["complete_objective_finite"]
        and flow["sparse_gradient_l1"] > 0
        and flow["active_direct_outside_support_max_error"] == 0
        and flow["direct_segmentation_max_error"] == 0
        and flow["parameter_state_unchanged"]
        and flow["velocity_evaluation_count"] == 2 * contract["diagnostic_flow_steps"]
    )
    control_pass = (
        flow["B_C_parameter_count_equal"]
        and flow["B_C_initial_state_identical"]
        and flow["A_C_dense_initial_state_identical"]
        and flow["B_C_velocity_max_difference"] > 1e-7
        and sampling["weights_equal"]
        and sampling["40_draw_order_equal"]
    )
    independent_test_pass = exposure["independent_confirmation_available"]
    decision = (
        "TOPOLOGY_CACHE_MISMATCH"
        if not topology_pass
        else "AUGMENTATION_MISMATCH"
        if not augmentation_pass
        else "COMPONENT_ISOLATION_FAILURE"
        if not isolation_pass
        else "FLOW_INTEGRATION_FAILURE"
        if not flow_pass
        else "MATCHED_ABLATION_CONTROL_INVALID"
        if not control_pass or not independent_test_pass
        else "HYBRID_TRAINING_INTEGRATION_READY"
    )
    summary = {
        "decision": decision,
        "branch": BRANCH,
        "experiment_commit": args.expected_commit,
        "parent_commit": provenance["parent_commit_sha"],
        "reviewed_parent_commit": PARENT,
        "topology_pass": topology_pass,
        "augmentation_pass": augmentation_pass,
        "component_isolation_pass": isolation_pass,
        "flow_pass": flow_pass,
        "matched_control_pass": control_pass,
        "mechanical_training_ready": topology_pass
        and augmentation_pass
        and isolation_pass
        and flow_pass
        and control_pass,
        "independent_test_pass": independent_test_pass,
        "historical_test_exposure": exposure,
        "audited_indexed_patches": len(qc),
        "topology": topology_rows,
        "isolation": isolation,
        "flow": flow,
        "sampler": sampling,
        "training": False,
        "optimizer_steps": 0,
        "checkpoint_loads": 0,
        "test_realizations_loaded": [],
        "source_changed_after_run_started": False,
        "private_artifacts_pushed": False,
        "data_or_checkpoints_pushed": False,
    }
    write_json(output / "v00332ac_summary.json", summary)
    report = f"""# v00332ac hybrid training/inference integration

Decision: `{decision}`

Branch `{BRANCH}`, commit `{args.expected_commit}`, reviewed parent `{PARENT}`.
No optimizer updates or checkpoint loads. Four full-section graphs were built once
from native observed AVO, RGT and support, using frozen v00332z parameters.
Topology/config/source/input hashes are saved privately. No test realization was loaded.

## Loader and augmentation

All {len(qc)} selected actual indexed patches passed cache/crop/coordinate checks:
{topology_pass}. Forced-flip AVO/RGT/graph and physics-halo checks: {augmentation_pass}.
Variable and empty graphs use a list collator and the actual training objective
routes that graph list to the optional hybrid model. Default dense calls are unchanged.
Gain/noise operate on patch AVO without regenerating topology. Matched A/B/C use
identical draw streams. The legacy augmentation had not flipped physics context;
the matched path now flips context, clean AVO and masks in all three conditions.
Sampler weights/order match on the bounded cohort: {sampling}.

## Component isolation

{json.dumps(isolation, indent=2)}

The old pre-decoder spatial GroupNorm coupled components even after masking the
final velocity. The new residual is a pointwise 1x1 projection after the unchanged
dense decoder, with no spatial normalization. Graph messages stay inside accepted
components and conflicting bilinear footprints receive no sparse contribution.

## Whole conditional flow

Zero-scale velocity error: {flow["zero_scale_velocity_max_error"]}.
Zero-scale whole-flow error: {flow["zero_scale_whole_flow_max_error"]}.
Finite gradients/flow/exact-PP and objective: {flow_pass}.
Instantaneous sparse residual outside support: {flow["active_direct_outside_support_max_error"]}.

{markdown_property_table(flow["properties"])}

Full-flow property changes can extend beyond instantaneous sparse support because
the subsequent dense CNN/GroupNorm evaluates a changed state. This is existing
conditional-flow feedback, not a new sparse edge or footprint spanning components.
Direct segmentation logits at the same state are unchanged; terminal segmentation
may change with the endpoint state. The exact-PP operator and losses are unchanged.
These initialized-model magnitudes measure mechanics only, not inversion performance.

## Matched ablation preparation

Frozen configuration: `configs/development_diagnostics_v00332ac.yaml` and resolved
configuration in the private contract. A is dense-only. B/C have equal initialized
parameters, graphs, edge descriptors, degrees, component restrictions and tensor
shapes; only B permutes source message keys/values within each safe component.
Queries/root features and destinations remain tied to their physical nodes. The
same source mapping is applied at each layer and flow evaluation with a fixed seed.
This control destroys message-source correspondence without changing node counts
or degrees; it is a payload control, not a claim to rewire topology.

Predeclare C-vs-B paired-by-seed whole-realization mean train-normalized elastic
RMSE as primary, with density/high-dip/fault/reservoir/unsupported-region and
segmentation/exact-PP reports separately. Same original weighted sampling, data
order, augmentation, objective/curriculum, optimizer, budget, validation-only
checkpoint criterion and tiling for all three variants. Immutable test IDs:
{splits["test"]}. Their arrays were not loaded here. Existing baseline prediction
filenames establish prior evaluation of {exposure["historically_evaluated_test_ids"]}.
No subset of these ten cases can be called genuinely untouched. Prior validation
cases remain exploratory; the old test split is secondary historical evidence.
A new, disjoint confirmation cohort is required and was not generated in this task.
The mechanical integration gates passed: {summary["mechanical_training_ready"]}.
The full readiness gate is blocked by the independent-test protocol, not unequal
B/C capacity or a numerical/mechanism failure.

Sparse support remains limited (v00332ab); no performance gain is established.
Future training must prepare caches for its entire train/validation cohorts before
launching. No multi-epoch run is authorized or performed in this readiness audit.
"""
    (output / "v00332ac_report.md").write_text(report)
    check_provenance(args.expected_commit)
    for path, digest in run_contract["input_sha256"].items():
        if sha(private / path) != digest:
            raise RuntimeError("Scientific input changed during audit")
    if historical_test_exposure(private, splits["test"]) != exposure:
        raise RuntimeError("Historical test exposure evidence changed during audit")
    output.rename(final)
    print(json.dumps({"decision": decision, "report": str(final / "v00332ac_report.md")}, indent=2))


if __name__ == "__main__":
    main()
