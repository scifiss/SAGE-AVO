#!/usr/bin/env python3
"""Run the bounded v00332o causal audit of the existing production GNN."""

from __future__ import annotations

import os

# This must be set before importing Torch/CUDA libraries.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import shutil
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
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.checkpoints import load_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332o.yaml"
ARCHITECTURES = ("rgt_gnn", "cartesian_gnn", "no_gnn")
PROPERTIES = ("vp", "vs", "density")


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
        / contract["frozen_inputs"]["source_experiment"]
    )
    return contract, dataset, experiment, source


def _resolved_config(contract: dict[str, Any], seed: int) -> dict[str, Any]:
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    budget = contract["bounded_screen"]
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
    config["training"]["checkpointing"]["whole_validation_every_epochs"] = 1000
    config["model"].pop("experimental_graph", None)
    config["observability"] = load_config(
        REPOSITORY / "configs" / "training_observability_v00332d.yaml"
    )
    config["observability"]["revision"] = f"{contract['revision']}-seed{seed}"
    config["observability"]["scientific_methodology_changed"] = False
    config["observability"]["diagnostics"]["flow_integration_steps"] = int(
        budget["flow_integration_steps"]
    )
    return config


def _run_path(experiment: Path, seed: int, architecture: str) -> Path:
    return experiment / "runs" / f"seed_{seed}_{architecture}"


def _state_path(experiment: Path, seed: int, architecture: str) -> Path:
    return experiment / "initial_states" / f"seed_{seed}_{architecture}.pt"


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _verify_determinism_environment() -> None:
    expected = ":4096:8"
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != expected:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be {expected!r} before CUDA starts; got {observed!r}"
        )


def _copy_common_state(source: dict[str, torch.Tensor], target: torch.nn.Module) -> dict[str, Any]:
    target_state = target.state_dict()
    copied = []
    for name, value in source.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone()
            copied.append(name)
    target.load_state_dict(target_state, strict=True)
    return {
        "copied_tensor_count": len(copied),
        "copied_tensor_names_sha256": canonical_hash(sorted(copied)),
        "uncopied_target_tensors": sorted(set(target_state) - set(copied)),
    }


def prepare(_: argparse.Namespace) -> None:
    contract, dataset, experiment, source = _roots()
    destination = experiment / "causal_gnn_experiment_contract.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared experiment: {destination}")
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    frozen = contract["frozen_inputs"]
    selection_source = source / "fixed_patch_selection.json"
    probes_source = source / "fixed_graph_probe_samples.json"
    if file_sha256(selection_source) != frozen["expected_selection_file_sha256"]:
        raise RuntimeError("Frozen fixed-patch selection hash mismatch")
    if file_sha256(probes_source) != frozen["expected_probe_file_sha256"]:
        raise RuntimeError("Frozen graph-probe manifest hash mismatch")
    selection = json.loads(selection_source.read_text(encoding="utf-8"))
    if canonical_hash(selection["train_indices"]) != frozen["expected_train_indices_sha256"]:
        raise RuntimeError("Frozen training-index digest mismatch")
    if canonical_hash(selection["validation_indices"]) != frozen["expected_validation_indices_sha256"]:
        raise RuntimeError("Frozen validation-index digest mismatch")
    experiment.mkdir(parents=True)
    (experiment / "initial_states").mkdir()
    shutil.copyfile(selection_source, experiment / "fixed_patch_selection.json")
    shutil.copyfile(probes_source, experiment / "fixed_graph_probe_samples.json")
    initializations = []
    for seed in map(int, contract["seeds"]):
        seed_everything(seed, deterministic_torch=True)
        config = _resolved_config(contract, seed)
        rgt = build_sage_avo_variant("full", **sage_avo_model_kwargs(config))
        cartesian = build_sage_avo_variant("no_rgt", **sage_avo_model_kwargs(config))
        cartesian.load_state_dict(rgt.state_dict(), strict=True)
        no_gnn = build_sage_avo_variant("no_gnn", **sage_avo_model_kwargs(config))
        no_gnn_common = _copy_common_state(dict(rgt.state_dict()), no_gnn)
        models = {"rgt_gnn": rgt, "cartesian_gnn": cartesian, "no_gnn": no_gnn}
        if state_sha256(dict(rgt.state_dict())) != state_sha256(dict(cartesian.state_dict())):
            raise RuntimeError("RGT and Cartesian initial tensor states differ")
        for architecture, model in models.items():
            path = _state_path(experiment, seed, architecture)
            torch.save(dict(model.state_dict()), path)
            initializations.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "state_path": str(path),
                    "state_file_sha256": file_sha256(path),
                    "model_state_sha256": state_sha256(dict(model.state_dict())),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "rgt_cartesian_tensor_identical": architecture in {"rgt_gnn", "cartesian_gnn"},
                    "no_gnn_common_initialization": no_gnn_common if architecture == "no_gnn" else None,
                }
            )
    payload = {
        "status": "PREPARED_NOT_TRAINED",
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "dataset_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
        "selection_sha256": file_sha256(experiment / "fixed_patch_selection.json"),
        "probe_sha256": file_sha256(experiment / "fixed_graph_probe_samples.json"),
        "initializations": initializations,
        "architecture_statement": (
            "The final production StratigraphicGraphEncoder is used unchanged: graph_mode rgt, "
            "the tensor-identical graph_mode cartesian control, and graph_mode none."
        ),
        "new_graph_mechanism_added": False,
        "new_attention_bias_added": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _load_probe_patch(dataset: Path, probe_manifest: Path) -> dict[str, np.ndarray]:
    probe = json.loads(probe_manifest.read_text(encoding="utf-8"))["patches"][0]
    with np.load(
        dataset / "realizations" / f"realization_{int(probe['realization_id']):07d}.npz",
        allow_pickle=False,
    ) as archive:
        top, left = int(probe["top"]), int(probe["left"])
        height, width = map(int, probe["raw_scale"])
        return {
            name: np.asarray(archive[name][..., top : top + height, left : left + width])
            for name in ("avo", "low", "rgt", "valid_mask")
        }


def smoke(args: argparse.Namespace) -> None:
    _verify_determinism_environment()
    contract, dataset, experiment, _ = _roots()
    seed_everything(int(contract["seeds"][0]), deterministic_torch=True)
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332o causal-GNN smoke test",
    )
    config = _resolved_config(contract, int(contract["seeds"][0]))
    normalization = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    patch = _load_probe_patch(dataset, experiment / "fixed_graph_probe_samples.json")
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    avo = torch.from_numpy(((patch["avo"] - x_mean) / x_std)[None]).to(device)
    low = torch.from_numpy(((patch["low"] - y_mean) / y_std)[None]).to(device)
    rgt = torch.from_numpy(patch["rgt"][None].astype(np.float32)).to(device)
    rows = []
    for architecture in ARCHITECTURES:
        definition = contract["architectures"][architecture]
        variant = str(definition["training_variant"])
        model = build_sage_avo_variant(variant, **sage_avo_model_kwargs(config)).to(device)
        state = torch.load(
            _state_path(experiment, int(contract["seeds"][0]), architecture),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        model.set_norm_stats(normalization)
        output = model(low, torch.full((1,), 0.5, device=device), avo, low, rgt)
        loss = output.velocity.square().mean() + output.segmentation_logits.square().mean()
        loss.backward()
        active_graph = [
            name
            for name, parameter in model.named_parameters()
            if name.startswith("graph.layers")
            and parameter.grad is not None
            and bool(parameter.grad.abs().sum() > 0)
        ]
        expected_graph = architecture != "no_gnn"
        if bool(active_graph) != expected_graph:
            raise RuntimeError(f"Graph gradient-path mismatch for {architecture}")
        rows.append(
            {
                "architecture": architecture,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "active_graph_gradient_tensor_count": len(active_graph),
                "finite_output": bool(torch.isfinite(output.velocity).all()),
            }
        )
        if not rows[-1]["finite_output"]:
            raise FloatingPointError(f"Nonfinite smoke output for {architecture}")
        del model, output, loss
        _release_cuda()

    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
    model.load_state_dict(
        torch.load(
            _state_path(experiment, int(contract["seeds"][0]), "rgt_gnn"),
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    model.set_norm_stats(normalization)
    model.eval()
    with torch.inference_mode():
        direct_sample = model.sample(avo, low, rgt, steps=2)
        repeated_sample = model.sample(avo, low, rgt, steps=2)
        bitwise_repeat_equal = bool(torch.equal(direct_sample, repeated_sample))
        if not bitwise_repeat_equal:
            raise RuntimeError("Deterministic CUDA repeat check was not bitwise equal")
        direct_output = model(
            direct_sample,
            torch.ones(1, device=device),
            avo,
            low,
            rgt,
        )
        direct = direct_sample[0].cpu().numpy() * y_std + y_mean
        direct_labels = direct_output.segmentation_logits[0].argmax(0).cpu().numpy()
    tiled, tiled_labels = infer_full_realization(
        model,
        avo=patch["avo"],
        low=patch["low"],
        rgt=patch["rgt"],
        normalization=normalization,
        patch_shape=patch["rgt"].shape,
        stride=patch["rgt"].shape,
        steps=2,
        batch_size=1,
        device=device,
        valid_mask=patch["valid_mask"],
    )
    maximum_difference = float(np.max(np.abs(direct - tiled)))
    tolerance = 1e-3
    if maximum_difference > tolerance:
        raise RuntimeError(
            f"Direct/single-tile verification failed: {maximum_difference} > {tolerance}"
        )
    label_disagreement_count = int(np.count_nonzero(direct_labels != tiled_labels))
    if label_disagreement_count:
        raise RuntimeError(
            "Direct/single-tile segmentation labels disagree at "
            f"{label_disagreement_count} pixels"
        )
    report = {
        "status": "PASS",
        "architectures": rows,
        "direct_vs_single_tile_maximum_absolute_difference": maximum_difference,
        "direct_vs_single_tile_tolerance": tolerance,
        "direct_vs_single_tile_label_disagreement_count": label_disagreement_count,
        "bitwise_repeat_equal": bitwise_repeat_equal,
        "inference_mode_verified_by_test": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    (experiment / "causal_gnn_smoke_test.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def train(args: argparse.Namespace) -> None:
    _verify_determinism_environment()
    contract, dataset, experiment, _ = _roots()
    smoke_path = experiment / "causal_gnn_smoke_test.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text())["status"] != "PASS":
        raise RuntimeError("Run and pass the v00332o smoke test before training")
    selection = json.loads(
        (experiment / "fixed_patch_selection.json").read_text(encoding="utf-8")
    )
    maximum = int(contract["bounded_screen"]["epochs"])
    for seed in map(int, contract["seeds"]):
        config = _resolved_config(contract, seed)
        for architecture in ARCHITECTURES:
            variant = str(contract["architectures"][architecture]["training_variant"])
            run = _run_path(experiment, seed, architecture)
            for epoch in range(_completed_epoch(run) + 1, maximum + 1):
                _release_cuda()
                train_controlled_variant(
                    repository=REPOSITORY,
                    config_path=CONTRACT_PATH,
                    config=config,
                    dataset_directory=dataset,
                    experiment_directory=experiment,
                    variant=variant,
                    device_name=args.device,
                    epochs_override=maximum,
                    max_train_batches=int(
                        contract["bounded_screen"]["train_batches_per_epoch"]
                    ),
                    max_validation_batches=int(
                        contract["bounded_screen"]["validation_batches_per_epoch"]
                    ),
                    run_name=run.name,
                    resume_from=(run / "last.pt") if (run / "last.pt").exists() else None,
                    stop_after_epoch=epoch,
                    fixed_train_indices=selection["train_indices"],
                    fixed_validation_indices=selection["validation_indices"],
                    initial_model_state=_state_path(experiment, seed, architecture),
                    finite_state_check_batches=(1, 16, 32),
                    abort_on_nonfinite=True,
                )
                print(
                    f"[v00332o] completed seed={seed} architecture={architecture} epoch={epoch}",
                    flush=True,
                )
    for seed in map(int, contract["seeds"]):
        run = _run_path(experiment, seed, "rgt_gnn")
        report = run / "mechanism_probe" / f"checkpoint_diagnostics_epoch_{maximum:04d}.json"
        if report.exists():
            continue
        _release_cuda()
        analyze_checkpoint(
            checkpoint_path=run / "last.pt",
            dataset_directory=dataset,
            run_directory=run,
            sample_manifest_path=experiment / "fixed_graph_probe_samples.json",
            output_directory=run / "mechanism_probe",
            device=args.device,
            maximum_patches=int(
                contract["bounded_screen"]["final_rgt_mechanism_probe_patch_count"]
            ),
            flow_steps=int(contract["bounded_screen"]["flow_integration_steps"]),
            include_whole_realizations=False,
        )
        print(f"[v00332o] completed RGT mechanism probe seed={seed}", flush=True)


def _segmentation_metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {"segmentation_accuracy": float(np.mean(prediction[mask] == truth[mask]))}
    ious = []
    for label in range(3):
        predicted = (prediction == label) & mask
        expected = (truth == label) & mask
        union = np.count_nonzero(predicted | expected)
        value = float(np.count_nonzero(predicted & expected) / union) if union else np.nan
        result[f"class_{label}_iou"] = value
        ious.append(value)
    result["miou"] = float(np.nanmean(ious))
    return result


def _complexity_masks(
    avo: np.ndarray,
    rgt: np.ndarray,
    labels: np.ndarray,
    reservoir: np.ndarray,
    plume: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    vertical_rgt, horizontal_rgt = np.gradient(rgt.astype(np.float64))
    dip = np.abs(horizontal_rgt) / (np.abs(vertical_rgt) + 1e-6)
    curvature = np.hypot(*np.gradient(vertical_rgt))
    angles = np.sin(np.deg2rad((10.0, 24.0, 38.0))) ** 2
    centered_angles = angles - angles.mean()
    centered_avo = avo.astype(np.float64) - avo.mean(axis=0, keepdims=True)
    avo_gradient = np.sum(centered_angles[:, None, None] * centered_avo, axis=0) / (
        np.sum(centered_angles**2) + 1e-12
    )
    avo_change = np.hypot(*np.gradient(avo_gradient))
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    dip_q25, dip_q75 = np.quantile(dip[valid], (0.25, 0.75))
    curvature_q75 = np.quantile(curvature[valid], 0.75)
    avo_q75 = np.quantile(avo_change[valid], 0.75)
    return {
        "all_valid": valid,
        "simple_low_dip": valid & (dip <= dip_q25) & (curvature < curvature_q75) & ~boundary,
        "high_dip": valid & (dip >= dip_q75),
        "high_rgt_curvature": valid & (curvature >= curvature_q75),
        "facies_boundary": valid & boundary,
        "reservoir": valid & reservoir.astype(bool),
        "plume": valid & plume.astype(bool),
        "high_avo_gradient": valid & (avo_change >= avo_q75),
    }


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    data_range = max(float(np.ptp(second)), 1e-12)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_first, mean_second = float(first.mean()), float(second.mean())
    variance_first, variance_second = float(first.var()), float(second.var())
    covariance = float(np.mean((first - mean_first) * (second - mean_second)))
    return float(
        ((2 * mean_first * mean_second + c1) * (2 * covariance + c2))
        / ((mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2))
    )


def evaluate_whole(args: argparse.Namespace) -> None:
    _verify_determinism_environment()
    contract, dataset, experiment, _ = _roots()
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332o whole-realization evaluation",
    )
    normalization = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    y_std = np.asarray(normalization["y_std"], dtype=np.float64)
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    whole = contract["whole_realization_evaluation"]
    rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    for seed in map(int, contract["seeds"]):
        config = _resolved_config(contract, seed)
        from sage_avo.forward.specification import forward_specification_from_mapping

        forward_specification = forward_specification_from_mapping(config)
        for architecture in ARCHITECTURES:
            run = _run_path(experiment, seed, architecture)
            if _completed_epoch(run) != int(contract["bounded_screen"]["epochs"]):
                raise RuntimeError(f"Incomplete training run: {run}")
            variant = str(contract["architectures"][architecture]["training_variant"])
            model = build_sage_avo_variant(variant, **sage_avo_model_kwargs(config)).to(device)
            model.set_norm_stats(normalization)
            load_checkpoint(run / "last.pt", model, restore_rng=False, map_location=device)
            model.eval()
            for realization_id in map(int, whole["realization_ids"]):
                output_path = run / "whole_realization" / f"realization_{realization_id:07d}.npz"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    with np.load(output_path, allow_pickle=False) as saved:
                        prediction = saved["prediction"]
                        predicted_labels = saved["segmentation_prediction"]
                else:
                    path = dataset / "realizations" / f"realization_{realization_id:07d}.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        prediction, predicted_labels = infer_full_realization(
                            model,
                            avo=archive["avo"],
                            low=archive["low"],
                            rgt=archive["rgt"],
                            normalization=normalization,
                            patch_shape=tuple(map(int, whole["patch_shape"])),
                            stride=tuple(map(int, whole["stride"])),
                            steps=int(whole["flow_integration_steps"]),
                            batch_size=int(whole["batch_size"]),
                            device=device,
                            valid_mask=archive["valid_mask"],
                        )
                        np.savez_compressed(
                            output_path,
                            prediction=prediction,
                            segmentation_prediction=predicted_labels,
                        )
                path = dataset / "realizations" / f"realization_{realization_id:07d}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    truth = np.asarray(archive["elastic"], dtype=np.float32)
                    labels = np.asarray(archive["segmentation"], dtype=np.int64)
                    valid = np.asarray(archive["valid_mask"], dtype=bool)
                    avo = np.asarray(archive["avo"], dtype=np.float32)
                    avo_clean = np.asarray(archive["avo_clean"], dtype=np.float32)
                    rgt = np.asarray(archive["rgt"], dtype=np.float32)
                    reservoir = np.asarray(archive["reservoir_mask"], dtype=bool)
                    plume = np.asarray(archive["plume_mask"], dtype=bool)
                masks = _complexity_masks(avo, rgt, labels, reservoir, plume, valid)
                prediction_tensor = torch.from_numpy(prediction[None])
                with torch.inference_mode():
                    modeled = forward_avo_three_band_spec_torch(
                        prediction_tensor[:, 0],
                        prediction_tensor[:, 1],
                        prediction_tensor[:, 2],
                        forward_specification,
                        sample_origin=0,
                    )[0].numpy()
                physics_error = ((modeled - x_mean) / x_std - (avo_clean - x_mean) / x_std) ** 2
                segmentation = _segmentation_metrics(predicted_labels, labels, valid)
                normalized_rmse = []
                for channel, name in enumerate(PROPERTIES):
                    error = prediction[channel][valid] - truth[channel][valid]
                    rmse = float(np.sqrt(np.mean(error**2)))
                    normalized_rmse.append(rmse / y_std[channel])
                    rows.append(
                        {
                            "seed": seed,
                            "architecture": architecture,
                            "realization_id": realization_id,
                            "property": name,
                            "rmse": rmse,
                            "normalized_rmse": rmse / y_std[channel],
                            "ssim": _global_ssim(prediction[channel][valid], truth[channel][valid]),
                            "exact_pp_rmse_clean_normalized": float(
                                np.sqrt(np.mean(physics_error[:, valid]))
                            ),
                            **segmentation,
                        }
                    )
                criterion = float(np.mean(normalized_rmse) - 0.1 * segmentation["miou"])
                for row in rows[-3:]:
                    row["checkpoint_criterion"] = criterion
                for stratum, mask in masks.items():
                    if not np.any(mask):
                        continue
                    stratum_segmentation = _segmentation_metrics(predicted_labels, labels, mask)
                    stratum_physics = float(np.sqrt(np.mean(physics_error[:, mask])))
                    for channel, name in enumerate(PROPERTIES):
                        error = prediction[channel][mask] - truth[channel][mask]
                        strata_rows.append(
                            {
                                "seed": seed,
                                "architecture": architecture,
                                "realization_id": realization_id,
                                "stratum": stratum,
                                "pixel_count": int(mask.sum()),
                                "property": name,
                                "normalized_rmse": float(np.sqrt(np.mean(error**2)) / y_std[channel]),
                                "exact_pp_rmse_clean_normalized": stratum_physics,
                                **stratum_segmentation,
                            }
                        )
                print(
                    f"[v00332o] evaluated seed={seed} architecture={architecture} "
                    f"realization={realization_id}",
                    flush=True,
                )
                del prediction, predicted_labels, prediction_tensor, modeled, physics_error
                _release_cuda()
            del model
            _release_cuda()
    _write_csv(experiment / "whole_realization_results.csv", rows)
    _write_csv(experiment / "complexity_stratified_results.csv", strata_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty results: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(_: argparse.Namespace) -> None:
    contract, _, experiment, _ = _roots()
    whole_path = experiment / "whole_realization_results.csv"
    strata_path = experiment / "complexity_stratified_results.csv"
    if not whole_path.exists() or not strata_path.exists():
        raise FileNotFoundError("Run whole-realization evaluation before final analysis")
    with whole_path.open(newline="", encoding="utf-8") as stream:
        whole_rows = list(csv.DictReader(stream))
    with strata_path.open(newline="", encoding="utf-8") as stream:
        strata_rows = list(csv.DictReader(stream))
    patch_rows = []
    maximum = int(contract["bounded_screen"]["epochs"])
    for seed in map(int, contract["seeds"]):
        for architecture in ARCHITECTURES:
            run = _run_path(experiment, seed, architecture)
            with (run / "training_log.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    patch_rows.append({"seed": seed, "architecture": architecture, **row})
    _write_csv(experiment / "bounded_patch_trends.csv", patch_rows)

    def grouped(rows: list[dict[str, str]], key: str) -> dict[tuple[int, str], float]:
        values: dict[tuple[int, str], list[float]] = {}
        for row in rows:
            pair = (int(row["seed"]), row["architecture"])
            values.setdefault(pair, []).append(float(row[key]))
        return {pair: float(np.mean(items)) for pair, items in values.items()}

    criterion = grouped(whole_rows, "checkpoint_criterion")
    physics = grouped(whole_rows, "exact_pp_rmse_clean_normalized")
    miou = grouped(whole_rows, "miou")
    property_rmse: dict[str, dict[tuple[int, str], float]] = {}
    property_ssim: dict[str, dict[tuple[int, str], float]] = {}
    for name in PROPERTIES:
        property_rmse[name] = grouped(
            [row for row in whole_rows if row["property"] == name], "normalized_rmse"
        )
        property_ssim[name] = grouped(
            [row for row in whole_rows if row["property"] == name], "ssim"
        )
    class_iou = {
        label: grouped(whole_rows, f"class_{label}_iou") for label in range(3)
    }
    full_property_rmse = {
        pair: float(np.mean([property_rmse[name][pair] for name in PROPERTIES]))
        for pair in criterion
    }
    paired_rows = []
    for seed in map(int, contract["seeds"]):
        for control in ("cartesian_gnn", "no_gnn"):
            row: dict[str, Any] = {
                "seed": seed,
                "comparison": f"rgt_gnn_minus_{control}",
                "checkpoint_criterion": criterion[(seed, "rgt_gnn")] - criterion[(seed, control)],
                "exact_pp_rmse_clean_normalized": physics[(seed, "rgt_gnn")] - physics[(seed, control)],
                "miou": miou[(seed, "rgt_gnn")] - miou[(seed, control)],
            }
            for name in PROPERTIES:
                row[f"{name}_normalized_rmse"] = (
                    property_rmse[name][(seed, "rgt_gnn")]
                    - property_rmse[name][(seed, control)]
                )
                row[f"{name}_ssim"] = (
                    property_ssim[name][(seed, "rgt_gnn")]
                    - property_ssim[name][(seed, control)]
                )
            row["full_property_normalized_rmse"] = (
                full_property_rmse[(seed, "rgt_gnn")]
                - full_property_rmse[(seed, control)]
            )
            for label in range(3):
                row[f"class_{label}_iou"] = (
                    class_iou[label][(seed, "rgt_gnn")]
                    - class_iou[label][(seed, control)]
                )
            paired_rows.append(row)
    _write_csv(experiment / "causal_gnn_paired_effects.csv", paired_rows)

    comparisons = {}
    higher_is_better = {
        "vp_ssim",
        "vs_ssim",
        "density_ssim",
        "miou",
        "class_0_iou",
        "class_1_iou",
        "class_2_iou",
    }
    for control in ("cartesian_gnn", "no_gnn"):
        selected = [row for row in paired_rows if row["comparison"].endswith(control)]
        comparisons[control] = {
            key: {
                "mean_delta": float(np.mean([float(row[key]) for row in selected])),
                "std_delta": float(np.std([float(row[key]) for row in selected])),
                "favorable_seed_count": int(
                    sum(
                        float(row[key]) > 0
                        if key in higher_is_better
                        else float(row[key]) < 0
                        for row in selected
                    )
                ),
                "seed_count": len(selected),
            }
            for key in (
                "vp_normalized_rmse",
                "vs_normalized_rmse",
                "density_normalized_rmse",
                "full_property_normalized_rmse",
                "vp_ssim",
                "vs_ssim",
                "density_ssim",
                "checkpoint_criterion",
                "exact_pp_rmse_clean_normalized",
                "miou",
                "class_0_iou",
                "class_1_iou",
                "class_2_iou",
            )
        }
        comparisons[control]["checkpoint_criterion"]["mean_relative_gain"] = float(
            -comparisons[control]["checkpoint_criterion"]["mean_delta"]
            / np.mean(
                [criterion[(int(seed), control)] for seed in contract["seeds"]]
            )
        )

    complex_names = {"high_dip", "high_rgt_curvature", "facies_boundary"}
    complexity_effects = []
    for seed in map(int, contract["seeds"]):
        for control in ("cartesian_gnn", "no_gnn"):
            for group_name, names in (
                ("simple", {"simple_low_dip"}),
                ("complex", complex_names),
            ):
                row: dict[str, Any] = {
                    "seed": seed,
                    "comparison": f"rgt_gnn_minus_{control}",
                    "complexity_group": group_name,
                }
                for prop in PROPERTIES:
                    relevant = [
                        item
                        for item in strata_rows
                        if int(item["seed"]) == seed
                        and item["stratum"] in names
                        and item["property"] == prop
                    ]
                    by_architecture = {
                        architecture: np.mean(
                            [
                                float(item["normalized_rmse"])
                                for item in relevant
                                if item["architecture"] == architecture
                            ]
                        )
                        for architecture in ("rgt_gnn", control)
                    }
                    row[f"{prop}_normalized_rmse"] = float(
                        by_architecture["rgt_gnn"] - by_architecture[control]
                    )
                for metric in (
                    "exact_pp_rmse_clean_normalized",
                    "miou",
                    "class_0_iou",
                    "class_1_iou",
                    "class_2_iou",
                ):
                    relevant = [
                        item
                        for item in strata_rows
                        if int(item["seed"]) == seed and item["stratum"] in names
                    ]
                    by_architecture = {
                        architecture: float(
                            np.mean(
                                [
                                    float(item[metric])
                                    for item in relevant
                                    if item["architecture"] == architecture
                                ]
                            )
                        )
                        for architecture in ("rgt_gnn", control)
                    }
                    row[metric] = (
                        by_architecture["rgt_gnn"] - by_architecture[control]
                    )
                complexity_effects.append(row)
    _write_csv(experiment / "complexity_paired_effects.csv", complexity_effects)

    criterion_vs_cart = comparisons["cartesian_gnn"]["checkpoint_criterion"]
    criterion_vs_none = comparisons["no_gnn"]["checkpoint_criterion"]
    minimum_relative_gain = float(contract["decision"]["minimum_mean_relative_criterion_gain"])
    rgt_specific = (
        criterion_vs_cart["favorable_seed_count"] == 3
        and criterion_vs_cart["mean_relative_gain"] >= minimum_relative_gain
    )
    rgt_global = (
        rgt_specific
        and criterion_vs_none["favorable_seed_count"] == 3
        and criterion_vs_none["mean_relative_gain"] >= minimum_relative_gain
    )
    graph_global = (
        criterion_vs_none["favorable_seed_count"] == 3
        and criterion_vs_none["mean_relative_gain"] >= minimum_relative_gain
    )
    complex_rows = [row for row in complexity_effects if row["complexity_group"] == "complex"]
    complex_favorable = all(
        np.mean([float(row[f"{name}_normalized_rmse"]) for name in PROPERTIES]) < 0
        for row in complex_rows
    )
    simple_rows = [row for row in complexity_effects if row["complexity_group"] == "simple"]
    simple_favorable = all(
        np.mean([float(row[f"{name}_normalized_rmse"]) for name in PROPERTIES]) < 0
        for row in simple_rows
    )
    property_signs = [
        comparisons["no_gnn"][f"{name}_normalized_rmse"]["favorable_seed_count"]
        for name in PROPERTIES
    ]

    final_patch_rows = [
        row for row in patch_rows if int(row["epoch"]) == maximum
    ]
    patch_criterion = {
        (int(row["seed"]), row["architecture"]): float(row["sample_criterion"])
        for row in final_patch_rows
    }
    patch_rgt_beats_no_gnn = all(
        patch_criterion[(seed, "rgt_gnn")] < patch_criterion[(seed, "no_gnn")]
        for seed in map(int, contract["seeds"])
    )
    patch_only_gain = patch_rgt_beats_no_gnn and not graph_global

    architecture_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    scalar_metrics = {
        "full_property_normalized_rmse": full_property_rmse,
        "checkpoint_criterion": criterion,
        "exact_pp_rmse_clean_normalized": physics,
        "miou": miou,
        **{
            f"{name}_normalized_rmse": property_rmse[name] for name in PROPERTIES
        },
        **{f"{name}_ssim": property_ssim[name] for name in PROPERTIES},
        **{f"class_{label}_iou": class_iou[label] for label in range(3)},
    }
    for architecture in ARCHITECTURES:
        architecture_metrics[architecture] = {}
        for metric, values in scalar_metrics.items():
            per_seed = {
                str(seed): values[(seed, architecture)]
                for seed in map(int, contract["seeds"])
            }
            array = np.asarray(list(per_seed.values()), dtype=np.float64)
            architecture_metrics[architecture][metric] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "per_seed": per_seed,
            }

    mechanism_rows: list[dict[str, Any]] = []
    for seed in map(int, contract["seeds"]):
        probe = _run_path(experiment, seed, "rgt_gnn") / "mechanism_probe"
        with (probe / "graph_learning_summary.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            graph_rows = list(csv.DictReader(stream))
        with (probe / "gradient_contributions.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            gradient_rows = list(csv.DictReader(stream))
        for layer in (1, 2):
            graph_row = next(
                row for row in graph_rows if int(row["layer"]) == layer
            )
            selected_gradients = [
                row
                for row in gradient_rows
                if row["parameter_group"] == f"transformerconv_layer_{layer}"
            ]
            raw_norms = np.asarray(
                [float(row["raw_gradient_norm"]) for row in selected_gradients]
            )
            weighted_norms = np.asarray(
                [float(row["weighted_gradient_norm"]) for row in selected_gradients]
            )
            mechanism_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "objective_count": len(selected_gradients),
                    "raw_gradient_norm_rss": float(np.linalg.norm(raw_norms)),
                    "weighted_gradient_norm_rss": float(np.linalg.norm(weighted_norms)),
                    "attention_entropy_normalized": float(
                        graph_row["attention_entropy_normalized"]
                    ),
                    "top_decile_attention_mass": float(
                        graph_row["top_decile_attention_mass"]
                    ),
                    "graph_embedding_rms": graph_row["graph_embedding_rms"],
                    "graph_reinjection_velocity_rms": graph_row[
                        "graph_reinjection_velocity_rms"
                    ],
                    "prediction_change_when_graph_reinjection_zeroed_velocity_rms": (
                        graph_row["graph_reinjection_velocity_rms"]
                    ),
                    "rgt_vs_cartesian_velocity_rms": graph_row[
                        "rgt_vs_cartesian_velocity_rms"
                    ],
                    "interpretation": "mechanism_diagnostic_only_not_causal_evidence",
                }
            )
    _write_csv(experiment / "rgt_mechanism_diagnostics.csv", mechanism_rows)
    if rgt_global:
        decision = "RGT_GNN_REPRODUCIBLE_GAIN"
    elif graph_global and not rgt_specific:
        decision = "GRAPH_GAIN_BUT_RGT_NOT_SPECIFIC"
    elif complex_favorable and not simple_favorable:
        decision = "GNN_COMPLEX_GEOLOGY_ONLY_GAIN"
    elif max(property_signs) > 0 and min(property_signs) < 3:
        decision = "GNN_TRADEOFF"
    else:
        decision = "GNN_NO_REPRODUCIBLE_GAIN"

    best_property = min(
        PROPERTIES,
        key=lambda name: comparisons["no_gnn"][f"{name}_normalized_rmse"][
            "mean_delta"
        ],
    )
    rgt_vs_no = comparisons["no_gnn"]
    rgt_vs_cartesian = comparisons["cartesian_gnn"]
    causal_answers = {
        "1_gnn_improves_whole_realization_inversion": (
            "yes_reproducibly"
            if graph_global
            else "not_reproducibly_under_this_bounded_screen"
        ),
        "2_rgt_improves_over_cartesian_routing": (
            "yes_reproducibly" if rgt_specific else "no_reproducible_advantage"
        ),
        "3_advantage_larger_in_complex_geology": (
            "yes" if complex_favorable and not simple_favorable else "not_demonstrated"
        ),
        "4_property_with_largest_mean_rgt_gain_over_no_gnn": best_property,
        "5_segmentation_benefits": {
            "mean_miou_delta": rgt_vs_no["miou"]["mean_delta"],
            "favorable_seed_count": rgt_vs_no["miou"]["favorable_seed_count"],
        },
        "6_physics_consistency_benefits": {
            "mean_exact_pp_rmse_delta": rgt_vs_no[
                "exact_pp_rmse_clean_normalized"
            ]["mean_delta"],
            "favorable_seed_count": rgt_vs_no[
                "exact_pp_rmse_clean_normalized"
            ]["favorable_seed_count"],
        },
        "7_effect_reproducible_across_seeds": {
            "rgt_vs_no_gnn_favorable_criterion_seeds": rgt_vs_no[
                "checkpoint_criterion"
            ]["favorable_seed_count"],
            "rgt_vs_cartesian_favorable_criterion_seeds": rgt_vs_cartesian[
                "checkpoint_criterion"
            ]["favorable_seed_count"],
        },
        "8_justifies_extra_graph_complexity": (
            "yes_for_controlled_followup_not_yet_for_production"
            if decision
            in {
                "RGT_GNN_REPRODUCIBLE_GAIN",
                "GRAPH_GAIN_BUT_RGT_NOT_SPECIFIC",
                "GNN_COMPLEX_GEOLOGY_ONLY_GAIN",
            }
            else "no"
        ),
    }
    summary = {
        "status": "BOUNDED_EXPERIMENT_COMPLETE",
        "decision": decision,
        "epochs": maximum,
        "seeds": contract["seeds"],
        "whole_realization_ids": contract["whole_realization_evaluation"]["realization_ids"],
        "architecture_whole_realization_metrics": architecture_metrics,
        "comparisons": comparisons,
        "mechanism_diagnostics": mechanism_rows,
        "causal_answers": causal_answers,
        "hypotheses": {
            "H1_complex_benefit_exceeds_simple": complex_favorable and not simple_favorable,
            "H2_rgt_topology_reproducibly_beats_cartesian": rgt_specific,
            "H3_graph_but_not_rgt_specific": graph_global and not rgt_specific,
            "H4_patch_only_gain": patch_only_gain,
        },
        "fault_vicinity_status": contract["complexity_strata"]["fault_vicinity"],
        "new_graph_mechanism_added": False,
        "production_training_started": False,
        "commit_or_push_performed": False,
    }
    (experiment / "causal_gnn_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    table_metrics = (
        "vp_normalized_rmse",
        "vs_normalized_rmse",
        "density_normalized_rmse",
        "full_property_normalized_rmse",
        "vp_ssim",
        "vs_ssim",
        "density_ssim",
        "exact_pp_rmse_clean_normalized",
        "miou",
        "class_0_iou",
        "class_1_iou",
        "class_2_iou",
        "checkpoint_criterion",
    )
    report_lines = [
        "# v00332o causal audit of the existing GNN",
        "",
        f"Decision: `{decision}`",
        "",
        (
            "This is a three-epoch, three-seed bounded screen on one tiled whole "
            "validation realization per seed. It is causal screening evidence, not "
            "final production or paper-quality evidence."
        ),
        "",
        "## Whole-realization mean ± population standard deviation across seeds",
        "",
        "| metric | RGT GNN | Cartesian GNN | No GNN |",
        "|---|---:|---:|---:|",
    ]
    for metric in table_metrics:
        values = []
        for architecture in ARCHITECTURES:
            item = architecture_metrics[architecture][metric]
            values.append(f"{item['mean']:.6g} ± {item['std']:.3g}")
        report_lines.append(f"| {metric} | " + " | ".join(values) + " |")
    report_lines.extend(
        [
            "",
            "## Causal questions",
            "",
            *[
                f"{index}. **{key}**: `{json.dumps(value, sort_keys=True)}`"
                for index, (key, value) in enumerate(causal_answers.items(), start=1)
            ],
            "",
            "## Interpretation guardrails",
            "",
            "- RGT and Cartesian models have identical parameter counts and tensor-identical initial states per seed.",
            "- The no-GNN control shares every shape-compatible initialized tensor but has a smaller graph-free architecture.",
            "- Attention and nonzero graph gradients establish activity only; they are not causal benefit.",
            "- Fault-vicinity stratification is unavailable because Stage-03 has no explicit fault mask.",
            "- No new graph mechanism, loss, production run, commit, or push was performed.",
            "",
        ]
    )
    (experiment / "causal_gnn_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def status(_: argparse.Namespace) -> None:
    contract, dataset, experiment, _ = _roots()
    print(
        json.dumps(
            {
                "dataset_ready": (dataset / "dataset_manifest.json").exists(),
                "prepared": (experiment / "causal_gnn_experiment_contract.json").exists(),
                "smoke_passed": (experiment / "causal_gnn_smoke_test.json").exists(),
                "runs": [
                    {
                        "seed": seed,
                        "architecture": architecture,
                        "completed_epoch": _completed_epoch(
                            _run_path(experiment, seed, architecture)
                        ),
                        "whole_outputs": len(
                            list(
                                (
                                    _run_path(experiment, seed, architecture)
                                    / "whole_realization"
                                ).glob("realization_*.npz")
                            )
                        ),
                    }
                    for seed in map(int, contract["seeds"])
                    for architecture in ARCHITECTURES
                ],
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(function=prepare)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--device", required=True)
    smoke_parser.set_defaults(function=smoke)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--device", required=True)
    train_parser.set_defaults(function=train)
    whole_parser = commands.add_parser("evaluate-whole")
    whole_parser.add_argument("--device", required=True)
    whole_parser.set_defaults(function=evaluate_whole)
    commands.add_parser("analyze").set_defaults(function=analyze)
    commands.add_parser("status").set_defaults(function=status)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
