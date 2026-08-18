"""Complete SAGE-AVO training, validation, and checkpoint workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from sage_avo.config import seed_everything
from sage_avo.forward import forward_specification_from_mapping
from sage_avo.data.augmentation import AugmentationConfig
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import PatchSamplingConfig, build_patch_sampling_weights
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.models.variants import (
    build_sage_avo_variant,
    sage_avo_model_kwargs,
    variant_definition,
)
from sage_avo.training.checkpoints import load_checkpoint, save_checkpoint
from sage_avo.training.engine import (
    ContrastiveSettings,
    PhysicsNormalization,
    PhysicsSettings,
    StepMetrics,
    train_epoch,
    validate_epoch,
)
from sage_avo.training.losses import AdaptiveTaskWeighter, LossWeights
from sage_avo.training.schedules import Curriculum
from sage_avo.training.selection import (
    CheckpointSelectionState,
    checkpoint_metadata,
    whole_realization_criterion,
    weighted_objective_contributions,
)

from .manifest import build_run_manifest, write_json


def _normalization_tensors(normalization: dict[str, list[float]]) -> PhysicsNormalization:
    def tensor(name: str) -> Tensor:
        return torch.tensor(normalization[name], dtype=torch.float32).view(1, 3, 1, 1)

    return PhysicsNormalization(
        x_mean=tensor("x_mean"),
        x_std=tensor("x_std"),
        y_mean=tensor("y_mean"),
        y_std=tensor("y_std"),
    )


def _class_weights(
    dataset: IndexedRealizationPatches,
    classes: int = 3,
    foreground_boost: float = 1.15,
) -> Tensor:
    """Compute inverse-frequency weights on valid training pixels only."""
    counts = np.zeros(classes, dtype=np.float64)
    for index in range(len(dataset)):
        fields = dataset.sampling_fields(index)
        labels = fields["segmentation"].numpy()
        valid = fields["mask"][0].numpy() > 0.5
        labels = labels[valid]
        labels = labels[(labels >= 0) & (labels < classes)]
        counts += np.bincount(labels.reshape(-1), minlength=classes)
    inverse = counts.sum() / (classes * np.maximum(counts, 1.0))
    normalized = inverse / inverse.mean()
    if classes > 1:
        normalized[1:] *= float(foreground_boost)
    return torch.tensor(normalized, dtype=torch.float32)


def _angles(definition: dict[str, Any]) -> tuple[float, ...]:
    start = float(definition["start"])
    stop = float(definition["stop"])
    step = float(definition["step"])
    count = int(round((stop - start) / step)) + 1
    return tuple(start + index * step for index in range(count))


def physics_settings_from_config(config: dict[str, Any]) -> PhysicsSettings:
    if "forward_model" in config:
        specification = forward_specification_from_mapping(config)
        return PhysicsSettings(
            angles_degrees=specification.angles_degrees,
            bands_degrees=tuple(
                (band.minimum_degrees, band.maximum_degrees)
                for band in specification.bands
            ),
            wavelet_hz=specification.wavelets[0].peak_frequency_hz,
            dt_seconds=specification.dt_seconds,
            wavelet_samples=specification.wavelets[0].samples,
            apply_mute=specification.apply_mute,
            mute_start=specification.mute_start,
            mute_end=specification.mute_end,
            taper_samples=specification.taper_samples,
            specification=specification,
        )
    forward = config["training"]["physics_forward"]
    bands = tuple(tuple(float(value) for value in band) for band in forward["bands_degrees"])
    return PhysicsSettings(
        angles_degrees=_angles(forward["angles_degrees"]),
        bands_degrees=bands,
        wavelet_hz=float(forward["wavelet_hz"]),
        dt_seconds=float(forward["dt_seconds"]),
        wavelet_samples=int(forward["wavelet_samples"]),
        apply_mute=bool(forward["front_mute"]["enabled"]),
        mute_start=tuple(float(value) for value in forward["front_mute"]["start"]),
        mute_end=tuple(float(value) for value in forward["front_mute"]["end"]),
        taper_samples=int(forward["front_mute"]["taper_samples"]),
    )


def loss_weights_from_config(config: dict[str, Any], physics_weight: float) -> LossWeights:
    values = config["training"]["loss_weights"]
    contrastive = config["training"]["contrastive_loss"]
    return LossWeights(
        inversion=float(values["inversion"]),
        flow_velocity=float(values["flow_velocity"]),
        full_property=float(values["full_property"]),
        ssim=float(values["ssim_initial"]),
        segmentation=float(values["segmentation"]),
        segmentation_cross_entropy=float(values["segmentation_cross_entropy"]),
        segmentation_dice=float(values["segmentation_dice"]),
        contrastive=float(contrastive["weight"]) if bool(contrastive["enabled"]) else 0.0,
        physics=float(physics_weight),
        structure=float(values["structure"]),
        density=float(values["density_initial"]),
    )


def curriculum_from_config(config: dict[str, Any]) -> Curriculum:
    values = config["training"]["curriculum"]
    return Curriculum(
        density_start=float(values["density_weight"]["start"]),
        density_end=float(values["density_weight"]["end"]),
        ssim_start=float(values["ssim_weight"]["start"]),
        ssim_end=float(values["ssim_weight"]["end"]),
        physics_multiplier_start=float(values["physics_multiplier"]["start"]),
        physics_multiplier_end=float(values["physics_multiplier"]["end"]),
        structure_multiplier_start=float(values["structure_multiplier"]["start"]),
        structure_multiplier_end=float(values["structure_multiplier"]["end"]),
    )


def _augmentation(config: dict[str, Any]) -> AugmentationConfig:
    values = config["training"]["augmentation"]
    return AugmentationConfig(
        horizontal_flip_probability=float(values["horizontal_flip_probability"]),
        avo_gain_probability=float(values["avo_gain_probability"]),
        avo_gain_minimum=float(values["avo_gain_range"][0]),
        avo_gain_maximum=float(values["avo_gain_range"][1]),
        avo_noise_probability=float(values["avo_noise_probability"]),
        avo_noise_standard_deviation=float(values["avo_noise_std_normalized"]),
    )


def _sampling(config: dict[str, Any]) -> PatchSamplingConfig:
    values = config["training"]["weighted_patch_sampling"]
    angles = tuple(float(value) for value in config["model"]["representative_angles_degrees"])
    return PatchSamplingConfig(
        foreground_boost=float(values["foreground_boost"]),
        structure_boost=float(values["structure_boost"]),
        avo_gradient_boost=float(values["avo_gradient_boost"]),
        foreground_fraction_threshold=float(values["foreground_fraction_threshold"]),
        upper_quantile=float(values["upper_quantile"]),
        representative_angles_degrees=angles,
    )


@torch.no_grad()
def _validation_sample_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    steps: int,
    max_batches: int,
    guidance_scale: float,
) -> dict[str, Any]:
    model.eval()
    squared_error = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)
    intersections = np.zeros(3, dtype=np.float64)
    unions = np.zeros(3, dtype=np.float64)
    predicted_counts = np.zeros(3, dtype=np.float64)
    target_counts = np.zeros(3, dtype=np.float64)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        values = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, Tensor)
        }
        prediction = model.sample(
            values["avo"],
            values["low"],
            values["rgt"],
            steps=steps,
            guidance_scale=guidance_scale,
            avo_mask=values["mask"],
        )
        mask = values["mask"].expand_as(prediction)
        error = ((prediction - values["target"]) ** 2 * mask).sum(dim=(0, 2, 3))
        squared_error += error.cpu().numpy()
        count += mask.sum(dim=(0, 2, 3)).cpu().numpy()
        final_time = torch.ones(prediction.shape[0], device=device)
        labels = model(
            prediction,
            final_time,
            values["avo"],
            values["low"],
            values["rgt"],
        ).segmentation_logits.argmax(dim=1)
        valid = values["mask"][:, 0] > 0.5
        for label in range(3):
            predicted = (labels == label) & valid
            target = (values["segmentation"] == label) & valid
            intersections[label] += (predicted & target).sum().item()
            unions[label] += (predicted | target).sum().item()
            predicted_counts[label] += predicted.sum().item()
            target_counts[label] += target.sum().item()
    normalized_rmse = np.sqrt(squared_error / np.maximum(count, 1.0))
    present = unions > 0
    miou = float(np.mean(intersections[present] / unions[present])) if present.any() else np.nan
    criterion = float(np.mean(normalized_rmse) - 0.1 * miou)
    class_iou = np.divide(
        intersections,
        unions,
        out=np.full(3, np.nan),
        where=unions > 0,
    )
    class_dice = np.divide(
        2.0 * intersections,
        predicted_counts + target_counts,
        out=np.full(3, np.nan),
        where=(predicted_counts + target_counts) > 0,
    )
    return {
        "criterion": criterion,
        "normalized_rmse": normalized_rmse.tolist(),
        "miou": miou,
        "class_iou": class_iou.tolist(),
        "class_dice": class_dice.tolist(),
    }


def _validation_sample_score(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    steps: int,
    max_batches: int,
    guidance_scale: float,
) -> tuple[float, list[float], float]:
    """Backward-compatible compact sampling score used by v002 callers."""
    metrics = _validation_sample_metrics(
        model,
        loader,
        device,
        steps=steps,
        max_batches=max_batches,
        guidance_scale=guidance_scale,
    )
    return metrics["criterion"], metrics["normalized_rmse"], metrics["miou"]


def _fixed_whole_validation_ids(
    dataset_root: Path,
    validation_ids: list[int],
    count: int,
) -> list[int]:
    """Choose the first sorted archive per geology group by a persisted rule."""
    selected: list[int] = []
    seen_groups: set[int] = set()
    for realization_id in sorted(map(int, validation_ids)):
        path = dataset_root / "realizations" / f"realization_{realization_id:07d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            group_id = int(
                archive["geology_realization_id"]
                if "geology_realization_id" in archive.files
                else realization_id
            )
        if group_id in seen_groups:
            continue
        selected.append(realization_id)
        seen_groups.add(group_id)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("No validation realizations are available for whole-section metrics")
    return selected


@torch.no_grad()
def _whole_realization_validation_metrics(
    model: torch.nn.Module,
    *,
    dataset_root: Path,
    realization_ids: list[int],
    normalization: dict[str, list[float]],
    patch_shape: tuple[int, int],
    stride: tuple[int, int],
    steps: int,
    batch_size: int,
    device: torch.device,
    guidance_scale: float,
) -> dict[str, Any]:
    """Evaluate deterministic complete validation sections in physical units."""
    y_std = np.asarray(normalization["y_std"], dtype=np.float64)
    per_realization: list[dict[str, Any]] = []
    for realization_id in realization_ids:
        path = dataset_root / "realizations" / f"realization_{realization_id:07d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            prediction, labels = infer_full_realization(
                model,
                avo=archive["avo"],
                low=archive["low"],
                rgt=archive["rgt"],
                normalization=normalization,
                patch_shape=patch_shape,
                stride=stride,
                steps=steps,
                batch_size=batch_size,
                device=device,
                valid_mask=archive["valid_mask"],
                guidance_scale=guidance_scale,
            )
            target = np.asarray(archive["elastic"], dtype=np.float64)
            target_labels = np.asarray(archive["segmentation"], dtype=np.int64)
            valid = np.asarray(archive["valid_mask"], dtype=bool)
        physical_rmse = np.asarray(
            [
                np.sqrt(
                    np.mean(
                        np.square(
                            prediction[channel][valid] - target[channel][valid]
                        )
                    )
                )
                for channel in range(3)
            ]
        )
        normalized_rmse = physical_rmse / y_std
        class_iou = []
        for label in range(3):
            predicted = (labels == label) & valid
            expected = (target_labels == label) & valid
            union = np.count_nonzero(predicted | expected)
            class_iou.append(
                float(np.count_nonzero(predicted & expected) / union)
                if union
                else np.nan
            )
        miou = float(np.nanmean(class_iou))
        per_realization.append(
            {
                "realization_id": int(realization_id),
                "physical_rmse": physical_rmse.tolist(),
                "normalized_rmse": normalized_rmse.tolist(),
                "class_iou": class_iou,
                "miou": miou,
            }
        )
    aggregate_rmse = np.mean(
        [record["normalized_rmse"] for record in per_realization], axis=0
    )
    aggregate_miou = float(np.mean([record["miou"] for record in per_realization]))
    return {
        "criterion": whole_realization_criterion(aggregate_rmse, aggregate_miou),
        "normalized_rmse": aggregate_rmse.tolist(),
        "miou": aggregate_miou,
        "per_realization": per_realization,
    }


def _metric_row(prefix: str, metrics: StepMetrics) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in metrics.__dict__.items()}


def train_controlled_variant(
    *,
    repository: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    dataset_directory: str | Path,
    experiment_directory: str | Path,
    variant: str,
    device_name: str | None = None,
    epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    run_name: str | None = None,
    resume_from: str | Path | None = None,
    allow_operator_validation_subset: bool = False,
) -> Path:
    """Train one SAGE-AVO condition with the configured complete procedure."""
    configured_physics_weight = float(config["training"]["loss_weights"]["physics"])
    definition = variant_definition(variant, physics_weight=configured_physics_weight)
    if definition.graph_mode is None:
        raise ValueError("The low-prior condition requires no training")
    seed = int(config["experiment"]["seed"])
    seed_everything(
        seed,
        deterministic_torch=bool(config["hardware"].get("deterministic_algorithms", True)),
    )
    device = torch.device(
        device_name
        or (
            "cuda"
            if torch.cuda.is_available() and config["hardware"]["preferred_device"] == "cuda"
            else "cpu"
        )
    )
    dataset_root = Path(dataset_directory)
    experiment_root = Path(experiment_directory)
    run_directory = experiment_root / "runs" / (run_name or variant)
    normalization = json.loads((dataset_root / "normalization.json").read_text(encoding="utf-8"))
    split_ids = json.loads((dataset_root / "split_ids.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        (dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    source_status = dataset_manifest.get("source_stage02_status")
    if source_status is not None and source_status != "complete" and not allow_operator_validation_subset:
        raise RuntimeError(
            "Production training requires a complete Stage-02 corpus; "
            f"the dataset records source status {source_status!r}. "
            "Use allow_operator_validation_subset=True only for an explicitly labeled sanity run."
        )
    prior = dataset_manifest["prior"]
    run_directory.mkdir(parents=True, exist_ok=True)
    training_config = config["training"]
    epochs = int(epochs_override or training_config["epochs"])

    augmentation_generator = torch.Generator().manual_seed(seed + 11)
    sampler_generator = torch.Generator().manual_seed(seed + 13)
    time_generator = torch.Generator().manual_seed(seed + 17)
    contrastive_generator = torch.Generator(device=device).manual_seed(seed + 19)
    train_dataset = IndexedRealizationPatches(
        dataset_root,
        "train",
        augment=bool(training_config["augmentation"]["enabled"]),
        augmentation_config=_augmentation(config),
        augmentation_generator=augmentation_generator,
    )
    validation_dataset = IndexedRealizationPatches(dataset_root, "validation")
    if bool(training_config["weighted_patch_sampling"]["enabled"]):
        patch_weights = build_patch_sampling_weights(train_dataset, _sampling(config))
        sampler = WeightedRandomSampler(
            patch_weights,
            num_samples=len(patch_weights),
            replacement=True,
            generator=sampler_generator,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        sampler=sampler,
        shuffle=shuffle,
        generator=sampler_generator if shuffle else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    physics_settings = physics_settings_from_config(config)
    model = build_sage_avo_variant(
        variant,
        **sage_avo_model_kwargs(config),
    ).to(device)
    model.set_norm_stats(normalization)
    base_weights = loss_weights_from_config(config, float(definition.physics_weight))
    curriculum = curriculum_from_config(config)
    is_v003 = int(config.get("schema_version", 2)) >= 3
    final_weights = curriculum.weights_for_epoch(base_weights, epochs - 1, epochs)
    contrastive_settings = ContrastiveSettings(
        temperature=float(training_config["contrastive_loss"]["temperature"]),
        max_samples=int(training_config["contrastive_loss"]["max_samples"]),
    )
    adaptive_config = training_config["adaptive_task_weighting"]
    adaptive_weighter = None
    if bool(adaptive_config["enabled"]):
        active_tasks = ["inversion", "segmentation"]
        if base_weights.contrastive > 0:
            active_tasks.append("contrastive")
        if base_weights.physics > 0:
            active_tasks.append("physics")
        if base_weights.structure > 0:
            active_tasks.append("structure")
        adaptive_weighter = AdaptiveTaskWeighter(active_tasks).to(device)
    optimizer_parameters = list(model.parameters())
    if adaptive_weighter is not None:
        optimizer_parameters.extend(adaptive_weighter.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config["scheduler_eta_min"]),
    )
    class_weights = _class_weights(
        train_dataset,
        classes=int(config["model"]["classes"]),
        foreground_boost=float(training_config["class_weight_foreground_boost"]),
    ).to(device)
    physics_normalization = _normalization_tensors(normalization)
    validation_time_grid = tuple(float(value) for value in training_config["validation_time_grid"])
    sampling_config = training_config["physics_guided_sampling"]
    guidance_scale = (
        float(sampling_config["guidance_scale"])
        if bool(sampling_config["enabled"])
        else 0.0
    )

    metric_definitions = {
        "checkpoint_criterion": training_config.get(
            "checkpoint_criterion", training_config.get("checkpoint_criteria")
        ),
        "validation_time_grid": list(validation_time_grid),
        "validation_sampling_batches": int(training_config["validation_sample_batches"]),
        "physics_guidance_scale": guidance_scale,
    }
    if is_v003:
        metric_definitions.update(
            {
                "fixed_objective_weights": final_weights.__dict__,
                "checkpoint_files": {
                    "fixed_objective": "best_fixed_objective.pt",
                    "sampling": "best_sampling.pt",
                    "segmentation": "best_segmentation.pt",
                    "whole_realization": "best_whole_realization.pt",
                    "last": "last.pt",
                },
            }
        )
    manifest_path = run_directory / "manifest.json"
    previous_manifest: dict[str, Any] = {}
    if resume_from is not None and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_flow = float(previous_manifest.get("best_validation_objective", float("inf")))
    best_sample = float(previous_manifest.get("best_sample_criterion", float("inf")))
    selection_state = CheckpointSelectionState.from_mapping(
        previous_manifest.get("checkpoint_selection_state") if is_v003 else None
    )
    manifest = build_run_manifest(
        repository=repository,
        config_path=config_path,
        seed=seed,
        split_ids=split_ids,
        model_variant=variant,
        checkpoint=None,
        training_epochs=epochs,
        normalization=normalization,
        prior_settings=prior,
        metric_definitions=metric_definitions,
        status="running",
    )
    manifest["algorithm_capabilities"] = config.get("capabilities", {})
    manifest["source_snapshot"] = config.get("source_snapshot")
    manifest["reference_training_contract"] = config.get(
        "reference_training_contract", config.get("final_005_settings", {})
    )
    manifest["source_stage02_status"] = source_status or "not_applicable"
    manifest["operator_validation_subset_allowed"] = bool(allow_operator_validation_subset)
    if is_v003:
        manifest["checkpoint_selection_state"] = selection_state.to_dict()
        checkpointing = training_config["checkpointing"]
        whole_validation_ids = _fixed_whole_validation_ids(
            dataset_root,
            split_ids["validation"],
            int(checkpointing["whole_validation_realization_count"]),
        )
        manifest["whole_validation_protocol"] = {
            "selection_rule": "first sorted archive ID per distinct geology group",
            "realization_ids": whole_validation_ids,
            "every_epochs": int(checkpointing["whole_validation_every_epochs"]),
            "criterion": training_config["checkpoint_criteria"]["whole_realization"],
            "test_split_used_for_selection": False,
        }
    if resume_from is not None:
        manifest["resumed_from_checkpoint"] = Path(resume_from).name
        if np.isfinite(best_flow):
            manifest["prior_best_validation_objective"] = best_flow
        if np.isfinite(best_sample):
            manifest["prior_best_sample_criterion"] = best_sample
    write_json(manifest_path, manifest)

    start_epoch = 0
    resumed_metrics: dict[str, Any] = {}
    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            adaptive_weighter=adaptive_weighter,
            restore_rng=True,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        resumed_metrics = checkpoint.get("metrics", {})
        if is_v003 and "checkpoint_selection_state" in resumed_metrics:
            selection_state = CheckpointSelectionState.from_mapping(
                resumed_metrics["checkpoint_selection_state"]
            )
        if not np.isfinite(best_flow) and "validation_objective" in resumed_metrics:
            best_flow = float(resumed_metrics["validation_objective"])
        if not np.isfinite(best_sample) and "sample_criterion" in resumed_metrics:
            best_sample = float(resumed_metrics["sample_criterion"])
        generator_states = checkpoint.get("rng_state", {}).get("generators", {})
        for name, generator in (
            ("augmentation", augmentation_generator),
            ("sampler", sampler_generator),
            ("time", time_generator),
            ("contrastive", contrastive_generator),
        ):
            if name in generator_states:
                generator.set_state(generator_states[name])

    metric_names = tuple(StepMetrics.__dataclass_fields__)
    fieldnames = list(
        ["epoch"]
        + [f"train_{name}" for name in metric_names]
        + [f"validation_{name}" for name in metric_names]
        + [
            "sample_criterion",
            "sample_rmse_vp_normalized",
            "sample_rmse_vs_normalized",
            "sample_rmse_density_normalized",
            "sample_miou",
            "learning_rate",
            "density_weight",
            "ssim_weight",
            "physics_weight",
            "structure_weight",
        ]
    )
    if is_v003:
        fieldnames.extend(
            [
                "validation_fixed_objective",
                "validation_weighted_inversion_contribution",
                "validation_weighted_segmentation_contribution",
                "validation_weighted_contrastive_contribution",
                "validation_weighted_physics_contribution",
                "validation_weighted_structure_contribution",
                "sample_class_0_iou",
                "sample_class_1_iou",
                "sample_class_2_iou",
                "sample_class_0_dice",
                "sample_class_1_dice",
                "sample_class_2_dice",
                "whole_validation_criterion",
                "whole_validation_rmse_vp_normalized",
                "whole_validation_rmse_vs_normalized",
                "whole_validation_rmse_density_normalized",
                "whole_validation_miou",
            ]
        )
    log_path = run_directory / "training_log.csv"
    append = resume_from is not None and log_path.exists()
    mode = "a" if append else "w"
    last_validation = float(resumed_metrics.get("validation_objective", np.nan))
    with log_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for epoch_index in range(start_epoch, epochs):
            effective_weights = curriculum.weights_for_epoch(base_weights, epoch_index, epochs)
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                physics_normalization,
                effective_weights,
                class_weights,
                float(training_config["gradient_clip"]),
                time_generator,
                physics_settings,
                contrastive_settings,
                contrastive_generator,
                adaptive_weighter,
                max_train_batches,
            )
            validation_metrics = validate_epoch(
                model,
                validation_loader,
                physics_normalization,
                effective_weights,
                time_grid=validation_time_grid,
                class_weights=class_weights,
                physics=physics_settings,
                contrastive=contrastive_settings,
                adaptive_weighter=adaptive_weighter,
                max_batches=max_validation_batches,
            )
            last_validation = validation_metrics.total
            generator_states = {
                "augmentation": augmentation_generator.get_state(),
                "sampler": sampler_generator.get_state(),
                "time": time_generator.get_state(),
                "contrastive": contrastive_generator.get_state(),
            }
            epoch_number = epoch_index + 1
            save_best_flow = False
            if not is_v003 and validation_metrics.total < best_flow:
                best_flow = validation_metrics.total
                save_best_flow = True
            fixed_contributions = weighted_objective_contributions(
                validation_metrics, final_weights
            )
            save_best_fixed = (
                selection_state.update(
                    "fixed_objective",
                    fixed_contributions["total"],
                    epoch_number,
                )
                if is_v003
                else False
            )
            sample_criterion = np.nan
            sample_rmse = [np.nan] * 3
            sample_miou = np.nan
            sample_class_iou = [np.nan] * 3
            sample_class_dice = [np.nan] * 3
            save_best_sample = False
            save_best_segmentation = False
            save_best_whole = False
            if epoch_number % int(training_config["validation_sample_every"]) == 0:
                sample_metrics = _validation_sample_metrics(
                    model,
                    validation_loader,
                    device,
                    steps=int(training_config["sample_steps_validation"]),
                    max_batches=(
                        max_validation_batches
                        or int(training_config["validation_sample_batches"])
                    ),
                    guidance_scale=guidance_scale,
                )
                sample_criterion = float(sample_metrics["criterion"])
                sample_rmse = list(sample_metrics["normalized_rmse"])
                sample_miou = float(sample_metrics["miou"])
                sample_class_iou = list(sample_metrics["class_iou"])
                sample_class_dice = list(sample_metrics["class_dice"])
                if is_v003:
                    save_best_sample = selection_state.update(
                        "sampling", sample_criterion, epoch_number
                    )
                    save_best_segmentation = selection_state.update(
                        "segmentation", sample_miou, epoch_number
                    )
                elif sample_criterion < best_sample:
                    best_sample = sample_criterion
                    save_best_sample = True
            whole_criterion = np.nan
            whole_rmse = [np.nan] * 3
            whole_miou = np.nan
            whole_metrics: dict[str, Any] | None = None
            if is_v003 and epoch_number % int(
                training_config["checkpointing"]["whole_validation_every_epochs"]
            ) == 0:
                whole_metrics = _whole_realization_validation_metrics(
                    model,
                    dataset_root=dataset_root,
                    realization_ids=whole_validation_ids,
                    normalization=normalization,
                    patch_shape=tuple(int(value) for value in config["patches"]["shape"]),
                    stride=tuple(int(value) for value in config["patches"]["stride"]),
                    steps=int(training_config["sample_steps_validation"]),
                    batch_size=int(training_config["batch_size"]),
                    device=device,
                    guidance_scale=guidance_scale,
                )
                whole_criterion = float(whole_metrics["criterion"])
                whole_rmse = list(whole_metrics["normalized_rmse"])
                whole_miou = float(whole_metrics["miou"])
                save_best_whole = selection_state.update(
                    "whole_realization", whole_criterion, epoch_number
                )
                write_json(
                    run_directory
                    / f"whole_validation_metrics_epoch_{epoch_number:04d}.json",
                    {
                        "epoch": epoch_number,
                        "selection_split": "validation",
                        "test_split_used_for_selection": False,
                        **whole_metrics,
                    },
                )
            row = {
                "epoch": epoch_number,
                **_metric_row("train", train_metrics),
                **_metric_row("validation", validation_metrics),
                "sample_criterion": sample_criterion,
                "sample_rmse_vp_normalized": sample_rmse[0],
                "sample_rmse_vs_normalized": sample_rmse[1],
                "sample_rmse_density_normalized": sample_rmse[2],
                "sample_miou": sample_miou,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "density_weight": effective_weights.density,
                "ssim_weight": effective_weights.ssim,
                "physics_weight": effective_weights.physics,
                "structure_weight": effective_weights.structure,
            }
            if is_v003:
                row.update(
                    {
                        "validation_fixed_objective": fixed_contributions["total"],
                        "validation_weighted_inversion_contribution": fixed_contributions[
                            "inversion"
                        ],
                        "validation_weighted_segmentation_contribution": fixed_contributions[
                            "segmentation"
                        ],
                        "validation_weighted_contrastive_contribution": fixed_contributions[
                            "contrastive"
                        ],
                        "validation_weighted_physics_contribution": fixed_contributions[
                            "physics"
                        ],
                        "validation_weighted_structure_contribution": fixed_contributions[
                            "structure"
                        ],
                        **{
                            f"sample_class_{index}_iou": sample_class_iou[index]
                            for index in range(3)
                        },
                        **{
                            f"sample_class_{index}_dice": sample_class_dice[index]
                            for index in range(3)
                        },
                        "whole_validation_criterion": whole_criterion,
                        "whole_validation_rmse_vp_normalized": whole_rmse[0],
                        "whole_validation_rmse_vs_normalized": whole_rmse[1],
                        "whole_validation_rmse_density_normalized": whole_rmse[2],
                        "whole_validation_miou": whole_miou,
                    }
                )
            writer.writerow(row)
            stream.flush()
            scheduler.step()
            if save_best_flow:
                save_checkpoint(
                    run_directory / "best_flow.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {"validation_objective": validation_metrics.total},
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            if save_best_fixed:
                save_checkpoint(
                    run_directory / "best_fixed_objective.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {
                        **checkpoint_metadata(
                            "fixed_objective",
                            fixed_contributions["total"],
                            epoch_number,
                        ),
                        "fixed_weight_contributions": fixed_contributions,
                    },
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            if save_best_sample:
                save_checkpoint(
                    run_directory / "best_sampling.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {
                        **(
                            checkpoint_metadata("sampling", sample_criterion, epoch_number)
                            if is_v003
                            else {}
                        ),
                        "sample_criterion": sample_criterion,
                        "normalized_rmse": sample_rmse,
                        "miou": sample_miou,
                        "class_iou": sample_class_iou,
                        "class_dice": sample_class_dice,
                    },
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            if save_best_segmentation:
                save_checkpoint(
                    run_directory / "best_segmentation.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {
                        **checkpoint_metadata(
                            "segmentation", sample_miou, epoch_number
                        ),
                        "miou": sample_miou,
                        "class_iou": sample_class_iou,
                        "class_dice": sample_class_dice,
                    },
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            if save_best_whole and whole_metrics is not None:
                save_checkpoint(
                    run_directory / "best_whole_realization.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {
                        **checkpoint_metadata(
                            "whole_realization", whole_criterion, epoch_number
                        ),
                        "normalized_rmse": whole_rmse,
                        "miou": whole_miou,
                        "realization_ids": whole_validation_ids,
                        "per_realization": whole_metrics["per_realization"],
                    },
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            periodic_interval = int(
                training_config.get("checkpointing", {}).get("periodic_interval_epochs", 0)
            )
            if is_v003 and periodic_interval > 0 and epoch_number % periodic_interval == 0:
                save_checkpoint(
                    run_directory / f"checkpoint_epoch_{epoch_number:04d}.pt",
                    model,
                    optimizer,
                    epoch_number,
                    {
                        "criterion_name": "periodic",
                        "criterion_formula": "fixed interval archival checkpoint",
                        "validation_fixed_objective": fixed_contributions["total"],
                    },
                    config,
                    scheduler=scheduler,
                    adaptive_weighter=adaptive_weighter,
                    generator_states=generator_states,
                )
            save_checkpoint(
                run_directory / "last.pt",
                model,
                optimizer,
                epoch_number,
                {
                    "validation_objective": validation_metrics.total,
                    "validation_fixed_objective": fixed_contributions["total"],
                    **(
                        {"checkpoint_selection_state": selection_state.to_dict()}
                        if is_v003
                        else {}
                    ),
                },
                config,
                scheduler=scheduler,
                adaptive_weighter=adaptive_weighter,
                generator_states=generator_states,
            )
            if is_v003:
                manifest["checkpoint_selection_state"] = selection_state.to_dict()
                manifest["last_completed_epoch"] = epoch_number
                write_json(manifest_path, manifest)

    generator_states = {
        "augmentation": augmentation_generator.get_state(),
        "sampler": sampler_generator.get_state(),
        "time": time_generator.get_state(),
        "contrastive": contrastive_generator.get_state(),
    }
    save_checkpoint(
        run_directory / "last.pt",
        model,
        optimizer,
        epochs,
        {
            "validation_objective": float(last_validation),
            **(
                {
                    "checkpoint_selection_state": selection_state.to_dict(),
                    "criterion_name": "last",
                    "criterion_formula": "last completed optimization epoch",
                }
                if is_v003
                else {}
            ),
        },
        config,
        scheduler=scheduler,
        adaptive_weighter=adaptive_weighter,
        generator_states=generator_states,
    )
    manifest["checkpoint"] = (
        "best_whole_realization.pt"
        if is_v003 and selection_state.best_epochs["whole_realization"] is not None
        else "best_sampling.pt"
    )
    manifest["status"] = "complete"
    if is_v003:
        manifest["best_fixed_objective"] = selection_state.best_values["fixed_objective"]
        manifest["best_sample_criterion"] = selection_state.best_values["sampling"]
        manifest["best_segmentation_miou"] = selection_state.best_values[
            "segmentation"
        ]
        manifest["best_whole_realization_criterion"] = selection_state.best_values[
            "whole_realization"
        ]
    else:
        manifest["best_validation_objective"] = best_flow
        manifest["best_sample_criterion"] = best_sample
    if is_v003:
        manifest["checkpoint_selection_state"] = selection_state.to_dict()
    write_json(manifest_path, manifest)
    return run_directory
