"""Stage-02 field-conditioned geology and exact synthetic AVO production."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sage_avo.forward import (
    AngleBand,
    ForwardConfig,
    apply_observation_perturbations,
    forward_avo_dense,
    forward_avo_dense_spec,
    forward_avo_madagascar,
    forward_specification_from_mapping,
    observation_config_from_mapping,
)
from sage_avo.geology import load_calibrated_dry_frame, make_field_conditioned_realization
from sage_avo.structure import estimate_pwd_dip, monotonicity_report

from .manifest import file_sha256, write_json


STAGE01_CHANNELS = {
    "sand_probability": ("usable", "sandprob.npy"),
    "porosity": ("usable", "poro.npy"),
    "elastic_background": ("usable", "elastic_background.npy"),
    "elastic_blend_weight": ("usable", "elastic_blend_weight.npy"),
    "reservoir_mask": ("usable", "reservoir_mask.npy"),
    "time_ms": ("usable", "reg_t.npy"),
    "cdp": ("usable", "good_cdps.npy"),
    "rgt": ("attributes", "rgt_tau.npy"),
    "strat_fraction": ("attributes", "strat_fraction_t6_t7.npy"),
    "horizon_top_ms": ("attributes", "horizons/t6_model_seismic_conformed_ms.npy"),
    "horizon_base_ms": ("attributes", "horizons/t7_model_seismic_conformed_ms.npy"),
}


def _stage01_directories(work_data_root: str | Path, dataset_id: str, version: str) -> dict[str, Path]:
    root = Path(work_data_root) / dataset_id
    return {
        "usable": root / "usable" / version,
        "attributes": root / "attributes" / version,
        "derived": root / "derived",
        "synthetic": root / "synthetic",
    }


def load_stage01_background(
    work_data_root: str | Path,
    dataset_id: str = "s01data",
    version: str = "v001",
) -> tuple[dict[str, np.ndarray], Any, dict[str, str]]:
    """Load the canonical Stage-01 artifact contract and reservoir RF model."""
    try:
        import joblib
    except ImportError as error:
        raise ImportError(
            "Loading Stage-01 field-conditioned models requires the 'field' dependencies; "
            "install sage-avo with `pip install -e '.[field]'`."
        ) from error

    directories = _stage01_directories(work_data_root, dataset_id, version)
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    missing: list[Path] = []
    for name, (kind, filename) in STAGE01_CHANNELS.items():
        path = directories[kind] / filename
        if name == "rgt":
            model_coordinate = directories[kind] / "rgt_model.npy"
            if model_coordinate.exists():
                path = model_coordinate
        if not path.exists():
            missing.append(path)
            continue
        arrays[name] = np.load(path, allow_pickle=False)
        hashes[f"{kind}/{version}/{path.name}"] = file_sha256(path)
    model_path = directories["derived"] / f"rf_models_{version}" / "elastic_model_set_reservoir.joblib"
    if not model_path.exists():
        missing.append(model_path)
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Stage 02 requires the complete Stage-01 local artifact contract:\n" + detail)
    reservoir_model = joblib.load(model_path)
    hashes[f"derived/rf_models_{version}/{model_path.name}"] = file_sha256(model_path)
    shape = arrays["sand_probability"].shape
    for name in ("porosity", "rgt", "strat_fraction", "reservoir_mask"):
        if arrays[name].shape != shape:
            raise ValueError(f"Stage-01 channel {name!r} has shape {arrays[name].shape}; expected {shape}")
    for name in ("elastic_background", "elastic_blend_weight"):
        if arrays[name].shape != (3, *shape):
            raise ValueError(f"Stage-01 channel {name!r} must have shape [3, time, trace]")
    for name in ("horizon_top_ms", "horizon_base_ms"):
        if arrays[name].shape != (shape[1],):
            raise ValueError(f"Stage-01 channel {name!r} must have one value per trace")
    return arrays, reservoir_model, hashes


def forward_config_from_mapping(config: dict[str, Any]) -> ForwardConfig:
    """Build the production forward definition from a versioned mapping."""
    spec = config["forward"]
    angle_spec = spec["angles_degrees"]
    angles = tuple(
        float(value)
        for value in range(
            int(angle_spec["start"]), int(angle_spec["stop"]) + 1, int(angle_spec["step"])
        )
    )
    bands = tuple(
        AngleBand(name, float(limits[0]), float(limits[1]))
        for name, limits in spec["production_bands"].items()
    )
    return ForwardConfig(
        angles_degrees=angles,
        bands=bands,
        wavelet_hz=float(spec["wavelet"]["frequency_hz"]),
        dt_seconds=float(spec["dt_seconds"]),
        wavelet_samples=int(spec["wavelet"]["samples"]),
        apply_mute=bool(spec["front_mute"]["enabled"]),
        mute_start=tuple(float(value) for value in spec["front_mute"]["start"]),
        mute_end=tuple(float(value) for value in spec["front_mute"]["end"]),
        taper_samples=int(spec["front_mute"]["taper_samples"]),
    )


def _realization_qc(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    elastic = payload["elastic"]
    numeric = [
        value
        for value in payload.values()
        if np.asarray(value).dtype.kind in {"b", "i", "u", "f", "c"}
    ]
    return {
        "all_required_channels_finite": bool(
            all(np.isfinite(value).all() for value in numeric)
        ),
        "vp_min_max_m_s": [float(np.min(elastic[0])), float(np.max(elastic[0]))],
        "vs_min_max_m_s": [float(np.min(elastic[1])), float(np.max(elastic[1]))],
        "density_min_max_g_cc": [float(np.min(elastic[2])), float(np.max(elastic[2]))],
        "sand_probability_min_max": [
            float(np.min(payload["sand_probability"])),
            float(np.max(payload["sand_probability"])),
        ],
        "porosity_min_max": [float(np.min(payload["porosity"])), float(np.max(payload["porosity"]))],
        "plume_pixel_count": int(payload["plume_mask"].sum()),
        "rgt_monotonicity": monotonicity_report(payload["rgt"]),
    }


def _deformed_horizon_ms(
    source_horizon_ms: np.ndarray,
    time_ms: np.ndarray,
    vertical_displacement: np.ndarray,
    horizontal_displacement: np.ndarray,
) -> np.ndarray:
    """Map a source horizon through the same inverse deformation as the image channels."""
    times = np.asarray(time_ms, dtype=float)
    height, width = vertical_displacement.shape
    rows = np.arange(height, dtype=float)
    columns = np.arange(width, dtype=float)
    source_rows = np.interp(np.asarray(source_horizon_ms, dtype=float), times, rows)
    output = np.empty(width, dtype=np.float32)
    for column in range(width):
        source_columns = column - horizontal_displacement[:, column]
        horizon_rows = np.interp(source_columns, columns, source_rows)
        mapped_source_rows = rows - vertical_displacement[:, column]
        destination_row = int(np.argmin(np.abs(mapped_source_rows - horizon_rows)))
        output[column] = np.float32(times[destination_row])
    return output


def _mask_horizons_ms(
    mask: np.ndarray,
    time_ms: np.ndarray,
    fallback_top_ms: np.ndarray,
    fallback_base_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover visible mask limits, retaining finite deformed surfaces when the interval exits the grid."""
    interval = np.asarray(mask, dtype=bool)
    times = np.asarray(time_ms, dtype=float)
    top = np.asarray(fallback_top_ms, dtype=np.float32).copy()
    base = np.asarray(fallback_base_ms, dtype=np.float32).copy()
    absent = 0
    for column in range(interval.shape[1]):
        samples = np.flatnonzero(interval[:, column])
        if samples.size:
            top[column] = times[samples[0]]
            base[column] = times[samples[-1]]
        else:
            absent += 1
    return top, base, absent


def generate_stage02_realization(
    *,
    realization_id: int,
    geology_realization_id: int | None = None,
    observation_variant_id: int = 0,
    arrays: dict[str, np.ndarray],
    reservoir_model: Any,
    fluid_calibration: Any | None = None,
    config: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Generate and save one complete, traceable Stage-02 realization package."""
    geology_seed = (
        int(realization_id)
        if geology_realization_id is None
        else int(geology_realization_id)
    )
    depth_m = None
    if fluid_calibration is not None:
        mapping = fluid_calibration.metadata["time_depth_linear_coefficients"]
        depth_by_row = (
            float(mapping["slope_m_per_ms"]) * np.asarray(arrays["time_ms"], dtype=float)
            + float(mapping["intercept_m"])
        )
        depth_m = np.broadcast_to(depth_by_row[:, None], arrays["porosity"].shape)
    geology = make_field_conditioned_realization(
        sand_probability_base=arrays["sand_probability"],
        porosity_base=arrays["porosity"],
        rgt_base=arrays["rgt"],
        strat_fraction_base=arrays["strat_fraction"],
        reservoir_mask_base=arrays["reservoir_mask"],
        elastic_background_base=arrays["elastic_background"],
        elastic_blend_weight_base=arrays["elastic_blend_weight"],
        reservoir_model=reservoir_model,
        seed=geology_seed,
        geology_config=config["geology"],
        fluid_config=config["fluid_substitution"],
        fluid_calibration=fluid_calibration,
        depth_m=depth_m,
    )
    is_v003 = "forward_model" in config
    forward_mapping = config["forward_model"] if is_v003 else config["forward"]
    backend = str(forward_mapping["local_backend"])
    specification = forward_specification_from_mapping(config) if is_v003 else None
    if backend == "numpy" and specification is not None:
        forward = forward_avo_dense_spec(*geology.elastic, specification)
    elif backend == "numpy":
        forward_config = forward_config_from_mapping(config)
        forward = forward_avo_dense(*geology.elastic, config=forward_config)
    elif backend == "madagascar" and specification is None:
        forward_config = forward_config_from_mapping(config)
        forward = forward_avo_madagascar(
            *geology.elastic,
            config=forward_config,
            time_origin_seconds=float(arrays["time_ms"][0]) / 1000.0,
            trace_origin=float(arrays["cdp"][0]),
        )
    elif backend == "madagascar":
        raise ValueError(
            "The v003 wavelet-bank contract requires the NumPy backend; "
            "Madagascar remains a separately validated production alternative"
        )
    else:
        raise ValueError("local_backend must be 'numpy' or 'madagascar'")
    clean_stacks = forward.stacks.astype(np.float32)
    if is_v003:
        observation_mapping = config["observation_perturbations"]
        noise_seed = int(realization_id) + int(observation_mapping["seed_offset"])
        observation = apply_observation_perturbations(
            clean_stacks,
            np.random.default_rng(noise_seed),
            observation_config_from_mapping(observation_mapping),
        )
        observed_stacks = observation.stacks
        band_standard_deviation = np.std(clean_stacks, axis=(1, 2), keepdims=True)
        noise_metadata = observation.metadata
    else:
        noise_config = config["forward"]["noise"]
        noise_fraction = float(noise_config["standard_deviation_fraction"])
        if bool(noise_config["enabled"]):
            noise_seed = int(realization_id) + int(noise_config["seed_offset"])
            noise_rng = np.random.default_rng(noise_seed)
            band_standard_deviation = np.std(clean_stacks, axis=(1, 2), keepdims=True)
            noise = noise_rng.standard_normal(clean_stacks.shape) * (
                noise_fraction * band_standard_deviation
            )
            observed_stacks = (clean_stacks + noise).astype(np.float32)
        else:
            noise_seed = None
            band_standard_deviation = np.std(clean_stacks, axis=(1, 2), keepdims=True)
            observed_stacks = clean_stacks
        noise_metadata = {
            "enabled": bool(noise_config["enabled"]),
            "standard_deviation_fraction": noise_fraction,
            "reference": str(noise_config["reference"]),
            "seed": noise_seed,
            "clean_band_standard_deviation": band_standard_deviation.reshape(-1).tolist(),
        }
    if is_v003:
        noise_metadata["seed"] = noise_seed
        noise_metadata["clean_band_standard_deviation"] = (
            band_standard_deviation.reshape(-1).tolist()
        )
    structural_stack = np.mean(observed_stacks[:2], axis=0)
    structural_stack = (structural_stack - structural_stack.mean()) / max(structural_stack.std(), 1e-8)
    if bool(config["structure_qc"]["recalculate_pwd_dip"]):
        pwd = estimate_pwd_dip(structural_stack)
        dip_pwd = pwd.dip
        structure_oriented = pwd.structure_oriented_seismic
    else:
        dip_pwd = np.zeros_like(structural_stack)
        structure_oriented = structural_stack
    fallback_top_ms = _deformed_horizon_ms(
        arrays["horizon_top_ms"],
        arrays["time_ms"],
        geology.vertical_displacement,
        geology.horizontal_displacement,
    )
    fallback_base_ms = _deformed_horizon_ms(
        arrays["horizon_base_ms"],
        arrays["time_ms"],
        geology.vertical_displacement,
        geology.horizontal_displacement,
    )
    horizon_top_ms, horizon_base_ms, absent_reservoir_traces = _mask_horizons_ms(
        geology.reservoir_mask,
        arrays["time_ms"],
        fallback_top_ms,
        fallback_base_ms,
    )

    payload = {
        "avo_dense": forward.seismic,
        "avo": observed_stacks,
        "avo_clean": clean_stacks,
        "elastic": geology.elastic,
        "elastic_brine": geology.elastic_brine,
        "delta": geology.delta,
        "sand_probability": geology.sand_probability,
        "porosity": geology.porosity,
        "rgt": geology.rgt,
        "dip_pwd": dip_pwd,
        "structural_stack": structural_stack.astype(np.float32),
        "structure_oriented": structure_oriented,
        "strat_fraction": geology.strat_fraction,
        "reservoir_mask": geology.reservoir_mask,
        "horizon_top_ms": horizon_top_ms,
        "horizon_base_ms": horizon_base_ms,
        "source_horizon_top_ms": arrays["horizon_top_ms"].astype(np.float32),
        "source_horizon_base_ms": arrays["horizon_base_ms"].astype(np.float32),
        "segmentation": geology.segmentation,
        "plume_mask": geology.plume_mask,
        "co2_saturation": geology.co2_saturation,
        "valid_mask": np.isfinite(geology.elastic).all(axis=0).astype(np.uint8),
        "angles_degrees": forward.angles_degrees,
        "time_ms": arrays["time_ms"].astype(np.float32),
        "cdp": arrays["cdp"].astype(np.int32),
    }
    if specification is not None:
        payload["forward_specification_sha256"] = np.asarray(specification.sha256)
        payload["wavelet_id_by_angle"] = np.asarray(
            [
                specification.wavelet_for_angle(float(angle)).wavelet_id
                for angle in specification.angles_degrees
            ]
        )
    qc = _realization_qc(payload)
    if not qc["all_required_channels_finite"]:
        nonfinite = [
            name
            for name, value in payload.items()
            if np.asarray(value).dtype.kind in {"b", "i", "u", "f", "c"}
            and not np.isfinite(value).all()
        ]
        raise ValueError(
            f"Realization {realization_id} contains non-finite required channels: {nonfinite}"
        )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"realization_{realization_id:07d}.npz"
    np.savez_compressed(
        path,
        realization_id=np.int64(realization_id),
        geology_realization_id=np.int64(geology_seed),
        observation_variant_id=np.int64(observation_variant_id),
        **payload,
    )
    metadata = {
        "generator_contract_version": 3 if is_v003 else 2,
        "realization_id": realization_id,
        "seed": geology_seed,
        "geology_realization_id": geology_seed,
        "observation_variant_id": int(observation_variant_id),
        "split_group_id": geology_seed,
        "generation_config_sha256": _configuration_sha256(config),
        "file": path.name,
        "forward_backend": backend,
        "forward_operator": forward_mapping["primary_operator"],
        "forward_specification": specification.to_dict() if specification is not None else None,
        "forward_specification_sha256": specification.sha256 if specification is not None else None,
        "observation_perturbations": noise_metadata,
        "geology": geology.metadata,
        "horizon_coordinate_method": {
            "visible_interval": "first_and_last_warped_reservoir_mask_sample",
            "interval_outside_time_window": "source_horizon_mapped_by_coherent_deformation_and_clipped_to_grid",
            "reservoir_absent_trace_count": absent_reservoir_traces,
        },
        "qc": qc,
    }
    metadata["archive_sha256"] = file_sha256(path)
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def _configuration_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _generate_realization_worker(arguments: dict[str, Any]) -> dict[str, Any]:
    """Pickle-safe worker for deterministic independent realization generation."""
    return generate_stage02_realization(**arguments)


def _existing_record(
    destination: Path,
    realization_id: int,
    configuration_sha256: str,
) -> dict[str, Any] | None:
    archive_path = destination / f"realization_{realization_id:07d}.npz"
    metadata_path = archive_path.with_suffix(".json")
    if not archive_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("realization_id", -1)) != realization_id:
        raise ValueError(f"Existing metadata has the wrong ID: {metadata_path}")
    if metadata.get("generation_config_sha256") != configuration_sha256:
        raise ValueError(f"Existing realization uses a different generation config: {archive_path}")
    if metadata.get("archive_sha256") != file_sha256(archive_path):
        raise ValueError(f"Existing realization hash mismatch: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        if int(archive["realization_id"]) != realization_id:
            raise ValueError(f"Existing archive has the wrong ID: {archive_path}")
        required = (
            "avo_dense",
            "avo",
            "avo_clean",
            "elastic",
            "elastic_brine",
            "delta",
            "sand_probability",
            "porosity",
            "rgt",
            "dip_pwd",
            "structural_stack",
            "structure_oriented",
            "strat_fraction",
            "reservoir_mask",
            "horizon_top_ms",
            "horizon_base_ms",
            "segmentation",
            "plume_mask",
            "co2_saturation",
            "valid_mask",
            "angles_degrees",
            "time_ms",
            "cdp",
        )
        if any(name not in archive.files for name in required):
            raise ValueError(f"Existing realization is incomplete: {archive_path}")
        if not all(np.isfinite(archive[name]).all() for name in required):
            raise ValueError(f"Existing realization is non-finite: {archive_path}")
        metadata_updated = False
        if "generator_contract_version" not in metadata:
            metadata["generator_contract_version"] = 2
            metadata_updated = True
        if "horizon_coordinate_method" not in metadata:
            reservoir_mask = np.asarray(archive["reservoir_mask"], dtype=bool)
            metadata["horizon_coordinate_method"] = {
                "visible_interval": "first_and_last_warped_reservoir_mask_sample",
                "interval_outside_time_window": (
                    "source_horizon_mapped_by_coherent_deformation_and_clipped_to_grid"
                ),
                "reservoir_absent_trace_count": int(
                    np.count_nonzero(reservoir_mask.sum(axis=0) == 0)
                ),
            }
            metadata_updated = True
    if metadata_updated:
        write_json(metadata_path, metadata)
    return metadata


def _write_generation_manifest(
    destination: Path,
    *,
    config: dict[str, Any],
    source_hashes: dict[str, str],
    requested: int,
    records: list[dict[str, Any]],
    status: str,
    workers: int,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["realization_id"]))
    manifest: dict[str, Any] = {
        "schema_version": 3 if "forward_model" in config else 2,
        "stage": "02_synthetic_avo_generation",
        "status": status,
        "master_seed": int(config["stage"]["seed"]),
        "member_seed_rule": (
            "geology_seed = geology_realization_id; observation_seed = "
            "realization_id + configured seed_offset"
            if "forward_model" in config
            else "realization_seed = realization_id"
        ),
        "generation_config_sha256": _configuration_sha256(config),
        "requested_realizations": requested,
        "generated_realizations": len(ordered),
        "output_version": str(config["outputs"]["version"]),
        "realization_ids": [record["realization_id"] for record in ordered],
        "split_group_ids": [
            int(record.get("split_group_id", record["realization_id"]))
            for record in ordered
        ],
        "source_artifact_hashes": source_hashes,
        "delta_convention": "DELTA is shaliness; P(sand) = 1 - DELTA",
        "exact_forward_operator": (
            config["forward_model"]["primary_operator"]
            if "forward_model" in config
            else config["forward"]["primary_operator"]
        ),
        "madagascar_production_alternative": (
            config["forward_model"]["production_alternative"]
            if "forward_model" in config
            else config["forward"]["production_alternative"]
        ),
        "production_bands": deepcopy(
            config["forward_model"]["bands"]
            if "forward_model" in config
            else config["forward"]["production_bands"]
        ),
        "observation_perturbations": deepcopy(
            config["observation_perturbations"]
            if "forward_model" in config
            else config["forward"]["noise"]
        ),
        "parallel_workers": workers,
        "source_snapshot": deepcopy(config.get("source_snapshot")),
        "records": ordered,
    }
    if ordered:
        first_path = destination / ordered[0]["file"]
        with np.load(first_path, allow_pickle=False) as archive:
            manifest["channels"] = {
                name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
                for name in archive.files
                if name
                not in {
                    "realization_id",
                    "geology_realization_id",
                    "observation_variant_id",
                }
            }
    write_json(destination / "manifest.json", manifest)
    return manifest


def generate_stage02_dataset(
    *,
    config: dict[str, Any],
    paths: dict[str, Any],
    output_directory: str | Path | None = None,
    realization_limit: int | None = None,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate the configured Stage-02 family from required Stage-01 artifacts."""
    stage = config["stage"]
    inputs = config["inputs"]
    arrays, reservoir_model, source_hashes = load_stage01_background(
        paths["work_data_root"], inputs["dataset_id"], inputs["structure_version"]
    )
    fluid_calibration = None
    fluid_mode = str(config["fluid_substitution"].get("mode", ""))
    calibrated_modes = {
        "calibrated_differential_gassmann",
        "constrained_local_gassmann",
    }
    if fluid_mode in calibrated_modes:
        relative = Path(str(config["fluid_substitution"]["calibration_artifact"]))
        calibration_path = (
            Path(paths["work_data_root"]) / str(inputs["dataset_id"]) / relative
        )
        if not calibration_path.exists() or not calibration_path.with_suffix(".json").exists():
            raise FileNotFoundError(
                f"Calibrated fluid mode requires {calibration_path} and its JSON sidecar"
            )
        fluid_calibration = load_calibrated_dry_frame(calibration_path)
        expected_id = str(config["fluid_substitution"]["calibration_id"])
        if fluid_calibration.calibration_id != expected_id:
            raise ValueError(
                f"Fluid calibration ID {fluid_calibration.calibration_id!r} does not match "
                f"configured ID {expected_id!r}"
            )
        validation_relative = Path(
            str(config["fluid_substitution"]["fluid_property_validation_artifact"])
        )
        validation_path = (
            Path(paths["work_data_root"]) / str(inputs["dataset_id"]) / validation_relative
        )
        if not validation_path.exists():
            raise FileNotFoundError(
                "Calibrated fluid production requires a validated pressure/"
                f"temperature/fluid-property validation artifact at {validation_path}"
            )
        fluid_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if fluid_validation.get("status") != "passed":
            raise ValueError("Fluid-property validation artifact has not passed")
        if fluid_validation.get("calibration_id") != fluid_calibration.calibration_id:
            raise ValueError("Fluid-property validation calibration ID does not match")
        if fluid_validation.get("pressure_temperature_source") in {None, "generic", "unavailable"}:
            raise ValueError("Fluid-property validation lacks a measured or independently validated P/T source")
        source_hashes[f"fluid_calibration/{calibration_path.name}"] = file_sha256(
            calibration_path
        )
        source_hashes[f"fluid_calibration/{calibration_path.with_suffix('.json').name}"] = (
            file_sha256(calibration_path.with_suffix(".json"))
        )
        source_hashes[f"fluid_calibration/{validation_path.name}"] = file_sha256(
            validation_path
        )
    requested = int(stage["realization_count"])
    count = requested if realization_limit is None else min(int(realization_limit), requested)
    if count < 1:
        raise ValueError("realization_limit must be positive when supplied")
    destination = Path(output_directory) if output_directory else (
        Path(paths["work_data_root"]) / inputs["dataset_id"] / config["outputs"]["directory"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive")
    offset = int(stage["realization_id_offset"])
    realization_ids = [offset + index for index in range(count)]
    observation_variants = int(stage.get("observation_variants_per_geology", 1))
    geology_count = int(stage.get("geology_realization_count", requested))
    if observation_variants < 1 or geology_count < 1:
        raise ValueError("Geology and observation-variant counts must be positive")
    if "forward_model" in config and requested != geology_count * observation_variants:
        raise ValueError(
            "stage.realization_count must equal geology_realization_count times "
            "observation_variants_per_geology"
        )
    configuration_sha256 = _configuration_sha256(config)
    records: list[dict[str, Any]] = []
    pending: list[int] = []
    for realization_id in realization_ids:
        existing = (
            _existing_record(destination, realization_id, configuration_sha256)
            if resume
            else None
        )
        if existing is None:
            pending.append(realization_id)
        else:
            records.append(existing)
    _write_generation_manifest(
        destination,
        config=config,
        source_hashes=source_hashes,
        requested=requested,
        records=records,
        status="generation_in_progress",
        workers=workers,
    )
    def arguments_for(realization_id: int) -> dict[str, Any]:
        member_index = realization_id - offset
        return {
            "realization_id": realization_id,
            "geology_realization_id": offset + member_index // observation_variants,
            "observation_variant_id": member_index % observation_variants,
            "arrays": arrays,
            "reservoir_model": reservoir_model,
            "fluid_calibration": fluid_calibration,
            "config": config,
            "output_directory": destination,
        }
    if workers == 1:
        iterator = map(
            _generate_realization_worker,
            (arguments_for(realization_id) for realization_id in pending),
        )
        for record in iterator:
            records.append(record)
            _write_generation_manifest(
                destination,
                config=config,
                source_hashes=source_hashes,
                requested=requested,
                records=records,
                status="generation_in_progress",
                workers=workers,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_generate_realization_worker, arguments_for(realization_id))
                for realization_id in pending[:workers]
            }
            # Keep only one task per worker in flight. The Stage-01 arrays and fitted
            # model are large; eagerly queueing every realization can backpressure
            # Python 3.9 multiprocessing pipes and delay progress-manifest updates.
            remaining = iter(pending[workers:])
            while futures:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    records.append(future.result())
                    _write_generation_manifest(
                        destination,
                        config=config,
                        source_hashes=source_hashes,
                        requested=requested,
                        records=records,
                        status="generation_in_progress",
                        workers=workers,
                    )
                    try:
                        realization_id = next(remaining)
                    except StopIteration:
                        continue
                    futures.add(
                        executor.submit(
                            _generate_realization_worker,
                            arguments_for(realization_id),
                        )
                    )
    status = "complete" if count == requested and len(records) == requested else "operator_validation_subset"
    return _write_generation_manifest(
        destination,
        config=config,
        source_hashes=source_hashes,
        requested=requested,
        records=records,
        status=status,
        workers=workers,
    )


def load_stage02_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
