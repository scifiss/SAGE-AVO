"""Stage-02 field-conditioned geology and exact synthetic AVO production."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sage_avo.forward import AngleBand, ForwardConfig, forward_avo_dense, forward_avo_madagascar
from sage_avo.geology import make_field_conditioned_realization
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
    directories = _stage01_directories(work_data_root, dataset_id, version)
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    missing: list[Path] = []
    for name, (kind, filename) in STAGE01_CHANNELS.items():
        path = directories[kind] / filename
        if not path.exists():
            missing.append(path)
            continue
        arrays[name] = np.load(path, allow_pickle=False)
        hashes[f"{kind}/{version}/{filename}"] = file_sha256(path)
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
        for name, limits in spec["current_bands"].items()
    )
    return ForwardConfig(
        angles_degrees=angles,
        bands=bands,
        wavelet_hz=float(spec["wavelet"]["frequency_hz"]),
        dt_seconds=float(spec["dt_seconds"]),
        wavelet_samples=int(spec["wavelet"]["samples"]),
        apply_mute=bool(spec["front_mute"]["enabled"]),
    )


def _realization_qc(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    elastic = payload["elastic"]
    return {
        "all_required_channels_finite": bool(
            all(np.isfinite(value).all() for value in payload.values())
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


def _mask_horizons_ms(mask: np.ndarray, time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover top/base coordinates from a coherently warped interval mask."""
    interval = np.asarray(mask, dtype=bool)
    times = np.asarray(time_ms, dtype=float)
    top = np.full(interval.shape[1], np.nan, dtype=np.float32)
    base = np.full(interval.shape[1], np.nan, dtype=np.float32)
    for column in range(interval.shape[1]):
        samples = np.flatnonzero(interval[:, column])
        if samples.size:
            top[column] = times[samples[0]]
            base[column] = times[samples[-1]]
    return top, base


def generate_stage02_realization(
    *,
    realization_id: int,
    arrays: dict[str, np.ndarray],
    reservoir_model: Any,
    config: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Generate and save one complete, traceable Stage-02 realization package."""
    geology = make_field_conditioned_realization(
        sand_probability_base=arrays["sand_probability"],
        porosity_base=arrays["porosity"],
        rgt_base=arrays["rgt"],
        strat_fraction_base=arrays["strat_fraction"],
        reservoir_mask_base=arrays["reservoir_mask"],
        elastic_background_base=arrays["elastic_background"],
        elastic_blend_weight_base=arrays["elastic_blend_weight"],
        reservoir_model=reservoir_model,
        seed=int(realization_id),
        geology_config=config["geology"],
        fluid_config=config["fluid_substitution"],
    )
    forward_config = forward_config_from_mapping(config)
    backend = str(config["forward"]["local_backend"])
    if backend == "numpy":
        forward = forward_avo_dense(*geology.elastic, config=forward_config)
    elif backend == "madagascar":
        forward = forward_avo_madagascar(
            *geology.elastic,
            config=forward_config,
            time_origin_seconds=float(arrays["time_ms"][0]) / 1000.0,
            trace_origin=float(arrays["cdp"][0]),
        )
    else:
        raise ValueError("forward.local_backend must be 'numpy' or 'madagascar'")
    structural_stack = np.mean(forward.stacks[:2], axis=0)
    structural_stack = (structural_stack - structural_stack.mean()) / max(structural_stack.std(), 1e-8)
    if bool(config["structure_qc"]["recalculate_pwd_dip"]):
        pwd = estimate_pwd_dip(structural_stack)
        dip_pwd = pwd.dip
        structure_oriented = pwd.structure_oriented_seismic
    else:
        dip_pwd = np.zeros_like(structural_stack)
        structure_oriented = structural_stack
    horizon_top_ms, horizon_base_ms = _mask_horizons_ms(
        geology.reservoir_mask, arrays["time_ms"]
    )

    payload = {
        "avo_dense": forward.seismic,
        "avo": forward.stacks,
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
    qc = _realization_qc(payload)
    if not qc["all_required_channels_finite"]:
        raise ValueError(f"Realization {realization_id} contains non-finite required channels")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"realization_{realization_id:07d}.npz"
    np.savez_compressed(path, realization_id=np.int64(realization_id), **payload)
    metadata = {
        "realization_id": realization_id,
        "seed": int(realization_id),
        "file": path.name,
        "forward_backend": backend,
        "forward_operator": config["forward"]["primary_operator"],
        "geology": geology.metadata,
        "qc": qc,
    }
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def generate_stage02_dataset(
    *,
    config: dict[str, Any],
    paths: dict[str, Any],
    output_directory: str | Path | None = None,
    realization_limit: int | None = None,
) -> dict[str, Any]:
    """Generate the configured Stage-02 realization family without toy fallbacks."""
    stage = config["stage"]
    inputs = config["inputs"]
    arrays, reservoir_model, source_hashes = load_stage01_background(
        paths["work_data_root"], inputs["dataset_id"], inputs["structure_version"]
    )
    requested = int(stage["realization_count"])
    count = requested if realization_limit is None else min(int(realization_limit), requested)
    if count < 1:
        raise ValueError("realization_limit must be positive when supplied")
    destination = Path(output_directory) if output_directory else (
        Path(paths["work_data_root"]) / inputs["dataset_id"] / config["outputs"]["directory"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    offset = int(stage["realization_id_offset"])
    records = [
        generate_stage02_realization(
            realization_id=offset + index,
            arrays=arrays,
            reservoir_model=reservoir_model,
            config=config,
            output_directory=destination,
        )
        for index in range(count)
    ]
    first_path = destination / records[0]["file"]
    with np.load(first_path, allow_pickle=False) as archive:
        channels = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
            if name != "realization_id"
        }
    manifest = {
        "schema_version": 1,
        "stage": "02_synthetic_avo_generation",
        "status": "complete" if count == requested else "operator_validation_subset",
        "requested_realizations": requested,
        "generated_realizations": count,
        "realization_ids": [record["realization_id"] for record in records],
        "source_artifact_hashes": source_hashes,
        "delta_convention": "DELTA is shaliness; P(sand) = 1 - DELTA",
        "exact_forward_operator": config["forward"]["primary_operator"],
        "madagascar_production_alternative": config["forward"]["production_alternative"],
        "legacy_bands": deepcopy(config["forward"]["legacy_bands"]),
        "current_bands": deepcopy(config["forward"]["current_bands"]),
        "channels": channels,
        "records": records,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_stage02_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
