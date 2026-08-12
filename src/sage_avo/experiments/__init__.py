"""Controlled experiment preparation, manifests, and orchestration."""

from .manifest import build_run_manifest, write_json
from .synthetic_generation import (
    forward_config_from_mapping,
    generate_stage02_dataset,
    generate_stage02_realization,
    load_stage01_background,
    load_stage02_manifest,
)
from .ml_dataset import build_stage03_dataset, validate_dataset_integrity

__all__ = [
    "build_run_manifest",
    "build_stage03_dataset",
    "forward_config_from_mapping",
    "generate_stage02_dataset",
    "generate_stage02_realization",
    "load_stage01_background",
    "load_stage02_manifest",
    "validate_dataset_integrity",
    "write_json",
]
