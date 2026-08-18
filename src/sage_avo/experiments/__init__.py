"""Controlled experiment preparation, manifests, and orchestration."""

from importlib import import_module

from .manifest import build_run_manifest, write_json

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


_LAZY_EXPORTS = {
    "build_stage03_dataset": (".ml_dataset", "build_stage03_dataset"),
    "validate_dataset_integrity": (".ml_dataset", "validate_dataset_integrity"),
    "forward_config_from_mapping": (".synthetic_generation", "forward_config_from_mapping"),
    "generate_stage02_dataset": (".synthetic_generation", "generate_stage02_dataset"),
    "generate_stage02_realization": (".synthetic_generation", "generate_stage02_realization"),
    "load_stage01_background": (".synthetic_generation", "load_stage01_background"),
    "load_stage02_manifest": (".synthetic_generation", "load_stage02_manifest"),
}


def __getattr__(name: str):
    """Preserve public re-exports without importing optional stage stacks eagerly."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    return getattr(import_module(module_name, __name__), attribute)
