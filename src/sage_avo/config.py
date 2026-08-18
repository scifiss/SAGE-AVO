"""Configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve an explicit shared forward-model configuration."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    shared_forward = config.get("forward_model_config")
    if shared_forward is not None:
        forward_path = source.parent / str(shared_forward)
        with forward_path.open("r", encoding="utf-8") as stream:
            forward_config = yaml.safe_load(stream)
        if not isinstance(forward_config, dict) or not isinstance(
            forward_config.get("forward_model"), dict
        ):
            raise ValueError(
                f"Shared forward configuration must define forward_model: {forward_path}"
            )
        if "forward_model" in config:
            raise ValueError(
                "Configuration may not define both forward_model_config and forward_model"
            )
        config["forward_model"] = forward_config["forward_model"]
    return config


def seed_everything(seed: int = 12345, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_torch:
        # Required by deterministic CUDA matrix products on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
