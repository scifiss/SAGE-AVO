"""State-dict checkpointing with explicit metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import random

import numpy as np
import torch
from torch import nn


def migrate_original_sage_avo_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map final-005 notebook/module keys to the modular SAGE-AVO names."""
    migrated: dict[str, torch.Tensor] = {}
    prefix_mapping = (
        ("time_embed.", "time_embedding."),
        ("cond_embed.", "condition_embedding."),
        ("enc.", "encoder."),
        ("gnn.", "graph."),
        ("dec.", "decoder."),
    )
    nested_mapping = (
        ("graph.node_proj.", "graph.node_projection."),
        ("graph.convs.", "graph.layers."),
        ("graph.norms.", "graph.normalizations."),
        ("graph.seg_decoder.", "graph.segmentation."),
    )
    for original_name, value in state_dict.items():
        name = original_name
        while name.startswith("module."):
            name = name[len("module.") :]
        for original, replacement in prefix_mapping:
            if name.startswith(original):
                name = replacement + name[len(original) :]
                break
        for original, replacement in nested_mapping:
            if name.startswith(original):
                name = replacement + name[len(original) :]
                break
        name = name.replace(".net.", ".network.")
        migrated[name] = value
    return migrated


def _cpu_byte_rng_state(state: Any) -> torch.Tensor:
    """Return the CPU byte buffer required by PyTorch RNG restore APIs."""
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def _normalize_rng_state_devices(checkpoint: dict[str, Any]) -> None:
    """Keep RNG buffers on CPU even when model tensors are mapped to CUDA."""
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        return
    if rng_state.get("torch_cpu") is not None:
        rng_state["torch_cpu"] = _cpu_byte_rng_state(rng_state["torch_cpu"])
    if rng_state.get("torch_cuda") is not None:
        rng_state["torch_cuda"] = [
            _cpu_byte_rng_state(state) for state in rng_state["torch_cuda"]
        ]
    generators = rng_state.get("generators")
    if isinstance(generators, dict):
        rng_state["generators"] = {
            name: _cpu_byte_rng_state(state) for name, state in generators.items()
        }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    adaptive_weighter: nn.Module | None = None,
    generator_states: dict[str, torch.Tensor] | None = None,
) -> None:
    """Save portable state dictionaries rather than pickled model objects."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "adaptive_weighter_state": (
                adaptive_weighter.state_dict() if adaptive_weighter is not None else None
            ),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "generators": generator_states or {},
            },
        },
        temporary,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    adaptive_weighter: nn.Module | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a portable training checkpoint, including optional resume state."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    _normalize_rng_state_devices(checkpoint)
    module = model.module if hasattr(model, "module") else model
    state = migrate_original_sage_avo_state_dict(checkpoint.get("model_state", checkpoint))
    module.load_state_dict(state)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if adaptive_weighter is not None and checkpoint.get("adaptive_weighter_state") is not None:
        adaptive_weighter.load_state_dict(checkpoint["adaptive_weighter_state"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    return checkpoint


def sampling_model(model: nn.Module) -> nn.Module:
    """Return the underlying model so custom sampling works under DataParallel."""
    return model.module if hasattr(model, "module") else model
