"""State-dict checkpointing with explicit metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Save portable state dictionaries rather than pickled model objects."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
        },
        destination,
    )


def sampling_model(model: nn.Module) -> nn.Module:
    """Return the underlying model so custom sampling works under DataParallel."""
    return model.module if hasattr(model, "module") else model
