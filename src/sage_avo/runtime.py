"""Runtime diagnostics and explicit torch-device selection."""

from __future__ import annotations

import sys
import warnings

import torch


def torch_runtime_report() -> dict[str, object]:
    """Return the interpreter and CUDA facts needed to diagnose kernel mismatches."""
    cuda_available = bool(torch.cuda.is_available())
    return {
        "sys.executable": sys.executable,
        "torch.__version__": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "torch.cuda.is_available()": cuda_available,
        "torch.cuda.get_device_name(0)": (
            torch.cuda.get_device_name(0) if cuda_available else "UNAVAILABLE"
        ),
    }


def print_torch_runtime() -> dict[str, object]:
    """Print and return the torch runtime report."""
    report = torch_runtime_report()
    for name, value in report.items():
        print(f"{name} = {value}")
    return report


def select_torch_device(
    requested: str | None = None,
    *,
    require_cuda: bool = False,
    context: str = "SAGE-AVO",
) -> torch.device:
    """Select a device visibly and optionally forbid an unintended CPU fallback."""
    selected = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{context} requested CUDA, but this Python process cannot access it. "
            "Check sys.executable and select the WSL 'Python (sage-avo CUDA)' kernel."
        )
    if require_cuda and selected.type != "cuda":
        raise RuntimeError(
            f"{context} requires CUDA; refusing an unannounced CPU fallback. "
            "Select the WSL 'Python (sage-avo CUDA)' kernel or pass an explicit CUDA device."
        )
    if selected.type == "cpu":
        warnings.warn(f"{context} selected CPU", RuntimeWarning, stacklevel=2)
    print(f"Selected device ({context}) = {selected}")
    return selected
