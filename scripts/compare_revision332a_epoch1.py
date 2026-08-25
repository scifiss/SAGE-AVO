#!/usr/bin/env python3
"""Compare interrupted and corrected epoch-1 checkpoints recursively and exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sage_avo.experiments.manifest import file_sha256, write_json


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _compare(left: Any, right: Any, path: str, differences: list[dict[str, Any]]) -> None:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        equal = left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        if not equal:
            maximum = None
            if left.shape == right.shape and left.dtype == right.dtype and left.numel():
                if left.is_floating_point() or left.is_complex():
                    maximum = float((left - right).abs().max())
            differences.append(
                {
                    "path": path,
                    "kind": "tensor",
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "maximum_absolute_difference": maximum,
                }
            )
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right):
            differences.append({"path": path, "kind": "numpy_array"})
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append(
                {
                    "path": path,
                    "kind": "mapping_keys",
                    "left_only": sorted(set(left) - set(right)),
                    "right_only": sorted(set(right) - set(left)),
                }
            )
        for key in left.keys() & right.keys():
            _compare(left[key], right[key], f"{path}.{key}", differences)
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            differences.append(
                {"path": path, "kind": "sequence_length", "left": len(left), "right": len(right)}
            )
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _compare(left_value, right_value, f"{path}[{index}]", differences)
        return
    try:
        equal = bool(left == right)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        differences.append(
            {"path": path, "kind": "scalar", "left": repr(left), "right": repr(right)}
        )


def run(arguments: argparse.Namespace) -> None:
    old_path = Path(arguments.old)
    corrected_path = Path(arguments.corrected)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    corrected = torch.load(corrected_path, map_location="cpu", weights_only=False)
    sections = (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "adaptive_weighter_state",
        "rng_state",
    )
    section_reports: dict[str, Any] = {}
    all_differences: list[dict[str, Any]] = []
    for section in sections:
        differences: list[dict[str, Any]] = []
        _compare(old.get(section), corrected.get(section), section, differences)
        section_reports[section] = {
            "exactly_equal": not differences,
            "difference_count": len(differences),
            "first_differences": differences[:20],
        }
        all_differences.extend(differences)
    equivalent = not all_differences and old.get("epoch") == corrected.get("epoch")
    status = (
        "OLD_EPOCH1_CHECKPOINT_EQUIVALENT"
        if equivalent
        else "OLD_EPOCH1_CHECKPOINT_NOT_EQUIVALENT"
    )
    report = {
        "schema_version": 1,
        "revision": "3.3.2a",
        "status": status,
        "comparison_is_exact": True,
        "old_checkpoint": {
            "path": str(old_path),
            "sha256": file_sha256(old_path),
            "epoch": old.get("epoch"),
        },
        "corrected_checkpoint": {
            "path": str(corrected_path),
            "sha256": file_sha256(corrected_path),
            "epoch": corrected.get("epoch"),
        },
        "sections": section_reports,
        "total_difference_count": len(all_differences),
        "model_state_sha256": {
            "old": {
                name: _tensor_sha256(value) for name, value in old["model_state"].items()
            },
            "corrected": {
                name: _tensor_sha256(value)
                for name, value in corrected["model_state"].items()
            },
        },
        "checkpoint_metrics": {"old": old.get("metrics"), "corrected": corrected.get("metrics")},
        "interpretation": (
            "Exact equality is required for reuse; the clean corrected production run remains "
            "mandatory regardless of this classification."
        ),
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(json.dumps({key: value for key, value in report.items() if key != "model_state_sha256"}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--old", required=True)
    root.add_argument("--corrected", required=True)
    root.add_argument("--output", required=True)
    return root


if __name__ == "__main__":
    run(parser().parse_args())
