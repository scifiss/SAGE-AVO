#!/usr/bin/env python3
"""Validate the Revision-3.3.2d matrix exact-PP operator before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sage_avo.forward.madagascar import reflectivity_gather_madagascar
from sage_avo.forward.torch_forward import (
    exact_zoeppritz_pp_closed_form,
    exact_zoeppritz_pp_matrix,
)
from sage_avo.forward.zoeppritz import reflectivity_gather

import run_revision332_production as revision332


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = revision332.PRIVATE
DATASET = revision332.DATASET
OUTPUT = PRIVATE / "revision332d" / "exact_zoeppritz_validation"
FAILURE_STATES = PRIVATE / "revision332d" / "former_failure_states"
FAILURE_REPLAY = PRIVATE / "revision332d" / "former_failure_replay.json"
TRUST_REGION_SELECTION = PRIVATE / "revision332d" / "residual_trust_region_selection.json"
ANGLES = np.arange(3.0, 46.0, dtype=np.float64)
REFERENCE_CASES = {
    "weak_positive": (2500.0, 1400.0, 2.25, 2600.0, 1450.0, 2.28),
    "weak_negative": (2600.0, 1450.0, 2.28, 2500.0, 1400.0, 2.25),
    "strong_ordinary": (2200.0, 1200.0, 2.10, 3000.0, 1650.0, 2.40),
    "near_critical": (2400.0, 1300.0, 2.15, 3400.0, 1850.0, 2.45),
    "postcritical": (2200.0, 1200.0, 2.10, 3600.0, 1950.0, 2.50),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    scale = float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float64) ** 2)))
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / max(scale, np.finfo(np.float64).tiny),
    }


def _coefficient_triplet(
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vp = np.stack((upper[:, 0], lower[:, 0]))
    vs = np.stack((upper[:, 1], lower[:, 1]))
    density = np.stack((upper[:, 2], lower[:, 2]))
    madagascar = reflectivity_gather_madagascar(vp, vs, density, ANGLES)[:, 1].T
    numpy_result = reflectivity_gather(vp, vs, density, ANGLES)[:, 1].T
    tensors = [torch.as_tensor(value.T[:, :, None], dtype=torch.float64) for value in (vp, vs, density)]
    angle_tensor = torch.as_tensor(ANGLES, dtype=torch.float64)
    matrix = exact_zoeppritz_pp_matrix(*tensors, angle_tensor)[:, :, 1, 0].detach().numpy()
    legacy = exact_zoeppritz_pp_closed_form(*tensors, angle_tensor)[:, :, 1, 0].detach().numpy()
    return madagascar, numpy_result, matrix, legacy


def _stage03_interfaces(count: int = 256) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    archives = sorted((DATASET / "realizations").glob("realization_*.npz"))
    if not archives:
        raise FileNotFoundError(DATASET / "realizations")
    rng = np.random.default_rng(12345)
    upper: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    ranges = np.stack((np.full(3, np.inf), np.full(3, -np.inf)))
    per_archive = max(1, int(np.ceil(count / len(archives))))
    for archive in archives:
        with np.load(archive) as data:
            elastic = np.asarray(data["elastic"], dtype=np.float64)
        flattened = elastic.reshape(3, -1)
        ranges[0] = np.minimum(ranges[0], flattened.min(axis=1))
        ranges[1] = np.maximum(ranges[1], flattened.max(axis=1))
        if len(upper) < count:
            for _ in range(min(per_archive, count - len(upper))):
                row = int(rng.integers(0, elastic.shape[1] - 1))
                column = int(rng.integers(0, elastic.shape[2]))
                upper.append(elastic[:, row, column])
                lower.append(elastic[:, row + 1, column])
    return (
        np.asarray(upper),
        np.asarray(lower),
        {
            "sample_count": len(upper),
            "minimum_vp_vs_density": ranges[0].tolist(),
            "maximum_vp_vs_density": ranges[1].tolist(),
            "selection_seed": 12345,
        },
    )


def _forward_validation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[np.ndarray]] = {
        name: [] for name in ("madagascar", "numpy", "matrix", "legacy")
    }
    for name, case in REFERENCE_CASES.items():
        upper = np.asarray([case[:3]])
        lower = np.asarray([case[3:]])
        results = _coefficient_triplet(upper, lower)
        mapped = dict(zip(aggregate, results))
        for values_name, values in mapped.items():
            aggregate[values_name].append(values.reshape(-1))
        for candidate in ("numpy", "matrix", "legacy"):
            rows.append(
                {
                    "population": name,
                    "reference": "madagascar_sfzoeppritz2",
                    "candidate": candidate,
                    **_metrics(mapped["madagascar"], mapped[candidate]),
                }
            )
        rows.append(
            {
                "population": name,
                "reference": "stage02_numpy",
                "candidate": "matrix",
                **_metrics(mapped["numpy"], mapped["matrix"]),
            }
        )

    upper, lower, range_report = _stage03_interfaces()
    results = _coefficient_triplet(upper, lower)
    mapped = dict(zip(aggregate, results))
    for values_name, values in mapped.items():
        aggregate[values_name].append(values.reshape(-1))
    for reference, candidate in (
        ("madagascar", "numpy"),
        ("madagascar", "matrix"),
        ("madagascar", "legacy"),
        ("numpy", "matrix"),
    ):
        rows.append(
            {
                "population": "stage03_physical_range",
                "reference": reference,
                "candidate": candidate,
                **_metrics(mapped[reference], mapped[candidate]),
            }
        )

    concatenated = {name: np.concatenate(values) for name, values in aggregate.items()}
    summary = {
        "angles_degrees": [3.0, 45.0, 1.0],
        "stage03_physical_range": range_report,
        "aggregate": {
            "madagascar_vs_numpy": _metrics(
                concatenated["madagascar"], concatenated["numpy"]
            ),
            "madagascar_vs_matrix": _metrics(
                concatenated["madagascar"], concatenated["matrix"]
            ),
            "numpy_vs_matrix": _metrics(concatenated["numpy"], concatenated["matrix"]),
            "legacy_vs_matrix": _metrics(concatenated["legacy"], concatenated["matrix"]),
        },
        "convention": (
            "Madagascar sfzoeppritz2 icoef=4 incp=y outp=y refl=y; real reflected "
            "P displacement coefficient; no polarity flip or interface shift"
        ),
    }
    return summary, rows


def _scalar_response(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    vp = torch.stack((values[0], values[3])).reshape(1, 2, 1)
    vs = torch.stack((values[1], values[4])).reshape(1, 2, 1)
    density = torch.stack((values[2], values[5])).reshape(1, 2, 1)
    angles = torch.tensor((3.0, 17.0, 31.0, 45.0), dtype=torch.float64)
    response = exact_zoeppritz_pp_matrix(vp, vs, density, angles)
    return response.square().mean()


def _finite_difference(case: tuple[float, ...]) -> dict[str, Any]:
    values = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in case)
    scalar = _scalar_response(values)
    analytic = torch.autograd.grad(scalar, values)
    records = []
    for index, (value, derivative) in enumerate(zip(case, analytic)):
        step = 1e-4 * max(abs(float(value)), 1.0)
        plus = list(case)
        minus = list(case)
        plus[index] += step
        minus[index] -= step
        plus_value = float(
            _scalar_response(tuple(torch.tensor(item, dtype=torch.float64) for item in plus))
        )
        minus_value = float(
            _scalar_response(tuple(torch.tensor(item, dtype=torch.float64) for item in minus))
        )
        numerical = (plus_value - minus_value) / (2.0 * step)
        analytic_value = float(derivative)
        records.append(
            {
                "parameter": ("vp1", "vs1", "rho1", "vp2", "vs2", "rho2")[index],
                "analytic": analytic_value,
                "finite_difference": numerical,
                "absolute_error": abs(analytic_value - numerical),
                "relative_error": abs(analytic_value - numerical)
                / max(abs(numerical), abs(analytic_value), 1e-14),
            }
        )
    return {
        "finite_forward": bool(torch.isfinite(scalar)),
        "finite_backward": all(bool(torch.isfinite(value)) for value in analytic),
        "maximum_absolute_error": max(record["absolute_error"] for record in records),
        "maximum_relative_error": max(record["relative_error"] for record in records),
        "parameters": records,
    }


def _gradient_validation() -> dict[str, Any]:
    results: dict[str, Any] = {}
    gradcheck_angles = torch.tensor((3.0, 17.0, 31.0, 45.0), dtype=torch.float64)
    for name in ("strong_ordinary", "near_critical", "postcritical"):
        case = REFERENCE_CASES[name]
        values = tuple(
            torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in case
        )

        def response(*arguments: torch.Tensor) -> torch.Tensor:
            vp = torch.stack((arguments[0], arguments[3])).reshape(1, 2, 1)
            vs = torch.stack((arguments[1], arguments[4])).reshape(1, 2, 1)
            density = torch.stack((arguments[2], arguments[5])).reshape(1, 2, 1)
            return exact_zoeppritz_pp_matrix(vp, vs, density, gradcheck_angles)

        gradcheck_passed = torch.autograd.gradcheck(
            response,
            values,
            eps=1e-4,
            atol=5e-5,
            rtol=5e-4,
        )
        results[name] = {
            "gradcheck_passed": bool(gradcheck_passed),
            "finite_difference": _finite_difference(case),
        }
    return results


def _failure_state_validation() -> dict[str, Any]:
    results: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in sorted(FAILURE_STATES.glob("*.npz")):
        with np.load(path) as data:
            vp = (
                torch.from_numpy(np.asarray(data["vp"]))
                .to(device=device, dtype=torch.float32)
                .requires_grad_(True)
            )
            vs = (
                torch.from_numpy(np.asarray(data["vs"]))
                .to(device=device, dtype=torch.float32)
                .requires_grad_(True)
            )
            density = (
                torch.from_numpy(np.asarray(data["density"]))
                .to(device=device, dtype=torch.float32)
                .requires_grad_(True)
            )
        input_finite = all(bool(torch.isfinite(value).all()) for value in (vp, vs, density))
        capture_role = "trigger_input" if "_trigger_input_" in path.stem else "first_nonfinite"
        if not input_finite:
            results[path.stem] = {
                "path": str(path),
                "shape": list(vp.shape),
                "capture_role": capture_role,
                "input_finite": False,
                "interpretation": "first observed nonfinite batch already contains NaN core predictions",
            }
            del vp, vs, density
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        angles = torch.as_tensor(ANGLES, device=device, dtype=torch.float32)
        physical_domain = {
            "minimum_vp": float(vp.min()),
            "minimum_vs": float(vs.min()),
            "minimum_density": float(density.min()),
            "nonpositive_vp_count": int((vp <= 0).sum()),
            "nonpositive_vs_count": int((vs <= 0).sum()),
            "nonpositive_density_count": int((density <= 0).sum()),
            "vp_not_greater_than_vs_count": int((vp <= vs).sum()),
        }
        outside_physical_domain = any(
            physical_domain[name] > 0
            for name in (
                "nonpositive_vp_count",
                "nonpositive_vs_count",
                "nonpositive_density_count",
                "vp_not_greater_than_vs_count",
            )
        )
        response = exact_zoeppritz_pp_matrix(vp, vs, density, angles)
        loss = response.square().mean()
        gradients = torch.autograd.grad(loss, (vp, vs, density))
        results[path.stem] = {
            "path": str(path),
            "shape": list(vp.shape),
            "device": str(device),
            "dtype": str(vp.dtype),
            "capture_role": capture_role,
            "input_finite": True,
            "physical_domain": physical_domain,
            "outside_physical_domain": outside_physical_domain,
            "finite_forward": bool(torch.isfinite(response).all()),
            "finite_backward": all(bool(torch.isfinite(value).all()) for value in gradients),
            "response_maximum_absolute": float(response.abs().max()),
            "gradient_maximum_absolute": [float(value.abs().max()) for value in gradients],
        }
        del response, loss, gradients, vp, vs, density
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def run(_: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    forward, rows = _forward_validation()
    gradients = _gradient_validation()
    failure_states = _failure_state_validation()
    matrix_vs_numpy = forward["aggregate"]["numpy_vs_matrix"]
    madagascar_vs_matrix = forward["aggregate"]["madagascar_vs_matrix"]
    former_prefixes = ("edge_aware_contrast_", "truth_edge_matching_", "no_aux_graph_loss_")
    trigger_states = [
        record
        for name, record in failure_states.items()
        if name.startswith(former_prefixes) and record.get("capture_role") == "trigger_input"
    ]
    base_passed = (
        matrix_vs_numpy["maximum_absolute_error"] <= 2e-8
        and madagascar_vs_matrix["maximum_absolute_error"] <= 1.1e-6
        and all(record["gradcheck_passed"] for record in gradients.values())
        and all(record["finite_difference"]["finite_backward"] for record in gradients.values())
    )
    all_triggers_finite = len(trigger_states) == 3 and all(
        record["finite_forward"] and record["finite_backward"] for record in trigger_states
    )
    failing_triggers = [
        record
        for record in trigger_states
        if not (record["finite_forward"] and record["finite_backward"])
    ]
    raw_mixed_batch_would_require_domain_control = (
        base_passed
        and len(trigger_states) == 3
        and bool(failing_triggers)
        and all(record.get("outside_physical_domain", False) for record in failing_triggers)
    )
    replay = json.loads(FAILURE_REPLAY.read_text(encoding="utf-8"))
    trust_selection = json.loads(TRUST_REGION_SELECTION.read_text(encoding="utf-8"))
    replay_passed = replay.get("status") == "FORMER_FAILURE_REPLAY_GO" and all(
        all(
            bool(row[key])
            for key in (
                "gradients_finite",
                "metrics_finite",
                "optimizer_state_finite",
                "parameters_finite",
            )
        )
        for row in replay.get("checks", [])
    )
    trust_region_required = not (
        replay_passed and trust_selection.get("status") == "NOT_SELECTED"
    )
    if base_passed and replay_passed and not trust_region_required:
        status = "ZOEPPRITZ_MATRIX_VALIDATION_GO"
    elif base_passed and all_triggers_finite:
        status = "ZOEPPRITZ_MATRIX_VALIDATION_GO"
    elif raw_mixed_batch_would_require_domain_control:
        status = "ZOEPPRITZ_MATRIX_VALIDATION_TRUST_REGION_REQUIRED"
    else:
        status = "ZOEPPRITZ_MATRIX_VALIDATION_NO_GO"
    report = {
        "schema_version": 1,
        "status": status,
        "operator": "exact_complex_boundary_condition_matrix_torch_linalg_solve",
        "forward_validation": forward,
        "gradient_validation": gradients,
        "former_failure_states": failure_states,
        "failure_state_count": len(trigger_states),
        "contaminated_first_nonfinite_state_count": sum(
            record.get("capture_role") == "first_nonfinite"
            for record in failure_states.values()
        ),
        "base_operator_validation_passed": base_passed,
        "raw_mixed_batch_trigger_states_finite": all_triggers_finite,
        "raw_mixed_batch_trigger_interpretation": (
            "Captured full-batch tensors contain physics-ineligible placeholder contexts and "
            "are retained as provenance; final masked physics subsets eligible samples before "
            "the nonlinear operator without changing the masked loss."
        ),
        "final_objective_failure_replay": {
            "path": str(FAILURE_REPLAY),
            "sha256": _file_sha256(FAILURE_REPLAY),
            "status": replay.get("status"),
            "passed": replay_passed,
            "checks": replay.get("checks", []),
        },
        "trust_region_selection": {
            "path": str(TRUST_REGION_SELECTION),
            "sha256": _file_sha256(TRUST_REGION_SELECTION),
            "status": trust_selection.get("status"),
        },
        "trust_region_required": trust_region_required,
        "acceptance": {
            "numpy_maximum_absolute_error": 2e-8,
            "madagascar_maximum_absolute_error": 1.1e-6,
            "gradcheck_required": True,
            "finite_final_objective_replay_required": True,
            "raw_ineligible_placeholder_contexts_are_operator_inputs": False,
        },
    }
    pd.DataFrame(rows).to_csv(OUTPUT / "forward_comparison.csv", index=False)
    (OUTPUT / "operator_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if status == "ZOEPPRITZ_MATRIX_VALIDATION_NO_GO":
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.set_defaults(function=run)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
