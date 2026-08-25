"""Deterministic support acceptance for calibrated synthetic fluid scenarios.

The acceptance layer rejects a complete candidate realization when its plume
state leaves the predeclared Revision-3.3 dry-frame contract.  It never clips,
masks, or repairs individual pixels.  The calibration and Candidate-B physics
remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    mineral_properties_vrh_strict,
    poisson_ratio_from_moduli,
)
from .rock_physics import elastic_moduli_gpa


@dataclass(frozen=True)
class SupportAcceptanceContract:
    """Immutable thresholds applied to every complete candidate realization."""

    calibration_id: str
    nearest_neighbor_threshold: float
    nearest_neighbor_training_quantile: float
    physical_depth_domain_m: tuple[float, float]
    physical_porosity_domain_fraction: tuple[float, float]
    minimum_overall_coverage: float
    minimum_class_coverage: float
    facies_shaliness_boundary: float
    depth_class_boundaries_m: tuple[float, float]
    dry_bulk_to_shear_range: tuple[float, float]
    dry_poisson_ratio_range: tuple[float, float]
    maximum_fixed_shear_error_gpa: float
    maximum_outside_plume_change: float
    scenario_pressure_range_mpa: tuple[float, float]
    scenario_temperature_range_c: tuple[float, float]
    scenario_salinity_range_fraction: tuple[float, float]
    scenario_brie_exponent_range: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable contract."""
        return {
            "calibration_id": self.calibration_id,
            "nearest_neighbor_threshold": self.nearest_neighbor_threshold,
            "nearest_neighbor_training_quantile": self.nearest_neighbor_training_quantile,
            "physical_depth_domain_m": list(self.physical_depth_domain_m),
            "physical_porosity_domain_fraction": list(
                self.physical_porosity_domain_fraction
            ),
            "minimum_overall_coverage": self.minimum_overall_coverage,
            "minimum_class_coverage": self.minimum_class_coverage,
            "facies_shaliness_boundary": self.facies_shaliness_boundary,
            "depth_class_boundaries_m": list(self.depth_class_boundaries_m),
            "dry_bulk_to_shear_range": list(self.dry_bulk_to_shear_range),
            "dry_poisson_ratio_range": list(self.dry_poisson_ratio_range),
            "maximum_fixed_shear_error_gpa": self.maximum_fixed_shear_error_gpa,
            "maximum_outside_plume_change": self.maximum_outside_plume_change,
            "scenario_pressure_range_mpa": list(self.scenario_pressure_range_mpa),
            "scenario_temperature_range_c": list(self.scenario_temperature_range_c),
            "scenario_salinity_range_fraction": list(
                self.scenario_salinity_range_fraction
            ),
            "scenario_brie_exponent_range": list(
                self.scenario_brie_exponent_range
            ),
        }


@dataclass(frozen=True)
class CandidateSupportReport:
    """Support and physical QC for one unmodified candidate realization."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    statistics: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "statistics": self.statistics,
        }


def calibration_support_threshold(
    calibration: CalibratedDryFrameModel,
    quantile: float,
) -> float:
    """Derive the approved self-neighbour distance threshold from calibration data."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("nearest-neighbour training quantile must lie in (0, 1)")
    features = np.asarray(calibration.features_standardized, dtype=float)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("Calibration support features are invalid")
    distances = cKDTree(features).query(features, k=2)[0][:, 1]
    return float(np.quantile(distances, quantile))


def support_contract_from_mapping(
    mapping: dict[str, Any],
    calibration: CalibratedDryFrameModel,
) -> SupportAcceptanceContract:
    """Build a support contract without changing any Revision-3.3 threshold."""
    quantile = float(mapping["nearest_neighbor_training_quantile"])
    expected_calibration = str(mapping["calibration_id"])
    if calibration.calibration_id != expected_calibration:
        raise ValueError(
            f"Support contract expects calibration {expected_calibration!r}, "
            f"received {calibration.calibration_id!r}"
        )

    def pair(name: str) -> tuple[float, float]:
        values = tuple(float(value) for value in mapping[name])
        if len(values) != 2 or values[1] <= values[0]:
            raise ValueError(f"{name} must contain two increasing values")
        return values

    contract = SupportAcceptanceContract(
        calibration_id=calibration.calibration_id,
        nearest_neighbor_threshold=calibration_support_threshold(calibration, quantile),
        nearest_neighbor_training_quantile=quantile,
        physical_depth_domain_m=pair("physical_depth_domain_m"),
        physical_porosity_domain_fraction=pair(
            "physical_porosity_domain_fraction"
        ),
        minimum_overall_coverage=float(mapping["minimum_overall_coverage"]),
        minimum_class_coverage=float(mapping["minimum_class_coverage"]),
        facies_shaliness_boundary=float(mapping["facies_shaliness_boundary"]),
        depth_class_boundaries_m=pair("depth_class_boundaries_m"),
        dry_bulk_to_shear_range=pair("dry_bulk_to_shear_range"),
        dry_poisson_ratio_range=pair("dry_poisson_ratio_range"),
        maximum_fixed_shear_error_gpa=float(
            mapping["maximum_fixed_shear_error_gpa"]
        ),
        maximum_outside_plume_change=float(
            mapping["maximum_outside_plume_change"]
        ),
        scenario_pressure_range_mpa=pair("scenario_pressure_range_mpa"),
        scenario_temperature_range_c=pair("scenario_temperature_range_c"),
        scenario_salinity_range_fraction=pair(
            "scenario_salinity_range_fraction"
        ),
        scenario_brie_exponent_range=pair("scenario_brie_exponent_range"),
    )
    for name, value in (
        ("minimum_overall_coverage", contract.minimum_overall_coverage),
        ("minimum_class_coverage", contract.minimum_class_coverage),
        ("facies_shaliness_boundary", contract.facies_shaliness_boundary),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    return contract


def deterministic_candidate_seed(
    master_seed: int,
    attempt_index: int,
    *,
    namespace: str,
) -> int:
    """Return the historical seed at attempt zero and a stable retry seed later."""
    if attempt_index < 0:
        raise ValueError("attempt_index cannot be negative")
    if attempt_index == 0:
        return int(master_seed)
    payload = f"{namespace}|{int(master_seed)}|{int(attempt_index)}".encode()
    # Keep retry seeds in the non-negative signed-int64 domain so they can be
    # represented losslessly in NumPy archives and across process boundaries.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _class_labels(
    shaliness: np.ndarray,
    depth_m: np.ndarray,
    contract: SupportAcceptanceContract,
) -> tuple[np.ndarray, np.ndarray]:
    facies = np.where(
        shaliness < contract.facies_shaliness_boundary,
        "clean_sand",
        "shaly_sand",
    )
    shallow, deep = contract.depth_class_boundaries_m
    depth = np.where(
        depth_m <= shallow,
        "shallow",
        np.where(depth_m <= deep, "middle", "deep"),
    )
    return facies, depth


def _inside(value: float, limits: tuple[float, float]) -> bool:
    return limits[0] <= value <= limits[1]


def evaluate_candidate_support(
    *,
    elastic: np.ndarray,
    elastic_brine: np.ndarray,
    shaliness: np.ndarray,
    plume_mask: np.ndarray,
    co2_saturation: np.ndarray,
    time_ms: np.ndarray,
    fluid_metadata: dict[str, Any],
    calibration: CalibratedDryFrameModel,
    physics: FluidRockPhysics,
    contract: SupportAcceptanceContract,
) -> CandidateSupportReport:
    """Evaluate one complete fluid candidate without changing any candidate pixel."""
    target = np.asarray(elastic, dtype=float)
    brine = np.asarray(elastic_brine, dtype=float)
    shale_grid = np.asarray(shaliness, dtype=float)
    plume = np.asarray(plume_mask, dtype=bool)
    saturation_grid = np.asarray(co2_saturation, dtype=float)
    if target.shape != brine.shape or target.shape[0] != 3:
        raise ValueError("elastic and elastic_brine must have shape [3, time, trace]")
    if (
        shale_grid.shape != plume.shape
        or saturation_grid.shape != plume.shape
        or target.shape[1:] != plume.shape
    ):
        raise ValueError("Candidate support arrays do not share one spatial shape")
    reasons: list[str] = []
    plume_pixels = int(plume.sum())
    if plume_pixels == 0:
        return CandidateSupportReport(
            accepted=False,
            rejection_reasons=("empty_plume",),
            statistics={"plume_pixels": 0},
        )
    shale = shale_grid[plume]
    saturation = saturation_grid[plume]
    density_brine = brine[2][plume]
    mineral_bulk, _, mineral_density = mineral_properties_vrh_strict(shale, physics)
    effective_porosity = (mineral_density - density_brine) / (
        mineral_density - physics.brine_density_g_cc
    )
    mapping = calibration.metadata["time_depth_linear_coefficients"]
    depth_by_row = (
        float(mapping["slope_m_per_ms"]) * np.asarray(time_ms, dtype=float)
        + float(mapping["intercept_m"])
    )
    depth = np.broadcast_to(depth_by_row[:, None], plume.shape)[plume]
    dry_bulk, frame_shear, distance = calibration.predict(
        effective_porosity, shale, depth
    )
    supported = distance <= contract.nearest_neighbor_threshold
    overall_coverage = float(supported.mean())
    facies, depth_classes = _class_labels(shale, depth, contract)
    class_rows: list[dict[str, object]] = []
    for facies_name, depth_name in sorted(set(zip(facies, depth_classes))):
        member = (facies == facies_name) & (depth_classes == depth_name)
        coverage = float(supported[member].mean())
        class_rows.append(
            {
                "facies": str(facies_name),
                "depth_class": str(depth_name),
                "pixels": int(member.sum()),
                "coverage": coverage,
            }
        )
        if coverage < contract.minimum_class_coverage:
            reasons.append(f"class_support:{facies_name}|{depth_name}")
    if overall_coverage < contract.minimum_overall_coverage:
        reasons.append("overall_support_coverage")
    depth_inside = (
        (depth >= contract.physical_depth_domain_m[0])
        & (depth <= contract.physical_depth_domain_m[1])
    )
    porosity_inside = (
        (effective_porosity >= contract.physical_porosity_domain_fraction[0])
        & (effective_porosity <= contract.physical_porosity_domain_fraction[1])
    )
    if not depth_inside.all():
        reasons.append("physical_depth_domain")
    if not porosity_inside.all():
        reasons.append("physical_effective_porosity_domain")
    dry_to_shear = dry_bulk / frame_shear
    dry_poisson = poisson_ratio_from_moduli(dry_bulk, frame_shear)
    dry_physical = (
        np.isfinite(dry_bulk)
        & np.isfinite(frame_shear)
        & (dry_bulk > 0.0)
        & (dry_bulk < mineral_bulk)
        & (frame_shear > 0.0)
        & (dry_to_shear >= contract.dry_bulk_to_shear_range[0])
        & (dry_to_shear <= contract.dry_bulk_to_shear_range[1])
        & (dry_poisson >= contract.dry_poisson_ratio_range[0])
        & (dry_poisson <= contract.dry_poisson_ratio_range[1])
    )
    if not dry_physical.all():
        reasons.append("dry_frame_physical")
    elastic_physical = (
        np.isfinite(target).all()
        and np.isfinite(brine).all()
        and np.all(target[0] > target[1])
        and np.all(target[1] > 0.0)
        and np.all(target[2] > 0.0)
    )
    if not elastic_physical:
        reasons.append("elastic_state_physical")
    target_bulk, target_shear = elastic_moduli_gpa(*target)
    brine_bulk, brine_shear = elastic_moduli_gpa(*brine)
    maximum_shear_error = float(np.max(np.abs(target_shear - brine_shear)))
    outside = ~plume
    maximum_outside_change = (
        float(np.max(np.abs(target[:, outside] - brine[:, outside])))
        if outside.any()
        else 0.0
    )
    if maximum_shear_error > contract.maximum_fixed_shear_error_gpa:
        reasons.append("fixed_shear_invariance")
    if maximum_outside_change > contract.maximum_outside_plume_change:
        reasons.append("outside_plume_change")
    forbidden_flags = {
        "feasibility_projection_used": False,
        "dry_bulk_clipping_used": False,
        "elastic_output_clipping_used": False,
        "direct_independent_elastic_delta_transfer": False,
        "shear_modulus_changed_by_fluid": False,
    }
    if any(
        fluid_metadata.get(name) is not value
        for name, value in forbidden_flags.items()
    ):
        reasons.append("projection_clipping_or_independent_delta")
    state = fluid_metadata.get("property_state", {})
    if state:
        pressure = float(state["brine"]["pressure_mpa"])
        temperature = float(state["brine"]["temperature_c"])
        salinity = float(state["brine"]["salinity_mass_fraction"])
        brie = float(state["brie_exponent"])
        scenario_valid = all(
            (
                _inside(pressure, contract.scenario_pressure_range_mpa),
                _inside(temperature, contract.scenario_temperature_range_c),
                _inside(salinity, contract.scenario_salinity_range_fraction),
                _inside(brie, contract.scenario_brie_exponent_range),
                state["co2"].get("phase") == "supercritical",
            )
        )
        if not scenario_valid:
            reasons.append("scenario_state")
    unique_reasons = tuple(dict.fromkeys(reasons))
    statistics: dict[str, Any] = {
        "plume_pixels": plume_pixels,
        "overall_support_coverage": overall_coverage,
        "support_threshold": contract.nearest_neighbor_threshold,
        "classes": class_rows,
        "depth_m": {
            "minimum": float(depth.min()),
            "median": float(np.median(depth)),
            "maximum": float(depth.max()),
            "outside_domain_pixels": int(np.count_nonzero(~depth_inside)),
        },
        "effective_porosity": {
            "minimum": float(effective_porosity.min()),
            "median": float(np.median(effective_porosity)),
            "maximum": float(effective_porosity.max()),
            "outside_domain_pixels": int(np.count_nonzero(~porosity_inside)),
        },
        "shaliness": {
            "minimum": float(shale.min()),
            "median": float(np.median(shale)),
            "maximum": float(shale.max()),
        },
        "saturation": {
            "minimum": float(saturation.min()),
            "median": float(np.median(saturation)),
            "maximum": float(saturation.max()),
        },
        "calibration_distance": {
            "minimum": float(distance.min()),
            "median": float(np.median(distance)),
            "p95": float(np.quantile(distance, 0.95)),
            "p99": float(np.quantile(distance, 0.99)),
            "maximum": float(distance.max()),
        },
        "dry_bulk_gpa": {
            "minimum": float(dry_bulk.min()),
            "median": float(np.median(dry_bulk)),
            "maximum": float(dry_bulk.max()),
        },
        "dry_to_shear": {
            "minimum": float(dry_to_shear.min()),
            "median": float(np.median(dry_to_shear)),
            "maximum": float(dry_to_shear.max()),
        },
        "dry_poisson_ratio": {
            "minimum": float(dry_poisson.min()),
            "median": float(np.median(dry_poisson)),
            "maximum": float(dry_poisson.max()),
        },
        "delta_ksat_gpa": {
            "minimum": float((target_bulk - brine_bulk)[plume].min()),
            "median": float(np.median((target_bulk - brine_bulk)[plume])),
            "maximum": float((target_bulk - brine_bulk)[plume].max()),
        },
        "maximum_fixed_shear_error_gpa": maximum_shear_error,
        "maximum_outside_plume_change": maximum_outside_change,
        "all_dry_frame_states_physical": bool(dry_physical.all()),
        "all_elastic_states_physical": bool(elastic_physical),
    }
    return CandidateSupportReport(
        accepted=not unique_reasons,
        rejection_reasons=unique_reasons,
        statistics=statistics,
    )


__all__ = [
    "CandidateSupportReport",
    "SupportAcceptanceContract",
    "calibration_support_threshold",
    "deterministic_candidate_seed",
    "evaluate_candidate_support",
    "support_contract_from_mapping",
]
