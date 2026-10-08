"""Thin checked adapters to existing authoritative scientific NumPy operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sage_avo.forward.pipeline import ForwardResult
    from sage_avo.forward.qc import ForwardAgreement
    from sage_avo.forward.specification import ForwardModelSpecification
    from sage_avo.geology.fluid_calibration import CalibratedDryFrameModel, FluidRockPhysics
    from sage_avo.geology.fluid_properties import FluidPropertyState
    from sage_avo.geology.rock_physics import ElasticProperties
    from sage_avo.geology.support import SupportAcceptanceContract
    from sage_avo.geology.synthetic import Deformation


@dataclass(frozen=True)
class CalibratedFluidApplication:
    """Approved Candidate-B output on a supplied baseline and declared region."""

    elastic: np.ndarray  # [3,time,trace]; channels Vp m/s, Vs m/s, density g/cc
    region_mask: np.ndarray  # [time, trace], unchanged exterior
    calibration_id: str
    maximum_calibration_distance: float
    method: str = "calibrated_differential_gassmann"


def _finite_array(
    name: str, values: np.ndarray, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _elastic_inputs(
    vp_m_s: np.ndarray, vs_m_s: np.ndarray, density_g_cc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vp = _finite_array("vp_m_s", vp_m_s)
    vs = _finite_array("vs_m_s", vs_m_s, vp.shape)
    density = _finite_array("density_g_cc", density_g_cc, vp.shape)
    if vp.ndim == 0 or np.any((vp <= vs) | (vs <= 0) | (density <= 0)):
        raise ValueError(
            "Elastic arrays must share a non-scalar shape with Vp > Vs > 0 and density > 0"
        )
    if np.any(vp**2 <= 4.0 * vs**2 / 3.0):
        raise ValueError("Elastic state has non-positive isotropic bulk modulus")
    return vp, vs, density


def make_coherent_deformation(
    shape: tuple[int, int], seed: int, **generator_parameters: Any
) -> Deformation:
    """Seed the existing generic synthetic fold/fault generator; units are samples."""
    from sage_avo.geology.synthetic import make_deformation

    if len(shape) != 2 or any(not isinstance(value, int) or value < 2 for value in shape):
        raise ValueError("shape must be two integer dimensions >= 2 in [time, trace] order")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return make_deformation(shape, np.random.default_rng(seed), **generator_parameters)


def warp_geological_arrays(
    arrays: Mapping[str, np.ndarray],
    deformation: Deformation,
    *,
    categorical_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Warp all supplied fields using one geometry; categorical fields use order 0."""
    from sage_avo.geology.synthetic import Deformation, warp_with_deformation

    if not isinstance(deformation, Deformation):
        raise TypeError("deformation must be a SAGE-AVO Deformation")
    vertical = _finite_array("vertical displacement", deformation.vertical_displacement)
    _finite_array("horizontal displacement", deformation.horizontal_displacement, vertical.shape)
    if vertical.ndim != 2 or not arrays:
        raise ValueError("deformation must be 2-D and arrays must not be empty")
    if set(categorical_keys) - set(arrays):
        raise ValueError("categorical_keys must name supplied arrays")
    output = {}
    for name, values in arrays.items():
        original = np.asarray(values)
        value = _finite_array(name, original)
        if value.ndim not in (2, 3) or value.shape[-2:] != vertical.shape:
            raise ValueError(
                f"{name} must be [time,trace] or [channel,time,trace] matching deformation"
            )
        categorical = name in categorical_keys
        output[name] = warp_with_deformation(
            original if categorical else value, deformation, order=0 if categorical else 1
        )
    return output


def elastic_moduli(
    vp_m_s: np.ndarray, vs_m_s: np.ndarray, density_g_cc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return saturated isotropic bulk K and shear G in GPa, without clipping."""
    from sage_avo.geology.rock_physics import elastic_moduli_gpa

    vp, vs, density = _elastic_inputs(vp_m_s, vs_m_s, density_g_cc)
    return elastic_moduli_gpa(vp, vs, density)


def elastic_from_moduli(
    bulk_gpa: np.ndarray, shear_gpa: np.ndarray, density_g_cc: np.ndarray
) -> ElasticProperties:
    """Return Vp/Vs m/s and density g/cc using the existing strict inverse."""
    from sage_avo.geology.fluid_calibration import elastic_from_gpa_strict

    bulk = _finite_array("bulk_gpa", bulk_gpa)
    shear = _finite_array("shear_gpa", shear_gpa, bulk.shape)
    density = _finite_array("density_g_cc", density_g_cc, bulk.shape)
    if bulk.ndim == 0:
        raise ValueError("moduli must use a non-scalar array shape")
    return elastic_from_gpa_strict(bulk, shear, density)


def brine_fluid_state(
    pressure_mpa: float, temperature_c: float, salinity_mass_fraction: float
) -> FluidPropertyState:
    """Evaluate the source Batzle–Wang NaCl-brine correlation within its support."""
    from sage_avo.geology.fluid_properties import batzle_wang_brine

    return batzle_wang_brine(pressure_mpa, temperature_c, salinity_mass_fraction)


def co2_fluid_state(
    pressure_mpa: float, temperature_c: float, *, require_supercritical: bool = True
) -> FluidPropertyState:
    """Evaluate pure CO2 using lazy CoolProp HEOS/Span–Wagner (single-phase)."""
    from sage_avo.geology.fluid_properties import span_wagner_co2

    return span_wagner_co2(pressure_mpa, temperature_c, require_supercritical=require_supercritical)


def _precritical(vp: np.ndarray, angles: np.ndarray) -> None:
    if vp.shape[0] < 2:
        raise ValueError("exact PP requires at least two time samples")
    maximum_sine = float(np.sin(np.deg2rad(angles[-1])))
    if np.any(maximum_sine * vp[1:] >= vp[:-1]):
        raise ValueError(
            "post-critical P transmission is outside the public real-valued PP contract"
        )


def exact_pp_reflectivity(
    vp_m_s: np.ndarray,
    vs_m_s: np.ndarray,
    density_g_cc: np.ndarray,
    angles_degrees: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Return dimensionless exact pre-critical PP coefficients [angle,time,trace]."""
    from sage_avo.forward.zoeppritz import reflectivity_gather

    vp, vs, density = _elastic_inputs(vp_m_s, vs_m_s, density_g_cc)
    if vp.ndim != 2:
        raise ValueError("elastic fields must have shape [time, trace]")
    angles = _finite_array("angles_degrees", np.asarray(angles_degrees))
    if (
        angles.ndim != 1
        or not len(angles)
        or angles[0] < 0
        or angles[-1] > 55
        or np.any(np.diff(angles) <= 0)
    ):
        raise ValueError("angles_degrees must increase strictly within [0, 55] degrees")
    _precritical(vp, angles)
    return reflectivity_gather(vp, vs, density, angles)


def simulate_three_band_avo(
    vp_m_s: np.ndarray,
    vs_m_s: np.ndarray,
    density_g_cc: np.ndarray,
    specification: ForwardModelSpecification,
    *,
    sample_origin: int = 0,
) -> ForwardResult:
    """Run the existing exact-PP spec: reflectivity, convolution, mute and bands."""
    from sage_avo.forward.pipeline import forward_avo_dense_spec
    from sage_avo.forward.specification import ForwardModelSpecification

    if not isinstance(specification, ForwardModelSpecification):
        raise TypeError("specification must be the existing ForwardModelSpecification")
    if not isinstance(sample_origin, int) or sample_origin < 0:
        raise ValueError("sample_origin must be a nonnegative global time-sample index")
    if tuple(band.name for band in specification.bands) != ("near", "mid", "far"):
        raise ValueError("three-band API requires ordered near/mid/far specification bands")
    vp, vs, density = _elastic_inputs(vp_m_s, vs_m_s, density_g_cc)
    if vp.ndim != 2:
        raise ValueError("elastic fields must have shape [time, trace]")
    _precritical(vp, np.asarray(specification.angles_degrees, dtype=float))
    return forward_avo_dense_spec(vp, vs, density, specification, sample_origin=sample_origin)


def compare_avo_outputs(reference: np.ndarray, candidate: np.ndarray) -> ForwardAgreement:
    """Compare matching [3,time,trace] amplitudes; neither input is truth."""
    from sage_avo.forward.qc import compare_forward_outputs

    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    if first.ndim != 3 or first.shape[0] != 3 or first.shape != second.shape:
        raise ValueError("reference and candidate must match [3,time,trace]")
    if not np.all(np.any(np.isfinite(first) & np.isfinite(second), axis=(1, 2))):
        raise ValueError("each band needs at least one pairwise-finite sample")
    return compare_forward_outputs(first, second)


def report_rgt_monotonicity(rgt: np.ndarray, *, tolerance: float = -1e-3) -> dict[str, float]:
    """Report coordinate monotonicity; never infer fault-safe identity from it."""
    from sage_avo.structure.rgt import monotonicity_report

    coordinate = _finite_array("rgt", rgt)
    if coordinate.ndim != 2 or coordinate.shape[0] < 2 or not np.isfinite(tolerance):
        raise ValueError("rgt must be finite [time,trace] with >=2 times and finite tolerance")
    return monotonicity_report(coordinate, tolerance)


def plot_elastic_comparison(
    truth: np.ndarray, low_prior: np.ndarray, prediction: np.ndarray
) -> Figure:
    """Return a Matplotlib figure from existing arrays; no recomputation."""
    from sage_avo.visualization.figures import plot_inversion_comparison

    first = _finite_array("truth", truth)
    second = _finite_array("low_prior", low_prior, first.shape)
    third = _finite_array("prediction", prediction, first.shape)
    if first.ndim != 3 or first.shape[0] != 3:
        raise ValueError("elastic arrays must be [3,time,trace] with Vp/Vs/density channels")
    return plot_inversion_comparison(first, second, third)


def substitute_calibrated_fluid(
    brine_elastic: np.ndarray,
    input_porosity: np.ndarray,
    shaliness: np.ndarray,
    co2_saturation: np.ndarray,
    depth_m: np.ndarray,
    region_mask: np.ndarray,
    *,
    calibration: CalibratedDryFrameModel | None = None,
    physics: FluidRockPhysics | None = None,
    support: SupportAcceptanceContract | None = None,
) -> CalibratedFluidApplication:
    """Apply approved Candidate B inside a declared region with conservative support QC.

    This per-pixel gate is not the complete Stage-03 realization-acceptance test.
    Caller must provide an approved calibration and matching support contract.
    """
    from sage_avo.geology.fluid_calibration import (
        CalibratedDryFrameModel,
        FluidRockPhysics,
        calibrated_differential_gassmann_substitution,
        poisson_ratio_from_moduli,
    )
    from sage_avo.geology.rock_physics import elastic_moduli_gpa
    from sage_avo.geology.support import SupportAcceptanceContract

    if not isinstance(calibration, CalibratedDryFrameModel):
        raise ValueError("calibration must be an approved CalibratedDryFrameModel")
    if not isinstance(physics, FluidRockPhysics):
        raise ValueError("physics must be an explicit FluidRockPhysics")
    if not isinstance(support, SupportAcceptanceContract):
        raise ValueError("support must be an approved SupportAcceptanceContract")
    if support.calibration_id != calibration.calibration_id:
        raise ValueError("support and dry-frame calibration identifiers disagree")
    baseline = _finite_array("brine_elastic", brine_elastic)
    if baseline.ndim != 3 or baseline.shape[0] != 3:
        raise ValueError("brine_elastic must have shape [3,time,trace]")
    vp, vs, density = _elastic_inputs(*baseline)
    shape = vp.shape
    if np.asarray(region_mask).dtype != np.dtype(bool):
        raise ValueError("region_mask must be explicitly boolean")
    region = np.asarray(region_mask)
    if region.shape != shape or not region.any():
        raise ValueError("region_mask must be nonempty and match [time,trace]")
    phi = _finite_array("input_porosity", input_porosity, shape)
    shale = _finite_array("shaliness", shaliness, shape)
    saturation = _finite_array("co2_saturation", co2_saturation, shape)
    depth = _finite_array("depth_m", depth_m, shape)
    if any(np.any((array < 0) | (array > 1)) for array in (phi, shale, saturation)):
        raise ValueError("porosity, shaliness and saturation must be fractions in [0,1]")
    if np.any(depth <= 0) or np.any(saturation[~region] != 0):
        raise ValueError("depth must be positive and saturation must be zero outside region_mask")
    active = region & (saturation > 0)
    output = baseline.copy()
    if not active.any():
        return CalibratedFluidApplication(output, region.copy(), calibration.calibration_id, 0.0)
    result = calibrated_differential_gassmann_substitution(
        vp[active],
        vs[active],
        density[active],
        phi[active],
        shale[active],
        saturation[active],
        depth[active],
        calibration,
        physics,
    )
    distance = result.nearest_calibration_distance
    if not np.isfinite(distance).all() or np.any(distance > support.nearest_neighbor_threshold):
        raise ValueError("substitution lies outside declared dry-frame calibration support")
    if np.any(
        (depth[active] < support.physical_depth_domain_m[0])
        | (depth[active] > support.physical_depth_domain_m[1])
    ):
        raise ValueError("substitution depth lies outside declared physical support")
    effective = result.effective_porosity
    if np.any(
        (effective < support.physical_porosity_domain_fraction[0])
        | (effective > support.physical_porosity_domain_fraction[1])
    ):
        raise ValueError("effective porosity lies outside declared physical support")
    ratio = result.dry_bulk_gpa / result.frame_shear_gpa
    poisson = poisson_ratio_from_moduli(result.dry_bulk_gpa, result.frame_shear_gpa)
    if np.any(
        (ratio < support.dry_bulk_to_shear_range[0]) | (ratio > support.dry_bulk_to_shear_range[1])
    ) or np.any(
        (poisson < support.dry_poisson_ratio_range[0])
        | (poisson > support.dry_poisson_ratio_range[1])
    ):
        raise ValueError("predicted dry frame lies outside declared physical support")
    target = np.stack((result.elastic.vp, result.elastic.vs, result.elastic.density))
    _, target_shear = elastic_moduli_gpa(*target)
    if np.max(np.abs(target_shear - result.rf_shear_gpa)) > support.maximum_fixed_shear_error_gpa:
        raise ValueError("fluid substitution violated fixed-shear support tolerance")
    output[:, active] = target
    return CalibratedFluidApplication(
        output, region.copy(), calibration.calibration_id, float(np.max(distance))
    )
