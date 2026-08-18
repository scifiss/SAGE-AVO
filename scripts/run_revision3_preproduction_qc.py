#!/usr/bin/env python3
"""Run the read-only Madagascar and fluid-magnitude v003 production gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from sage_avo.config import load_config
from sage_avo.experiments.manifest import write_json
from sage_avo.forward import reflectivity_gather_madagascar
from sage_avo.forward.torch_forward import exact_zoeppritz_pp
from sage_avo.forward.zoeppritz import reflectivity_gather
from sage_avo.geology.rock_physics import (
    elastic_moduli_gpa,
    inverse_gassmann_dry_bulk,
    local_inverse_gassmann_substitution,
    mineral_bulk_modulus_vrh,
)


REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATION_ID = "v003_validation8_stage01v003"
PERCENTILES = (1.0, 5.0, 50.0, 95.0, 99.0)


INTERFACE_CASES = {
    "weak_positive": (2500.0, 1400.0, 2.25, 2600.0, 1450.0, 2.28),
    "weak_negative": (2600.0, 1450.0, 2.28, 2500.0, 1400.0, 2.25),
    "strong_ordinary": (2200.0, 1200.0, 2.10, 3000.0, 1650.0, 2.40),
    "near_critical": (2400.0, 1300.0, 2.15, 3400.0, 1850.0, 2.45),
    "postcritical": (2200.0, 1200.0, 2.10, 3600.0, 1950.0, 2.50),
}


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(actual, dtype=float) - np.asarray(reference, dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1e-15),
    }


def madagascar_qc(destination: Path) -> dict[str, Any]:
    angles = np.arange(3.0, 46.0, dtype=float)
    records: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, float | str]] = []
    aggregate: dict[str, list[np.ndarray]] = {
        "madagascar": [],
        "numpy": [],
        "torch": [],
    }
    for name, values in INTERFACE_CASES.items():
        vp1, vs1, rho1, vp2, vs2, rho2 = values
        vp = np.asarray([[vp1], [vp1], [vp2], [vp2], [vp2]])
        vs = np.asarray([[vs1], [vs1], [vs2], [vs2], [vs2]])
        density = np.asarray([[rho1], [rho1], [rho2], [rho2], [rho2]])
        madagascar_full = reflectivity_gather_madagascar(vp, vs, density, angles)
        numpy_full = reflectivity_gather(vp, vs, density, angles)
        torch_full = (
            exact_zoeppritz_pp(
                torch.tensor(vp[None], dtype=torch.float64),
                torch.tensor(vs[None], dtype=torch.float64),
                torch.tensor(density[None], dtype=torch.float64),
                torch.tensor(angles, dtype=torch.float64),
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        coefficients = {
            "madagascar": madagascar_full[:, 2, 0].astype(float),
            "numpy": numpy_full[:, 2, 0].astype(float),
            "torch": torch_full[:, 2, 0].astype(float),
        }
        for key, values_array in coefficients.items():
            aggregate[key].append(values_array)
        critical = (
            float(np.degrees(np.arcsin(vp1 / vp2))) if vp2 > vp1 else None
        )
        record: dict[str, Any] = {
            "case": name,
            "upper": {"vp_m_s": vp1, "vs_m_s": vs1, "density_g_cc": rho1},
            "lower": {"vp_m_s": vp2, "vs_m_s": vs2, "density_g_cc": rho2},
            "pp_critical_angle_degrees": critical,
            "contains_postcritical_samples": bool(
                critical is not None and np.any(angles > critical)
            ),
            "madagascar_vs_numpy": _error_metrics(
                coefficients["madagascar"], coefficients["numpy"]
            ),
            "madagascar_vs_torch": _error_metrics(
                coefficients["madagascar"], coefficients["torch"]
            ),
            "numpy_vs_torch": _error_metrics(
                coefficients["numpy"], coefficients["torch"]
            ),
            "madagascar_off_interface_maximum_absolute": float(
                np.max(np.abs(np.delete(madagascar_full, 2, axis=1)))
            ),
            "polarity_sign_mismatch_count": int(
                np.count_nonzero(
                    np.sign(coefficients["madagascar"])
                    != np.sign(coefficients["numpy"])
                )
            ),
        }
        records.append(record)
        for index, angle in enumerate(angles):
            coefficient_rows.append(
                {
                    "case": name,
                    "angle_degrees": float(angle),
                    "madagascar": float(coefficients["madagascar"][index]),
                    "numpy": float(coefficients["numpy"][index]),
                    "torch": float(coefficients["torch"][index]),
                }
            )
    concatenated = {key: np.concatenate(value) for key, value in aggregate.items()}
    summary = {
        "convention": (
            "Madagascar sfzoeppritz2 icoef=4 incp=y outp=y refl=y; real reflected "
            "P displacement coefficient; angle-fast RSF transposed to [angle,time,trace]; "
            "no polarity flip and no interface sample shift"
        ),
        "angles_degrees": [3.0, 45.0, 1.0],
        "cases": records,
        "aggregate": {
            "madagascar_vs_numpy": _error_metrics(
                concatenated["madagascar"], concatenated["numpy"]
            ),
            "madagascar_vs_torch": _error_metrics(
                concatenated["madagascar"], concatenated["torch"]
            ),
            "numpy_vs_torch": _error_metrics(
                concatenated["numpy"], concatenated["torch"]
            ),
        },
    }
    pd.DataFrame(coefficient_rows).to_csv(
        destination / "madagascar_numpy_torch_coefficients.csv", index=False
    )
    write_json(destination / "madagascar_numpy_torch_comparison.json", summary)
    return summary


def _fluid_rows(realization_directory: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in sorted(realization_directory.glob("realization_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            plume = np.asarray(archive["plume_mask"], dtype=bool)
            rows, columns = np.where(plume)
            brine = np.asarray(archive["elastic_brine"], dtype=float)
            corrected = np.asarray(archive["elastic"], dtype=float)
            delta = corrected - brine
            records.append(
                pd.DataFrame(
                    {
                        "realization_id": int(archive["realization_id"]),
                        "time_index": rows,
                        "trace_index": columns,
                        "co2_saturation": archive["co2_saturation"][plume],
                        "porosity": archive["porosity"][plume],
                        "shaliness_delta": archive["delta"][plume],
                        "vp_brine_m_s": brine[0][plume],
                        "vs_brine_m_s": brine[1][plume],
                        "density_brine_g_cc": brine[2][plume],
                        "vp_co2_m_s": corrected[0][plume],
                        "vs_co2_m_s": corrected[1][plume],
                        "density_co2_g_cc": corrected[2][plume],
                        "delta_vp_m_s": delta[0][plume],
                        "delta_vs_m_s": delta[1][plume],
                        "delta_density_g_cc": delta[2][plume],
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def _binned_changes(data: pd.DataFrame, variable: str) -> list[dict[str, Any]]:
    bins = pd.qcut(data[variable], q=5, duplicates="drop")
    rows = []
    for interval, group in data.groupby(bins, observed=True):
        rows.append(
            {
                "minimum": float(interval.left),
                "maximum": float(interval.right),
                "count": int(len(group)),
                "median_delta_vp_m_s": float(group["delta_vp_m_s"].median()),
                "median_delta_vs_m_s": float(group["delta_vs_m_s"].median()),
                "median_delta_density_g_cc": float(
                    group["delta_density_g_cc"].median()
                ),
            }
        )
    return rows


def _feasibility_qc(
    realization_directory: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    fluid = config["fluid_substitution"]
    adjusted_fractions = []
    plume_adjusted = 0
    plume_total = 0
    dry_bulk_projected = 0
    output_bound_clips = np.zeros(3, dtype=int)
    zero_errors = np.zeros(3, dtype=float)
    plume_saturation_at_configured_minimum = 0
    plume_saturation_at_configured_maximum = 0
    plume_porosity_at_generation_floor = 0
    for path in sorted(realization_directory.glob("realization_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            brine = np.asarray(archive["elastic_brine"], dtype=float)
            porosity = np.asarray(archive["porosity"], dtype=float)
            shaliness = np.asarray(archive["delta"], dtype=float)
            saturation = np.asarray(archive["co2_saturation"], dtype=float)
            plume = np.asarray(archive["plume_mask"], dtype=bool)
            saturation_limits = tuple(float(value) for value in fluid["co2_saturation"])
            plume_saturation_at_configured_minimum += int(
                np.count_nonzero(
                    plume & np.isclose(saturation, saturation_limits[0], atol=1e-7)
                )
            )
            plume_saturation_at_configured_maximum += int(
                np.count_nonzero(
                    plume & np.isclose(saturation, saturation_limits[1], atol=1e-7)
                )
            )
            plume_porosity_at_generation_floor += int(
                np.count_nonzero(plume & np.isclose(porosity, 0.002, atol=1e-8))
            )
            kwargs = {
                name: fluid[name]
                for name in (
                    "quartz_bulk_modulus_gpa",
                    "clay_bulk_modulus_gpa",
                    "brine_bulk_modulus_gpa",
                    "co2_bulk_modulus_gpa",
                    "brine_density_g_cc",
                    "co2_density_g_cc",
                    "brie_exponent",
                    "compatibility_margin",
                )
            }
            result = local_inverse_gassmann_substitution(
                brine[0], brine[1], brine[2], porosity, shaliness, saturation, **kwargs
            )
            zero_result = local_inverse_gassmann_substitution(
                brine[0],
                brine[1],
                brine[2],
                porosity,
                shaliness,
                np.zeros_like(saturation),
                **kwargs,
            )
            for channel, values in enumerate(
                (zero_result.elastic.vp, zero_result.elastic.vs, zero_result.elastic.density)
            ):
                zero_errors[channel] = max(
                    zero_errors[channel], float(np.max(np.abs(values - brine[channel])))
                )
            saturated_bulk, _ = elastic_moduli_gpa(brine[0], brine[1], brine[2])
            mineral_prior = mineral_bulk_modulus_vrh(
                shaliness,
                quartz_bulk_modulus_gpa=float(fluid["quartz_bulk_modulus_gpa"]),
                clay_bulk_modulus_gpa=float(fluid["clay_bulk_modulus_gpa"]),
            )
            phi = np.clip(porosity, 1e-4, 0.6)
            denominator = 1.0 / np.maximum(saturated_bulk, 1e-8) - (
                phi / float(fluid["brine_bulk_modulus_gpa"])
            )
            limit = np.where(
                denominator > 1e-10,
                (1.0 - phi) / denominator,
                mineral_prior,
            )
            effective = np.maximum(
                saturated_bulk * 1.001,
                np.minimum(
                    mineral_prior,
                    float(fluid["compatibility_margin"]) * limit,
                ),
            )
            adjusted = effective < mineral_prior * (1.0 - 1e-8)
            adjusted_fractions.append(float(np.mean(adjusted)))
            plume_adjusted += int(np.count_nonzero(adjusted & plume))
            plume_total += int(np.count_nonzero(plume))
            dry_raw = inverse_gassmann_dry_bulk(
                saturated_bulk,
                phi,
                effective,
                float(fluid["brine_bulk_modulus_gpa"]),
            )
            dry_bulk_projected += int(
                np.count_nonzero(
                    plume
                    & (
                        ~np.isfinite(dry_raw)
                        | (dry_raw < 0.0)
                        | (dry_raw > 0.999 * effective)
                    )
                )
            )
            raw = (result.elastic.vp, result.elastic.vs, result.elastic.density)
            bounds = (
                fluid["vp_bounds_m_s"],
                fluid["vs_bounds_m_s"],
                fluid["density_bounds_g_cc"],
            )
            for channel, (values, limits) in enumerate(zip(raw, bounds)):
                output_bound_clips[channel] += int(
                    np.count_nonzero(
                        plume & ((values < float(limits[0])) | (values > float(limits[1])))
                    )
                )
    return {
        "compatibility_margin": float(fluid["compatibility_margin"]),
        "adjusted_mineral_fraction_by_realization": adjusted_fractions,
        "adjusted_mineral_fraction_range": [
            min(adjusted_fractions),
            max(adjusted_fractions),
        ],
        "plume_pixels_using_mineral_feasibility_projection": plume_adjusted,
        "plume_pixel_count": plume_total,
        "plume_adjusted_fraction": plume_adjusted / plume_total,
        "plume_pixels_requiring_dry_bulk_clipping": dry_bulk_projected,
        "plume_pixels_requiring_final_vp_vs_density_bound_clipping": (
            output_bound_clips.tolist()
        ),
        "plume_saturation_at_configured_minimum_count": (
            plume_saturation_at_configured_minimum
        ),
        "plume_saturation_at_configured_maximum_count": (
            plume_saturation_at_configured_maximum
        ),
        "plume_porosity_at_generation_floor_count": plume_porosity_at_generation_floor,
        "zero_saturation_maximum_absolute_vp_vs_density_error": zero_errors.tolist(),
        "formula": {
            "compatibility_denominator": "D = 1/Ksat - phi/Kbrine",
            "mineral_limit": "Klimit = (1 - phi)/D when D > 1e-10",
            "effective_mineral": (
                "Kmin_eff = max(1.001*Ksat, min(Kmin_VRH, "
                "compatibility_margin*Klimit))"
            ),
            "reported_0.9421_to_0.9603_meaning": (
                "fraction of all image pixels whose Kmin_eff is below Kmin_VRH; "
                "it is not a multiplicative compatibility factor"
            ),
        },
    }


def fluid_qc(
    realization_directory: Path,
    config: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    data = _fluid_rows(realization_directory)
    outliers = data[data["delta_vp_m_s"] < -600.0].copy()
    data.to_csv(destination / "fluid_all_plume_pixels.csv", index=False)
    outliers.to_csv(destination / "fluid_delta_vp_below_minus600.csv", index=False)
    percentile_table = data[
        ["delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc"]
    ].quantile(np.asarray(PERCENTILES) / 100.0)
    percentile_table.index = [f"p{value:g}" for value in PERCENTILES]
    percentile_table.to_csv(destination / "fluid_delta_percentiles.csv")
    correlations: dict[str, Any] = {}
    for predictor in ("co2_saturation", "porosity"):
        correlations[predictor] = {}
        for response in (
            "delta_vp_m_s",
            "delta_vs_m_s",
            "delta_density_g_cc",
        ):
            correlations[predictor][response] = {
                "pearson": float(data[predictor].corr(data[response])),
                "spearman": float(spearmanr(data[predictor], data[response]).statistic),
            }

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    responses = (
        ("delta_vp_m_s", "ΔVp (m/s)"),
        ("delta_vs_m_s", "ΔVs (m/s)"),
        ("delta_density_g_cc", "Δdensity (g/cc)"),
    )
    scatter = None
    for axis, (column, label) in zip(axes, responses):
        scatter = axis.scatter(
            data["co2_saturation"],
            data[column],
            c=data["porosity"],
            cmap="viridis",
            s=11,
            alpha=0.65,
            linewidths=0,
        )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_xlabel("CO₂ saturation")
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
    assert scatter is not None
    figure.colorbar(scatter, ax=axes, label="Porosity", shrink=0.86)
    figure.suptitle(
        "Revision-3 fluid-magnitude gate — all plume pixels in 8 validation realizations"
    )
    figure_path = destination / "fluid_saturation_delta_colored_by_porosity.png"
    figure.savefig(figure_path, dpi=260, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "realization_count": int(data["realization_id"].nunique()),
        "plume_pixel_count": int(len(data)),
        "percentiles": percentile_table.to_dict(orient="index"),
        "delta_vp_below_minus600_count": int(len(outliers)),
        "delta_vp_below_minus600_fraction": float(len(outliers) / len(data)),
        "outlier_file": "fluid_delta_vp_below_minus600.csv",
        "correlations": correlations,
        "saturation_quintiles": _binned_changes(data, "co2_saturation"),
        "porosity_quintiles": _binned_changes(data, "porosity"),
        "feasibility_and_clipping": _feasibility_qc(realization_directory, config),
        "figure": figure_path.name,
    }
    write_json(destination / "fluid_magnitude_qc.json", summary)
    return summary


def main() -> None:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    validation_root = (
        Path(paths["private_artifact_root"]) / "revision3" / VALIDATION_ID
    )
    destination = validation_root / "preproduction_qc"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (validation_root / "configs" / "synthetic_resolved.json").read_text()
    )
    madagascar = madagascar_qc(destination)
    fluid = fluid_qc(validation_root / "stage02" / "realizations", config, destination)
    combined = {
        "scope": "read-only pre-production gate on existing v003 validation artifacts",
        "madagascar_cross_validation": madagascar,
        "fluid_magnitude_qc": fluid,
    }
    write_json(destination / "preproduction_qc_summary.json", combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
