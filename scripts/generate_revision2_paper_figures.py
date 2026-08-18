#!/usr/bin/env python3
"""Generate CPU-only Revision-2 paper Figures 12--23 from Stage-02/03 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from sage_avo.config import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
PATHS_FILE = REPOSITORY / "configs" / "paths.yaml"
if not PATHS_FILE.exists():
    raise FileNotFoundError("Create ignored configs/paths.yaml from paths.example.yaml")
PRIVATE_ROOT = Path(load_config(PATHS_FILE)["private_artifact_root"])
STAGE02 = (
    PRIVATE_ROOT
    / "stage_artifacts"
    / "stage02"
    / "v002_production100_overlap"
    / "realizations"
)
STAGE03 = (
    PRIVATE_ROOT
    / "stage_artifacts"
    / "stage03"
    / "ds_v002_production100_multiscale"
    / "dataset"
)
OUTPUT = PRIVATE_ROOT / "figures" / "revision2" / "paper"
BANDS = ((3.0, 17.0), (17.0, 31.0), (31.0, 45.0))
BAND_NAMES = ("Near 3–17°", "Mid 17–31°", "Far 31–45°")
PROPERTY_NAMES = ("Vp", "Vs", "Density")
PROPERTY_UNITS = ("m/s", "m/s", "g/cc")
FACIES_CMAP = ListedColormap(("#495057", "#e9c46a", "#d62828"))
FACIES_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), FACIES_CMAP.N)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 9,
            "savefig.facecolor": "white",
        }
    )


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _stage02_path(realization_id: int) -> Path:
    return STAGE02 / f"realization_{realization_id:07d}.npz"


def _stage03_path(realization_id: int) -> Path:
    return STAGE03 / "realizations" / f"realization_{realization_id:07d}.npz"


def _extent(data: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    return (
        float(data["cdp"][0]),
        float(data["cdp"][-1]),
        float(data["time_ms"][-1]),
        float(data["time_ms"][0]),
    )


def _image(
    axis: plt.Axes,
    values: np.ndarray,
    data: dict[str, np.ndarray],
    title: str,
    *,
    cmap: str | matplotlib.colors.Colormap = "viridis",
    label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: matplotlib.colors.Normalize | None = None,
    horizons: bool = False,
) -> matplotlib.image.AxesImage:
    image = axis.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        extent=_extent(data),
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        interpolation="nearest",
    )
    if horizons:
        axis.plot(data["cdp"], data["horizon_top_ms"], color="white", lw=0.8)
        axis.plot(data["cdp"], data["horizon_base_ms"], color="white", lw=0.8)
    axis.set_title(title)
    axis.set_xlabel("CDP")
    axis.set_ylabel("TWT (ms)")
    if label:
        plt.colorbar(image, ax=axis, shrink=0.78, pad=0.02, label=label)
    return image


def _symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    limit = float(np.nanpercentile(np.abs(values), percentile))
    return max(limit, np.finfo(float).eps)


def _property_limits(data: dict[str, np.ndarray]) -> list[tuple[float, float]]:
    return [
        (
            float(np.nanpercentile(data["elastic"][channel], 1)),
            float(np.nanpercentile(data["elastic"][channel], 99)),
        )
        for channel in range(3)
    ]


def _save(figure: plt.Figure, number: int, stem: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"figure{number:02d}_{stem}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def figure12(data: dict[str, np.ndarray], realization_id: int) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    limit = _symmetric_limit(data["structural_stack"])
    _image(
        axes[0, 0],
        data["structural_stack"],
        data,
        "(a) Near/mid structural stack",
        cmap="gray",
        label="standardized amplitude",
        vmin=-limit,
        vmax=limit,
        horizons=True,
    )
    _image(
        axes[0, 1],
        data["structure_oriented"],
        data,
        "(b) PySeistr structure-oriented image",
        cmap="gray",
        label="standardized amplitude",
        vmin=-limit,
        vmax=limit,
        horizons=True,
    )
    dip_limit = _symmetric_limit(data["dip_pwd"], 98.5)
    _image(
        axes[1, 0],
        data["dip_pwd"],
        data,
        "(c) Recalculated two-pass PySeistr PWD dip",
        cmap="seismic",
        label="slope (sample/trace)",
        vmin=-dip_limit,
        vmax=dip_limit,
        horizons=True,
    )
    _image(
        axes[1, 1],
        data["rgt"],
        data,
        "(d) Coherently warped Stage-01 RGT",
        cmap="turbo",
        label="RGT coordinate",
        horizons=True,
    )
    figure.suptitle(
        f"Figure 12. Structure, PWD and warped RGT registration — realization {realization_id}",
        fontsize=13,
    )
    return _save(figure, 12, "structure_pwd_warped_rgt")


def figure13(data: dict[str, np.ndarray], realization_id: int) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    panels = (
        (data["delta"], "(a) DELTA = shaliness", "viridis", "fraction", None),
        (data["sand_probability"], "(b) P(sand) = 1 − DELTA", "viridis", "probability", None),
        (data["porosity"], "(c) Porosity", "magma", "fraction", None),
        (data["strat_fraction"], "(d) Reservoir stratigraphic fraction", "turbo", "fraction", None),
        (data["reservoir_mask"], "(e) Warped reservoir support", "gray_r", "mask", None),
        (data["segmentation"], "(f) Facies / plume labels", FACIES_CMAP, "0 shale, 1 sand, 2 CO₂", FACIES_NORM),
    )
    for axis, (array, title, cmap, label, norm) in zip(axes.flat, panels):
        _image(
            axis,
            array,
            data,
            title,
            cmap=cmap,
            label=label,
            norm=norm,
            horizons=True,
        )
    complement_error = float(np.max(np.abs(data["delta"] + data["sand_probability"] - 1.0)))
    figure.suptitle(
        f"Figure 13. Registered geological attributes — realization {realization_id}; "
        f"max |DELTA + P(sand) − 1| = {complement_error:.1e}",
        fontsize=13,
    )
    return _save(figure, 13, "geological_attributes")


def figure14(data: dict[str, np.ndarray], realization_id: int) -> Path:
    limits = _property_limits(data)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for channel, axis in enumerate(axes):
        _image(
            axis,
            data["elastic"][channel],
            data,
            f"({chr(97 + channel)}) {PROPERTY_NAMES[channel]}",
            cmap="viridis",
            label=PROPERTY_UNITS[channel],
            vmin=limits[channel][0],
            vmax=limits[channel][1],
            horizons=True,
        )
    figure.suptitle(
        f"Figure 14. Field-conditioned elastic realization after fluid substitution — {realization_id}",
        fontsize=13,
    )
    return _save(figure, 14, "elastic_properties")


def figure15(data: dict[str, np.ndarray], realization_id: int) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 12), constrained_layout=True)
    limits = _property_limits(data)
    for row in range(3):
        _image(
            axes[row, 0],
            data["elastic_brine"][row],
            data,
            f"({chr(97 + 3 * row)}) Brine {PROPERTY_NAMES[row]}",
            cmap="viridis",
            label=PROPERTY_UNITS[row],
            vmin=limits[row][0],
            vmax=limits[row][1],
            horizons=True,
        )
        _image(
            axes[row, 1],
            data["elastic"][row],
            data,
            f"({chr(98 + 3 * row)}) CO₂-substituted {PROPERTY_NAMES[row]}",
            cmap="viridis",
            label=PROPERTY_UNITS[row],
            vmin=limits[row][0],
            vmax=limits[row][1],
            horizons=True,
        )
        difference = data["elastic"][row] - data["elastic_brine"][row]
        limit = _symmetric_limit(difference, 99.5)
        _image(
            axes[row, 2],
            difference,
            data,
            f"({chr(99 + 3 * row)}) Substituted − brine",
            cmap="seismic",
            label=PROPERTY_UNITS[row],
            vmin=-limit,
            vmax=limit,
            horizons=True,
        )
        axes[row, 2].contour(
            data["cdp"],
            data["time_ms"],
            data["plume_mask"],
            levels=(0.5,),
            colors="black",
            linewidths=0.7,
        )
    figure.suptitle(
        f"Figure 15. Gassmann/Brie CO₂ substitution within plume support — realization {realization_id}",
        fontsize=13,
    )
    return _save(figure, 15, "gassmann_fluid_substitution")


def figure16(data: dict[str, np.ndarray], realization_id: int) -> Path:
    angles = data["angles_degrees"]
    requested = (3.0, 24.0, 45.0)
    indices = [int(np.argmin(np.abs(angles - angle))) for angle in requested]
    selected = data["avo_dense"][indices]
    limit = _symmetric_limit(selected, 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for axis, angle_index, panel in zip(axes, indices, "abc"):
        angle = float(angles[angle_index])
        _image(
            axis,
            data["avo_dense"][angle_index],
            data,
            f"({panel}) Exact PP response at {angle:g}°",
            cmap="seismic",
            label="amplitude",
            vmin=-limit,
            vmax=limit,
            horizons=True,
        )
    figure.suptitle(
        f"Figure 16. Dense-angle exact isotropic PP Zoeppritz response — realization {realization_id}",
        fontsize=13,
    )
    return _save(figure, 16, "exact_zoeppritz_dense_angles")


def figure17(data: dict[str, np.ndarray], realization_id: int) -> Path:
    limit = _symmetric_limit(np.concatenate((data["avo_clean"], data["avo"])), 99.5)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for column, name in enumerate(BAND_NAMES):
        _image(
            axes[0, column],
            data["avo_clean"][column],
            data,
            f"({chr(97 + column)}) Clean {name}",
            cmap="seismic",
            label="amplitude",
            vmin=-limit,
            vmax=limit,
            horizons=True,
        )
        _image(
            axes[1, column],
            data["avo"][column],
            data,
            f"({chr(100 + column)}) Observed {name} + 3% noise",
            cmap="seismic",
            label="amplitude",
            vmin=-limit,
            vmax=limit,
            horizons=True,
        )
    figure.suptitle(
        f"Figure 17. Production near/mid/far AVO bands — realization {realization_id}",
        fontsize=13,
    )
    return _save(figure, 17, "production_angle_bands")


def figure18(data: dict[str, np.ndarray], realization_id: int) -> Path:
    angles = data["angles_degrees"]
    reservoir = data["reservoir_mask"].astype(bool)
    reservoir_amplitude = np.where(reservoir[None], np.abs(data["avo_dense"]), -np.inf)
    _, sample, trace = np.unravel_index(
        int(np.argmax(reservoir_amplitude)), reservoir_amplitude.shape
    )
    gather = data["avo_dense"][:, :, trace].T
    limit = _symmetric_limit(gather, 99.5)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    image = axes[0].imshow(
        gather,
        aspect="auto",
        cmap="seismic",
        extent=(float(angles[0]), float(angles[-1]), float(data["time_ms"][-1]), float(data["time_ms"][0])),
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axes[0].axhline(float(data["time_ms"][sample]), color="black", lw=1.0, ls="--")
    axes[0].set(
        title=f"(a) Dense angle gather at CDP {int(data['cdp'][trace])}",
        xlabel="Angle (degrees)",
        ylabel="TWT (ms)",
    )
    plt.colorbar(image, ax=axes[0], shrink=0.8, label="amplitude")
    curve = data["avo_dense"][:, sample, trace]
    axes[1].plot(angles, curve, "o-", ms=2.8, lw=1.0, color="#264653", label="dense exact response")
    colors = ("#2a9d8f", "#e9c46a", "#e76f51")
    errors = []
    for band_index, ((minimum, maximum), name, color) in enumerate(
        zip(BANDS, BAND_NAMES, colors)
    ):
        selected = (angles >= minimum) & (angles <= maximum)
        dense_mean = float(curve[selected].mean())
        saved_mean = float(data["avo_clean"][band_index, sample, trace])
        errors.append(abs(dense_mean - saved_mean))
        axes[1].axvspan(minimum, maximum, color=color, alpha=0.10)
        axes[1].hlines(saved_mean, minimum, maximum, colors=color, lw=3, label=name)
    axes[1].axhline(0.0, color="0.5", lw=0.7)
    axes[1].set(
        title=f"(b) Band means at {float(data['time_ms'][sample]):.0f} ms",
        xlabel="Angle (degrees)",
        ylabel="amplitude",
    )
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle(
        f"Figure 18. Inclusive band averaging of the exact dense response — {realization_id}; "
        f"max saved/derived mismatch {max(errors):.1e}",
        fontsize=13,
    )
    return _save(figure, 18, "dense_response_band_averaging")


def figure19(realization_ids: list[int]) -> Path:
    loaded = [_load(_stage02_path(value)) for value in realization_ids]
    structural_limit = max(_symmetric_limit(item["structural_stack"], 99.0) for item in loaded)
    figure, axes = plt.subplots(2, len(realization_ids), figsize=(20, 8.2), constrained_layout=True)
    for column, (realization_id, data) in enumerate(zip(realization_ids, loaded)):
        metadata = json.loads(_stage02_path(realization_id).with_suffix(".json").read_text())
        faults = metadata["geology"]["deformation"]["fault_count"]
        plume = metadata["geology"]["fluid"]["plume_pixels"]
        _image(
            axes[0, column],
            data["structural_stack"],
            data,
            f"ID {realization_id}\n{faults} faults",
            cmap="gray",
            vmin=-structural_limit,
            vmax=structural_limit,
            horizons=True,
        )
        _image(
            axes[1, column],
            data["segmentation"],
            data,
            f"facies; {plume} plume pixels",
            cmap=FACIES_CMAP,
            norm=FACIES_NORM,
            horizons=True,
        )
    figure.suptitle(
        "Figure 19. Structural and facies diversity at fixed realization-ID quantiles",
        fontsize=13,
    )
    return _save(figure, 19, "structural_facies_diversity")


def figure20(realization_ids: list[int]) -> Path:
    loaded = [_load(_stage02_path(value)) for value in realization_ids]
    vp_min = min(float(np.percentile(item["elastic"][0], 1)) for item in loaded)
    vp_max = max(float(np.percentile(item["elastic"][0], 99)) for item in loaded)
    far_limit = max(_symmetric_limit(item["avo"][2], 99.5) for item in loaded)
    figure, axes = plt.subplots(2, len(realization_ids), figsize=(20, 8.2), constrained_layout=True)
    for column, (realization_id, data) in enumerate(zip(realization_ids, loaded)):
        _image(
            axes[0, column],
            data["elastic"][0],
            data,
            f"Vp — ID {realization_id}",
            cmap="viridis",
            vmin=vp_min,
            vmax=vp_max,
            horizons=True,
        )
        _image(
            axes[1, column],
            data["avo"][2],
            data,
            "Far AVO 31–45°",
            cmap="seismic",
            vmin=-far_limit,
            vmax=far_limit,
            horizons=True,
        )
    figure.suptitle(
        "Figure 20. Elastic and exact-physics AVO diversity at fixed realization-ID quantiles",
        fontsize=13,
    )
    return _save(figure, 20, "elastic_avo_diversity")


def figure21(
    split_ids: dict[str, list[int]],
    patch_index: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    names = ("train", "validation", "test")
    colors = ("#2a9d8f", "#e9c46a", "#e76f51")
    realization_counts = [len(split_ids[name]) for name in names]
    patch_counts = [int((patch_index["split"] == name).sum()) for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    bars = axes[0].bar(names, realization_counts, color=colors)
    axes[0].bar_label(bars)
    axes[0].set(title="(a) Realization-level split", ylabel="realizations")
    bars = axes[1].bar(names, patch_counts, color=colors)
    axes[1].bar_label(bars)
    axes[1].set(title="(b) Deterministic patch index", ylabel="patch rows")
    for row, (name, color) in enumerate(zip(names, colors)):
        ids = sorted(int(value) for value in split_ids[name])
        axes[2].scatter(ids, np.full(len(ids), row), s=28, color=color, label=name)
    axes[2].set_yticks(range(3), names)
    axes[2].set(title="(c) Disjoint realization IDs", xlabel="realization ID")
    axes[2].grid(axis="x", alpha=0.25)
    train = set(split_ids["train"])
    validation = set(split_ids["validation"])
    test = set(split_ids["test"])
    overlap = len((train & validation) | (train & test) | (validation & test))
    figure.suptitle(
        "Figure 21. Leakage-safe Stage-03 data contract — "
        f"overlap = {overlap}; normalization fit = {manifest['normalization_fit']}",
        fontsize=13,
    )
    return _save(figure, 21, "leakage_safe_dataset_split")


def _resize(array: np.ndarray, output_shape: tuple[int, int], order: int) -> np.ndarray:
    factors = (output_shape[0] / array.shape[0], output_shape[1] / array.shape[1])
    return zoom(array, factors, order=order)


def figure22(patch_index: pd.DataFrame, split_ids: dict[str, list[int]]) -> Path:
    realization_id = sorted(int(value) for value in split_ids["validation"])[0]
    rows = patch_index[
        (patch_index["split"] == "validation")
        & (patch_index["realization_id"] == realization_id)
    ]
    data = _load(_stage03_path(realization_id))
    selected = []
    for scale in range(3):
        candidates = rows[rows["scale_index"] == scale].copy()
        candidates["reservoir_support"] = candidates.apply(
            lambda row: np.count_nonzero(
                data["segmentation"][
                    int(row.top) : int(row.top + row.raw_height),
                    int(row.left) : int(row.left + row.raw_width),
                ]
            ),
            axis=1,
        )
        selected.append(
            candidates.sort_values(
                ["reservoir_support", "top", "left"],
                ascending=[False, True, True],
            ).iloc[0]
        )
    figure, axes = plt.subplots(3, 3, figsize=(14.5, 12), constrained_layout=True)
    vp_min, vp_max = np.percentile(data["elastic"][0], (1, 99))
    for column, row in enumerate(selected):
        top, left = int(row.top), int(row.left)
        height, width = int(row.raw_height), int(row.raw_width)
        overview = axes[0, column].imshow(
            data["elastic"][0], aspect="auto", cmap="viridis", vmin=vp_min, vmax=vp_max
        )
        axes[0, column].add_patch(
            Rectangle((left, top), width, height, fill=False, color="white", lw=1.5)
        )
        axes[0, column].set_title(
            f"({chr(97 + column)}) raw {height}×{width}; ID {realization_id}"
        )
        axes[0, column].set(xlabel="trace sample", ylabel="time sample")
        plt.colorbar(overview, ax=axes[0, column], shrink=0.75, label="Vp (m/s)")
        spatial = np.s_[top : top + height, left : left + width]
        vp = _resize(data["elastic"][0][spatial], (50, 100), order=1)
        image = axes[1, column].imshow(
            vp, aspect="auto", cmap="viridis", vmin=vp_min, vmax=vp_max
        )
        axes[1, column].set_title(f"({chr(100 + column)}) Vp resized to 50×100 (bilinear)")
        axes[1, column].set(xlabel="tensor trace", ylabel="tensor time")
        plt.colorbar(image, ax=axes[1, column], shrink=0.75, label="Vp (m/s)")
        facies = _resize(data["segmentation"][spatial], (50, 100), order=0)
        image = axes[2, column].imshow(
            facies, aspect="auto", cmap=FACIES_CMAP, norm=FACIES_NORM
        )
        axes[2, column].set_title(f"({chr(103 + column)}) labels resized to 50×100 (nearest)")
        axes[2, column].set(xlabel="tensor trace", ylabel="tensor time")
        plt.colorbar(image, ax=axes[2, column], shrink=0.75, label="0 shale, 1 sand, 2 CO₂")
    figure.suptitle(
        "Figure 22. Multiscale physical patches retain raw extent and resize metadata",
        fontsize=13,
    )
    return _save(figure, 22, "multiscale_patch_contract")


def figure23(split_ids: dict[str, list[int]]) -> tuple[Path, int]:
    test_ids = sorted(int(value) for value in split_ids["test"])
    realization_id = test_ids[len(test_ids) // 2]
    data = _load(_stage03_path(realization_id))
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for row in range(3):
        truth = data["elastic"][row]
        prior = data["low"][row]
        lower, upper = np.percentile(truth, (1, 99))
        image = axes[row, 0].imshow(
            truth, aspect="auto", cmap="viridis", vmin=lower, vmax=upper
        )
        axes[row, 0].set_title(f"({chr(97 + 3 * row)}) Truth {PROPERTY_NAMES[row]}")
        plt.colorbar(image, ax=axes[row, 0], shrink=0.75, label=PROPERTY_UNITS[row])
        image = axes[row, 1].imshow(
            prior, aspect="auto", cmap="viridis", vmin=lower, vmax=upper
        )
        axes[row, 1].set_title(f"({chr(98 + 3 * row)}) Truth-derived 2-Hz prior")
        plt.colorbar(image, ax=axes[row, 1], shrink=0.75, label=PROPERTY_UNITS[row])
        residual = truth - prior
        limit = _symmetric_limit(residual, 99.0)
        image = axes[row, 2].imshow(
            residual, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit
        )
        axes[row, 2].set_title(f"({chr(99 + 3 * row)}) Truth − prior")
        plt.colorbar(image, ax=axes[row, 2], shrink=0.75, label=PROPERTY_UNITS[row])
        for axis in axes[row]:
            axis.set(xlabel="trace sample", ylabel="time sample")
    figure.suptitle(
        f"Figure 23. Supplied low-frequency elastic prior contract — test realization {realization_id}",
        fontsize=13,
    )
    return _save(figure, 23, "two_hz_prior_truth_disclosure"), realization_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    _style()
    manifest02 = json.loads((STAGE02 / "manifest.json").read_text(encoding="utf-8"))
    manifest03 = json.loads((STAGE03 / "dataset_manifest.json").read_text(encoding="utf-8"))
    split_ids = json.loads((STAGE03 / "split_ids.json").read_text(encoding="utf-8"))
    patch_index = pd.read_csv(STAGE03 / "patch_index.csv")
    ids = sorted(int(value) for value in manifest02["realization_ids"])
    if manifest02["status"] != "complete" or len(ids) != 100:
        raise RuntimeError("Figures require the complete 100-realization Stage-02 corpus")
    if manifest03["status"] != "complete" or manifest03["realization_count"] != 100:
        raise RuntimeError("Figures require the complete 100-realization Stage-03 dataset")
    if tuple(tuple(manifest02["production_bands"][name]) for name in ("near", "mid", "far")) != BANDS:
        raise ValueError("Stage-02 angle bands do not match the Revision-2 production contract")
    representative_id = ids[len(ids) // 2]
    diversity_ids = [ids[index] for index in np.linspace(0, len(ids) - 1, 5, dtype=int)]
    representative = _load(_stage02_path(representative_id))

    paths: dict[int, Path] = {
        12: figure12(representative, representative_id),
        13: figure13(representative, representative_id),
        14: figure14(representative, representative_id),
        15: figure15(representative, representative_id),
        16: figure16(representative, representative_id),
        17: figure17(representative, representative_id),
        18: figure18(representative, representative_id),
        19: figure19(diversity_ids),
        20: figure20(diversity_ids),
        21: figure21(split_ids, patch_index, manifest03),
        22: figure22(patch_index, split_ids),
    }
    paths[23], test_id = figure23(split_ids)
    messages = {
        12: "Registration of the synthetic structural stack, two-pass PySeistr PWD dip, and coherently warped Stage-01 RGT.",
        13: "DELTA=shaliness, complementary sand probability, porosity, reservoir coordinate/support, and facies/plume registration.",
        14: "Field-conditioned Vp/Vs/density truth used by the production synthetic experiment.",
        15: "Reservoir-confined CO2 fluid substitution and its elastic-property changes relative to the brine state.",
        16: "Dense 3-45 degree primary synthetic response from the exact isotropic PP Zoeppritz operator.",
        17: "Inclusive 3-17, 17-31, and 31-45 degree stacks before and after configured 3% observation noise.",
        18: "Saved production band stacks equal inclusive averages of the dense exact-angle response.",
        19: "Structural/facies diversity under a fixed realization-ID quantile rule across the 100-member corpus.",
        20: "Elastic and far-angle AVO diversity under the same fixed realization-ID quantile rule.",
        21: "Realization-level 70/20/10 split, zero split overlap, and training-only normalization provenance.",
        22: "40x80, 50x100, and 64x128 raw patches mapped to 50x100 tensors with interpolation semantics preserved.",
        23: "Explicit disclosure that the supplied 2-Hz elastic prior is smoothed from synthetic truth.",
    }
    source_stages = {number: "02" if number <= 20 else "03" for number in paths}
    selections = {
        **{number: f"median sorted corpus ID {representative_id}" for number in range(12, 19)},
        19: f"fixed sorted-ID quantiles {diversity_ids}",
        20: f"fixed sorted-ID quantiles {diversity_ids}",
        21: "all 100 persisted realization IDs and all 20,000 patch rows",
        22: (
            "maximum reservoir-support validation patch per scale for the smallest "
            "sorted validation ID; top/left order breaks ties"
        ),
        23: f"median sorted test ID {test_id}",
    }
    from PIL import Image

    records: list[dict[str, Any]] = []
    for number, path in sorted(paths.items()):
        with Image.open(path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (300, 300))
        records.append(
            {
                "figure_number": number,
                "filename": path.name,
                "source_stage": source_stages[number],
                "source_artifacts": "completed Stage-02/03 production NPZ + manifests/index",
                "selection_rule": selections[number],
                "scientific_message": messages[number],
                "angle_bands_degrees": "near 3-17; mid 17-31; far 31-45",
                "forward_operator": "exact_pp_zoeppritz_numpy" if number in range(16, 21) else "not directly displayed",
                "structure_contract": "two-pass PySeistr PWD; coherently warped Stage-01 RGT" if number in (12, 19) else "not directly displayed",
                "prior_contract": "truth-derived 2-Hz low-frequency elastic prior" if number in (21, 22, 23) else "not directly displayed",
                "field_private_data_shown": False,
                "private_or_field_derived": True,
                "public_redistribution_needs_verification": True,
                "width_pixels": width,
                "height_pixels": height,
                "dpi_x": round(float(dpi[0]), 2),
                "dpi_y": round(float(dpi[1]), 2),
                "sha256": _sha256(path),
            }
        )
    index_path = OUTPUT / "paper_figure_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    print(pd.DataFrame(records)[["figure_number", "filename", "selection_rule"]].to_string(index=False))
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
