"""Build the public research notebooks from canonical cell sources.

This maintainer utility keeps notebook JSON deterministic. It is not part of the
scientific runtime; every code cell calls the installed ``sage_avo`` package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def write(name: str, cells: list) -> None:
    for index, cell in enumerate(cells):
        identity = f"{name}\0{index}\0{cell.cell_type}\0{cell.source}".encode()
        cell["id"] = hashlib.sha256(identity).hexdigest()[:16]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


COMMON_ROOT = """
from pathlib import Path

def find_repository_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "sage_avo").exists():
            return candidate
    raise RuntimeError("SAGE-AVO repository root not found; start the kernel within an installed checkout.")

ROOT = find_repository_root()
"""


write(
    "02_synthetic_avo_generation.ipynb",
    [
        md(
            """
            # 02 — Field-conditioned synthetic geology and exact AVO generation

            | Item | Definition |
            |---|---|
            | **Scientific purpose** | Turn the calibrated Stage-01 structural/elastic background into geologically diverse, physics-generated AVO realizations. |
            | **Inputs** | Stage-01 Vp, Vs, density, DELTA/P(sand), porosity, RGT, reservoir mask, stratigraphic fraction, blend weights, and the fitted reservoir elastic model. |
            | **Outputs** | Complete realization packages containing geology, brine and substituted elastic properties, dense-angle exact PP AVO, three angle stacks, PWD dip, masks, coordinates, and provenance. |
            | **Data availability** | Algorithms and configuration schemas are public. Licensed S01 inputs and generated arrays remain local. |
            | **Local data requirements** | `work_data_root` and `private_artifact_root` are defined in ignored `configs/paths.yaml`. Execution stops if the required Stage-01 artifacts are unavailable. |
            | **Software requirements** | `pip install -e ".[field,ml,notebooks]"`; Madagascar is optional and used only for independent reference-path cross-checking. |
            | **Approximate runtime** | Minutes per realization on CPU; the configured 100-realization production family is an offline generation job. |
            | **Pipeline position** | Consumes Notebook 01; produces the full realizations consumed by Notebook 03. |

            The main forward operator is the **exact PP Zoeppritz solution** followed by wavelet convolution and angle-domain mute/taper. Shuey/Aki–Richards intercept and gradient are compact diagnostics and later model features—not substitutes for the generation physics.
            """
        ),
        code(
            COMMON_ROOT
            + """
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.config import load_config, seed_everything
from sage_avo.experiments import (
    generate_stage02_dataset,
    load_stage01_background,
    load_stage02_manifest,
)
from sage_avo.forward import (
    ForwardConfig,
    forward_avo_dense,
    forward_avo_madagascar,
    forward_specification_from_mapping,
    madagascar_availability,
)

paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError(
        "Missing local configuration: create configs/paths.yaml from "
        "configs/paths.example.yaml and define the authorized input and artifact roots."
    )
paths = load_config(paths_file)
private_root = Path(paths["private_artifact_root"])
validation_root_text = os.getenv("SAGE_AVO_REVISION3_VALIDATION_ROOT", "").strip()
if validation_root_text:
    validation_root = Path(validation_root_text)
    workflow = json.loads((validation_root / "configs" / "synthetic_resolved.json").read_text())
    realization_dir = validation_root / "stage02" / "realizations"
    figure_dir = validation_root / "figures" / "stage02"
else:
    workflow = load_config(ROOT / "configs" / "synthetic_s01_v0032.yaml")
    support_contract = load_config(ROOT / "configs" / "revision331_support_acceptance.yaml")
    workflow["stage"].update({
        "name": "field_conditioned_synthetic_avo_v00331_support_aware",
        "geology_realization_count": 100,
        "observation_variants_per_geology": 1,
        "realization_count": 100,
        "realization_id_offset": 3_400_000,
        "member_master_seeds": list(range(3_400_000, 3_400_100)),
    })
    workflow["fluid_substitution"].update({
        "enabled": True,
        "mode": "calibrated_differential_gassmann",
        "calibration_id": "v0033_58a5fe39a11c4fe66431",
        "calibration_artifact": "derived/fluid_models_v0033/calibrated_dry_frame_scenario_ensemble.npz",
        "fluid_property_validation_artifact": "derived/fluid_models_v0033/fluid_property_validation.json",
    })
    workflow["support_aware_acceptance"] = support_contract
    workflow["outputs"].update({
        "version": "v00331_production100_support_aware",
        "directory": "synthetic/v00331_production100_support_aware/realizations",
    })
    realization_dir = private_root / "stage_artifacts" / "stage02" / workflow["outputs"]["version"] / "realizations"
    figure_dir = private_root / "figures" / "revision331" / "stage02_production"
seed_everything(int(workflow["stage"]["seed"]))
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            r"""
            ## 1. Stage-01 contract and conventions

            All channels share the Stage-01 time/CDP grid. `elastic_background` and `elastic_blend_weight` have shape `[3, time, trace]`; the other image channels have shape `[time, trace]`.

            The canonical convention is

            \[
            \mathrm{DELTA}=\text{shaliness},\qquad P(\mathrm{sand})=1-\mathrm{DELTA}.
            \]

            The interface stores both channels, enforces `DELTA + P(sand) = 1`, and records the convention in every realization manifest. The configured 0.30 sand threshold follows the calibrated Stage-01 reservoir probability distribution; it is not a generic 0.5 classifier threshold.
            """
        ),
        code(
            """
stage01, reservoir_model, source_hashes = load_stage01_background(
    paths["work_data_root"],
    workflow["inputs"]["dataset_id"],
    workflow["inputs"]["structure_version"],
)
contract = pd.DataFrame(
    [{"channel": name, "shape": value.shape, "dtype": value.dtype} for name, value in stage01.items()]
)
display(contract)
print(f"Hashed source artifacts: {len(source_hashes)}")
print(
    "Reservoir P(sand) range:",
    np.nanmin(stage01["sand_probability"][stage01["reservoir_mask"].astype(bool)]),
    np.nanmax(stage01["sand_probability"][stage01["reservoir_mask"].astype(bool)]),
)
"""
        ),
        md(
            """
            ## 2. Deterministic geological realization

            A geology-realization ID is its geological random seed. One coherent deformation field is applied to every Stage-01 channel, so horizons, RGT, facies, porosity, masks, and elastic background remain registered. The deformation combines smooth folds with optional finite-length fault displacement. Correlated Gaussian fields perturb P(sand), porosity, saturation, and coupled bulk/shear/density properties before forward modeling. Observation-variant IDs use separate post-forward seeds; every variant of one geology is kept in one ML split.

            The trained Stage-01 random-forest relationship maps `[DELTA, porosity, stratigraphic fraction]` to reservoir Vp/Vs/density. Warped regional elastic background is preserved outside the reservoir and blended only across the saved transition weights; this prevents the block artifacts produced by assigning a constant exterior.
            """
        ),
        code(
            """
print("Configured realizations:", workflow["stage"]["realization_count"])
print("Geological deformation parameters:", {
    key: value for key, value in workflow["geology"].items()
    if "fold" in key or "fault" in key
})
print("Sand facies P(sand) threshold:", workflow["geology"]["sand_facies_probability_threshold"])
print("Fluid substitution:", workflow["fluid_substitution"])
"""
        ),
        md(
            """
            ## 3. CO₂ scenario and fluid substitution

            CO₂ saturation is introduced only in connected, sufficiently thick reservoir sand. Fluid substitution is selected explicitly by the versioned configuration and recorded in each realization manifest. Production-eligible modes preserve a common dry frame and transfer the fluid-induced bulk-modulus and density response while keeping shear modulus invariant; they also require a validated pressure/temperature/fluid-property artifact. Compatibility modes that overwrite or project the empirical random-forest brine state are excluded from production claims. Both brine and substituted elastic cubes are retained for physical QC.
            """
        ),
        md(
            """
            ## 4. Exact dense-angle forward response

            For each elastic interface and each configured angle, the shared production specification solves the exact isotropic PP Zoeppritz system, retaining the real component of the complex post-critical solution. The configured wavelet bank is convolved with explicit constant-zero same-length boundaries, then the global-time front mute/taper is applied. Dense responses retain all 43 angles from 3° through 45°. The identical serialized specification is consumed by the differentiable Stage-04 operator.

            Production angle bands are centralized in configuration with declared shared endpoints: near `3–17°`, mid `17–31°`, and far `31–45°`. The overlap at 17° and 31° is intentional and hashed into the forward contract. Compact P/G representative angles are the band midpoints (`10°`, `24°`, `38°`) rather than an independent forward convention.
            """
        ),
        code(
            """
forward_definition = forward_specification_from_mapping(workflow)
display(
    pd.DataFrame(
        [{"band": b.name, "minimum_deg": b.minimum_degrees, "maximum_deg": b.maximum_degrees}
         for b in forward_definition.bands]
    )
)
print("Forward specification SHA-256:", forward_definition.sha256)
print("Production bands:", workflow["forward_model"]["bands"])
print("Post-forward observation perturbations:", workflow["observation_perturbations"])
print("Dense angles:", forward_definition.angles_degrees)
"""
        ),
        md(
            """
            ## 5. Generate the realization family

            The default call creates the complete configured family. `SAGE_AVO_STAGE02_LIMIT` creates an explicitly labeled `operator_validation_subset`; corpus-level analysis accepts only a manifest labeled as complete. `SAGE_AVO_REUSE_STAGE02=1` reopens an existing immutable artifact set.
            """
        ),
        code(
            """
limit_text = os.getenv("SAGE_AVO_STAGE02_LIMIT", "").strip()
realization_limit = int(limit_text) if limit_text else None
reuse = os.getenv("SAGE_AVO_REUSE_STAGE02", "0") == "1"
workers = int(os.getenv("SAGE_AVO_STAGE02_WORKERS", "1"))
manifest_path = realization_dir / "manifest.json"

if reuse and manifest_path.exists():
    manifest = load_stage02_manifest(manifest_path)
else:
    manifest = generate_stage02_dataset(
        config=workflow,
        paths=paths,
        output_directory=realization_dir,
        realization_limit=realization_limit,
        workers=workers,
        resume=reuse,
    )

display(pd.Series({key: manifest[key] for key in (
    "status", "requested_realizations", "generated_realizations", "exact_forward_operator"
)}).to_frame("value"))
"""
        ),
        md(
            """
            ## 6. Deterministic realization QC

            The representative realization is the smallest generated ID—a documented rule independent of visual appearance. The panels verify channel registration, the DELTA/P(sand) complement, plume support, corrected local fluid substitution, exact near/mid/far response, and recalculated PWD dip. RGT is coherently warped from Stage 01; dip is recalculated for structural QC rather than replacing the warped RGT graph coordinate.
            """
        ),
        code(
            """
representative_id = min(manifest["realization_ids"])
representative_path = realization_dir / f"realization_{representative_id:07d}.npz"
with np.load(representative_path, allow_pickle=False) as archive:
    realization = {name: archive[name] for name in archive.files}

panels = [
    (realization["sand_probability"], "P(sand)", "viridis"),
    (realization["delta"], "DELTA (shaliness)", "viridis_r"),
    (realization["porosity"], "Porosity", "viridis"),
    (realization["co2_saturation"], "CO₂ saturation", "magma"),
    (realization["elastic"][0], "Vp", "viridis"),
    (realization["elastic"][1], "Vs", "viridis"),
    (realization["elastic"][2], "Density", "viridis"),
    (realization["rgt"], "Warped RGT", "turbo"),
    (realization["avo"][0], "Near AVO", "gray"),
    (realization["avo"][1], "Mid AVO", "gray"),
    (realization["avo"][2], "Far AVO", "gray"),
    (realization["dip_pwd"], "Recalculated PWD dip", "coolwarm"),
]
fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
for axis, (array, title, cmap) in zip(axes.flat, panels):
    image = axis.imshow(array, aspect="auto", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("Trace")
    axis.set_ylabel("Time sample")
    fig.colorbar(image, ax=axis, shrink=0.72)
for axis in axes[:2].flat:
    top_sample = np.interp(realization["horizon_top_ms"], realization["time_ms"], np.arange(realization["time_ms"].size))
    base_sample = np.interp(realization["horizon_base_ms"], realization["time_ms"], np.arange(realization["time_ms"].size))
    axis.plot(top_sample, color="white", linewidth=0.8, label="warped T6")
    axis.plot(base_sample, color="black", linewidth=0.8, label="warped T7")
fig.suptitle(f"Stage-02 field-conditioned realization {representative_id}")
qc_path = figure_dir / "stage02_representative_realization.png"
fig.savefig(qc_path, dpi=300, bbox_inches="tight")
plt.show()

print("max |DELTA + P(sand) - 1| =", np.max(np.abs(
    realization["delta"] + realization["sand_probability"] - 1.0
)))
print("Saved figure:", qc_path)
"""
        ),
        md(
            """
            ## 7. Mandatory Stage-02/Stage-04 operator round trip

            The saved clean three-band AVO is regenerated from the saved truth elastic model with the differentiable Torch operator and the same hashed forward specification. This checks exact PP reflectivity—including post-critical complex slowness—wavelet convolution, global mute/taper, and inclusive band stacking. Reported tolerances are numerical, not visually judged.
            """
        ),
        code(
            """
import torch
from sage_avo.forward import forward_avo_three_band_spec_torch

elastic64 = torch.from_numpy(realization["elastic"].astype(np.float64))
reproduced = forward_avo_three_band_spec_torch(
    elastic64[0][None], elastic64[1][None], elastic64[2][None], forward_definition
)[0].detach().numpy()
difference = reproduced - realization["avo_clean"]
round_trip = {
    "maximum_absolute_error": float(np.max(np.abs(difference))),
    "rmse": float(np.sqrt(np.mean(difference**2))),
    "relative_rmse": float(
        np.sqrt(np.mean(difference**2))
        / max(np.sqrt(np.mean(realization["avo_clean"] ** 2)), 1e-15)
    ),
}
display(pd.Series(round_trip).to_frame("value"))
"""
        ),
        md(
            """
            ## 8. Madagascar reference-path cross-check

            If Madagascar is installed, the same elastic crop is passed through `sfzoeppritz2 → sftransp → sfricker1 → sftransp`. Correlation is the principal diagnostic because `sfricker1` and the NumPy implementation use different wavelet normalization conventions. This reference check does not change the backend recorded in the realization manifest.
            """
        ),
        code(
            """
availability = madagascar_availability()
print(availability)
if availability.available:
    crop = realization["elastic"][:, 80:130, 20:120]
    controlled_wavelet = forward_definition.wavelets[0]
    madagascar_definition = ForwardConfig(
        angles_degrees=forward_definition.angles_degrees,
        bands=forward_definition.bands,
        wavelet_hz=controlled_wavelet.peak_frequency_hz,
        dt_seconds=forward_definition.dt_seconds,
        wavelet_samples=controlled_wavelet.samples,
        apply_mute=forward_definition.apply_mute,
        mute_start=forward_definition.mute_start,
        mute_end=forward_definition.mute_end,
        taper_samples=forward_definition.taper_samples,
    )
    numpy_forward = forward_avo_dense(*crop, config=madagascar_definition)
    rsf_forward = forward_avo_madagascar(*crop, config=madagascar_definition)
    comparisons = []
    for band, numpy_stack, rsf_stack in zip(
        numpy_forward.band_names, numpy_forward.stacks, rsf_forward.stacks
    ):
        comparisons.append({
            "band": band,
            "correlation": np.corrcoef(numpy_stack.ravel(), rsf_stack.ravel())[0, 1],
            "standard_deviation_ratio_rsf_to_numpy": rsf_stack.std() / numpy_stack.std(),
        })
    display(pd.DataFrame(comparisons))
else:
    print("Madagascar cross-check skipped; exact NumPy Zoeppritz remains the configured operator.")
"""
        ),
        md("## 9. Saved-channel manifest"),
        code(
            """
channel_table = pd.DataFrame(
    [{"channel": name, **definition} for name, definition in manifest["channels"].items()]
)
display(channel_table)
"""
        ),
        md(
            """
            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | `realization_XXXXXXX.npz` | dense 43-angle AVO; 3-band AVO; 3-channel elastic; geological/structural masks | One deterministic field-conditioned geological and exact-physics experiment | Notebook 03 |
            | per-realization JSON | provenance + deformation/fluid parameters + QC | Reproducibility record | Notebooks 03 and 05 |
            | `manifest.json` | channel schema, hashes, IDs, conventions | Immutable Stage-02 dataset contract | Notebook 03 |

            ## Scientific checks

            - Shared deformation keeps geology, RGT, masks, and elastic fields registered.
            - `DELTA + P(sand) = 1` is checked numerically.
            - Elastic and AVO channels are finite and physically bounded in each saved QC record.
            - Zero saturation reproduces the local RF brine state; outside-plume values remain unchanged; dry-frame shear is retained.
            - Corrected CO₂ substitution is confined to connected reservoir sand; brine and substituted elastic cubes are both retained.
            - Exact dense-angle Zoeppritz is the primary operator; compact P/G approximations are not used to generate training observations.
            - Wavelet, convolution, mute, angle bands, and post-forward perturbations are persisted per realization.
            - Observation variants share an explicit geology split-group ID.
            - Optional Madagascar correlation checks the independent reference route.

            ## Next stage

            Notebook 03 consumes the immutable realization IDs and complete saved channels. It splits at the **realization level**, constructs the disclosed truth-derived low-frequency elastic prior, and extracts traceable multiscale patches without leakage.
            """
        ),
    ],
)


write(
    "03_ml_dataset_construction.ipynb",
    [
        md(
            """
            # 03 — Leakage-safe ML dataset construction

            | Item | Definition |
            |---|---|
            | **Scientific purpose** | Convert complete Stage-02 realizations into normalized, traceable training/validation/test tensors without realization leakage. |
            | **Inputs** | Stage-02 exact-physics realizations and manifest. |
            | **Outputs** | Immutable realization files with low-frequency priors, realization split IDs, training-only normalization, multiscale patch index, integrity report, and dataset manifest. |
            | **Data availability** | Schemas and algorithms are public; generated realization arrays remain local. |
            | **Local data requirements** | Ignored `configs/paths.yaml` and a completed or explicitly subset-labeled Stage-02 artifact directory. Execution stops if Stage-02 artifacts are unavailable. |
            | **Software requirements** | `pip install -e ".[ml,notebooks]"`. |
            | **Approximate runtime** | Minutes for indexing and prior construction; storage and time scale with realization count. |
            | **Pipeline position** | Consumes Notebook 02; supplies the immutable tensors used by Notebooks 04 and 05. |

            This experiment is **AVO-guided refinement of a supplied low-frequency elastic prior**. It is not unconstrained AVO-only absolute-property inversion.
            """
        ),
        code(
            COMMON_ROOT
            + """
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.config import load_config, seed_everything
from sage_avo.data import IndexedRealizationPatches
from sage_avo.experiments import build_stage03_dataset, validate_dataset_integrity

paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Missing local configuration: configs/paths.yaml (template: configs/paths.example.yaml).")
paths = load_config(paths_file)
validation_root_text = os.getenv("SAGE_AVO_REVISION3_VALIDATION_ROOT", "").strip()
if validation_root_text:
    validation_root = Path(validation_root_text)
    workflow = json.loads((validation_root / "configs" / "dataset_resolved.json").read_text())
    stage02_dir = validation_root / "stage02" / "realizations"
    dataset_dir = validation_root / "stage03" / "dataset"
    figure_dir = validation_root / "figures" / "stage03"
else:
    workflow = load_config(ROOT / "configs" / "ml_dataset_s01_v00331.yaml")
    private_root = Path(paths["private_artifact_root"])
    stage02_dir = private_root / "stage_artifacts" / "stage02" / workflow["inputs"]["synthetic_version"] / "realizations"
    dataset_dir = private_root / "stage_artifacts" / "stage03" / workflow["outputs"]["version"] / "dataset"
    figure_dir = private_root / "figures" / "revision3" / "stage03"
seed_everything(int(workflow["stage"]["seed"]))
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            """
            ## 1. Validate the realization contract

            Required channels are near/mid/far AVO, Vp/Vs/density targets, RGT, segmentation, DELTA, P(sand), porosity, plume mask, and valid mask. Shapes and finite values are checked before any split or patch is produced. Realization IDs—not patches—are the independent sampling units.
            """
        ),
        code(
            """
source_manifest_path = stage02_dir / "manifest.json"
if not source_manifest_path.exists():
    raise FileNotFoundError(f"Stage-02 manifest not found: {source_manifest_path}")
source_manifest = json.loads(source_manifest_path.read_text())
display(pd.Series({
    "source_status": source_manifest["status"],
    "realizations": source_manifest["generated_realizations"],
    "exact_forward_operator": source_manifest["exact_forward_operator"],
    "delta_convention": source_manifest["delta_convention"],
}).to_frame("value"))
"""
        ),
        md(
            """
            ## 2. Split before patching

            A seeded permutation assigns complete **geology realization groups** to train/validation/test (70/20/10 in production). Every wavelet/noise/observation variant of a geology inherits that group assignment. Patch coordinates are generated only afterward, so neither geological structure nor alternative observations of it can leak across splits.
            """
        ),
        md(
            r"""
            ## 3. Disclosed low-frequency prior

            The synthetic prior is derived from each target/truth elastic cube using a Gaussian approximation to a 2 Hz low-pass filter. With `dt = 0.004 s` and `sigma_constant = 0.133`,

            \[
            \sigma_t=\frac{0.133}{f_c\,\Delta t}=16.625\ \text{samples},
            \qquad \sigma_x=2\sigma_t=33.25\ \text{traces}.
            \]

            These exact parameters and the boundary mode are saved in `dataset_manifest.json`. Normalization is fitted from full **training realizations only**, then applied unchanged to validation and test.
            """
        ),
        code(
            """
prior = workflow["prior"]
sigma_time = prior["sigma_constant"] / (prior["cutoff_hz"] * prior["dt_seconds"])
sigma_trace = sigma_time * prior["lateral_sigma_ratio"]
display(pd.Series({**prior, "sigma_time_samples": sigma_time, "sigma_trace_samples": sigma_trace}).to_frame("value"))
"""
        ),
        md(
            """
            ## 4. Build the immutable dataset

            Production candidates deliberately cover facies boundaries, high dip, high RGT change, high AVA-gradient change, reservoir/background, and multiple depth bins with minimum separation and coordinate deduplication. Uniform random sampling remains an explicit control. Raw patches are sampled at 40×80, 50×100, and 64×128 physical extents, then resized to 50×100 tensors. Continuous channels use bilinear interpolation; class labels and masks use nearest-neighbor interpolation. Every row preserves observation and geology IDs, origin, raw size, tensor size, resize factors, global mute origin, wavelet ID, and convolution halo.
            """
        ),
        code(
            """
manifest = build_stage03_dataset(
    config=workflow,
    paths=paths,
    source_directory=stage02_dir,
    output_directory=dataset_dir,
)
integrity = validate_dataset_integrity(dataset_dir)
display(pd.Series(integrity).to_frame("value"))
"""
        ),
        md("## 5. Split, normalization, and patch metadata audit"),
        code(
            """
split_ids = json.loads((dataset_dir / "split_ids.json").read_text())
split_group_ids = json.loads((dataset_dir / "split_group_ids.json").read_text())
normalization = json.loads((dataset_dir / "normalization.json").read_text())
patch_index = pd.read_csv(dataset_dir / "patch_index.csv")

display(pd.DataFrame({name: pd.Series(values) for name, values in split_ids.items()}))
display(pd.DataFrame({name: pd.Series(values) for name, values in split_group_ids.items()}))
display(pd.DataFrame(normalization, index=["near/Vp", "mid/Vs", "far/density"]))
display(patch_index.groupby(["split", "raw_height", "raw_width"]).size().rename("patches").to_frame())
display(patch_index.head())

sets = {name: set(values) for name, values in split_ids.items()}
assert sets["train"].isdisjoint(sets["validation"])
assert sets["train"].isdisjoint(sets["test"])
assert sets["validation"].isdisjoint(sets["test"])
group_sets = {name: set(values) for name, values in split_group_ids.items()}
assert group_sets["train"].isdisjoint(group_sets["validation"])
assert group_sets["train"].isdisjoint(group_sets["test"])
assert group_sets["validation"].isdisjoint(group_sets["test"])
"""
        ),
        md(
            """
            ## 6. Tensor contract and representative patches

            For class-channel QC, one patch is selected in each split by a reproducible stratified rule: seeded random selection among patches containing sand or plume. This prevents an all-background panel while remaining independent of model performance. AVO and elastic values are normalized using the saved training statistics in the loader; RGT and categorical targets retain their native meanings. Metadata enables predictions to be traced back to the full realization.
            """
        ),
        code(
            """
datasets = {split: IndexedRealizationPatches(dataset_dir, split) for split in ("train", "validation", "test")}
rng = np.random.default_rng(int(workflow["stage"]["seed"]))
samples = {}
selected_indices = {}
for split, dataset in datasets.items():
    candidates = [index for index in range(len(dataset)) if (dataset[index]["segmentation"] > 0).any()]
    if not candidates:
        raise ValueError(f"No sand/plume patch is available for {split} QC")
    selected_indices[split] = int(rng.choice(candidates))
    samples[split] = dataset[selected_indices[split]]
print({split: len(dataset) for split, dataset in datasets.items()})
print("Seeded class-QC patch indices:", selected_indices)
print({key: tuple(value.shape) for key, value in samples["train"].items() if hasattr(value, "shape")})

columns = [
    ("avo", 0, "Near AVO"), ("avo", 1, "Mid AVO"), ("avo", 2, "Far AVO"),
    ("low", 0, "Low-frequency Vp"), ("target", 0, "Target Vp"),
    ("target", 1, "Target Vs"), ("target", 2, "Target density"),
    ("segmentation", None, "Facies/plume"), ("rgt", None, "RGT"),
]
fig, axes = plt.subplots(3, len(columns), figsize=(20, 7), constrained_layout=True)
for row, split in enumerate(("train", "validation", "test")):
    sample = samples[split]
    for col, (key, channel, title) in enumerate(columns):
        values = sample[key].numpy()
        panel = values[channel] if channel is not None else values
        cmap = "gray" if key == "avo" else ("tab10" if key == "segmentation" else "viridis")
        axes[row, col].imshow(panel, aspect="auto", cmap=cmap)
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if row == 0: axes[row, col].set_title(title)
        if col == 0: axes[row, col].set_ylabel(split)
qc_path = figure_dir / "stage03_split_patch_contract.png"
fig.savefig(qc_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved figure:", qc_path)
"""
        ),
        md("## 7. Distribution QC without patch-pooling claims"),
        code(
            """
rows = []
for split, dataset in datasets.items():
    for index in np.linspace(0, len(dataset) - 1, min(40, len(dataset)), dtype=int):
        sample = dataset[int(index)]
        for channel, name in enumerate(("Vp", "Vs", "density")):
            rows.append({
                "split": split,
                "property": name,
                "normalized_mean": float(sample["target"][channel].mean()),
                "normalized_std": float(sample["target"][channel].std()),
            })
distribution = pd.DataFrame(rows)
display(distribution.groupby(["split", "property"])[["normalized_mean", "normalized_std"]].agg(["mean", "std"]))
"""
        ),
        md(
            """
            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | `realizations/*.npz` | full images with AVO, truth, prior, RGT, masks/classes | Immutable whole-realization evaluation unit | Notebooks 04–05 |
            | `split_ids.json` | realization-ID lists | Leakage-safe partition | Notebooks 04–05 |
            | `split_group_ids.json` | geology-group ID lists | Keeps all observation variants of one geology together | Notebooks 04–05 |
            | `normalization.json` | 3-channel means/stds | Training-only normalization transform | Notebooks 04–05 |
            | `patch_index.csv` | one row per multiscale patch | Traceable sampling and resize metadata | Notebook 04 |
            | `dataset_manifest.json` | prior, channels, split, integrity | Complete ML task contract | Notebooks 04–05 |

            ## Scientific checks

            - Required channels, matching dimensions, and finite values are validated before splitting.
            - Train/validation/test observation IDs and geology-group IDs are asserted disjoint.
            - Patch rows are checked against their assigned realization split.
            - Diverse candidate categories/depth bins are counted; minimum separation and zero duplicate coordinates are checked.
            - Low-frequency priors are explicitly labeled truth-derived and their smoothing constants are saved.
            - Normalization statistics use training realizations only.
            - Continuous and categorical resize modes are separated, while raw physical sizes and scale factors remain in metadata.
            - Exact seismic physics loss is restricted to native 50×100 patches with truth halo/global sample origin; multiscale patches retain all supervised losses.

            ## Next stage

            Notebook 04 consumes the normalized near/mid/far AVO, truth-derived low-frequency Vp/Vs/density prior, RGT, valid masks, and elastic/segmentation targets. It trains controlled SAGE-AVO variants against the same immutable split and checkpoint rule.
            """
        ),
    ],
)


write(
    "04_sage_avo_training.ipynb",
    [
        md(
            """
            # 04 — Complete SAGE-AVO model and training

            | Item | Definition |
            |---|---|
            | **Scientific purpose** | Train a structure-aware graph/CNN model to refine a supplied low-frequency elastic prior using near/mid/far AVO. |
            | **Inputs** | Notebook-03 patch index, train-only normalization, realization splits, AVO, low prior, RGT, elastic targets, segmentation targets, and masks. |
            | **Outputs** | Criterion-specific fixed-objective, sampled, segmentation, whole-realization, periodic, and resumable final checkpoints with complete raw/weighted metric logs. |
            | **Data availability** | Architecture, losses, and orchestration are public; datasets and checkpoints remain private/local. |
            | **Local data requirements** | Completed Stage-03 artifacts and adequate PyTorch/PyG compute. Execution stops if the dataset contract is unavailable. |
            | **Software requirements** | `pip install -e ".[ml,notebooks]"` with PyTorch and PyTorch Geometric. |
            | **Approximate runtime** | Operator checks: seconds to minutes. Production 120-epoch SAGE-AVO training is a GPU-scale job. |
            | **Pipeline position** | Consumes Notebook 03; produces matched checkpoints and manifests evaluated in Notebook 05. |

            The implemented transport is a deterministic straight-path conditional residual flow from the low-frequency prior toward the target. It is **not** a calibrated probabilistic posterior. The graph module is PyTorch Geometric `TransformerConv` graph attention/message passing—not a full-image Vision Transformer.
            """
        ),
        code(
            COMMON_ROOT
            + """
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sage_avo.config import load_config, seed_everything
from sage_avo.data import IndexedRealizationPatches
from sage_avo.models import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.models.sage_avo import angular_features
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.engine import PhysicsNormalization, train_step
from sage_avo.training.flow import straight_path
from sage_avo.experiments.training import (
    curriculum_from_config,
    loss_weights_from_config,
    physics_settings_from_config,
)

workflow_path = ROOT / "configs" / "final_training_v00332d.yaml"
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Missing local configuration: configs/paths.yaml (template: configs/paths.example.yaml).")
paths = load_config(paths_file)
private_root = Path(paths["private_artifact_root"])
validation_root_text = os.getenv("SAGE_AVO_REVISION3_VALIDATION_ROOT", "").strip()
if validation_root_text:
    validation_root = Path(validation_root_text)
    workflow_path = validation_root / "configs" / "training_resolved.json"
    workflow = json.loads(workflow_path.read_text())
    dataset_dir = validation_root / "stage03" / "dataset"
    experiment_dir = validation_root / "stage04" / "sage_avo_s01_v003_stage01v003_validation8"
    figure_dir = validation_root / "figures" / "stage04"
else:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_revision332d_final_training import _configuration as resolve_final_training

    workflow, observability = resolve_final_training()
    final_contract = load_config(workflow_path)
    if final_contract["immutable_dataset"] != "ds_v00331_production100_support_aware":
        raise RuntimeError("Final training contract does not select the immutable v00331 dataset")
    dataset_dir = private_root / "stage_artifacts" / "stage03" / final_contract["immutable_dataset"] / "dataset"
    experiment_dir = private_root / "stage_artifacts" / "stage04" / "sage_avo_s01_v00332d_final_production"
    figure_dir = private_root / "figures" / "revision332d" / "stage04"
seed_everything(int(workflow["experiment"]["seed"]))
print_torch_runtime()
device = select_torch_device(require_cuda=True, context="Notebook 04 model execution")
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            """
            ## 1. Immutable data contract

            Each training item contains normalized low/mid/high AVO `[3,H,W]`, normalized low-frequency Vp/Vs/density `[3,H,W]`, normalized elastic target `[3,H,W]`, RGT `[H,W]`, segmentation `[H,W]`, valid mask `[1,H,W]`, and traceability metadata. Normalization is fitted only on training realizations and its elastic statistics are also installed in the model for physical-unit forward modeling and optional sampling guidance.
            """
        ),
        code(
            """
if not (dataset_dir / "dataset_manifest.json").exists():
    raise FileNotFoundError(f"Stage-03 dataset manifest not found: {dataset_dir / 'dataset_manifest.json'}")
dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
normalization = json.loads((dataset_dir / "normalization.json").read_text())
split_ids = json.loads((dataset_dir / "split_ids.json").read_text())
train_data = IndexedRealizationPatches(dataset_dir, "train")
validation_data = IndexedRealizationPatches(dataset_dir, "validation")
sample = train_data[0]
display(pd.Series({
    "train_patches": len(train_data),
    "validation_patches": len(validation_data),
    "split_unit": dataset_manifest["split_unit"],
    "prior_truth_derived": dataset_manifest["prior"]["truth_derived"],
}).to_frame("value"))
print({key: tuple(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)})
"""
        ),
        md(
            """
            ## 2. Compact AVO summaries

            The three stacks are retained as image channels. A least-squares line in `sin²(theta)` additionally yields intercept `P` and gradient `G`; near–2×mid+far supplies curvature. These summaries condition graph edges and node features. They are feature extraction—not the Stage-02 forward model.
            """
        ),
        code(
            """
representative_angles = tuple(float(value) for value in workflow["model"]["representative_angles_degrees"])
with torch.no_grad():
    angular, gradient = angular_features(sample["avo"].unsqueeze(0), representative_angles)
print("[near, mid, far, P, G, curvature] shape:", tuple(angular.shape))

fig, axes = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
for axis, panel, title in zip(
    axes,
    [sample["avo"][0], sample["avo"][1], sample["avo"][2], angular[0, 3], angular[0, 4]],
    ["Near", "Mid", "Far", "Intercept P", "Gradient G"],
):
    axis.imshow(panel, aspect="auto", cmap="coolwarm")
    axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
feature_path = figure_dir / "stage04_avo_feature_contract.png"
fig.savefig(feature_path, dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        md(
            """
            ## 3. CNN, RGT-steered graph, and reinjection

            1. A CNN encodes the current elastic state, time, three AVO bands, and low-frequency prior.
            2. Every image sample is a graph node.
            3. Edges include vertical trace neighbors and bidirectional lateral neighbors. For an RGT-steered edge, the adjacent-trace endpoint is chosen within ±3 time samples by minimum RGT mismatch; the no-RGT control uses Cartesian lateral neighbors.
            4. Edge attributes decrease with local AVO-gradient contrast.
            5. Two `TransformerConv` layers perform graph attention/message passing and return learned final-layer attention.
            6. Graph features are reshaped to the image grid, reinjected into CNN features, and decoded into elastic transport velocity; a second decoder predicts shale/sand/plume classes.

            The canonical architecture uses a two-block CNN, an RGT-steered dynamic graph, two four-head `TransformerConv` layers, graph-to-image residual reinjection, an elastic velocity decoder, and a convolutional segmentation decoder.
            """
        ),
        code(
            """
model_config = workflow["model"]
full = build_sage_avo_variant("full", **sage_avo_model_kwargs(workflow)).to(device).eval()
full.set_norm_stats(normalization)
display(pd.Series({
    "graph_mode": full.graph_mode,
    "trainable_parameters": sum(parameter.numel() for parameter in full.parameters()),
    "graph_layers": model_config["graph_layers"],
    "graph_heads": model_config["graph_heads"],
}).to_frame("value"))
with torch.no_grad():
    state = sample["low"].unsqueeze(0).to(device)
    output = full(
        state,
        torch.zeros(1, device=device),
        sample["avo"].unsqueeze(0).to(device),
        state,
        sample["rgt"].unsqueeze(0).to(device),
    )
print("elastic velocity:", tuple(output.velocity.shape))
print("segmentation logits:", tuple(output.segmentation_logits.shape))
print("graph embedding:", tuple(output.embeddings.shape))
print("directed graph edges:", output.edge_indices[0].shape[1])
print("edge attributes:", tuple(output.edge_weights[0].shape))
print("learned attention:", tuple(output.attention_weights[0].shape))
"""
        ),
        md(
            r"""
            ## 4. Deterministic conditional residual transport

            Training samples a time `t ~ Uniform(0,1)` and constructs

            \[
            x_t=(1-t)x_{low}+t y,\qquad u^*=y-x_{low}.
            \]

            The network predicts the straight-path velocity conditioned on AVO, the supplied prior, and RGT. Inference starts at `x_low` and integrates the learned velocity from `t=0` to `1` with Heun/RK2 steps. Optional physics guidance differentiates exact-PP AVO mismatch with respect to the current elastic state and corrects selected trajectory steps. The production configuration sets `guidance_scale=0.0`. No stochastic base distribution or posterior calibration is implemented.
            """
        ),
        code(
            """
t = torch.tensor([0.35], device=device)
low_device = sample["low"].unsqueeze(0).to(device)
target_device = sample["target"].unsqueeze(0).to(device)
state, target_velocity = straight_path(
    low_device, target_device, t
)
assert torch.allclose(target_velocity, target_device - low_device)
print("state and velocity:", tuple(state.shape), tuple(target_velocity.shape))
"""
        ),
        md(
            r"""
            ## 5. Complete training objective

            \[
            L = 0.65L_{flow}+0.20L_{property}+w_{ssim}L_{SSIM}
                +0.30L_{seg}+w_{phys}L_{Zoeppritz}.
            \]

            `L_flow` fits residual velocity and uses a density weight increasing from 2.0 to 3.5. `L_property` and masked SSIM supervise the teacher-forced proxy endpoint `x_low + velocity`; this proxy is distinct from the Heun-integrated endpoint used for whole-realization inference. SSIM decreases from 0.15 to 0.05. Segmentation combines masked class-weighted cross-entropy and masked Dice. `L_Zoeppritz` compares the native central crop with stored **clean** Stage-02 bands while using truth elastic halo, global sample origin, and the identical hashed exact-PP/wavelet/mute contract. Complex post-critical slowness follows the Stage-02 convention. Multiscale patches retain supervised losses but receive zero exact-physics mask. The RGT graph architecture and graph reinjection remain active, but the former auxiliary graph-smoothness objective is scientifically retired: v00332d fixes its coefficient to zero and uses `no_aux_graph_loss`. Physics decays to 70% of its initial weight. Legacy self-instance contrastive loss and adaptive task weighting are implemented capabilities but disabled.

            Stage 02 and Stage 04 use the same declared shared-endpoint bands (`3–17`, `17–31`, `31–45`). The overlap at 17° and 31° is intentional. Compact P/G representative angles are the corresponding band midpoints (`10°`, `24°`, `38°`).
            """
        ),
        code(
            """
training = workflow["training"]
display(pd.Series(training["loss_weights"], name="weight").to_frame())
display(pd.DataFrame(training["curriculum"]).T)
display(pd.DataFrame.from_dict(workflow["capabilities"], orient="index"))
display(pd.Series({
    "optimizer": "AdamW",
    "learning_rate": training["learning_rate"],
    "weight_decay": training["weight_decay"],
    "scheduler": "cosine annealing",
    "epochs": training["epochs"],
    "checkpoint_criteria": training["checkpoint_criteria"],
}).to_frame("value"))
"""
        ),
        md(
            """
            ## 6. Real-batch operator validation

            This cell performs one optimization step on a real Stage-03 batch using the production model and losses. It checks gradients, exact differentiable forward consistency, graph construction, and tensor contracts; it is not reported as a trained result.
            """
        ),
        code(
            """
native_index = int(train_data.index.index[train_data.index["physics_eligible"] == 1][0])
loader = DataLoader(Subset(train_data, [native_index]), batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(loader))
assert bool(batch["physics_eligible"].all())
operator_model = build_sage_avo_variant(
    "full",
    **sage_avo_model_kwargs(workflow),
).to(device)
operator_model.set_norm_stats(normalization)
optimizer = torch.optim.AdamW(operator_model.parameters(), lr=float(training["learning_rate"]))
as_tensor = lambda name: torch.tensor(normalization[name], dtype=torch.float32).view(1, 3, 1, 1)
physics_normalization = PhysicsNormalization(
    x_mean=as_tensor("x_mean"), x_std=as_tensor("x_std"),
    y_mean=as_tensor("y_mean"), y_std=as_tensor("y_std"),
)
physics_settings = physics_settings_from_config(workflow)
base_weights = loss_weights_from_config(workflow, physics_weight=training["loss_weights"]["physics"])
weights = curriculum_from_config(workflow).weights_for_epoch(base_weights, 0, training["epochs"])
operator_metrics = train_step(
    operator_model, batch, optimizer, physics_normalization, weights,
    gradient_clip=float(training["gradient_clip"]),
    time_generator=torch.Generator().manual_seed(int(workflow["experiment"]["seed"]) + 17),
    physics=physics_settings,
)
display(pd.Series(operator_metrics.__dict__).to_frame("one-step value"))
assert all(np.isfinite(value) for value in operator_metrics.__dict__.values())
"""
        ),
        md(
            """
            ## 7. Weighted sampling, augmentation, and production training

            Training uses replacement sampling weighted by reservoir-facies fraction, RGT-gradient complexity, and absolute AVO gradient. Registered augmentation applies horizontal geological flips, mild normalized AVO gain, and mild normalized noise. Validation uses no augmentation and the deterministic interior-time grid `[0.2, 0.5, 0.8]`.

            The final v00332d production run completed the mandatory epoch-100 review. Whole-realization validation selected epoch 40; epochs 101–120 are not scientifically justified and this notebook does not resume training. Every completed epoch logged raw loss components, current weighted terms, and the fixed-final-weight objective. Deterministic sampled metrics are separate. At configured intervals, fixed complete validation realizations were tiled and scored without test data. The run writes `best_fixed_objective.pt`, `best_sampling.pt`, `best_segmentation.pt`, `best_whole_realization.pt`, periodic checkpoints, and `last.pt`; every file records its criterion formula and resume state.
            """
        ),
        code(
            """
print("Read-only final-method notebook: training/resume is intentionally disabled.")
print("Selected production checkpoint: epoch 40 best whole-realization.")

run_dir = experiment_dir / "runs" / (
    "full_2epoch_cuda_sanity" if validation_root_text else "full"
)
manifest_file = run_dir / "manifest.json"
display(pd.Series({
    "manifest": manifest_file.exists(),
    "best_fixed_objective_checkpoint": (run_dir / "best_fixed_objective.pt").exists(),
    "best_sampling_checkpoint": (run_dir / "best_sampling.pt").exists(),
    "best_segmentation_checkpoint": (run_dir / "best_segmentation.pt").exists(),
    "best_whole_realization_checkpoint": (run_dir / "best_whole_realization.pt").exists(),
    "resumable_last_checkpoint": (run_dir / "last.pt").exists(),
    "status": json.loads(manifest_file.read_text()).get("status") if manifest_file.exists() else "not_generated",
}).to_frame("value"))
"""
        ),
        md(
            """
            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | `runs/full/best_fixed_objective.pt` | complete checkpoint state | Minimum fixed-final-weight patch-validation objective | Notebook 05 |
            | `runs/full/best_sampling.pt` | model/optimizer/scheduler/RNG state | Checkpoint selected by sampled elastic RMSE and segmentation mIoU | Notebook 05 |
            | `runs/full/best_segmentation.pt` | complete checkpoint state | Maximum deterministic sampled macro mIoU | Notebook 05 |
            | `runs/full/best_whole_realization.pt` | complete checkpoint state | Preferred fixed whole-validation-section criterion | Notebook 05 |
            | `runs/full/last.pt` | complete resumable state | Exact continuation point after the last completed epoch | training resume |
            | `training_log.csv` | epoch-level objective terms and validation criteria | Optimization/QC history | Notebook 05 |
            | `manifest.json` | seed, split IDs, normalization, prior, commit/config hash | Reproducibility and comparability record | Notebook 05 |

            ## Scientific checks

            - A production-shape real batch passes the CNN, RGT graph, `TransformerConv`, dual decoders, differentiable exact-PP physics loss, and backward optimization.
            - The straight-path target is asserted to equal `truth − low prior`.
            - Physics guidance is executable with installed train-only normalization; zero guidance follows the identical unguided Heun path.
            - Weighted sampling and training-only augmentation are configured explicitly and excluded from validation.
            - Raw losses, current/fixed weighted contributions, sampled metrics, per-class metrics, and whole-validation metrics are logged separately.
            - Criterion names/formulas and all checkpoint-selection minima/maxima are restored on resume; test data never selects a checkpoint.
            - Operator validation is kept distinct from completed model training and scientific performance.

            ## Next stage

            Notebook 05 consumes a criterion-selected full-model checkpoint for whole-realization synthetic inference and field deployment/QC. Controlled ablation and baseline comparisons require complete matched checkpoints.
            """
        ),
    ],
)


write(
    "05_evaluation_and_field_application.ipynb",
    [
        md(
            """
            # 05 — Controlled evaluation, interpretability, and field deployment/QC

            | Item | Definition |
            |---|---|
            | **Scientific purpose** | Evaluate matched synthetic experiments, expose the graph mechanism, reconstruct whole images, and deploy the trained model to the field line with conservative QC. |
            | **Inputs** | Notebook-03 test realizations and normalization; Notebook-04 controlled checkpoints; Stage-01 real AVO, RGT, low-frequency elastic model, coordinates, and local wells. |
            | **Outputs** | Per-realization/summary/paired metrics, documented representative figures, whole-image predictions, field consistency plots, and model/prior sensitivity products. |
            | **Data availability** | Evaluation code and figure definitions are public. Checkpoints and field/private-derived arrays remain local. |
            | **Local data requirements** | Completed controlled checkpoints plus authorized Stage-01 artifacts for field deployment. Numerical tables remain empty until matched artifacts exist. |
            | **Software requirements** | `pip install -e ".[field,ml,notebooks]"`. |
            | **Approximate runtime** | Synthetic inference: minutes per variant/realization; field tiling and sensitivity scale with checkpoints and integration steps. |
            | **Pipeline position** | Final stage consuming Notebooks 01, 03, and 04. |

            Field wells contributed to upstream model construction. Results are therefore described as **field deployment and QC** or **field consistency assessment**, never independent field validation. Ensemble/checkpoint/prior-cutoff spread is sensitivity, not calibrated posterior uncertainty.
            """
        ),
        code(
            COMMON_ROOT
            + """
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sage_avo.config import load_config, seed_everything
from sage_avo.data import PriorDefinition, make_low_frequency_prior
from sage_avo.evaluation import (
    field_well_consistency,
    load_passing_field_calibration,
    prepare_calibrated_field_observation,
)
from sage_avo.evaluation.controlled import evaluate_controlled_ablation
from sage_avo.evaluation.inference import infer_full_realization, load_normalization
from sage_avo.evaluation.sensitivity import ensemble_sensitivity
from sage_avo.experiments.prediction import load_controlled_model, predict_controlled_variant
from sage_avo.forward import forward_avo_dense_spec, forward_specification_from_mapping
from sage_avo.forward.qc import compare_forward_outputs
from sage_avo.models import LEARNED_VARIANTS
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.visualization import plot_inversion_comparison
from sage_avo.visualization.publication import graph_mechanism_figure

workflow_path = ROOT / "configs" / "final_training_v00332d.yaml"
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Missing local configuration: configs/paths.yaml (template: configs/paths.example.yaml).")
paths = load_config(paths_file)
private_root = Path(paths["private_artifact_root"])
validation_root_text = os.getenv("SAGE_AVO_REVISION3_VALIDATION_ROOT", "").strip()
if validation_root_text:
    validation_root = Path(validation_root_text)
    workflow_path = validation_root / "configs" / "training_resolved.json"
    workflow = json.loads(workflow_path.read_text())
    dataset_dir = validation_root / "stage03" / "dataset"
    experiment_dir = validation_root / "stage04" / "sage_avo_s01_v003_stage01v003_validation8"
    figure_dir = validation_root / "figures" / "stage05"
else:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_revision332d_final_training import _configuration as resolve_final_training

    workflow, observability = resolve_final_training()
    final_contract = load_config(workflow_path)
    dataset_dir = private_root / "stage_artifacts" / "stage03" / final_contract["immutable_dataset"] / "dataset"
    experiment_dir = private_root / "stage_artifacts" / "stage04" / "sage_avo_s01_v00332d_final_production"
    figure_dir = private_root / "figures" / "revision332d" / "stage05"
seed_everything(int(workflow["experiment"]["seed"]))
print_torch_runtime()
device = select_torch_device(require_cuda=True, context="Notebook 05 model inference")
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            """
            ## Part A — Available epoch-40 baseline evaluation

            The available baseline comparison is low-frequency-prior-only versus the validation-selected epoch-40 full SAGE-AVO checkpoint. It is runnable before optional ablation checkpoints exist. RMSE, MAE, R², SSIM, Dice/F1, and mIoU are first computed per realization; pooled summaries and paired realization-level bootstrap intervals are secondary.

            The test split is used only for final evaluation, never for checkpoint selection. Missing no-GNN, no-RGT, and no-physics variants remain explicitly unavailable and are excluded rather than represented by unmatched values.
            """
        ),
        code(
            """
if not (dataset_dir / "dataset_manifest.json").exists():
    raise FileNotFoundError(f"Stage-03 dataset manifest not found: {dataset_dir / 'dataset_manifest.json'}")
checkpoints = {
    variant: experiment_dir / "runs" / variant / "best_whole_realization.pt"
    for variant in LEARNED_VARIANTS
}
checkpoints["full"] = experiment_dir / "runs" / "full" / "best_whole_realization.pt"
if validation_root_text:
    checkpoints["full"] = (
        experiment_dir
        / "runs"
        / "full_2epoch_cuda_sanity"
        / "best_whole_realization.pt"
    )
checkpoint_status = pd.DataFrame([
    {"variant": variant, "checkpoint": path.name, "available": path.exists()}
    for variant, path in checkpoints.items()
])
display(checkpoint_status)
all_checkpoints_available = bool(checkpoint_status["available"].all())
baseline_output_dir = private_root / "stage_artifacts" / "stage05" / "v00332d_epoch40_baseline"
baseline_ready = bool(checkpoints["full"].exists())
"""
        ),
        md("### A1. Full-versus-prior whole-test prediction generation"),
        code(
            """
run_baseline = os.getenv("SAGE_AVO_RUN_BASELINE_EVALUATION", "0") == "1"
if run_baseline:
    if not baseline_ready:
        raise FileNotFoundError(f"Selected full-model checkpoint is missing: {checkpoints['full']}")
    for variant in ("low_prior", "full"):
        predict_controlled_variant(
            repository=ROOT,
            config_path=workflow_path,
            config=workflow,
            dataset_directory=dataset_dir,
            experiment_directory=experiment_dir,
            prediction_directory=baseline_output_dir,
            checkpoint_path=checkpoints["full"] if variant == "full" else None,
            variant=variant,
            device_name=str(device),
            require_cuda=True,
            inference_batch_size=2,
        )
else:
    print("Baseline evaluation is disabled (SAGE_AVO_RUN_BASELINE_EVALUATION=0).")
    print("It requires only the available epoch-40 full checkpoint and immutable test split.")
"""
        ),
        md("### A2. Metrics and non-cherry-picked representative selection"),
        code(
            """
baseline_manifests = [
    baseline_output_dir / "predictions" / variant / "manifest.json"
    for variant in ("low_prior", "full")
]
baseline_metrics_available = all(path.exists() for path in baseline_manifests)
if baseline_metrics_available:
    summary, per_realization, paired, representative_id = evaluate_controlled_ablation(
        experiment_directory=baseline_output_dir,
        dataset_directory=dataset_dir,
        bootstrap_repetitions=int(workflow["evaluation"]["bootstrap_repetitions"]),
        bootstrap_confidence=float(workflow["evaluation"]["bootstrap_confidence"]),
        seed=int(workflow["experiment"]["seed"]),
        variants=("low_prior", "full"),
    )
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(baseline_output_dir / "baseline_summary.csv", index=False)
    per_realization.to_csv(baseline_output_dir / "baseline_per_realization.csv", index=False)
    paired.to_csv(baseline_output_dir / "baseline_paired_improvements.csv", index=False)
    display(summary)
    display(paired)
    print("Representative test realization (median full-model Vp RMSE):", representative_id)
else:
    summary = per_realization = paired = pd.DataFrame()
    representative_id = None
    display(pd.DataFrame(columns=["variant", "domain", "metric", "mean", "std", "n_realizations"]))
    print("Run Part A1 to generate the full-versus-prior baseline results.")

controlled_prediction_manifests = [
    experiment_dir / "predictions" / variant / "manifest.json"
    for variant in ("low_prior", *LEARNED_VARIANTS)
]
controlled_metrics_available = all(path.exists() for path in controlled_prediction_manifests)
"""
        ),
        md(
            r"""
            ## Part B — Scientifically interpretable graph evidence

            The representative synthetic realization is fixed by median full-model Vp RMSE—not visual appeal. A normalized forward pass through the criterion-selected checkpoint returns base edge indices, AVO-gradient edge attributes, learned final-layer mean-head `TransformerConv` attention, and node embeddings. Panel (f) displays only the strongest 15% of learned attention edges. This threshold is visualization only; all graph edges remain active in the network. The saved forward-contract JSON records all four returned tensors' dimensions/ranges.

            The graph-benefit panel is

            \[
            |Vp_{noGNN}-Vp_{true}|-|Vp_{full}-Vp_{true}|,
            \]

            so positive values mean graph propagation reduces absolute error. Latent embedding norm is not used as the primary scientific evidence.
            """
        ),
        code(
            """
if controlled_metrics_available:
    realization_path = dataset_dir / "realizations" / f"realization_{representative_id:07d}.npz"
    with np.load(realization_path) as archive:
        avo = archive["avo"]
        rgt = archive["rgt"]
        truth = archive["elastic"]
    with np.load(experiment_dir / "predictions" / "full" / f"realization_{representative_id:07d}.npz") as archive:
        full_prediction = archive["elastic"]
    with np.load(experiment_dir / "predictions" / "no_gnn" / f"realization_{representative_id:07d}.npz") as archive:
        no_gnn_prediction = archive["elastic"]

    graph_mechanism_figure(
        workflow,
        experiment_dir,
        dataset_dir,
        representative_id,
        figure_dir,
        device,
    )
    print("Saved actual-attention graph mechanism figure and forward-contract JSON.")
else:
    print("Graph-benefit figure unavailable: matched full/no-GNN predictions are required.")
"""
        ),
        md("## Part C — Whole-image synthetic inference"),
        code(
            """
if controlled_metrics_available:
    with np.load(realization_path) as archive:
        low_prior = archive["low"]
    figure = plot_inversion_comparison(truth, low_prior, full_prediction)
    whole_path = figure_dir / "stage05_whole_image_synthetic.png"
    figure.savefig(whole_path, dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("Whole-image comparison unavailable: a controlled full-model prediction is required.")
"""
        ),
        md(
            """
            ## Part D — Field deployment and QC

            The deployment input comprises real low/mid/high AVO stacks, Stage-01 RGT, and a field elastic background passed through the **same configured 2-Hz prior builder** used for synthetic data. The SEG-Y export axis remains described as the **configured gather-coordinate header** until acquisition/export metadata independently establish that it is true incidence angle.

            Raw field AVA is never normalized directly with synthetic statistics. A versioned amplitude/phase/polarity/wavelet transfer must be recorded in a calibration manifest that satisfies explicit polarity, phase, spectrum, amplitude, percentile-overlap, and spatial-stability criteria. Missing, failed, unapproved, or forward-hash-mismatched manifests block inference. After calibration, whole-section inference uses the synthetic training normalization, common tiling/Hann stitching, and deterministic transport. Local well curves are non-blind field-consistency overlays.
            """
        ),
        code(
            """
dataset_id = workflow["field_application"]["dataset_id"]
version = workflow["field_application"]["stage01_version"]
field_root = Path(paths["work_data_root"]) / dataset_id
field_files = {
    "avo_near": field_root / "usable" / version / "real_avo" / "AVO_low_real.npy",
    "avo_mid": field_root / "usable" / version / "real_avo" / "AVO_mid_real.npy",
    "avo_far": field_root / "usable" / version / "real_avo" / "AVO_high_real.npy",
    "low_source": field_root / "usable" / version / "elastic_background.npy",
    "rgt": field_root / "attributes" / version / "rgt_tau.npy",
    "time_ms": field_root / "usable" / version / "reg_t.npy",
    "cdp": field_root / "usable" / version / "good_cdps.npy",
    "line_xy": field_root / "usable" / version / "line_xy.npy",
}
missing_field = [path for path in field_files.values() if not path.exists()]
if missing_field:
    raise FileNotFoundError("Stage-01 field deployment channels are missing:\\n" + "\\n".join(map(str, missing_field)))
loaded_field = {name: np.load(path, allow_pickle=False) for name, path in field_files.items()}
field = {
    "avo": np.stack([loaded_field.pop("avo_near"), loaded_field.pop("avo_mid"), loaded_field.pop("avo_far")]),
    "low": make_low_frequency_prior(
        loaded_field.pop("low_source"),
        PriorDefinition(**workflow["field_application"]["low_frequency_prior"]),
    ),
    **loaded_field,
}
display(pd.DataFrame([{"channel": name, "shape": value.shape} for name, value in field.items()]))

fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
for axis, panel, title in zip(
    axes.flat,
    [*field["avo"], field["rgt"], *field["low"], field["low"][0] - field["low"][0].mean(axis=0)],
    ["Real near AVO", "Real mid AVO", "Real far AVO", "RGT", "Low Vp", "Low Vs", "Low density", "Vp vertical variation"],
):
    axis.imshow(panel, aspect="auto", cmap="gray" if "AVO" in title else "viridis")
    axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
field_input_path = figure_dir / "stage05_field_input_contract.png"
fig.savefig(field_input_path, dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        md("### D1. Whole-section checkpoint inference"),
        code(
            """
field_prediction = field_segmentation = calibrated_field_avo = None
run_field_inference = os.getenv("SAGE_AVO_RUN_FIELD_INFERENCE", "0") == "1"
calibration_manifest = Path(
    os.getenv(
        "SAGE_AVO_FIELD_CALIBRATION_MANIFEST",
        private_root / "revision3" / "field_calibration_v003.json",
    )
)
forward_specification = forward_specification_from_mapping(workflow)
if run_field_inference and checkpoints["full"].exists():
    calibration = load_passing_field_calibration(
        calibration_manifest,
        expected_forward_specification_sha256=forward_specification.sha256,
    )
    calibrated_field_avo = prepare_calibrated_field_observation(
        field["avo"],
        calibration_manifest=calibration_manifest,
        expected_forward_specification_sha256=forward_specification.sha256,
    )
    print("Calibration record:", calibration["approved_by"], calibration["manifest_sha256"])
    model = load_controlled_model("full", workflow, checkpoints["full"], device)
    field_prediction, field_segmentation = infer_full_realization(
        model,
        avo=calibrated_field_avo, low=field["low"], rgt=field["rgt"],
        normalization=load_normalization(dataset_dir),
        patch_shape=tuple(workflow["patches"]["shape"]),
        stride=tuple(workflow["patches"]["stride"]),
        steps=int(workflow["training"]["sample_steps_test"]),
        batch_size=min(2, int(workflow["training"]["batch_size"])),
        device=device,
    )
    np.savez_compressed(
        private_root / "stage_artifacts" / "stage05_field_prediction.npz",
        elastic=field_prediction, segmentation=field_segmentation,
        time_ms=field["time_ms"], cdp=field["cdp"],
    )
else:
    print("Field inference is inactive: enable it explicitly after the checkpoint and calibration contracts are satisfied.")
    print("A passing versioned calibration manifest is required; implicit identity transfer is prohibited.")
"""
        ),
        md(
            """
            ### D2. Wells, forward seismic QC, and far-angle behavior

            When a prediction exists, local processed wells are overlaid in time/CDP coordinates. Exact forward-modeled near/mid/far stacks from the predicted elastic section are compared with real stacks using per-band robust amplitude fitting and correlation. Near, mid, and far statistics are reported separately; weak far-angle agreement is retained and discussed rather than hidden.
            """
        ),
        code(
            """
if field_prediction is not None:
    well_qc, well_overlays = field_well_consistency(
        field_prediction,
        time_ms=field["time_ms"],
        line_xy=field["line_xy"],
        wells_directory=field_root / "usable" / version / "wells",
    )
    display(well_qc)
    well_qc.to_csv(private_root / "stage_artifacts" / "stage05_field_well_consistency.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for channel, (axis, name) in enumerate(zip(axes, ("Vp", "Vs", "density"))):
        image = axis.imshow(
            field_prediction[channel], aspect="auto", cmap="viridis",
            extent=[0, field_prediction.shape[2] - 1, field["time_ms"][-1], field["time_ms"][0]],
        )
        for overlay in well_overlays:
            if overlay["channel"] == channel:
                axis.scatter(
                    np.full_like(overlay["time_ms"], overlay["trace_index"]),
                    overlay["time_ms"], c=overlay["observed"], cmap="viridis",
                    vmin=np.nanpercentile(field_prediction[channel], 2),
                    vmax=np.nanpercentile(field_prediction[channel], 98), s=2,
                )
        axis.set(title=f"Predicted {name} with processed-well overlays", xlabel="Trace", ylabel="TWT (ms)")
        fig.colorbar(image, ax=axis, shrink=0.75)
    well_path = figure_dir / "stage05_field_prediction_with_wells.png"
    fig.savefig(well_path, dpi=300, bbox_inches="tight")
    plt.show()

    modeled = forward_avo_dense_spec(*field_prediction, forward_specification).stacks
    agreement = compare_forward_outputs(calibrated_field_avo, modeled)
    forward_qc = pd.DataFrame({
        "band": ("near", "mid", "far"),
        "fitted_amplitude_scale": agreement.scale,
        "correlation": agreement.correlation,
        "normalized_rmse_after_scale": agreement.normalized_rmse,
    })
    display(forward_qc)
    forward_qc.to_csv(private_root / "stage_artifacts" / "stage05_field_forward_qc.csv", index=False)
else:
    print("Forward field QC unavailable: no field prediction has been generated.")
"""
        ),
        md(
            """
            ### D3. Model/prior sensitivity

            Checkpoint ensembles, alternative justified prior cutoffs, and controlled model variants may be propagated through the same inference code. Their spread is reported as **model/prior sensitivity**, **predictive sensitivity**, or **ensemble sensitivity**. It is not a calibrated posterior uncertainty distribution.
            """
        ),
        code(
            """
sensitivity_dir = experiment_dir / "field_sensitivity"
run_sensitivity = os.getenv("SAGE_AVO_RUN_FIELD_SENSITIVITY", "0") == "1"
if run_sensitivity:
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    if not all_checkpoints_available:
        raise FileNotFoundError("Matched controlled checkpoints are required for model-variant sensitivity.")
    if calibrated_field_avo is None:
        calibrated_field_avo = prepare_calibrated_field_observation(
            field["avo"],
            calibration_manifest=calibration_manifest,
            expected_forward_specification_sha256=forward_specification.sha256,
        )
    for variant, checkpoint in checkpoints.items():
        model = load_controlled_model(variant, workflow, checkpoint, device)
        member, _ = infer_full_realization(
            model,
            avo=calibrated_field_avo, low=field["low"], rgt=field["rgt"],
            normalization=load_normalization(dataset_dir),
            patch_shape=tuple(workflow["patches"]["shape"]),
            stride=tuple(workflow["patches"]["stride"]),
            steps=int(workflow["training"]["sample_steps_test"]),
            batch_size=min(2, int(workflow["training"]["batch_size"])),
            device=device,
        )
        np.savez_compressed(sensitivity_dir / f"member_{variant}.npz", elastic=member)

sensitivity_files = sorted(sensitivity_dir.glob("member_*.npz"))
if len(sensitivity_files) >= 2:
    members = np.stack([np.load(path)["elastic"] for path in sensitivity_files])
    sensitivity = ensemble_sensitivity(members)
    print("Sensitivity members:", len(members), "shape:", sensitivity["standard_deviation"].shape)
    np.savez_compressed(sensitivity_dir / "model_variant_sensitivity_summary.npz", **sensitivity)
else:
    print("Sensitivity maps unavailable: at least two predeclared model/prior members are required.")
    print("SAGE_AVO_RUN_FIELD_SENSITIVITY=1 activates sensitivity evaluation when those members exist.")
"""
        ),
        md(
            """
            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | controlled prediction packages | full test images per variant | Matched whole-realization benchmark predictions | metric/figure pipeline |
            | per-realization/summary/paired CSVs | metric tables | Performance and paired ablation evidence | scientific reporting |
            | graph mechanism figure | 2×4 high-resolution panel | AVO/RGT/edge mechanism and graph error reduction | scientific reporting |
            | whole-image synthetic figure | truth/prior/prediction/error for Vp/Vs/density | Spatial inversion performance | scientific reporting |
            | field prediction/QC package | whole section + coordinates + QC | Field deployment and consistency assessment | scientific reporting |
            | sensitivity maps | ensemble spread | Model/prior sensitivity, not posterior uncertainty | discussion |

            ## Scientific checks

            - Numerical tables require all matched controlled prediction manifests; absent results remain empty rather than being filled with unmatched values.
            - Metrics are calculated per realization and paired by realization before aggregation.
            - The representative case is chosen by median full-model Vp RMSE.
            - Graph figures use returned base edges, edge attributes, learned attention, and embeddings from the actual normalized model forward pass; only the strongest 15% of attention is drawn.
            - Whole-image inference uses common tiling, overlap, integration, normalization, and checkpoint rules.
            - Field wells are treated as non-blind consistency overlays.
            - Field inference is blocked unless a versioned calibration manifest passes and matches the forward specification hash.
            - Forward QC preserves band-specific behavior, including weak far-angle agreement.
            - Ensemble spread is labeled sensitivity rather than calibrated posterior uncertainty.

            ## Next stage

            This is the final computational stage. Its artifacts support downstream scientific communication when benchmark completeness and redistribution status are recorded in the artifact index.
            """
        ),
    ],
)
