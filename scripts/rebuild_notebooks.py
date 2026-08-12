"""Build the four public research notebooks from reviewed cell sources.

This maintainer utility keeps notebook JSON deterministic. It is not part of the
scientific runtime; every code cell calls the installed ``sage_avo`` package.
"""

from __future__ import annotations

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
    raise RuntimeError("Run this notebook from the installed SAGE-AVO repository.")

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
            | **Local/private-data requirements** | Configure `work_data_root` and `private_artifact_root` in ignored `configs/paths.yaml`. No synthetic toy fallback is used when Stage-01 artifacts are absent. |
            | **Software requirements** | `pip install -e ".[field,ml,notebooks]"`; Madagascar is optional and used only for historical production-path cross-checking. |
            | **Approximate runtime** | Minutes per realization on CPU; the configured 200-realization production family is an offline generation job. |
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
    forward_config_from_mapping,
    generate_stage02_dataset,
    load_stage01_background,
    load_stage02_manifest,
)
from sage_avo.forward import forward_avo_dense, forward_avo_madagascar, madagascar_availability

workflow = load_config(ROOT / "configs" / "synthetic_s01.yaml")
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError(
        "Create ignored configs/paths.yaml from configs/paths.example.yaml and point it "
        "to the licensed Stage-01 artifacts and private output root."
    )
paths = load_config(paths_file)
seed_everything(int(workflow["stage"]["seed"]))

private_root = Path(paths["private_artifact_root"])
realization_dir = private_root / "stage_artifacts" / "stage02" / "realizations"
figure_dir = private_root / "figures" / "stage02"
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

            Historical development code sometimes assigned the sand-probability field directly to `DELTA`. The production interface converts explicitly and never propagates that reversed convention. The field-calibrated probability range also makes a 0.5 threshold inappropriate here; the configured 0.30 threshold is supported by the Stage-01 reservoir distribution and is recorded in every realization manifest.
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

            A realization ID is also its random seed. One coherent deformation field is applied to every Stage-01 channel, so horizons, RGT, facies, porosity, masks, and elastic background remain registered. The deformation combines smooth folds with optional finite-length fault displacement. Correlated Gaussian fields then perturb P(sand) and porosity inside the warped reservoir.

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

            CO₂ saturation is introduced only in connected, sufficiently thick reservoir sand. The implementation follows the historical field-conditioned rock-physics path: Hertz–Mindlin dry-frame moduli, Brie mixing for the brine/CO₂ fluid modulus, then Gassmann saturation. Density follows volumetric fluid replacement. Both the brine baseline and substituted elastic cubes are saved, making the physical change auditable.
            """
        ),
        md(
            """
            ## 4. Exact dense-angle forward response

            For each elastic interface and each configured angle, the production operator solves the exact isotropic PP Zoeppritz system. A Ricker wavelet is convolved along time, then the historical front mute/taper is applied. Dense responses retain all 43 angles from 3° through 45°.

            Angle bands are centralized in configuration. Historical products used overlapping endpoints (`3–17`, `17–31`, `31–45`). New products use non-overlapping integer-angle bands (`3–17`, `18–31`, `32–45`). Both definitions are written to the manifest; old products are not silently relabeled.
            """
        ),
        code(
            """
forward_definition = forward_config_from_mapping(workflow)
display(
    pd.DataFrame(
        [{"band": b.name, "minimum_deg": b.minimum_degrees, "maximum_deg": b.maximum_degrees}
         for b in forward_definition.bands]
    )
)
print("Legacy bands:", workflow["forward"]["legacy_bands"])
print("Dense angles:", forward_definition.angles_degrees)
"""
        ),
        md(
            """
            ## 5. Generate the realization family

            The default call creates the complete configured family. `SAGE_AVO_STAGE02_LIMIT` is an explicit operator-validation control for local development; a limited manifest is labeled `operator_validation_subset` and must not be presented as a completed corpus. `SAGE_AVO_REUSE_STAGE02=1` reopens an existing immutable local result.
            """
        ),
        code(
            """
limit_text = os.getenv("SAGE_AVO_STAGE02_LIMIT", "").strip()
realization_limit = int(limit_text) if limit_text else None
reuse = os.getenv("SAGE_AVO_REUSE_STAGE02", "0") == "1"
manifest_path = realization_dir / "manifest.json"

if reuse and manifest_path.exists():
    manifest = load_stage02_manifest(manifest_path)
else:
    manifest = generate_stage02_dataset(
        config=workflow,
        paths=paths,
        output_directory=realization_dir,
        realization_limit=realization_limit,
    )

display(pd.Series({key: manifest[key] for key in (
    "status", "requested_realizations", "generated_realizations", "exact_forward_operator"
)}).to_frame("value"))
"""
        ),
        md(
            """
            ## 6. Deterministic realization QC

            The representative realization is the smallest generated ID—a documented rule independent of visual appearance. The panels verify channel registration, the DELTA/P(sand) complement, plume support, exact near/mid/far response, and recalculated PWD dip. RGT is coherently warped from Stage 01; the historical synthetic workflow recalculated dip rather than reintegrating RGT.
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
print("private figure:", qc_path)
"""
        ),
        md(
            """
            ## 7. Historical Madagascar production-path cross-check

            If Madagascar is installed, the same elastic crop is passed through `sfzoeppritz2 → sftransp → sfricker1 → sftransp`. Correlation is the principal diagnostic because the historical `sfricker1` normalization differs from the NumPy wavelet normalization. This is a verification path; it does not change the manifest’s chosen local backend.
            """
        ),
        code(
            """
availability = madagascar_availability()
print(availability)
if availability.available:
    crop = realization["elastic"][:, 80:130, 20:120]
    numpy_forward = forward_avo_dense(*crop, config=forward_definition)
    rsf_forward = forward_avo_madagascar(*crop, config=forward_definition)
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
        md("## 8. Saved-channel manifest"),
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
            - CO₂ substitution is confined to connected reservoir sand; brine and substituted elastic cubes are both retained.
            - Exact dense-angle Zoeppritz is the primary operator; compact P/G approximations are not used to generate training observations.
            - Current and legacy angle bands remain separately identified.
            - Optional Madagascar correlation checks the historical production route.

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
            | **Local/private-data requirements** | Ignored `configs/paths.yaml` and a completed or explicitly subset-labeled Stage-02 artifact directory. No toy fallback is used. |
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.config import load_config, seed_everything
from sage_avo.data import IndexedRealizationPatches
from sage_avo.experiments import build_stage03_dataset, validate_dataset_integrity

workflow = load_config(ROOT / "configs" / "ml_dataset_s01.yaml")
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Create ignored configs/paths.yaml from configs/paths.example.yaml.")
paths = load_config(paths_file)
seed_everything(int(workflow["stage"]["seed"]))

private_root = Path(paths["private_artifact_root"])
stage02_dir = private_root / "stage_artifacts" / "stage02" / "realizations"
dataset_dir = private_root / "stage_artifacts" / "stage03" / "dataset"
figure_dir = private_root / "figures" / "stage03"
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
    raise FileNotFoundError("Run Notebook 02 against the licensed/generated inputs first.")
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

            A seeded permutation assigns entire realizations to train/validation/test (70/15/15 in the production configuration). Patch coordinates are generated only after this assignment. Consequently, no geological deformation, plume scenario, or trace from one realization can leak across splits.
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

            Raw patches are sampled at 40×80, 50×100, and 64×128 physical extents with configured proportions, then resized to 50×100 tensors. Continuous channels use bilinear interpolation; class labels and masks use nearest-neighbor interpolation. Every row preserves realization ID, origin, raw size, tensor size, and both resize factors.
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
normalization = json.loads((dataset_dir / "normalization.json").read_text())
patch_index = pd.read_csv(dataset_dir / "patch_index.csv")

display(pd.DataFrame({name: pd.Series(values) for name, values in split_ids.items()}))
display(pd.DataFrame(normalization, index=["near/Vp", "mid/Vs", "far/density"]))
display(patch_index.groupby(["split", "raw_height", "raw_width"]).size().rename("patches").to_frame())
display(patch_index.head())

sets = {name: set(values) for name, values in split_ids.items()}
assert sets["train"].isdisjoint(sets["validation"])
assert sets["train"].isdisjoint(sets["test"])
assert sets["validation"].isdisjoint(sets["test"])
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
print("private figure:", qc_path)
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
            | `normalization.json` | 3-channel means/stds | Training-only normalization transform | Notebooks 04–05 |
            | `patch_index.csv` | one row per multiscale patch | Traceable sampling and resize metadata | Notebook 04 |
            | `dataset_manifest.json` | prior, channels, split, integrity | Complete ML task contract | Notebooks 04–05 |

            ## Scientific checks

            - Required channels, matching dimensions, and finite values are validated before splitting.
            - Train/validation/test realization-ID sets are asserted disjoint.
            - Patch rows are checked against their assigned realization split.
            - Low-frequency priors are explicitly labeled truth-derived and their smoothing constants are saved.
            - Normalization statistics use training realizations only.
            - Continuous and categorical resize modes are separated, while raw physical sizes and scale factors remain in metadata.

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
            # 04 — SAGE-AVO model and controlled training

            | Item | Definition |
            |---|---|
            | **Scientific purpose** | Train a structure-aware graph/CNN model to refine a supplied low-frequency elastic prior using near/mid/far AVO. |
            | **Inputs** | Notebook-03 patch index, train-only normalization, realization splits, AVO, low prior, RGT, elastic targets, segmentation targets, and masks. |
            | **Outputs** | Controlled full/no-GNN/no-RGT/no-physics checkpoints, training logs, checkpoint-selection metadata, and run manifests. |
            | **Data availability** | Architecture, losses, and orchestration are public; datasets and checkpoints remain private/local. |
            | **Local/private-data requirements** | Completed Stage-03 artifacts and adequate PyTorch/PyG compute. No randomly generated data fallback is used. |
            | **Software requirements** | `pip install -e ".[ml,notebooks]"` with PyTorch and PyTorch Geometric. |
            | **Approximate runtime** | Operator checks: seconds to minutes. Production 120-epoch controlled training: GPU-scale, dependent on hardware. |
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
from torch.utils.data import DataLoader

from sage_avo.config import load_config, seed_everything
from sage_avo.data import IndexedRealizationPatches
from sage_avo.models import ALL_VARIANTS, LEARNED_VARIANTS, build_sage_avo_variant
from sage_avo.models.sage_avo import angular_features
from sage_avo.training.engine import PhysicsNormalization, train_step
from sage_avo.training.flow import straight_path
from sage_avo.training.losses import LossWeights
from sage_avo.experiments.training import train_controlled_variant

workflow_path = ROOT / "configs" / "sage_avo_s01.yaml"
workflow = load_config(workflow_path)
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Create ignored configs/paths.yaml from configs/paths.example.yaml.")
paths = load_config(paths_file)
seed_everything(int(workflow["experiment"]["seed"]))

private_root = Path(paths["private_artifact_root"])
dataset_dir = private_root / "stage_artifacts" / "stage03" / "dataset"
experiment_dir = private_root / "stage_artifacts" / "stage04" / "experiments"
figure_dir = private_root / "figures" / "stage04"
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            """
            ## 1. Immutable data contract

            Each training item contains normalized low/mid/high AVO `[3,H,W]`, normalized low-frequency Vp/Vs/density `[3,H,W]`, normalized elastic target `[3,H,W]`, RGT `[H,W]`, segmentation `[H,W]`, valid mask `[1,H,W]`, and traceability metadata. All controlled variants use the same realization split, patch index, normalization, optimizer schedule, and checkpoint criterion.
            """
        ),
        code(
            """
if not (dataset_dir / "dataset_manifest.json").exists():
    raise FileNotFoundError("Run Notebook 03 first; the training notebook never creates a toy dataset.")
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
with torch.no_grad():
    angular, gradient = angular_features(sample["avo"].unsqueeze(0))
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
            4. Edge weights decrease with local AVO-gradient contrast.
            5. Two `TransformerConv` layers perform graph attention/message passing.
            6. Graph features are reshaped to the image grid, reinjected into CNN features, and decoded into elastic transport velocity; a second decoder predicts shale/sand/plume classes.

            The no-GNN variant retains the CNN and local segmentation decoder. This isolates graph contribution without changing the dataset or target.
            """
        ),
        code(
            """
model_config = workflow["model"]
models = {
    variant: build_sage_avo_variant(
        variant,
        hidden_channels=int(model_config["hidden_channels"]),
        graph_layers=int(model_config["graph_layers"]),
        graph_heads=int(model_config["graph_heads"]),
        max_rgt_shift=int(model_config["max_rgt_shift_samples"]),
        classes=int(model_config["classes"]),
    )
    for variant in LEARNED_VARIANTS
}
display(pd.DataFrame([
    {
        "variant": name,
        "graph_mode": model.graph_mode,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    for name, model in models.items()
]))

full = models["full"].eval()
with torch.no_grad():
    state = sample["low"].unsqueeze(0)
    output = full(state, torch.zeros(1), sample["avo"].unsqueeze(0), state, sample["rgt"].unsqueeze(0))
print("elastic velocity:", tuple(output.velocity.shape))
print("segmentation logits:", tuple(output.segmentation_logits.shape))
print("graph embedding:", tuple(output.embeddings.shape))
print("directed graph edges:", output.edge_indices[0].shape[1])
"""
        ),
        md(
            r"""
            ## 4. Deterministic conditional residual transport

            Training samples a time `t ~ Uniform(0,1)` and constructs

            \[
            x_t=(1-t)x_{low}+t y,\qquad u^*=y-x_{low}.
            \]

            The network predicts the straight-path velocity conditioned on AVO, the supplied prior, and RGT. Inference starts at `x_low` and integrates the learned velocity from `t=0` to `1` with Heun steps. No stochastic base distribution or posterior calibration is implemented.
            """
        ),
        code(
            """
t = torch.tensor([0.35])
state, target_velocity = straight_path(
    sample["low"].unsqueeze(0), sample["target"].unsqueeze(0), t
)
assert torch.allclose(target_velocity, sample["target"].unsqueeze(0) - sample["low"].unsqueeze(0))
print("state and velocity:", tuple(state.shape), tuple(target_velocity.shape))
"""
        ),
        md(
            r"""
            ## 5. Complete training objective

            \[
            L = w_f L_{flow}+w_p L_{property}+w_s L_{segmentation}
                +w_{phys}L_{Zoeppritz}+w_g L_{graph}.
            \]

            `L_flow` fits residual velocity; `L_property` supervises the reconstructed full elastic state; segmentation combines weighted cross-entropy and Dice; `L_Zoeppritz` compares observed bands with a differentiable exact-PP forward response from predicted Vp/Vs/density; and `L_graph` penalizes elastic contrast preferentially along high-weight graph edges. The no-physics variant alone sets `w_phys=0`.
            """
        ),
        code(
            """
training = workflow["training"]
display(pd.Series(training["loss_weights"], name="weight").to_frame())
display(pd.Series({
    "optimizer": "AdamW",
    "learning_rate": training["learning_rate"],
    "weight_decay": training["weight_decay"],
    "scheduler": "cosine annealing",
    "epochs": training["epochs"],
    "checkpoint_criterion": training["checkpoint_criterion"],
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
loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(loader))
operator_model = build_sage_avo_variant(
    "full",
    hidden_channels=int(model_config["hidden_channels"]),
    graph_layers=int(model_config["graph_layers"]),
    graph_heads=int(model_config["graph_heads"]),
    max_rgt_shift=int(model_config["max_rgt_shift_samples"]),
    classes=int(model_config["classes"]),
)
optimizer = torch.optim.AdamW(operator_model.parameters(), lr=float(training["learning_rate"]))
as_tensor = lambda name: torch.tensor(normalization[name], dtype=torch.float32).view(1, 3, 1, 1)
physics_normalization = PhysicsNormalization(
    x_mean=as_tensor("x_mean"), x_std=as_tensor("x_std"),
    y_mean=as_tensor("y_mean"), y_std=as_tensor("y_std"),
)
weights = LossWeights(**{name: float(value) for name, value in training["loss_weights"].items()})
operator_metrics = train_step(
    operator_model, batch, optimizer, physics_normalization, weights,
    gradient_clip=float(training["gradient_clip"]),
    time_generator=torch.Generator().manual_seed(int(workflow["experiment"]["seed"]) + 17),
)
display(pd.Series(operator_metrics.__dict__).to_frame("one-step value"))
assert all(np.isfinite(value) for value in operator_metrics.__dict__.values())
"""
        ),
        md(
            """
            ## 7. Production controlled training

            Set `SAGE_AVO_RUN_FULL_TRAINING=1` on suitable compute to train every learned condition. Each run writes a manifest before training, a per-epoch CSV, `best_flow.pt`, `best_sampling.pt`, and `last.pt`. The selected evaluation checkpoint is the shared sampled-validation criterion; seeds and split IDs are embedded in each manifest.
            """
        ),
        code(
            """
run_full_training = os.getenv("SAGE_AVO_RUN_FULL_TRAINING", "0") == "1"
run_directories = {}
if run_full_training:
    for variant in LEARNED_VARIANTS:
        run_directories[variant] = train_controlled_variant(
            repository=ROOT,
            config_path=workflow_path,
            config=workflow,
            dataset_directory=dataset_dir,
            experiment_directory=experiment_dir,
            variant=variant,
        )
else:
    print("Production training not requested in this execution.")
    print("Set SAGE_AVO_RUN_FULL_TRAINING=1 to train:", LEARNED_VARIANTS)

status_rows = []
for variant in LEARNED_VARIANTS:
    run_dir = experiment_dir / "runs" / variant
    manifest_file = run_dir / "manifest.json"
    status_rows.append({
        "variant": variant,
        "manifest": manifest_file.exists(),
        "best_sampling_checkpoint": (run_dir / "best_sampling.pt").exists(),
        "status": json.loads(manifest_file.read_text()).get("status") if manifest_file.exists() else "pending",
    })
display(pd.DataFrame(status_rows))
"""
        ),
        md(
            """
            ## 8. HCTNet baseline context

            HCTNet remains important prior work and is implemented in `sage_avo.models.hctnet`. A headline comparison is intentionally excluded until HCTNet is retrained with the identical truth-derived prior, realization split, normalization, masks, inference tiling, and checkpoint-selection rule. Historical HCTNet results are therefore historical/non-controlled evidence, not entries in the controlled table.

            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | `runs/<variant>/best_sampling.pt` | state dictionaries + config/metrics | Checkpoint selected by the common sampled-validation rule | Notebook 05 |
            | `training_log.csv` | epoch-level objective terms and validation criteria | Optimization/QC history | Notebook 05 |
            | `manifest.json` | seed, split IDs, normalization, prior, commit/config hash | Reproducibility and comparability record | Notebook 05 |

            ## Scientific checks

            - A production-shape real batch passes the CNN, RGT graph, `TransformerConv`, dual decoders, differentiable exact-PP physics loss, and backward optimization.
            - The straight-path target is asserted to equal `truth − low prior`.
            - Controlled variants differ only in graph mode or physics-loss weight.
            - Checkpoint selection, split, normalization, masks, optimizer, and schedule are shared.
            - Operator validation is kept distinct from completed model training and scientific performance.

            ## Next stage

            Notebook 05 requires matched `best_sampling.pt` checkpoints for the learned variants. It generates whole-realization predictions, reports per-realization controlled metrics, selects a representative test case by median full-model Vp RMSE, visualizes actual graph edges, and then performs conservative field deployment/QC.
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
            | **Local/private-data requirements** | Completed controlled checkpoints plus licensed Stage-01 artifacts for field deployment. No fabricated metric placeholders are populated. |
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
from sage_avo.evaluation import field_well_consistency
from sage_avo.evaluation.controlled import evaluate_controlled_ablation
from sage_avo.evaluation.inference import infer_full_realization, load_normalization
from sage_avo.evaluation.sensitivity import ensemble_sensitivity
from sage_avo.experiments.prediction import load_controlled_model, predict_controlled_variant
from sage_avo.forward import ForwardConfig, forward_avo_three_band
from sage_avo.forward.qc import compare_forward_outputs
from sage_avo.models import LEARNED_VARIANTS
from sage_avo.models.graph import build_rgt_edges
from sage_avo.models.sage_avo import angular_features
from sage_avo.structure.graph import GraphEdges
from sage_avo.visualization import plot_graph_mechanism, plot_inversion_comparison

workflow_path = ROOT / "configs" / "sage_avo_s01.yaml"
workflow = load_config(workflow_path)
paths_file = ROOT / "configs" / "paths.yaml"
if not paths_file.exists():
    raise FileNotFoundError("Create ignored configs/paths.yaml from configs/paths.example.yaml.")
paths = load_config(paths_file)
seed_everything(int(workflow["experiment"]["seed"]))

private_root = Path(paths["private_artifact_root"])
dataset_dir = private_root / "stage_artifacts" / "stage03" / "dataset"
experiment_dir = private_root / "stage_artifacts" / "stage04" / "experiments"
figure_dir = private_root / "figures" / "stage05"
figure_dir.mkdir(parents=True, exist_ok=True)
"""
        ),
        md(
            """
            ## Part A — Controlled synthetic evaluation

            The required conditions are low-frequency-prior-only, full SAGE-AVO, no-GNN, no-RGT-steering, and no-physics-loss. Learned variants are trained on the same realization split and evaluated on complete test realizations. RMSE, MAE, R², SSIM, Dice/F1, and mIoU are first computed per realization; pooled summaries and paired realization-level bootstrap intervals are secondary.

            HCTNet enters this table only after a matched retraining with identical prior, split, normalization, masks, tiling, and checkpoint rule. Otherwise it remains explicitly historical/non-controlled.
            """
        ),
        code(
            """
if not (dataset_dir / "dataset_manifest.json").exists():
    raise FileNotFoundError("Run Notebook 03 first.")
checkpoints = {
    variant: experiment_dir / "runs" / variant / "best_sampling.pt"
    for variant in LEARNED_VARIANTS
}
checkpoint_status = pd.DataFrame([
    {"variant": variant, "checkpoint": path.name, "available": path.exists()}
    for variant, path in checkpoints.items()
])
display(checkpoint_status)
all_checkpoints_available = bool(checkpoint_status["available"].all())
"""
        ),
        md("### A1. Whole-test prediction generation"),
        code(
            """
run_predictions = os.getenv("SAGE_AVO_RUN_EVALUATION", "0") == "1"
if run_predictions:
    if not all_checkpoints_available:
        missing = [str(path) for path in checkpoints.values() if not path.exists()]
        raise FileNotFoundError("Controlled evaluation requested but checkpoints are missing:\\n" + "\\n".join(missing))
    for variant in ("low_prior", *LEARNED_VARIANTS):
        predict_controlled_variant(
            repository=ROOT,
            config_path=workflow_path,
            config=workflow,
            dataset_directory=dataset_dir,
            experiment_directory=experiment_dir,
            variant=variant,
        )
else:
    print("Evaluation is pending unless complete matched checkpoints/predictions already exist.")
    print("Set SAGE_AVO_RUN_EVALUATION=1 after Notebook 04 production training.")
"""
        ),
        md("### A2. Metrics and non-cherry-picked representative selection"),
        code(
            """
prediction_manifests = [
    experiment_dir / "predictions" / variant / "manifest.json"
    for variant in ("low_prior", *LEARNED_VARIANTS)
]
metrics_available = all(path.exists() for path in prediction_manifests)
if metrics_available:
    summary, per_realization, paired, representative_id = evaluate_controlled_ablation(
        experiment_directory=experiment_dir,
        dataset_directory=dataset_dir,
        bootstrap_repetitions=int(workflow["evaluation"]["bootstrap_repetitions"]),
        bootstrap_confidence=float(workflow["evaluation"]["bootstrap_confidence"]),
        seed=int(workflow["experiment"]["seed"]),
    )
    summary.to_csv(experiment_dir / "controlled_summary.csv", index=False)
    per_realization.to_csv(experiment_dir / "controlled_per_realization.csv", index=False)
    paired.to_csv(experiment_dir / "controlled_paired_improvements.csv", index=False)
    display(summary)
    display(paired)
    print("Representative test realization (median full-model Vp RMSE):", representative_id)
else:
    summary = per_realization = paired = pd.DataFrame()
    representative_id = None
    display(pd.DataFrame(columns=["variant", "domain", "metric", "mean", "std", "n_realizations"]))
    print("No controlled numerical claims are emitted until every matched prediction manifest exists.")
"""
        ),
        md(
            r"""
            ## Part B — Scientifically interpretable graph evidence

            The representative synthetic realization is fixed by median full-model Vp RMSE—not visual appeal. Panel (f) displays only the strongest 15% of graph edges ranked by the **actual edge weight used during message passing**. This threshold is visualization only; all graph edges remain active in the network.

            The graph-benefit panel is

            \[
            |Vp_{noGNN}-Vp_{true}|-|Vp_{full}-Vp_{true}|,
            \]

            so positive values mean graph propagation reduces absolute error. Latent embedding norm is not used as the primary scientific evidence.
            """
        ),
        code(
            """
if metrics_available:
    realization_path = dataset_dir / "realizations" / f"realization_{representative_id:07d}.npz"
    with np.load(realization_path) as archive:
        avo = archive["avo"]
        rgt = archive["rgt"]
        truth = archive["elastic"]
    with np.load(experiment_dir / "predictions" / "full" / f"realization_{representative_id:07d}.npz") as archive:
        full_prediction = archive["elastic"]
    with np.load(experiment_dir / "predictions" / "no_gnn" / f"realization_{representative_id:07d}.npz") as archive:
        no_gnn_prediction = archive["elastic"]

    normalized_avo = torch.from_numpy(avo[None].astype(np.float32))
    _, gradient = angular_features(normalized_avo)
    edge_index = build_rgt_edges(torch.from_numpy(rgt[None].astype(np.float32)), max_shift=3, steered=True)[0]
    flat_gradient = gradient[0, 0].flatten()
    contrast = torch.abs(flat_gradient[edge_index[0]] - flat_gradient[edge_index[1]])
    weight = torch.exp(-contrast / (contrast.std() + 1e-6))
    graph_edges = GraphEdges(
        source=edge_index[0].numpy(), destination=edge_index[1].numpy(), weight=weight.numpy()
    )
    figure = plot_graph_mechanism(
        avo, rgt, gradient[0, 0].numpy(), graph_edges,
        full_prediction[0], no_gnn_prediction[0], truth[0],
    )
    graph_path = figure_dir / "stage05_graph_mechanism.png"
    figure.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("Graph-benefit figure pending matched full/no-GNN predictions; no surrogate embedding plot is substituted.")
"""
        ),
        md("## Part C — Whole-image synthetic inference"),
        code(
            """
if metrics_available:
    with np.load(realization_path) as archive:
        low_prior = archive["low"]
    figure = plot_inversion_comparison(truth, low_prior, full_prediction)
    whole_path = figure_dir / "stage05_whole_image_synthetic.png"
    figure.savefig(whole_path, dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("Truth/prior/prediction/error figure pending the controlled full-model checkpoint.")
"""
        ),
        md(
            """
            ## Part D — Field deployment and QC

            The deployment input comprises real low/mid/high AVO stacks, Stage-01 RGT, and the Stage-01 field-conditioned low-frequency Vp/Vs/density model. The SEG-Y export axis remains described as the **configured gather-coordinate header** until acquisition/export metadata independently establish that it is true incidence angle.

            Whole-section inference uses the same normalization, patch size, overlap, Hann stitching, and deterministic transport integrator as synthetic evaluation. Local well curves are overlays for field consistency assessment; because they contributed upstream, they are not blind validation.
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
    "low": field_root / "usable" / version / "elastic_background.npy",
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
field_prediction = field_segmentation = None
if checkpoints["full"].exists():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_controlled_model("full", workflow, checkpoints["full"], device)
    field_prediction, field_segmentation = infer_full_realization(
        model,
        avo=field["avo"], low=field["low"], rgt=field["rgt"],
        normalization=load_normalization(dataset_dir),
        patch_shape=tuple(workflow["patches"]["shape"]),
        stride=tuple(workflow["patches"]["stride"]),
        steps=int(workflow["training"]["sample_steps_test"]),
        batch_size=int(workflow["training"]["batch_size"]),
        device=device,
    )
    np.savez_compressed(
        private_root / "stage_artifacts" / "stage05_field_prediction.npz",
        elastic=field_prediction, segmentation=field_segmentation,
        time_ms=field["time_ms"], cdp=field["cdp"],
    )
else:
    print("Field prediction pending the controlled full-model checkpoint; the field inputs are not replaced by toy data.")
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

    modeled = forward_avo_three_band(*field_prediction, config=ForwardConfig())
    agreement = compare_forward_outputs(field["avo"], modeled)
    forward_qc = pd.DataFrame({
        "band": ("near", "mid", "far"),
        "fitted_amplitude_scale": agreement.scale,
        "correlation": agreement.correlation,
        "normalized_rmse_after_scale": agreement.normalized_rmse,
    })
    display(forward_qc)
    forward_qc.to_csv(private_root / "stage_artifacts" / "stage05_field_forward_qc.csv", index=False)
else:
    print("Forward field QC is pending; no correlation values are fabricated.")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant, checkpoint in checkpoints.items():
        model = load_controlled_model(variant, workflow, checkpoint, device)
        member, _ = infer_full_realization(
            model,
            avo=field["avo"], low=field["low"], rgt=field["rgt"],
            normalization=load_normalization(dataset_dir),
            patch_shape=tuple(workflow["patches"]["shape"]),
            stride=tuple(workflow["patches"]["stride"]),
            steps=int(workflow["training"]["sample_steps_test"]),
            batch_size=int(workflow["training"]["batch_size"]),
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
    print("At least two predeclared checkpoint/prior/model variants are required; sensitivity maps remain pending.")
    print("Set SAGE_AVO_RUN_FIELD_SENSITIVITY=1 after controlled training.")
"""
        ),
        md(
            """
            ## Stage outputs

            | artifact | shape/type | scientific meaning | consumed by |
            |---|---|---|---|
            | controlled prediction packages | full test images per variant | Matched whole-realization benchmark predictions | metric/figure pipeline |
            | per-realization/summary/paired CSVs | metric tables | Performance and paired ablation evidence | paper/meeting tables |
            | graph mechanism figure | 2×4 high-resolution panel | AVO/RGT/edge mechanism and graph error reduction | IMAGE/paper |
            | whole-image synthetic figure | truth/prior/prediction/error for Vp/Vs/density | Spatial inversion performance | IMAGE/paper |
            | field prediction/QC package | whole section + coordinates + QC | Field deployment and consistency assessment | IMAGE/paper |
            | sensitivity maps | ensemble spread | Model/prior sensitivity, not posterior uncertainty | discussion |

            ## Scientific checks

            - Numerical tables require all matched controlled prediction manifests; absent results remain empty, not relabeled historical values.
            - Metrics are calculated per realization and paired by realization before aggregation.
            - The representative case is chosen by median full-model Vp RMSE.
            - Only the strongest 15% of actual edge weights are drawn; all edges remain active in message passing.
            - Whole-image inference uses common tiling, overlap, integration, normalization, and checkpoint rules.
            - Field wells are treated as non-blind consistency overlays.
            - Forward QC preserves band-specific behavior, including weak far-angle agreement.
            - Ensemble spread is labeled sensitivity rather than calibrated posterior uncertainty.

            ## Next stage

            This is the final notebook. Its artifacts feed the IMAGE presentation and manuscript only after redistribution rights and controlled-run completeness are verified in the private figure index.
            """
        ),
    ],
)
