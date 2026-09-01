# Controlled five-condition benchmark

## Scientific question

How much information does SAGE-AVO add beyond a supplied low-frequency elastic
prior, and what are the incremental contributions of graph propagation, RGT
steering, and differentiable AVO consistency?

The final lineage is Stage-02 `v00331_production100_support_aware`, immutable
Stage-03 `ds_v00331_production100_support_aware`, and the v00332d training
contract in `configs/final_training_v00332d.yaml` resolved over
`configs/sage_avo_s01_v0031.yaml`. Notebooks 02–05 expose this final lineage.

## Matched conditions

1. `low_prior`: supplied Vp/Vs/density prior without learned refinement;
2. `full`: RGT-steered TransformerConv graph with physics loss;
3. `no_gnn`: matched CNN conditioning/decoder without graph message passing;
4. `no_rgt`: matched graph branch with Cartesian lateral connectivity;
5. `no_physics`: full architecture with the physics-loss weight set to zero.

Any external architecture is included only after retraining with the identical
prior, realization split, normalization, masks, inference tiling, and
checkpoint-selection rule. Unmatched external values are not controlled
comparisons.

## Prior and inverse-problem scope

Synthetic Vp, Vs, and density priors are generated per realization by applying
a Gaussian filter to target/truth properties:

```text
sigma_time = 0.133 / (2.0 Hz * 0.004 s) = 16.625 samples
sigma_lateral = 2 * sigma_time = 33.25 traces
boundary mode = reflect
```

This oracle truth-derived prior makes the task AVO-guided refinement of a
supplied low-frequency model, not AVO-only absolute-property inversion.

## Immutable shared artifacts

Notebook 02 produces field-conditioned realizations with exact dense-angle
Zoeppritz observations. Notebook 03 then creates, once:

- a split of complete realization IDs before patching;
- one truth-derived prior per realization;
- training-realization-only normalization;
- one multiscale patch index with raw size and resize metadata;
- split, dataset, prior, and source-manifest hashes.

Every learned condition reads these exact files. Complete test realizations use
the same tiling, Hann blending, integration steps, masks, and metrics.

## Training and evaluation

Notebook 04 exposes the final training contract. The completed full-model
production interface is `scripts/run_revision332d_final_training.py`; future
matched component-removal runs must use a predeclared common harness over the
same immutable v00331 dataset. Each run writes its seed, split IDs,
normalization, prior definition, config hash, optimizer schedule, and separate
fixed-objective, sampling, segmentation, whole-realization, periodic, and last
checkpoints.

Notebook 05 exposes the evaluation contract. The production driver first
generates deterministic whole-test predictions and then reports RMSE,
MAE, R², SSIM, mIoU, macro Dice/F1, and class IoU per realization before
aggregation. Paired bootstrap intervals compare the full model with each
control. Positive paired improvement is defined to mean that the full model is
better.

At present only the full v00332d checkpoint is final. Matched final no-GNN,
Cartesian/no-RGT, no-physics, and faithful-HCTNet checkpoints are still missing;
bounded gates, v002 results, sanity runs, and historical HCTNet values are not
paper-quality controlled evidence.

## Runnable epoch-40 baseline

The public driver can inspect prerequisites, generate deterministic predictions
for the supplied low-frequency prior and validation-selected epoch-40 full
checkpoint, and evaluate those two conditions:

```bash
PYTHONPATH=src python scripts/run_final_baseline.py status
PYTHONPATH=src python scripts/run_final_baseline.py predict --device cuda:0
PYTHONPATH=src python scripts/run_final_baseline.py evaluate
```

The defaults resolve private data/checkpoint locations through
`configs/paths.yaml`; `--dataset`, `--checkpoint`, and `--output` provide
explicit overrides. This is a baseline comparison, not evidence for the graph,
RGT, or physics ablations.

Before running model cells, install and select the WSL conda kernel explicitly:

```bash
conda run -n sage-avo python -m ipykernel install --user \
  --name sage-avo --display-name "Python (sage-avo CUDA)"
```

In Jupyter choose **Python (sage-avo CUDA)**. Notebooks 04 and 05 print
`sys.executable`, the PyTorch/CUDA build, CUDA availability, GPU name, and the
selected device before model work. They refuse to silently run production model
work on CPU. NumPy preprocessing, file I/O, and optional Madagascar operations
remain on CPU.

The 4-GB matched-control contract uses batch size 2. Mixed precision remains
disabled until the differentiable exact-PP operator and graph scatter/attention
path pass a dedicated FP16/BF16 numerical-equivalence and finite-gradient gate;
silently changing precision would invalidate the matched scientific contract.

## Predeclared matched controls

`configs/matched_training_v00332e.yaml` freezes the shared training contract and
the only allowed differences among `full`, `no_gnn`, `no_rgt`, and
`no_physics`. Preparation records the contract but never starts training:

```bash
PYTHONPATH=src python scripts/run_matched_training_v00332e.py status
PYTHONPATH=src python scripts/run_matched_training_v00332e.py prepare
PYTHONPATH=src python scripts/run_matched_training_v00332e.py train --variant no_gnn --device cuda:0
```

Each training command requires an explicit variant and device and refuses to
overwrite an existing run. The optional experimental objectives remain off so
these runs isolate the advertised architecture/physics components.

For laptop-safe execution, stop at an absolute epoch boundary and resume later:

```bash
# New run or resume to epoch 1.
PYTHONPATH=src python scripts/run_matched_training_v00332e.py train \
  --variant full --device cuda --until-epoch 1

# Resume the same run from last.pt and stop after epoch 3.
PYTHONPATH=src python scripts/run_matched_training_v00332e.py train \
  --variant full --device cuda --until-epoch 3
```

Every completed epoch writes an atomic `last.pt`, fsyncs `training_log.csv`, and
preserves optimizer, scheduler, task-weighter, and RNG states. If power is lost
mid-epoch, the next command repeats only that incomplete epoch. Durable batch
progress is appended every 50 optimizer steps to `training_progress.log`.
`--archive-incomplete` is required to preserve and replace a crash directory
that has no completed-epoch checkpoint.

The representative test realization is the ID whose full-model Vp RMSE is
closest to the test-set median. The rule and ID are saved before figure
generation.

## Development harness boundary

`scripts/run_revision3_validation.py` creates a bounded eight-realization,
two-epoch operator/science validation. It exercises the production functions
but is distinct from the 100-realization experiment and is not eligible for
performance claims.
