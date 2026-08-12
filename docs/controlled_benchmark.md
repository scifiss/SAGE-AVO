# Controlled five-condition benchmark

## Question

How much information does SAGE-AVO add beyond a supplied low-frequency elastic
prior, and what are the incremental contributions of graph propagation, RGT
steering, and differentiable AVO consistency?

The experiment configuration is `configs/controlled_ablation_v1.yaml`. A new
configuration version represents any change to a variant or shared condition.

## Conditions

1. `low_prior`: no inference; prediction is the supplied Vp/Vs/density prior.
2. `full`: RGT-steered TransformerConv graph and physics loss.
3. `no_gnn`: CNN conditioning/decoder without graph message passing.
4. `no_rgt`: the same graph branch with Cartesian same-time horizontal edges.
5. `no_physics`: full architecture with physics-loss weight exactly zero.

The primary benchmark isolates the incremental contributions of the SAGE-AVO
components under a single shared experimental contract.

## Prior

Vp, Vs, and density priors are generated independently per channel by applying
a 2-D Gaussian filter to synthetic truth:

```text
sigma_time = 0.133 / (cutoff_hz * dt_seconds)
           = 0.133 / (2.0 * 0.004)
           = 16.625 samples

sigma_lateral = 2 * sigma_time = 33.25 traces
```

Boundary mode is `reflect`. This is an oracle truth-derived background model,
so the experiment measures elastic-prior refinement rather than AVO-only
absolute inversion.

## Immutable shared artifacts

`prepare-data` creates, once:

- 100 versioned realizations;
- a 70/20/10 realization-level split;
- exact-Zoeppritz near/mid/far AVO;
- one truth-derived prior per realization;
- training-only normalization;
- one fixed patch-index table;
- split, dataset, prior, and forward manifests.

Every learned condition reads these same files. Complete test realizations are
evaluated with the same tiled Hann blending and metrics.

## Commands

The ML extras provide PyTorch Geometric and the training dependencies:

```bash
python -m pip install -e ".[ml]"
```

The smoke configuration validates the data harness with small CPU artifacts:

```bash
python scripts/run_controlled_ablation.py \
  --experiment-name controlled_ablation_smoke prepare-data --smoke
```

The complete experiment runs as:

```bash
python scripts/run_controlled_ablation.py prepare-data
python scripts/run_controlled_ablation.py train --variant all --device cuda:0
python scripts/run_controlled_ablation.py predict --variant all --device cuda:0
python scripts/run_controlled_ablation.py evaluate
python scripts/run_controlled_ablation.py figures --device cuda:0
python scripts/run_controlled_ablation.py status
```

## Metrics and statistics

The evaluator writes:

- `results/controlled_ablation_metrics.csv` — pooled and mean ± standard
  deviation across complete test realizations;
- `results/per_realization_metrics.csv` — one realization/variant/metric row;
- `results/paired_ablation_comparisons.csv` — paired full-versus-comparator
  improvements with bootstrap confidence intervals.

Elastic metrics are RMSE, MAE, R², and Gaussian-window SSIM. Segmentation metrics
are pooled/per-realization mIoU, macro Dice, and class-wise IoU. A positive
paired improvement always means the full model is better. Confidence intervals
describe variation within this finite synthetic test family.

## Representative realization

The representative figure uses the test ID whose full-model Vp RMSE is closest
to the test-set median. The ID and selection rule are written to
`representative_realization.json` after test prediction is complete.

## Run manifests

Every trained and prediction condition records timestamp, config checksum, seed,
split IDs, variant, checkpoint, epochs, normalization, prior definition, metric
definition, hardware, and git commit when available under:

```text
results/experiments/controlled_ablation_v1/
```
