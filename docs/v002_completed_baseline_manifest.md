# Completed v002 baseline: immutable archival record

## Status

> Computationally complete 120-epoch v002 baseline. Demonstrates stable
> optimization, active conditional graph learning, and improvement over the
> supplied 2-Hz prior on diagnostic validation examples. It is not the final
> corrected scientific experiment because fluid substitution, patch-domain
> forward consistency, and field-domain calibration are superseded in v003.

The frozen archival package is maintained outside the distributed repository as
`archives/v002_completed_baseline_20260816/`. Its manifest SHA-256 is
`e2cb46c732efa674eabf6b31f5d2228a4391c6ca70b551b346a76d5580ca2059`.
The archive includes the exact source commit (`abea749`), worktree diff,
configurations, Stage-02/03 manifests, normalization, checkpoints, logs, and
all completed epoch-120 diagnostic figures.

## Checkpoints are criteria, not a universal ranking

| immutable file | epoch | actual selection rule | SHA-256 |
|---|---:|---|---|
| `best_sampling.pt` | 52 | minimum mean normalized elastic RMSE minus `0.1 × mIoU` | `99b917087587d1fb82b5ae9889b3f53aab21a5239640a55063af9233d2ee85a5` |
| `best_flow.pt` | 120 | minimum epoch-dependent weighted validation total | `510e7dc8efcc500721ee5f1f5759027eeb6d443425eba88d180ad045b90d556c` |
| `last.pt` | 120 | final optimizer/model/RNG state | `10829ceaed5ec270781438b444c1ac86ddb6d86e7da1a27af416e3ff6914d13f` |

The `best_flow.pt` name is inaccurate: raw validation flow was lowest at epoch
94, but the file was selected using the dynamically weighted total. Subsequent
experiment contracts therefore use criterion-accurate names such as
`best_fixed_objective.pt`.
Epoch 29 (best sampled segmentation mIoU) and epoch 94 (best raw validation
flow and SSIM) were not persisted and cannot be reconstructed.

`best_flow.pt` and `last.pt` have different container hashes because their
optimizer/RNG payloads differ, but their model-state SHA-256 is identical:
`ef8d935393f09b604d1f41958b84650b6e5c55f5ad698ab0755bc26744abefd1`.

## Common fixed evaluation protocol

The common fixed evaluation protocol uses:

- the same train-derived normalization and valid masks;
- 20 deterministic Heun steps and zero physics-guidance scale;
- one native 50×100 patch per sorted validation realization, chosen as the
  first indexed native patch without consulting labels or model error;
- a fixed 20-patch suite spanning all 20 validation realization IDs;
- fixed final loss weights for raw objective comparison;
- one predetermined whole validation realization (ID 1000004) and one
  predetermined whole test realization (ID 1000002), tiled identically for
  both unique model states.

### Fixed 20-patch sampling suite

| checkpoint | Vp RMSE (m/s) | Vs RMSE (m/s) | density RMSE (g/cc) | class-0/1/2 IoU | mIoU | sampling criterion |
|---|---:|---:|---:|---|---:|---:|
| epoch 52 | 102.38 | 89.92 | 0.01306 | 0.962 / 0.176 / 0.663 | 0.601 | 0.3496 |
| epoch 120 | 106.60 | 90.97 | 0.01305 | 0.959 / 0.147 / 0.630 | 0.579 | 0.3591 |

Epoch 52 is better under this sampling criterion and for Vp, Vs, and
segmentation. Epoch 120 is marginally better for density and is better under
the fixed-final-weight interior-time objective (`0.06377` versus `0.06520`).
This is genuine task/criterion competition, not a single “best checkpoint.”

### Deterministic whole-realization diagnostics

| split / realization | checkpoint | Vp RMSE | Vs RMSE | density RMSE | Vp/Vs/density improvement over 2-Hz prior | mIoU |
|---|---|---:|---:|---:|---|---:|
| validation / 1000004 | epoch 52 | 80.19 m/s | 71.54 m/s | 0.01049 g/cc | 43.3% / 39.3% / 28.5% | 0.597 |
| validation / 1000004 | epoch 120 | 85.22 m/s | 73.50 m/s | 0.01073 g/cc | 39.7% / 37.6% / 26.8% | 0.583 |
| test / 1000002 | epoch 52 | 84.35 m/s | 76.38 m/s | 0.01140 g/cc | 43.4% / 40.4% / 25.5% | 0.589 |
| test / 1000002 | epoch 120 | 91.11 m/s | 76.43 m/s | 0.01141 g/cc | 38.9% / 40.3% / 25.4% | 0.583 |

These two images are deterministic archival diagnostics, not a corpus-wide
test claim. Corpus-level reporting requires per-realization aggregation over
the complete held-out split.

## Metric-ordering reconciliation

The training log sampled the first 16 validation batches: 128 indexed patches,
all from realization 1000064, with a mixture of scales. The completed-run
figure used one different patch: realization 1000004 at `(top=53, left=40)`,
selected for maximum non-background support. The apparent Vp ranking reversal
therefore came from different samples and aggregation, not from converting to
physical units. For one property, linear denormalization multiplies RMSE by its
positive training standard deviation and cannot reverse checkpoint ordering.
A regression test enforces that invariant.

## Segmentation audit

All Stage-03 `valid_mask` pixels are valid in v002; masking does not remove a
class selectively. Whole-realization label frequencies are:

| split | background | sand | CO₂/plume |
|---|---:|---:|---:|
| train | 97.041% | 2.296% | 0.663% |
| validation | 97.119% | 2.199% | 0.682% |
| test | 97.023% | 2.420% | 0.557% |

The implementation computes inverse-frequency weights on valid training
patches, normalizes them, and then applies the configured 1.15 foreground
boost. The reproduced weights are `[0.02338, 0.90862, 2.51449]`; their implied
sampled-patch frequencies are about `[96.13%, 2.84%, 1.03%]`. The weighting is
implemented as configured. The observed foreground decline is consistent with severe
imbalance plus competition between elastic, physics, graph, and segmentation
objectives. It is not attributable to a mask bug. The v002 log excludes
per-class metrics for each epoch, and epoch 29 was not persisted; consequently,
the per-class degradation trajectory is not recoverable.

## Graph evidence

The archive preserves the actual RGT-conditioned graph, graph-reinjection
effect, RGT-versus-Cartesian steering effect, learned `TransformerConv`
attention, and graph-branch parameter activity. These show that the graph
branch is active and functional. They do **not** establish an accuracy benefit;
that requires matched v003 no-GNN and no-RGT ablations.

## Known v002 scientific limitations

1. The absolute Hertz–Mindlin fluid overwrite is not locally matched to the
   Random-Forest brine background.
2. Patch-local physics convolution lacks a native vertical halo and restarts
   the shallow mute at each patch top.
3. Patch candidates are uniform and include coordinate duplicates.
4. Field amplitudes, phase, polarity, spectrum, and wavelet transfer were not
   calibrated, so real-field inference is not reported.
5. Vp/Vs/density are improved relative to a truth-derived 2-Hz prior; this is
   prior refinement, not unconstrained AVO-only absolute inversion.

The machine-readable common-protocol tables and predictions are retained with
the archival package and are not distributed in this release.
