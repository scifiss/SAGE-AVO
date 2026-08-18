# Controlled five-condition benchmark

## Scientific question

How much information does SAGE-AVO add beyond a supplied low-frequency elastic
prior, and what are the incremental contributions of graph propagation, RGT
steering, and differentiable AVO consistency?

The production contract is `configs/sage_avo_s01_v003.yaml`; Notebooks 02–05 expose
the complete data-generation, dataset, training, and evaluation sequence.

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

Notebook 04 exposes the training contract; the versioned production interface is
`scripts/run_revision3_production.py train --variant ...`. Each run writes its seed, split IDs,
normalization, prior definition, config hash, optimizer schedule, and separate
fixed-objective, sampling, segmentation, whole-realization, periodic, and last
checkpoints.

Notebook 05 exposes the evaluation contract. The production driver first
generates deterministic whole-test predictions and then reports RMSE,
MAE, R², SSIM, mIoU, macro Dice/F1, and class IoU per realization before
aggregation. Paired bootstrap intervals compare the full model with each
control. Positive paired improvement is defined to mean that the full model is
better.

The representative test realization is the ID whose full-model Vp RMSE is
closest to the test-set median. The rule and ID are saved before figure
generation.

## Development harness boundary

`scripts/run_revision3_validation.py` creates a bounded eight-realization,
two-epoch operator/science validation. It exercises the production functions
but is distinct from the 100-realization experiment and is not eligible for
performance claims.
