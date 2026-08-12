# Methodology

## Scientific objective

SAGE-AVO refines supplied low-frequency Vp, Vs, and density models. It uses
near-, mid-, and far-angle AVO stacks as data evidence and relative geologic
time (RGT) as a structural coordinate. The output is a higher-resolution
elastic model and, when enabled, a facies/CO₂ segmentation.

This is a conditional inverse problem. The low-frequency elastic model is part
of the input; the method is not positioned as AVO-only absolute inversion.

## Field-conditioned structural model

The research workflow stacks a 2-D seismic line, estimates local slopes with
plane-wave destruction, performs structure-oriented smoothing, and integrates
the refined slope field into RGT. Horizons and wells constrain reference RGT
levels. Facies and porosity are interpolated in Wheeler/RGT coordinates before
being unflattened to seismic time.

The structural utilities are dependency-light. PySeistr/PWD is an optional
field-processing dependency; its outputs are evaluated through RGT
monotonicity, horizon mistie, and well-residual diagnostics.

## Facies convention

The source well workflow treats lower normalized DELTA as cleaner sand.
SAGE-AVO therefore uses one explicit convention everywhere:

```text
DELTA = 1 - P(sand)
```

`sage_avo.geology.conventions` is the single implementation of this mapping.

## Synthetic geology and fluid substitution

Synthetic members coherently perturb sand probability, porosity, and RGT using
folds, correlated displacement fields, and faults. Reservoir-constrained plume
masks may drive fluid substitution. These realizations are diverse members of a
field-conditioned geological family, not independent regional geology.

Rock-physics utilities expose modulus conversion and Gassmann substitution.
Field deployment uses dataset-specific mineral/fluid properties and calibrated
depth-domain assumptions.

## AVO forward modeling

Synthetic reflectivity uses exact isotropic P-P Zoeppritz equations, followed
by a zero-phase Ricker wavelet, angle-dependent front mute, and mean stacking.
The public defaults use non-overlapping inclusive bands:

- near: 3–17°;
- mid: 18–31°;
- far: 32–45°.

Shuey-derived intercept and gradient are compact network features and
diagnostics. They are not substituted for exact Zoeppritz synthetic generation.

The NumPy and differentiable Torch operators share assumptions. Experiments
that include Madagascar use a matched test model and per-band
`compare_forward_outputs` diagnostics.

## Model

At flow time `t`, the model receives:

- current normalized elastic state;
- normalized three-band AVO;
- normalized low-frequency elastic prior;
- RGT;
- continuous time `t`.

A CNN extracts local spatial features. The three AVO bands are augmented with
Shuey-style intercept `P`, gradient `G`, and angular curvature. Each image pixel
becomes a graph node. For each pixel, an adjacent-trace edge selects the sample
within ±3 vertical indices whose RGT is closest. Trace-wise vertical edges are
also included. TransformerConv layers propagate information over these edges.

Edge attributes decay with the difference in AVO gradient. Similar-gradient
nodes therefore exchange information more strongly. TransformerConv graph
attention operates on the RGT-steered graph branch.

## Deterministic conditional residual transport

Training uses the straight path

```text
x_t = (1 - t) low + t target
u_t = target - low
```

The network estimates `u_t`. Inference begins at the low-frequency prior and
integrates the learned velocity with Heun's method. There is no random latent
initial state, so this is deterministic conditional residual transport rather
than a fully probabilistic flow posterior.

## Objective

The multitask objective combines:

- masked flow-velocity MSE;
- masked full-property MSE;
- class-weighted cross-entropy and multiclass Dice;
- differentiable exact-Zoeppritz AVO consistency;
- edge-weighted structural smoothness.

Sampling metrics and flow loss are checkpointed separately because lower flow
loss does not guarantee a better integrated inversion.

## Evaluation design

The controlled synthetic comparison uses shared input/prior conditions and
includes:

1. low-frequency prior only;
2. full SAGE-AVO;
3. no GNN;
4. no RGT steering;
5. no physics loss.

Additional architectures enter the comparison under the same low-frequency
prior, split, normalization, mask, patch index, inference, and checkpoint rule.
