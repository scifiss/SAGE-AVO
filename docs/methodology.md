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
monotonicity, horizon mistie, and well-residual diagnostics. Raw PWD RGT and
its monotonic repair are preserved. An optional regularized horizon refinement
is reported separately and is disabled by default: horizon residual reduction
is not treated as sufficient reason to distort seismic-following structure.

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

The production-eligible fluid design constructs a well-calibrated physical dry
frame, computes matched brine and CO₂ states on that frame, and transfers only
the fluid-induced bulk-modulus and density changes to the RF-conditioned brine
background. Shear modulus remains fixed. Eligibility additionally requires a
versioned pressure/temperature/fluid-property validation artifact. Projected
inverse-Gassmann and absolute Hertz–Mindlin overwrite modes are retained only
for artifact compatibility and are excluded from production claims.

Spatial variability is applied to correlated geological and elastic quantities
before forward modeling. Seismic noise, gain, phase/polarity, coherent noise,
and weakened/missing far angles are observation perturbations applied after
the clean exact-physics response; their realization metadata is retained.

## AVO forward modeling

Synthetic reflectivity uses exact isotropic P-P Zoeppritz equations, followed
by a zero-phase Ricker wavelet, angle-dependent front mute, and mean stacking.
The production defaults use declared inclusive shared-endpoint bands:

- near: 3–17°;
- mid: 17–31°;
- far: 31–45°.

The shared endpoints at 17° and 31° are intentional. Compact P/G summaries use
the arithmetic band midpoints 10°, 24°, and 38°.

Shuey-derived intercept and gradient are compact network features and
diagnostics. They are not substituted for exact Zoeppritz synthetic generation.

The NumPy and differentiable Torch operators consume one serialized forward
specification. Native physics-loss patches carry absolute sample origin and a
vertical wavelet halo, so neither the shallow mute nor convolution is restarted
at each patch top. Experiments that include Madagascar use a matched test model
and per-band `compare_forward_outputs` diagnostics.

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
loss does not guarantee a better integrated inversion. The versioned training
contract also writes criterion-specific fixed-objective, segmentation, and
deterministic whole-realization validation checkpoints. Test data never select
checkpoints.

## Field-domain gate

The field prior uses the same declared 2-Hz builder as the synthetic prior but
starts from the field-conditioned elastic background; it is not truth-derived.
Raw field AVO is not passed directly through synthetic normalization. A
versioned calibration manifest must satisfy declared polarity, phase, spectrum,
amplitude, percentile-overlap, and spatial-stability diagnostics before
inference. Any amplitude/phase/wavelet transfer is explicit and recorded rather
than inferred silently.

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
