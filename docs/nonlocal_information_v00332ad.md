# v00332ad: nonlocal information decision study

This is a frozen-representation information screen, **not** a GNN redesign or
SAGE-AVO training run. RGT-specific value remains under controlled investigation.

## Frozen protocol

- Parent: `64f3c9f9243c84ec61abcd32e06d027f63d7ace2`.
- Branch: `experiment/v00332ad-nonlocal-information-value`.
- Existing validated dense v00332d epoch-40 best-whole-realization checkpoint.
- Immutable v00331 production100 support-aware train/validation/test split.
- Fit probes on all 70 training realizations; use the existing 20 validation
  realizations as **exploratory development**, not independent confirmation.
  Test is not evaluated. Verify both realization and geology-group disjointness.
- Observable flow time zero only: state is supplied low-frequency prior.
  Execute the unchanged current model's upstream encoder modules, not the graph.
  A fixture verifies exact equality with a hook on normal dense-model forward.
- Production tile size/stride and Hann blending; model parameters frozen and
  digest checked before/after extraction. FP32 CUDA batch one, no CPU fallback.
- 384 truth-free common-support query points/case and inward-facing lateral
  offsets 16/32/64 traces identical in all modes. Native RGT inverse, Cartesian
  same-time, and disrupted tau (within-source-trace derangement) correspondence.
  Record plateaus, support exclusion, delta-tau, and native shift discontinuity.
  Confidence is descriptive; fault/reservoir truth is evaluation only.

## Capacity and attribution

Targets are three training-normalized elastic residuals relative to the
supplied prior. Local inputs: current 64-channel CNN feature, three AVA bands,
three prior properties, normalized tau/gradients, and coordinates.
Remote inputs: sampled current CNN features and observed AVA, **not elastic truth**.

All four primary probes have 64 regressors plus intercept, 195 coefficients:
local-only receives 64 local PCs; local + remote receives the same first 48
local PCs plus 16 remote PCs. All remote modes share one training-only PCA
basis. This replaces local dimensions, rather than enlarging model capacity.
Report ranks/effective degrees to expose remaining effective-capacity differences.
PCA/whitening, shared ridge regularization selection, and all statistics use
training data only. Shared alpha is selected averaging the primary four
objectives on a seeded realization-disjoint 56/14 internal training split.
Then refit on all training realizations; no development tuning.

Descriptive controls: supplied prior alone, prior-only residual calibration,
equal-capacity remote-prior-only and remote-AVA-waveform probes. Supporting
conditional-innovation probes keep ALL 64 local PCs, residualize remote PCs
against them using training-only linear regression, and add 16 innovation
features. Those three probes have identical expanded capacity and are reported
separately; they cannot replace the fixed-capacity primary decision.

Property RMSE in physical units and training-standard-deviation units;
joint score is mean property NRMSE. Average per-realization scores equally;
95% paired bootstrap intervals use realizations as sampling units. Report
high-dip/reservoir/fault-adjacent/ordinary and observable confidence strata,
including unfavorable differences. Training-only dip/jump thresholds;
minimum four query samples/case/region. Full-elastic R2 uses the training-mean
elastic predictor as reference to quantify how much structure the prior explains.

## Predeclared decision

Adequacy: >=12 development cases, >=50% common support in every case, sufficient
feature rank, nondegenerate training target variance. Otherwise `PROBE_INCONCLUSIVE`.

Remote support requires >=1% mean relative improvement and a positive 95%
paired lower bound over BOTH local-only and disrupted probes. RGT-specific
support additionally requires that gate over Cartesian. This is a conjunction
of necessary conditions, not a selective regional success claim.

- RGT-specific support: `RGT_NONLOCAL_INFORMATION_SUPPORTED`.
- Credible remote support without established RGT superiority:
  `NONLOCAL_INFORMATION_USEFUL_BUT_NOT_RGT_SPECIFIC` (not equivalence).
- Neither: `NONLOCAL_INFORMATION_NOT_ESTABLISHED` (not proof of uselessness).

All decisions are exploratory due to checkpoint/validation reuse. A positive
synthetic public fixture must demonstrate that the probe detects remote-only
signal. Linear/PCA bottlenecks may still miss nonlinear geological information.
GroupNorm already supplies tile-wide context to the query CNN representation;
the prior is synthetically truth-derived by the frozen dataset procedure.
No endpoint-error or graph-architecture conclusion follows from time-zero probes.
Remote-prior controls are essential before attributing any gain to seismic.

## Execution and reproducibility

Validate, commit and push source/config/tests **before** private execution.
Use `scripts/study_nonlocal_information_v00332ad.py freeze` with explicit dataset,
checkpoint, Stage-02 sidecar directory, private output, and pushed commit SHA.
Then `... run --output PRIVATE_OUTPUT`. Runtime verifies exact branch/SHA,
clean tracked tree, source/config/test hashes, and frozen input metadata hashes.
The private contract records exact split IDs and protected file hashes.
Private per-realization caches allow resuming interrupted feature extraction
without rerunning completed cases. Results/probe weights/caches remain private.
No GNN optimizer, SAGE-AVO training, graph construction changes, commit marker,
main merge, or subsequent experiment is authorized. Stop after this study.
