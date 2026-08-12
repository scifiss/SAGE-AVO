# Data and reproducibility

## Publicly runnable components

The following require no proprietary data, Madagascar, or GPU:

- exact NumPy Zoeppritz reflectivity and three-band forward modeling;
- Shuey P/G diagnostics;
- RGT graph construction and strongest-edge selection;
- synthetic geological perturbation;
- normalization, realization splitting, metrics, and plotting;
- unit tests and `scripts/smoke_test.py`.

The PyTorch/PyG network can run on CPU for small examples and uses a GPU for
practical training. Madagascar is an optional forward-model implementation used
for matched-operator validation.

## Private field inputs

Field SEG-Y, horizons, LAS logs, checkpoints, and derived private artifacts are
not distributed. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and
provide local paths. The real path file and all common geoscience binaries are
ignored by git.

## Experiment contract

Every experiment artifact records the configuration and Git commit. Complete
realization IDs are split before patch extraction, and normalization statistics
are fitted on the training split. Multiscale patch records include raw size and
resize factors; prior records include cutoff and construction method.

Model variants share the same prior, mask, patch index, and inference tiling.
Best-flow and best-sampling checkpoints are stored separately. Evaluation
reports per-realization and pooled statistics. Matched elastic inputs provide
cross-implementation validation for NumPy, Torch, and Madagascar forward
operators.

## Truth-derived priors

The synthetic low-frequency priors are Gaussian-filtered versions of synthetic
truth. This oracle construction isolates residual-detail refinement and is
recorded in the experiment manifests, tables, and figure captions. Priors from
low-frequency inversion or independently perturbed backgrounds define a
separate, more realistic evaluation setting.

## Expected runtimes

- unit tests and smoke test: under one minute on CPU;
- Stage-01 cached field workflow: typically under two minutes on CPU;
- synthetic dataset generation: hours and optional Madagascar;
- full model training: hours on a CUDA GPU;
- field workflow: dependent on private data and local processing software.
