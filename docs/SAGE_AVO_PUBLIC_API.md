# SAGE-AVO scientific Python API v0.1

`sage_avo.api` is a small, application-independent facade over existing SAGE-AVO
numerical code. It is **not** an inversion service, a model checkpoint API, a
field-validation claim, or a replacement for the research pipeline. Consumers
own orchestration, storage, plots, and any independent architecture. The
[eligibility matrix](SAGE_AVO_API_ELIGIBILITY.md) records why other internals
were not promoted.

## Install, version and discovery

Pin an exact SAGE-AVO Git commit or a released `sage-avo` package version. The
package metadata currently requires Python ≥3.10, NumPy ≥1.24,<2, SciPy ≥1.10,
Pandas ≥2, Matplotlib ≥3.7, PyYAML ≥6 and CoolProp ≥7.1,<8; Torch/PyG and field
readers remain optional. No dependency constraint was changed for this API.

```python
from sage_avo import api

print(api.API_VERSION)                    # public schema, independent of package version
for operation in api.list_operations():  # metadata only; never executes kernels
    print(operation["identifier"], operation["eligibility"], operation["available"])
print(api.provenance("elastic_moduli"))
```

`list_operations(include_unavailable=False)` returns only callable entries.
The installed `catalog.json` is also machine-readable. Returned dictionaries
are detached copies; mutating them cannot change the catalog. Catalog discovery
does not import Torch, PyG, CoolProp, Matplotlib or initialize a GPU. Individual
operators load only their relevant dependencies when called. Missing CoolProp
raises an actionable `ImportError` from the authoritative CO₂ implementation.

`provenance()` distinguishes the API schema version, installed package version,
optional Git source revision, operator/formulation version, structured bounded
validation status, optional forward specification ID and SHA-256, optional
calibration ID, and SHA-256 of a caller-supplied JSON-serializable input
configuration. A wheel installed without `.git` has no Git revision; pin the
package version or an exact Git commit separately. The API schema is pre-1.0:
incompatible facade changes may increment its minor version; operator versions
track scientific contracts independently. Internal imports and old workflows
remain unchanged. Experimental branches are not part of this stability promise.

## Common numerical conventions

All section arrays use `[time_sample, trace]`; channels-first arrays use
`[channel, time_sample, trace]`. Inputs are caller-owned and are never mutated.
The facade requires finite inputs for numerical operations, except
`compare_avo_outputs`, which uses pairwise-finite samples by band. No implicit
field data, calibration artifact, checkpoint, mask, spatial CRS, training
statistics, or seismic amplitude unit is assumed. Floating calculations use
NumPy `float64` where the source does; `ForwardResult` arrays intentionally use
source `float32` outputs. Seeded deformation uses a local NumPy generator and
does not change global random state. CPU execution is sufficient throughout.

### Public independent operators

| Entry point (operator version) | Inputs, output and applicability |
| --- | --- |
| `make_coherent_deformation` (`synthetic-deformation-v1`) | Integer `(time,trace)` shape ≥2, nonnegative seed, optional **existing** generator parameters (fold/fault displacement in samples). Returns source `Deformation`: two `[time,trace]` displacement arrays in sample/trace coordinates plus seed-independent geometry metadata. Same seed/inputs reproduce it. It is generic synthetic deformation, not a field realization. Invalid shape/seed raises. |
| `warp_geological_arrays` (`shared-warp-v1`) | Nonempty mapping of finite `[time,trace]` or `[channel,time,trace]` arrays, one `Deformation`, explicit `categorical_keys`. Returns new arrays with same keys/shapes. Continuous values use source order-1 interpolation, categorical masks order-0, boundary mode `nearest`. This operation does not enforce geological/elastic bounds; mask and support interpretation remain the consumer’s responsibility. Invalid shape/key raises. |
| `elastic_moduli` (`elastic-gpa-v1`) | Matching finite Vp/Vs in m/s and density in g/cc, `Vp>Vs>0`, positive isotropic bulk modulus. Returns `(K,G)` in GPa, same shape, `float64`; no clipping. Invalid physical states raise. Calls `elastic_moduli_gpa`. |
| `elastic_from_moduli` (`elastic-strict-inverse-v1`) | Matching finite positive K/G in GPa and density in g/cc. Returns source `ElasticProperties` with Vp/Vs m/s and density g/cc. Uses `elastic_from_gpa_strict`, **not** the legacy inverse that floors values. Invalid states raise. |
| `brine_fluid_state` (`batzle-wang-1992`) | Pressure 5–60 MPa, temperature 20–100 °C, NaCl mass fraction 0–0.32. Returns source `FluidPropertyState` with density g/cc, acoustic velocity m/s, modulus GPa and model provenance. Out-of-support states raise; correlation is empirical. |
| `co2_fluid_state` (`coolprop-span-wagner-v1`) | Pressure MPa and temperature °C inside CoolProp fluid limits, supercritical by default. Returns `FluidPropertyState`; two-phase/unknown/critical states and extrapolation raise. CoolProp loads only on invocation. A pinned CoolProp version matters for exact reproducibility. |
| `exact_pp_reflectivity` (`numpy-zoeppritz-matrix-v1`) | Matching finite isotropic Vp/Vs m/s, density g/cc `[time,trace]`, ≥2 times, strictly increasing angles 0–55°. Returns dimensionless `float64` `[angle,time,trace]` interface coefficients; first time sample is zero. Facade rejects post-critical P transmission. The internal complex solve returns a real component post-critical; this facade does **not** advertise that as a full complex coefficient. Singular systems raise from NumPy. No wavelet or mute is applied. |
| `simulate_three_band_avo` (`forward-model-spec-v003-numpy`) | Same elastic arrays, existing `ForwardModelSpecification` with ordered near/mid/far bands and unique angle–wavelet mapping, nonnegative global `sample_origin`. Returns source `ForwardResult`: `float32` reflectivity and convolved seismic `[angle,time,trace]`, stacks `[3,time,trace]`, angle degrees and band names. The spec controls wavelet ID, phase, sampling, `constant_zero_same` convolution, inclusive shared 17°/31° endpoints, amplitude normalization and angle-dependent front mute. Sample origin is in the **whole-section** time frame. Patch callers must supply the required wavelet halo; a cropped patch cannot invent missing neighboring reflectivity. The facade rejects post-critical states; source functions are otherwise unchanged. |
| `compare_avo_outputs` (`forward-agreement-v1`) | Two aligned `[3,time,trace]` near/mid/far arrays in the same amplitude units. Requires at least one pairwise-finite sample per band. Returns source `ForwardAgreement` (scale, correlation, normalized RMSE per band). Constant bands may yield `NaN` correlation as in source. Neither input is declared truth. |
| `report_rgt_monotonicity` (`rgt-qc-v1`) | Finite `[time,trace]` RGT, ≥2 time samples, tolerance in the caller’s RGT units (default −0.001). Returns `fraction_bad`, `worst_step`. The caller supplies the RGT coordinate reference. Passing monotonicity does **not** establish fault-safe geological identity. |
| `plot_elastic_comparison` (`array-figure-v1`) | Matching finite `[3,time,trace]` truth/prior/prediction channels (Vp/Vs m/s, density g/cc). Returns a Matplotlib `Figure`; caller saves/closes it. It renders existing arrays only and inherits source titles and percentile scaling. |

`ForwardModelSpecification`, `WaveletSpecification`, `AngleBand` and
`forward_specification_from_mapping` are re-exported existing contracts, not a
competing `ForwardSpec`. For the controlled v003 physics identity, construct
the specification from an explicit copy of the controlled mapping in your own
configuration, then record `specification.sha256`. The public examples use a
clearly marked synthetic fixture. Do not assume an arbitrary specification is
the production one merely because its operator can execute.

### Conditional calibrated operation

`substitute_calibrated_fluid` (`candidate-b-calibrated-differential-v1`)
requires all of:

- Brine baseline `[3,time,trace]` (Vp/Vs m/s, density g/cc), matching finite
  input porosity, DELTA/shaliness, CO₂ saturation fractions, depth in metres,
  and an explicit nonempty boolean region. Saturation outside the region must
  be exactly zero.
- A caller-supplied, **approved** `CalibratedDryFrameModel`, explicit
  `FluidRockPhysics`, and matching `SupportAcceptanceContract`. Nothing is
  inferred, loaded, fabricated, or silently replaced by local inverse Gassmann.

The adapter invokes only the source
`calibrated_differential_gassmann_substitution` on saturated region pixels,
then assembles those values into a copy of the baseline. Zero saturation and
all pixels outside the region remain exactly unchanged. Its result contains
`elastic` `[3,time,trace]`, the declared mask, calibration ID, maximum nearest
calibration distance, and method ID. It rejects calibration-ID mismatch,
nearest-neighbor distance beyond the supplied threshold, out-of-domain depth
or effective porosity, dry-frame ratio/Poisson violations, and fixed-shear
tolerance violations. This is a conservative **per-pixel** gate, not the
complete Stage-03 candidate support/coverage acceptance. A synthetic fixture
demonstrates mechanics only; it is never a field-approved calibration.

Field-conditioned geological realization, dry-frame scenario families,
historical fluid modes, PWD/horizon field calibration, Torch/PyG inference,
dynamic graphs, and publication/checkpoint pipelines remain unavailable in the
v0 facade. Their reasons appear in the eligibility matrix and catalog.

## Examples, verification and compatibility

Run the four independent synthetic examples from `examples/public_api/` after
installing the package. They cover shared deformation, fluid-state evaluation
and conditional synthetic-fixture substitution, exact PP plus three-band AVO,
and array-only revisualization. No example loads private data or a checkpoint.

Focused tests compare facade values directly against authoritative source
operators (exact equality where wrappers pass the same inputs), check physical
invariants, deterministic seeding, zero-saturation identity, exterior
preservation, invalid-state rejection, and optional import boundaries. The
external-consumer smoke test runs from outside this repository against a
committed/pinned package, not from `PYTHONPATH=src`. Unit parity does not imply
independent field validation, calibrated uncertainty, or superiority of an
experimental graph/inversion method.

The facade adds no REST/MCP endpoint, agent registry, planner, model training,
GPU initialization, release, or changes to existing numerical kernels. Existing
internal import paths remain available, but are not all stable public contracts.
