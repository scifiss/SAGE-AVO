# Public scientific API eligibility (main `44fec5e`)

This matrix was completed before the `sage_avo.api` facade. “Stable” means a bounded,
testable numerical operation on caller-supplied arrays; it does **not** assert field
validation, universal geology, or that every internal default is physically admissible.

| Candidate operator / source | Eligibility | Public v0 disposition and reason |
| --- | --- | --- |
| `make_deformation`, `warp_with_deformation` (`geology.synthetic`) | STABLE_NUMERICAL | Expose with an explicit seed and one shared deformation for all channels. Generic synthetic geometry only; not a field-conditioned realization. |
| `make_field_conditioned_realization` (`geology.synthetic`) | REQUIRES_FIELD_DATA | Catalog only. Requires Stage-01 backgrounds, reservoir model, masks, configuration, and often a calibrated fluid model. |
| `apply_co2_fluid_substitution` / legacy HM modes (`geology.synthetic`) | COMPATIBILITY_ONLY | Exclude. Historical absolute/compatibility formulations are not the approved calibrated differential method. |
| `moduli_from_velocities`, `elastic_moduli_gpa`, `elastic_from_gpa` (`geology.rock_physics`) | STABLE_NUMERICAL | Expose strict-unit conversion through validated inputs; the internal generic inverse clips nonphysical inputs, so the facade uses existing strict conversion for the inverse. |
| `local_inverse_gassmann_substitution`, `matched_hm_delta_substitution` (`geology.rock_physics`) | COMPATIBILITY_ONLY | Exclude as public substitution defaults. They are not the approved calibrated differential method. |
| `hertz_mindlin_end_member`, `hashin_shtrikman_sand_frame`, matched families (`geology.dry_frame`) | REQUIRES_CALIBRATION | Catalog only. Projection-free scenario families, not measured field pressure or posterior dry-frame estimates. |
| `batzle_wang_brine` (`geology.fluid_properties`) | STABLE_NUMERICAL | Expose with source support bounds: 5–60 MPa, 20–100 °C, NaCl mass fraction 0–0.32. |
| `span_wagner_co2` (`geology.fluid_properties`) | STABLE_NUMERICAL | Expose with lazy CoolProp dependency and the source’s single-phase/supercritical checks. |
| `sample_fluid_scenario` (`geology.fluid_properties`) | REQUIRES_CALIBRATION | Catalog only; scenario ranges and seed policy belong to a declared application contract. |
| `calibrated_differential_gassmann_substitution` (`geology.fluid_calibration`) | REQUIRES_CALIBRATION | Conditional facade only. Requires an actual `CalibratedDryFrameModel`, `FluidRockPhysics`, declared region, depth and a `SupportAcceptanceContract`; no implicit calibration or alternate formulation. |
| `constrained_local_gassmann_substitution` (`geology.fluid_calibration`) | COMPATIBILITY_ONLY | Exclude from default facade; separate Candidate A, not approved Candidate B. |
| `evaluate_candidate_support` (`geology.support`) | REQUIRES_FIELD_DATA | Catalog only. Complete-realization acceptance additionally needs field time/depth, plume, metadata and scenario contract. |
| `build_horizon_conditioned_fields`, `predict_elastic_fields` (`geology.field_conditioning`) | REQUIRES_FIELD_DATA | Catalog only. Well/horizon observations and calibrated elastic models are required. |
| `monotonicity_report` (`structure.rgt`) | STABLE_NUMERICAL | Expose finite-array QC. RGT monotonicity is not evidence of fault-safe reflector identity. |
| `repair_rgt_monotonicity` (`structure.rgt`) | STABLE_NUMERICAL | Retain internal path for now; isotonic repair changes supplied coordinates and needs explicit scientific review by a consumer. |
| PWD estimation, horizon refinement/projection (`structure.rgt`, `structure.horizons`, `structure.calibration`) | REQUIRES_FIELD_DATA | Catalog only. Depend on seismic/well/horizon geometry and optional field packages; not generic geology. |
| `build_rgt_graph` and later reflector-component/dynamic-GNN work (`structure.graph`, experimental branches) | EXPERIMENTAL | Exclude. RGT alignment alone does not prove fault-safe geological identity; no experimental branch is merged. |
| `zoeppritz_pp`, `reflectivity_gather` (`forward.zoeppritz`) | STABLE_NUMERICAL | Expose exact real-valued PP coefficients on a validated pre-critical domain. Internal post-critical complex solve returns only its real component; facade rejects post-critical states rather than presenting that as a full complex coefficient. |
| `ForwardModelSpecification`, `forward_avo_dense_spec` (`forward.specification`, `forward.pipeline`) | STABLE_NUMERICAL | Expose the existing spec and NumPy execution unchanged, with validated elastic inputs, sample origin, and spec hash. Bands retain inclusive shared endpoints. |
| `compare_forward_outputs` (`forward.qc`) | STABLE_NUMERICAL | Expose for independent `[band,time,trace]` outputs; comparison does not establish ground truth. |
| Torch exact-PP/forward (`forward.torch_forward`) | OPTIONAL_GPU_ML | Exclude from NumPy facade. Existing optional parity tests remain authoritative. |
| Madagascar reference path (`forward.madagascar`) | REQUIRES_FIELD_DATA | Exclude from small facade; optional external executable and reference-workflow conventions. |
| `plot_inversion_comparison` (`visualization.figures`) | STABLE_NUMERICAL | Expose lazily for caller-supplied arrays; visualization never recomputes geology or AVO. |
| Publication figure pipeline (`visualization.publication`) | REQUIRES_FIELD_DATA | Exclude. It loads experiment outputs/checkpoints and is not an independent array-only renderer. |

The v0 facade does not expose production inversion, experimental graph topology,
learned models, training, checkpoint loading, or field-conditioned generation.
