# Optional hybrid reflector graph: v00332ac

The research path preserves the dense SAGE-AVO model and frozen v00332z tracker.
`TopologyCache.prepare` constructs one graph from full native observed AVO, RGT
and valid support. Its private JSON records native coordinates, accepted component
IDs/edges, observable input hashes, frozen parameter hashes and diagnostic-source
hashes. Existing inconsistent caches fail rather than being regenerated.

`IndexedRealizationPatches(topology_cache=cache, matched_augmentation=True)` adds
an aligned graph after obtaining the original realization/crop/resize metadata.
`collate_sparse_patches` stacks tensors and retains graphs as a variable-length
list. The training engine transfers the list to the selected device and supplies
it to the hybrid forward call. The normal tensor-only loader/forward call remains
available with the existing defaults. Sampler scores, replacement behavior and
dedicated random streams remain unchanged.

The matched augmentation path uses the same flip draw for all tensors and graph
coordinates. It also flips native physics halos, clean physics observations and
masks, which legacy augmentation did not synchronize. This correction applies
equally to future A/B/C runs. Gain and noise change patch observations without
rerunning detection. Inference obtains graphs through `cache.tile_provider` after
verifying the full native observation identity. Both paths induce graphs using
only endpoints whose CNN features are available in the crop.

The previous pre-decoder injection is preserved as `LegacyDecoderHybridSAGEAVO`
solely for the component-coupling audit. Its spatial GroupNorm couples sparse
components. `HybridSparseSAGEAVO` instead samples current pre-GNN CNN features,
applies two component-local TransformerConv layers, deposits only uncontested
bilinear footprints and adds a pointwise post-decoder three-channel velocity
residual. There is no spatial normalization on this added residual. Gamma zero
delegates to the identical dense model.

Direct sparse contributions stay on that footprint and cannot pass from one
accepted component to another. During Heun integration, however, changed states
enter the existing dense CNN and spatial normalization on subsequent evaluations.
Endpoint property and segmentation changes can consequently extend beyond the
instantaneous sparse footprint. The audit measures this effect explicitly;
instantaneous locality does not establish whole-trajectory locality.

The frozen A/B/C proposal is `configs/development_diagnostics_v00332ac.yaml`.
`resolve_hybrid_config` resolves the shared final objective and
`build_hybrid_condition` initializes equal dense weights for all conditions and
equal sparse parameters for B/C, without advancing external random streams.
A uses dense SAGE-AVO. C uses genuine node source payloads. B permutes active
source keys/values within each accepted component at every sparse layer; target
queries/root features, endpoints, degrees, descriptors and shapes stay identical.
This deliberately disrupts message-source geological correspondence while
retaining the same graph and parameter capacity. It is not a graph-rewiring claim.

`train_controlled_variant(..., variant="full", hybrid_condition="A" | "B" | "C",
topology_cache=cache)` is the optional future entry point. Each condition has its
own run directory/manifest and cannot resume another condition. Prepare full
train/validation caches first. The immutable test split is reserved for later
evaluation after the protocol is frozen; reused validation cases are exploratory.
Historical test exposure needs confirmation before claiming independence.

The diagnostic command verifies the pushed commit, clean tracked tree and
source/config/test hashes before creating private artifacts:

```bash
PYTHONPATH=src python scripts/check_hybrid_integration_v00332ac.py \
  --private-root "$SAGE_PRIVATE_ROOT" \
  --expected-commit "$SAGE_EXPERIMENT_COMMIT" \
  --device cuda
```

This command performs cache, loader, augmentation, component and full-flow QC,
including one isolated backward through the complete objective. It creates no
optimizer, loads no checkpoint and does not train. Results remain private.
