# Scientific design safeguards

The versioned SAGE-AVO workflow separates compatibility modes from the
scientifically supported production defaults.

| Failure mode | Scientific consequence | Implemented safeguard |
|---|---|---|
| Horizon levels could be forced into RGT to reduce pick RMSE. | Lower mistie could be purchased by distorting seismic-following structure. | Preserve raw and monotonic-repaired RGT; make bounded, smoothed horizon refinement optional, separately measured, and disabled by default. |
| Independently fitted RF Vp/Vs/density and porosity were treated as a jointly Gassmann-consistent brine state. | Inverse Gassmann required widespread mineral-modulus projection and produced unsupported near-zero dry bulk moduli. | Calibrate a physical dry-frame family to wells, calculate matched brine/CO₂ states on the same frame, transfer only bulk-modulus and density changes to the RF background, and require an independently validated pressure/temperature/fluid-property artifact. |
| Elastic-label variation could be treated as independent pixel noise. | Vp, Vs, and density could violate geological correlation and rock-physics consistency. | Perturb correlated geological/dry-frame/elastic quantities before physics; apply traceable seismic noise, gain, phase, and far-angle degradation only after forward modeling. |
| Stage-02 and patch-domain training physics used separately encoded operator assumptions. | Mute origin and convolution boundaries could change with patch location; the loss could disagree with stored AVO. | Use one serialized exact-PP specification, absolute sample origins, native physics patches, and vertical convolution halos; enforce a NumPy/Torch round trip. |
| JSON key sorting could reverse the near/mid/far band order. | Replaying a resolved configuration could attach the wrong representative angles to input channels. | Canonicalize bands by physical lower angle and regression-test sorted-JSON replay. |
| Uniform candidates underrepresented difficult structures. | Training patches could miss facies boundaries, dip, RGT change, and AVO-gradient transitions. | Restore deterministic diverse categories, depth bins, minimum separation, and coordinate deduplication; retain uniform sampling as a controlled option. |
| Observation variants could be split independently. | The same geology could leak across train, validation, and test. | Assign a geology-group ID and split all wavelet/noise variants as one group. |
| Raw field bands could be normalized with synthetic statistics without a domain check. | Apparent field predictions could reflect polarity, phase, spectrum, or amplitude mismatch. | Require a passing saved calibration manifest and explicit transfer before synthetic normalization and inference. |
| Graph illustrations could use reconstructed or surrogate connectivity. | A visually plausible graph was not necessarily the graph used by the model. | Return base edges, edge attributes, learned TransformerConv attention, and embeddings from the normalized forward pass and consume those exact tensors in figures. |
| A checkpoint filename implied raw flow-loss selection although it tracked a composite objective. | “Best” was scientifically ambiguous, and objectives peaked at different epochs. | Keep immutable v002 names, document their true criteria, and use criterion-specific v003 checkpoints with whole-realization validation as the preferred final rule. |
| Target-conditioned synthetic diversity could be described too broadly. | Claims could exceed the modeled geology and local Zoeppritz observation physics. | State the target-field-conditioned scope and define salt/subsalt work as a separate wave-equation/RTM/FWI-style domain. |

The bounded v003 run is an implementation and operator validation only; it is
not a performance benchmark. Production eligibility is determined by the
machine-readable physical, operator, dataset, and execution criteria encoded in
the corresponding manifests.
