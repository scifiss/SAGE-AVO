# Controlled benchmark figure captions

## Main synthetic inversion

Truth, truth-derived 2 Hz low-frequency prior, full SAGE-AVO prediction, and
absolute error for the test realization whose full-model Vp RMSE is closest to
the test-set median. Common color scales are used for truth, prior, and
prediction within each elastic property. SAGE-AVO refines a supplied prior; it
does not estimate absolute elastic properties from AVO alone.

## Controlled ablation

Vp predictions for the low-prior-only, full SAGE-AVO, no-GNN, no-RGT-steering,
and no-physics conditions on the same median-rule test realization. Every
learned condition uses the same dataset, split, prior, normalization, patch
index, inference tiling, metric implementation, and checkpoint-selection rule.

## Graph mechanism and benefit

Near-, mid-, and far-angle normalized AVO; RGT; Shuey-style AVO gradient; graph
edges; full-model Vp; and graph error reduction for the central training-scale
patch of the median-rule test realization. Positive benefit is
`|Vp_noGNN − Vp_true| − |Vp_full − Vp_true|` and indicates lower full-model
absolute error.

For visualization, only the strongest 15% of graph edges ranked by edge weight
are displayed; all graph edges are used during message passing.

## Synthetic training diversity

Three systematically spaced training realization IDs show variation in
three-band AVO, Vp, Vs, density, facies/plume, and RGT. The sampling domain is
one procedural, field-conditioned geological family.
