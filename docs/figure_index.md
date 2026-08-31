# Figure index

| Figure | Generator | Scientific role |
|---|---|---|
| End-to-end workflow | `figures/workflow.svg` | Repository and method overview |
| Synthetic diversity | `plot_training_diversity` | Show the range of the field-conditioned training family |
| Main inversion | `plot_inversion_comparison` | Truth vs low prior vs SAGE-AVO vs absolute error |
| Ablation (planned pending matched checkpoints) | `plot_ablation_metrics` | Compare required prior/full/component-removal variants only after all final matched runs exist |
| Graph mechanism | `plot_graph_mechanism` | AVO, RGT, strongest weighted edges, and graph benefit |

For graph visualization, only the strongest 15% of graph edges ranked by the
actual message-passing edge weight are displayed; all graph edges are used
during message passing.

Once matched full and no-GNN checkpoints exist, synthetic graph benefit is defined as:

```text
|Vp_noGNN - Vp_true| - |Vp_full - Vp_true|
```

Positive values indicate reduced full-model error. Without field truth, the
corresponding panel is `Vp_full - Vp_noGNN` and is labeled graph sensitivity.
