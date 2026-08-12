# Result artifact contract

Generated result CSVs are ignored by Git. The evaluation stage writes:

- `controlled_ablation_metrics.csv` — aggregate metrics by model variant;
- `per_realization_metrics.csv` — realization-level metrics;
- `paired_ablation_comparisons.csv` — paired improvements and confidence
  intervals;
- `field_forward_qc.csv` — per-band field forward-consistency metrics.

Each table is associated with the experiment manifest that fixes the dataset,
split, prior, mask, inference settings, checkpoint rule, and metric definitions.
