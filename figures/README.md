# Figures

`workflow.svg` summarizes the SAGE-AVO method. Experiment figures are generated
from versioned private or synthetic artifacts with:

```bash
python scripts/make_figures.py --help
```

Generated PNG/PDF files are ignored by default; selected distributable figures
are allow-listed in `.gitignore`.

The controlled evaluation generates four primary figures:

- `main_synthetic_inversion.{png,pdf}`;
- `controlled_ablation.{png,pdf}`;
- `graph_mechanism_and_benefit.{png,pdf}`;
- `synthetic_training_diversity.{png,pdf}`.

`scripts/make_figures.py` provides the metrics-to-figure entry point. Figure
captions and source artifact identifiers are recorded with each evaluation.
