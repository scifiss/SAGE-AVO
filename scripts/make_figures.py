#!/usr/bin/env python3
"""Generate lightweight figures from versioned CSV/NPZ artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sage_avo.visualization import plot_ablation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPOSITORY / "results" / "example_metrics.csv",
        help="Ablation metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "figures" / "example_ablation.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    table = pd.read_csv(arguments.metrics)
    table = table.dropna(subset=["rmse_vp", "rmse_vs", "rmse_density", "miou"])
    figure = plot_ablation_metrics(table)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    print(f"Saved {arguments.output}")


if __name__ == "__main__":
    main()
