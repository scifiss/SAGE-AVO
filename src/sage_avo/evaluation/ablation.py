"""Ablation-table assembly with the required low-prior baseline."""

from __future__ import annotations

import pandas as pd


REQUIRED_VARIANTS = ("low_prior", "full", "no_gnn", "no_rgt", "no_physics")


def validate_ablation_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate required variants and return them in publication order."""
    if "model" not in table:
        raise ValueError("Ablation table requires a model column")
    missing = set(REQUIRED_VARIANTS) - set(table["model"])
    if missing:
        raise ValueError(f"Missing required ablations: {sorted(missing)}")
    order = {name: index for index, name in enumerate(REQUIRED_VARIANTS)}
    return table.assign(_order=table["model"].map(order)).sort_values("_order").drop(columns="_order")
