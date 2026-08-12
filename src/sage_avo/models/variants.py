"""Controlled model variants for the five-condition benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from .sage_avo import SAGEAVO


LEARNED_VARIANTS = ("full", "no_gnn", "no_rgt", "no_physics")
ALL_VARIANTS = ("low_prior",) + LEARNED_VARIANTS


@dataclass(frozen=True)
class VariantDefinition:
    name: str
    graph_mode: str | None
    physics_weight: float | None


def variant_definition(name: str, physics_weight: float = 0.5) -> VariantDefinition:
    """Return the only intended difference for a controlled condition."""
    definitions = {
        "low_prior": VariantDefinition("low_prior", None, None),
        "full": VariantDefinition("full", "rgt", physics_weight),
        "no_gnn": VariantDefinition("no_gnn", "none", physics_weight),
        "no_rgt": VariantDefinition("no_rgt", "cartesian", physics_weight),
        "no_physics": VariantDefinition("no_physics", "rgt", 0.0),
    }
    if name not in definitions:
        raise ValueError(f"Unknown variant {name!r}; expected one of {tuple(definitions)}")
    return definitions[name]


def build_sage_avo_variant(
    name: str,
    *,
    hidden_channels: int = 64,
    graph_layers: int = 2,
    graph_heads: int = 4,
    max_rgt_shift: int = 3,
    classes: int = 3,
) -> SAGEAVO:
    """Build a learned ablation with all shared hyperparameters held constant."""
    definition = variant_definition(name)
    if definition.graph_mode is None:
        raise ValueError("low_prior has no neural model")
    return SAGEAVO(
        hidden_channels=hidden_channels,
        graph_layers=graph_layers,
        graph_heads=graph_heads,
        max_rgt_shift=max_rgt_shift,
        graph_mode=definition.graph_mode,
        classes=classes,
    )
