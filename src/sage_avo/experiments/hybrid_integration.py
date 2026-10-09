"""Predeclared A/B/C configuration and identical initial dense/sparse weights."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from sage_avo.config import load_config
from sage_avo.models.hybrid_sparse import HybridSparseSAGEAVO
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs


def resolve_hybrid_config(contract_path: str | Path) -> dict:
    path = Path(contract_path)
    contract = load_config(path)
    config = deepcopy(load_config(path.parent / contract["base_training_config"]))
    config["dataset"]["directory"] = contract["immutable_dataset"]
    config["experiment"]["name"] = contract["revision"]
    config["experiment"]["seed"] = int(contract["seeds"][0])
    config["training"]["epochs"] = contract["future_matched"]["epochs"]
    config["training"]["batch_size"] = contract["future_matched"]["batch_size"]
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"].update(enabled=False, weight=0.0)
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"].update(enabled=False, guidance_scale=0.0)
    config["hybrid_sparse"] = deepcopy(contract["sparse_branch"])
    config["hybrid_sparse"]["matched_augmentation"] = True
    return config


def build_hybrid_condition(config: dict, condition: str):
    """Dense initialization is identical in A/B/C; B/C sparse weights match.

    Keep initialization draws separate from the sampler, augmentation, time
    and optimizer streams. No model training or optimizer is created here.
    """
    if condition not in {"A", "B", "C"}:
        raise ValueError("Hybrid condition must be A, B or C")
    with torch.random.fork_rng(devices=[]):
        # Constructors allocate on CPU. torch.manual_seed would also reseed CUDA
        # outside this CPU-only fork, altering an unrelated execution stream.
        torch.default_generator.manual_seed(int(config["experiment"]["seed"]))
        dense = build_sage_avo_variant("full", **sage_avo_model_kwargs(config))
        if condition == "A":
            return dense
        sparse = config["hybrid_sparse"]
        return HybridSparseSAGEAVO(
            dense,
            int(config["model"]["hidden_channels"]),
            gamma=float(sparse["gamma"]),
            message_control="source_permuted" if condition == "B" else "genuine",
            control_seed=int(sparse["control_seed"]),
        )
