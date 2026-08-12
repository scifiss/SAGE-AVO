#!/usr/bin/env python3
"""Run a small CPU-only public workflow without private data or Madagascar."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sage_avo.evaluation import elastic_metrics
from sage_avo.forward import ForwardConfig, forward_avo_three_band, shuey_intercept_gradient
from sage_avo.geology import make_synthetic_geology
from sage_avo.structure import build_rgt_graph, integrate_dip_to_rgt


def main() -> None:
    height, width = 48, 20
    rows, columns = np.indices((height, width))
    dip = 0.2 * np.sin(columns / 6.0)
    rgt = integrate_dip_to_rgt(dip)
    sand = np.clip(0.5 + 0.35 * np.sin(rgt / 6.0), 0.0, 1.0)
    porosity = np.clip(0.08 + 0.18 * sand, 0.02, 0.35)
    geology = make_synthetic_geology(sand, porosity, rgt, seed=7, max_faults=2)

    vp = 2700.0 + 800.0 * geology.delta - 150.0 * geology.porosity
    vs = 1450.0 + 550.0 * geology.delta - 100.0 * geology.porosity
    density = 2.10 + 0.28 * geology.delta - 0.10 * geology.porosity
    avo = forward_avo_three_band(vp, vs, density, ForwardConfig(apply_mute=False))
    _, gradient = shuey_intercept_gradient(avo)
    graph = build_rgt_graph(geology.rgt, gradient)
    report = elastic_metrics(vp + 10.0, vp)

    assert avo.shape == (3, height, width)
    assert np.isfinite(avo).all()
    assert graph.weight.size > 0
    assert np.isclose(report["rmse"], 10.0)
    print("SAGE-AVO smoke test passed")
    print(f"  AVO shape: {avo.shape}")
    print(f"  graph edges: {graph.weight.size}")
    print(f"  demonstration Vp RMSE: {report['rmse']:.2f} m/s")


if __name__ == "__main__":
    main()
