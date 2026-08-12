#!/usr/bin/env python3
"""Compare matched three-band outputs from Torch/NumPy/Madagascar workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sage_avo.forward.qc import compare_forward_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference [3, time, trace] .npy file")
    parser.add_argument("candidate", type=Path, help="Candidate [3, time, trace] .npy file")
    arguments = parser.parse_args()
    agreement = compare_forward_outputs(np.load(arguments.reference), np.load(arguments.candidate))
    for index, name in enumerate(("near", "mid", "far")):
        print(
            f"{name:>4}: scale={agreement.scale[index]:.6g}, "
            f"r={agreement.correlation[index]:.4f}, "
            f"NRMSE={agreement.normalized_rmse[index]:.4f}"
        )


if __name__ == "__main__":
    main()
