#!/usr/bin/env python3
"""Prepare and run the v00332m midpoint elastic-attention diagnostic."""

from pathlib import Path

import run_development_diagnostics_v00332l as runner


runner.CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "development_diagnostics_v00332m.yaml"
)


if __name__ == "__main__":
    runner.main()
