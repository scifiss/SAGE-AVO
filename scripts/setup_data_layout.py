#!/usr/bin/env python3
"""Create the empty canonical data hierarchy without copying private data."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIRECTORIES = [
    "gom/raw",
    "gom/synthetic/syn_v001_clean",
    "gom/synthetic/syn_v002_fieldcal_noiseRMO",
    "gom/datasets/ds_v001_syn2d_ang3_p50x100_s10x25_sliceHoldout",
    "gom/attributes",
    "gom/usable",
    "sleipner/raw/horizons",
    "sleipner/raw/offsets",
    "sleipner/raw/wells",
    "sleipner/datasets/ds_v001_field2d_ang3_p50x100_s10x25_blockHoldout",
    "sleipner/attributes/horizons_mapped",
    "sleipner/attributes/dip",
    "sleipner/attributes/rgt",
    "sleipner/usable/stacks",
    "sleipner/usable/velocity",
    "sleipner/usable/angles",
    "avo/s01/bundles/v001",
    "s01data/raw/well",
    "s01data/raw/horizon",
    "s01data/raw/seismic",
    "s01data/bundles/v001",
    "s01data/derived/rf_models_v001",
]
VERSIONS = ("v001",)
for version in VERSIONS:
    for stage in ("synthetic", "datasets", "attributes", "derived", "usable"):
        DIRECTORIES.append(f"s01data/{stage}/{version}")


def main() -> None:
    for relative in DIRECTORIES:
        (DATA / relative).mkdir(parents=True, exist_ok=True)
    print(f"Created/verified {len(DIRECTORIES)} data directories below {DATA}")


if __name__ == "__main__":
    main()
