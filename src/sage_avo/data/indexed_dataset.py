"""Lazy fixed-index patches shared by all learned ablation variants."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class IndexedRealizationPatches(Dataset[dict[str, Tensor]]):
    """Extract deterministic patches from immutable full-realization files."""

    def __init__(self, dataset_directory: str | Path, split: str) -> None:
        self.root = Path(dataset_directory)
        index = pd.read_csv(self.root / "patch_index.csv")
        self.index = index[index["split"] == split].reset_index(drop=True)
        if self.index.empty:
            raise ValueError(f"No patch rows found for split {split!r}")
        self.normalization = json.loads(
            (self.root / "normalization.json").read_text(encoding="utf-8")
        )

    def __len__(self) -> int:
        return len(self.index)

    @lru_cache(maxsize=4)
    def _load(self, realization_id: int) -> dict[str, np.ndarray]:
        path = self.root / "realizations" / f"realization_{realization_id:04d}.npz"
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.index.iloc[index]
        realization_id = int(row["realization_id"])
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["raw_height"]), int(row["raw_width"])
        arrays = self._load(realization_id)
        spatial = np.s_[top : top + height, left : left + width]
        x_mean = np.asarray(self.normalization["x_mean"], dtype=np.float32)[:, None, None]
        x_std = np.asarray(self.normalization["x_std"], dtype=np.float32)[:, None, None]
        y_mean = np.asarray(self.normalization["y_mean"], dtype=np.float32)[:, None, None]
        y_std = np.asarray(self.normalization["y_std"], dtype=np.float32)[:, None, None]
        avo = (arrays["avo"][(slice(None),) + spatial] - x_mean) / x_std
        target = (arrays["elastic"][(slice(None),) + spatial] - y_mean) / y_std
        low = (arrays["low"][(slice(None),) + spatial] - y_mean) / y_std
        return {
            "avo": torch.from_numpy(np.asarray(avo, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "low": torch.from_numpy(np.asarray(low, dtype=np.float32)),
            "rgt": torch.from_numpy(np.asarray(arrays["rgt"][spatial], dtype=np.float32)),
            "mask": torch.from_numpy(np.asarray(arrays["mask"][spatial][None], dtype=np.float32)),
            "segmentation": torch.from_numpy(
                np.asarray(arrays["segmentation"][spatial], dtype=np.int64)
            ),
            "realization_id": torch.tensor(realization_id, dtype=torch.int64),
        }
