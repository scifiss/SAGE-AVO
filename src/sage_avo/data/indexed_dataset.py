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

from .patches import resize_channels_first


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
        matches = self.index[self.index["realization_id"] == realization_id]
        if "realization_file" in matches.columns:
            filename = str(matches.iloc[0]["realization_file"])
        else:
            filename = f"realization_{realization_id:04d}.npz"
        path = self.root / "realizations" / filename
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.index.iloc[index]
        realization_id = int(row["realization_id"])
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["raw_height"]), int(row["raw_width"])
        output_height = int(row.get("output_height", height))
        output_width = int(row.get("output_width", width))
        output_shape = (output_height, output_width)
        arrays = self._load(realization_id)
        spatial = np.s_[top : top + height, left : left + width]
        x_mean = np.asarray(self.normalization["x_mean"], dtype=np.float32)[:, None, None]
        x_std = np.asarray(self.normalization["x_std"], dtype=np.float32)[:, None, None]
        y_mean = np.asarray(self.normalization["y_mean"], dtype=np.float32)[:, None, None]
        y_std = np.asarray(self.normalization["y_std"], dtype=np.float32)[:, None, None]
        avo_raw = arrays["avo"][(slice(None),) + spatial]
        target_raw = arrays["elastic"][(slice(None),) + spatial]
        low_raw = arrays["low"][(slice(None),) + spatial]
        rgt_raw = arrays["rgt"][spatial][None]
        mask_name = "valid_mask" if "valid_mask" in arrays else "mask"
        mask_raw = arrays[mask_name][spatial][None]
        segmentation_raw = arrays["segmentation"][spatial][None]
        if output_shape != (height, width):
            avo_raw = resize_channels_first(avo_raw, output_shape, order=1)
            target_raw = resize_channels_first(target_raw, output_shape, order=1)
            low_raw = resize_channels_first(low_raw, output_shape, order=1)
            rgt_raw = resize_channels_first(rgt_raw, output_shape, order=1)
            mask_raw = resize_channels_first(mask_raw, output_shape, order=0)
            segmentation_raw = resize_channels_first(segmentation_raw, output_shape, order=0)
        avo = (avo_raw - x_mean) / x_std
        target = (target_raw - y_mean) / y_std
        low = (low_raw - y_mean) / y_std
        return {
            "avo": torch.from_numpy(np.asarray(avo, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "low": torch.from_numpy(np.asarray(low, dtype=np.float32)),
            "rgt": torch.from_numpy(np.asarray(rgt_raw[0], dtype=np.float32)),
            "mask": torch.from_numpy(np.asarray(mask_raw, dtype=np.float32)),
            "segmentation": torch.from_numpy(np.asarray(segmentation_raw[0], dtype=np.int64)),
            "realization_id": torch.tensor(realization_id, dtype=torch.int64),
            "top": torch.tensor(top, dtype=torch.int64),
            "left": torch.tensor(left, dtype=torch.int64),
            "raw_shape": torch.tensor((height, width), dtype=torch.int64),
            "output_shape": torch.tensor(output_shape, dtype=torch.int64),
            "resize_scale": torch.tensor(
                (height / output_height, width / output_width), dtype=torch.float32
            ),
        }
