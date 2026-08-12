"""Optional Torch dataset for versioned SAGE-AVO patch archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .normalization import NormalizationStats


class SAGEPatchDataset(Dataset[dict[str, Tensor]]):
    """Load explicit AVO/target/prior/RGT/mask arrays from an NPZ archive.

    Expected keys are ``X``, ``Y``, ``low``, ``rgt``, ``mask``, and
    ``segmentation``. Optional raw patch-scale fields are returned when present.
    """

    def __init__(self, path: str | Path, stats: NormalizationStats, augment: bool = False) -> None:
        archive = np.load(Path(path), mmap_mode="r")
        required = {"X", "Y", "low", "rgt", "mask", "segmentation"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
        self.arrays = {key: archive[key] for key in archive.files}
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return int(self.arrays["X"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        avo = np.asarray(self.arrays["X"][index, :3], dtype=np.float32)
        target = np.asarray(self.arrays["Y"][index], dtype=np.float32)
        low = np.asarray(self.arrays["low"][index], dtype=np.float32)
        rgt = np.asarray(self.arrays["rgt"][index], dtype=np.float32)
        mask = np.asarray(self.arrays["mask"][index, :1], dtype=np.float32)
        segmentation = np.asarray(self.arrays["segmentation"][index], dtype=np.int64)
        x_mean = np.asarray(self.stats.x_mean[:3], dtype=np.float32)[:, None, None]
        x_std = np.asarray(self.stats.x_std[:3], dtype=np.float32)[:, None, None]
        y_mean = np.asarray(self.stats.y_mean, dtype=np.float32)[:, None, None]
        y_std = np.asarray(self.stats.y_std, dtype=np.float32)[:, None, None]
        avo = (avo - x_mean) / x_std
        target = (target - y_mean) / y_std
        low = (low - y_mean) / y_std

        if self.augment and torch.rand(()) < 0.5:
            avo = avo[..., ::-1].copy()
            target = target[..., ::-1].copy()
            low = low[..., ::-1].copy()
            rgt = rgt[..., ::-1].copy()
            mask = mask[..., ::-1].copy()
            segmentation = segmentation[..., ::-1].copy()

        item = {
            "avo": torch.from_numpy(avo),
            "target": torch.from_numpy(target),
            "low": torch.from_numpy(low),
            "rgt": torch.from_numpy(rgt),
            "mask": torch.from_numpy(mask),
            "segmentation": torch.from_numpy(segmentation),
        }
        for key in ("realization_id", "raw_height", "raw_width"):
            if key in self.arrays:
                item[key] = torch.as_tensor(self.arrays[key][index])
        return item
