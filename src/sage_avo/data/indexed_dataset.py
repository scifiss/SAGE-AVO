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

from .augmentation import AugmentationConfig, augment_patch
from .patches import resize_channels_first


class IndexedRealizationPatches(Dataset[dict[str, Tensor]]):
    """Extract deterministic patches from immutable full-realization files."""

    def __init__(
        self,
        dataset_directory: str | Path,
        split: str,
        *,
        augment: bool = False,
        augmentation_config: AugmentationConfig = AugmentationConfig(),
        augmentation_generator: torch.Generator | None = None,
    ) -> None:
        self.root = Path(dataset_directory)
        index = pd.read_csv(self.root / "patch_index.csv")
        self.index = index[index["split"] == split].reset_index(drop=True)
        if self.index.empty:
            raise ValueError(f"No patch rows found for split {split!r}")
        self.normalization = json.loads(
            (self.root / "normalization.json").read_text(encoding="utf-8")
        )
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config
        self.augmentation_generator = augmentation_generator

    def __len__(self) -> int:
        return len(self.index)

    @lru_cache(maxsize=128)
    def _load(self, realization_id: int) -> dict[str, np.ndarray]:
        """Load one immutable realization, caching the complete production split in RAM."""
        matches = self.index[self.index["realization_id"] == realization_id]
        if "realization_file" in matches.columns:
            filename = str(matches.iloc[0]["realization_file"])
        else:
            filename = f"realization_{realization_id:04d}.npz"
        path = self.root / "realizations" / filename
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def _raw_patch(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, int | tuple[int, int]]]:
        row = self.index.iloc[index]
        realization_id = int(row["realization_id"])
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["raw_height"]), int(row["raw_width"])
        output_height = int(row.get("output_height", height))
        output_width = int(row.get("output_width", width))
        output_shape = (output_height, output_width)
        arrays = self._load(realization_id)
        spatial = np.s_[top : top + height, left : left + width]
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
        arrays_out = {
            "avo": np.asarray(avo_raw, dtype=np.float32),
            "target": np.asarray(target_raw, dtype=np.float32),
            "low": np.asarray(low_raw, dtype=np.float32),
            "rgt": np.asarray(rgt_raw[0], dtype=np.float32),
            "mask": np.asarray(mask_raw, dtype=np.float32),
            "segmentation": np.asarray(segmentation_raw[0], dtype=np.int64),
        }
        metadata: dict[str, int | tuple[int, int]] = {
            "realization_id": realization_id,
            "top": top,
            "left": left,
            "raw_shape": (height, width),
            "output_shape": output_shape,
        }
        if "physics_eligible" in row.index:
            eligible = bool(int(row["physics_eligible"]))
            halo = int(row.get("convolution_halo_samples", 0))
            context_height = output_height + 2 * halo
            context_shape = (3, context_height, output_width)
            if eligible:
                if output_shape != (height, width):
                    raise ValueError("Physics-eligible patches must remain on their native grid")
                if "avo_clean" not in arrays:
                    raise ValueError("Physics-eligible v003 data require avo_clean")
                context_top = top - halo
                context_bottom = top + height + halo
                clipped_top = max(context_top, 0)
                clipped_bottom = min(context_bottom, arrays["elastic"].shape[1])
                context = arrays["elastic"][
                    :, clipped_top:clipped_bottom, left : left + width
                ]
                pad_top = clipped_top - context_top
                pad_bottom = context_bottom - clipped_bottom
                context = np.pad(
                    context,
                    ((0, 0), (pad_top, pad_bottom), (0, 0)),
                    mode="edge",
                )
                if context.shape != context_shape:
                    raise ValueError("Physics context does not match its fixed tensor contract")
                physics_avo = arrays["avo_clean"][
                    :, top : top + height, left : left + width
                ]
                physics_mask = mask_raw.astype(np.float32)
            else:
                context = np.broadcast_to(
                    np.asarray((3000.0, 1700.0, 2.4), dtype=np.float32)[:, None, None],
                    context_shape,
                ).copy()
                physics_avo = np.zeros((3, output_height, output_width), dtype=np.float32)
                physics_mask = np.zeros((1, output_height, output_width), dtype=np.float32)
                context_top = top - halo
            arrays_out.update(
                {
                    "physics_context": np.asarray(context, dtype=np.float32),
                    "physics_avo": np.asarray(physics_avo, dtype=np.float32),
                    "physics_mask": np.asarray(physics_mask, dtype=np.float32),
                }
            )
            metadata.update(
                {
                    "physics_eligible": int(eligible),
                    "physics_context_sample_origin": int(context_top),
                    "physics_core_start": halo,
                    "native_dt_seconds": float(row["native_dt_seconds"]),
                    "mute_origin_seconds": float(row["mute_origin_seconds"]),
                }
            )
        return arrays_out, metadata

    def sampling_fields(self, index: int) -> dict[str, Tensor]:
        """Return deterministic pre-normalization fields for sampler scoring."""
        arrays, _ = self._raw_patch(index)
        return {
            "avo": torch.from_numpy(arrays["avo"]),
            "rgt": torch.from_numpy(arrays["rgt"]),
            "segmentation": torch.from_numpy(arrays["segmentation"]),
            "mask": torch.from_numpy(arrays["mask"]),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        arrays, metadata = self._raw_patch(index)
        x_mean = np.asarray(self.normalization["x_mean"], dtype=np.float32)[:, None, None]
        x_std = np.asarray(self.normalization["x_std"], dtype=np.float32)[:, None, None]
        y_mean = np.asarray(self.normalization["y_mean"], dtype=np.float32)[:, None, None]
        y_std = np.asarray(self.normalization["y_std"], dtype=np.float32)[:, None, None]
        avo = (arrays["avo"] - x_mean) / x_std
        target = (arrays["target"] - y_mean) / y_std
        low = (arrays["low"] - y_mean) / y_std
        height, width = metadata["raw_shape"]
        output_height, output_width = metadata["output_shape"]
        item = {
            "avo": torch.from_numpy(np.asarray(avo, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "low": torch.from_numpy(np.asarray(low, dtype=np.float32)),
            "rgt": torch.from_numpy(arrays["rgt"]),
            "mask": torch.from_numpy(arrays["mask"]),
            "segmentation": torch.from_numpy(arrays["segmentation"]),
            "realization_id": torch.tensor(metadata["realization_id"], dtype=torch.int64),
            "top": torch.tensor(metadata["top"], dtype=torch.int64),
            "left": torch.tensor(metadata["left"], dtype=torch.int64),
            "raw_shape": torch.tensor((height, width), dtype=torch.int64),
            "output_shape": torch.tensor((output_height, output_width), dtype=torch.int64),
            "resize_scale": torch.tensor(
                (height / output_height, width / output_width), dtype=torch.float32
            ),
        }
        if "physics_context" in arrays:
            item.update(
                {
                    "physics_context": torch.from_numpy(
                        (arrays["physics_context"] - y_mean) / y_std
                    ),
                    "physics_avo": torch.from_numpy(
                        (arrays["physics_avo"] - x_mean) / x_std
                    ),
                    "physics_mask": torch.from_numpy(arrays["physics_mask"]),
                    "physics_eligible": torch.tensor(
                        metadata["physics_eligible"], dtype=torch.bool
                    ),
                    "physics_context_sample_origin": torch.tensor(
                        metadata["physics_context_sample_origin"], dtype=torch.int64
                    ),
                    "physics_core_start": torch.tensor(
                        metadata["physics_core_start"], dtype=torch.int64
                    ),
                    "native_dt_seconds": torch.tensor(
                        metadata["native_dt_seconds"], dtype=torch.float32
                    ),
                    "mute_origin_seconds": torch.tensor(
                        metadata["mute_origin_seconds"], dtype=torch.float32
                    ),
                }
            )
        if self.augment:
            item = augment_patch(
                item,
                self.augmentation_config,
                generator=self.augmentation_generator,
            )
        return item
