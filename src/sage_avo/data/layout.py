"""Canonical, versioned data layout for SAGE-AVO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_OUTPUT_KINDS = ("usable", "attributes", "derived", "synthetic", "datasets", "bundles")


@dataclass(frozen=True)
class DataLayout:
    """Resolve immutable raw inputs separately from writable derived products.

    Parameters
    ----------
    work_root:
        Root containing dataset folders such as ``s01data`` and ``sleipner``.
    dataset:
        Dataset identifier below ``work_root``.
    version:
        Version applied to usable, attribute, derived, synthetic, dataset, and
        bundle outputs.
    raw_root:
        Optional external, read-only raw dataset folder. When omitted, raw data
        resolve to ``work_root / dataset / raw``.
    """

    work_root: Path
    dataset: str
    version: str
    raw_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.dataset or Path(self.dataset).name != self.dataset:
            raise ValueError("dataset must be one safe path component")
        if not self.version or Path(self.version).name != self.version:
            raise ValueError("version must be one safe path component")
        object.__setattr__(self, "work_root", Path(self.work_root).expanduser().resolve())
        if self.raw_root is not None:
            object.__setattr__(self, "raw_root", Path(self.raw_root).expanduser().resolve())

    @property
    def dataset_root(self) -> Path:
        return self.work_root / self.dataset

    @property
    def raw(self) -> Path:
        return self.raw_root if self.raw_root is not None else self.dataset_root / "raw"

    @property
    def usable(self) -> Path:
        return self.dataset_root / "usable" / self.version

    @property
    def attributes(self) -> Path:
        return self.dataset_root / "attributes" / self.version

    @property
    def derived(self) -> Path:
        return self.dataset_root / "derived" / self.version

    @property
    def synthetic(self) -> Path:
        return self.dataset_root / "synthetic" / self.version

    @property
    def datasets(self) -> Path:
        return self.dataset_root / "datasets" / self.version

    @property
    def bundles(self) -> Path:
        return self.dataset_root / "bundles" / self.version

    def output(self, kind: str, *parts: str) -> Path:
        """Return a versioned output path without creating it."""
        if kind not in _OUTPUT_KINDS:
            raise ValueError(f"kind must be one of {_OUTPUT_KINDS}")
        return getattr(self, kind).joinpath(*parts)

    def ensure_outputs(self, *kinds: str) -> tuple[Path, ...]:
        """Create only explicitly requested writable output directories."""
        requested = kinds or _OUTPUT_KINDS
        paths = tuple(self.output(kind) for kind in requested)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def to_dict(self) -> dict[str, str]:
        return {
            "work_root": str(self.work_root),
            "dataset": self.dataset,
            "version": self.version,
            "dataset_root": str(self.dataset_root),
            "raw": str(self.raw),
            **{kind: str(self.output(kind)) for kind in _OUTPUT_KINDS},
        }
