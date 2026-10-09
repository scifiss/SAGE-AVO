"""Private observable-only topology cache shared by indexed patches and tiles.

The loader never runs detection. Prepare caches explicitly from full, native,
unaugmented observations before constructing a dataset. A stale cache fails
closed rather than silently changing topology during training or inference.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from torch.utils.data import default_collate

from sage_avo.models.hybrid_sparse import SparsePatchGraph, crop_accepted_graph


def numpy_json_value(value: Any) -> Any:
    """Preserve the frozen graph's native path arrays without changing geometry."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported topology metadata type: {type(value).__name__}")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=numpy_json_value
        ).encode()
    ).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def observable_hashes(avo: np.ndarray, rgt: np.ndarray, support: np.ndarray) -> dict[str, str]:
    if avo.shape != (3, *rgt.shape) or support.shape != rgt.shape:
        raise ValueError("Native AVO/RGT/support shapes must agree")
    return {
        name: array_hash(value)
        for name, value in (("avo", avo), ("rgt", rgt), ("support", support))
    }


def frozen_topology_sources() -> dict[str, str]:
    """Hash every tracked diagnostic implementation used by the frozen tracker."""
    root = Path(__file__).resolve().parents[1] / "diagnostics"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.py"))
    }


class TopologyCache:
    """One authoritative graph per realization, with input/config/source identity."""

    def __init__(self, directory: str | Path, frozen: dict[str, Any], config: dict[str, Any]):
        self.directory = Path(directory)
        self.frozen = deepcopy(frozen)
        self.config = deepcopy(config)
        self.config_hash = canonical_hash({"frozen": frozen, "config": config})
        self.source_hashes = frozen_topology_sources()
        self._verified_inputs: dict[int, dict[str, str]] = {}

    def prepare(
        self, realization_id: int, *, avo: np.ndarray, rgt: np.ndarray, support: np.ndarray
    ) -> dict[str, Any]:
        """Build once using only the three explicitly named observable arrays."""
        hashes = observable_hashes(avo, rgt, support)
        path = self.directory / f"realization_{int(realization_id)}.json"
        if path.exists():
            record = self.load(realization_id)
            if record["observable_hashes"] != hashes:
                raise ValueError("Topology cache observations mismatch")
            return record
        # Reuse the frozen v00332z algorithm; no truth-bearing archive is passed.
        from sage_avo.diagnostics.gap_tolerant_graph import (
            candidate_links,
            detect_events,
            score_links,
            sparse_graph,
            track_paths,
        )
        from sage_avo.diagnostics.native_rgt_graph import NativeRGT
        from sage_avo.diagnostics.rgt_topology_repair import structural_fields
        from sage_avo.diagnostics.skeleton_graph import sample

        events, _ = detect_events(
            avo,
            rgt,
            support.astype(bool),
            self.config["strong_detector"],
            self.frozen["weak_detector"],
            self.config,
        )
        curvature = structural_fields(rgt)["curvature"]
        for event in events:
            event["structural_curvature"] = float(
                sample(curvature, np.asarray([[event["time"], event["trace"]]]))[0]
            )
        candidates = candidate_links(
            NativeRGT(rgt),
            events,
            self.frozen["search_radius"],
            self.config["maximum_gap_traces"],
        )
        links = score_links(candidates, self.frozen["scales"], self.config)
        tracks = track_paths(events, links, self.config)
        graph = sparse_graph(
            events,
            tracks,
            self.frozen["node_spacing"],
            self.config,
            self.frozen["curvature_threshold"],
        )
        record = {
            "schema": 1,
            "algorithm": "frozen_v00332z_observable_event_tracking",
            "realization_id": int(realization_id),
            "geometry": {"shape": list(rgt.shape), "coordinates": "native_time_sample_trace"},
            "observable_hashes": hashes,
            "config_sha256": self.config_hash,
            "source_sha256": self.source_hashes,
            "nodes": graph["nodes"],
            "edges": graph["edges"],
        }
        record["topology_sha256"] = canonical_hash(record)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Publish a complete cache atomically without replacing an existing one.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".topology-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, sort_keys=True, allow_nan=False, default=numpy_json_value)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if self.load(realization_id)["topology_sha256"] != record["topology_sha256"]:
                    raise ValueError("A concurrent topology writer produced a different graph")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.load(realization_id)

    @lru_cache(maxsize=128)
    def load(self, realization_id: int) -> dict[str, Any]:
        path = self.directory / f"realization_{int(realization_id)}.json"
        record = json.loads(path.read_text())
        payload = {key: value for key, value in record.items() if key != "topology_sha256"}
        if record["topology_sha256"] != canonical_hash(payload):
            raise ValueError("Topology cache content hash mismatch")
        if (
            record["schema"] != 1
            or record["realization_id"] != int(realization_id)
            or record["config_sha256"] != self.config_hash
            or record["source_sha256"] != self.source_hashes
        ):
            raise ValueError("Topology cache identity/config/source mismatch")
        return record

    def validate_observations(
        self, realization_id: int, *, avo: np.ndarray, rgt: np.ndarray, support: np.ndarray
    ) -> None:
        hashes = observable_hashes(avo, rgt, support)
        if self.load(realization_id)["observable_hashes"] != hashes:
            raise ValueError("Topology cache observations mismatch")
        self._verified_inputs[int(realization_id)] = hashes

    def patch(
        self,
        realization_id: int,
        *,
        top: int,
        left: int,
        raw_shape: tuple[int, int],
        output_shape: tuple[int, int],
    ) -> SparsePatchGraph:
        record = self.load(realization_id)
        height, width = record["geometry"]["shape"]
        if top < 0 or left < 0 or top + raw_shape[0] > height or left + raw_shape[1] > width:
            raise ValueError("Topology crop lies outside the native section")
        return crop_accepted_graph(
            record["nodes"],
            record["edges"],
            top=top,
            left=left,
            raw_shape=raw_shape,
            output_shape=output_shape,
        )

    def tile_provider(
        self,
        realization_id: int,
        patch_shape: tuple[int, int],
        *,
        avo: np.ndarray,
        rgt: np.ndarray,
        support: np.ndarray,
    ):
        self.validate_observations(realization_id, avo=avo, rgt=rgt, support=support)

        def provide(positions: list[tuple[int, int]]) -> list[SparsePatchGraph]:
            return [
                self.patch(
                    realization_id,
                    top=top,
                    left=left,
                    raw_shape=patch_shape,
                    output_shape=patch_shape,
                )
                for top, left in positions
            ]

        return provide


def collate_sparse_patches(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack ordinary tensors and keep variable/empty graphs in batch order."""
    has_graph = ["sparse_graph" in item for item in items]
    if any(has_graph) and not all(has_graph):
        raise ValueError("Batch mixes graph-bearing and dense-only patches")
    result = default_collate(
        [{key: value for key, value in item.items() if key != "sparse_graph"} for item in items]
    )
    if all(has_graph):
        result["sparse_graphs"] = [item["sparse_graph"] for item in items]
    return result
