"""Deterministic diverse patch candidates for the v003 production dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchCandidateConfig:
    mode: str = "diverse"
    depth_bins: int = 4
    minimum_separation_samples: float = 8.0
    top_quantile: float = 0.75
    maximum_attempt_multiplier: int = 100
    categories: tuple[str, ...] = (
        "facies_boundary",
        "high_dip",
        "high_rgt_change",
        "high_avo_gradient_change",
        "reservoir",
        "background",
    )


@dataclass(frozen=True)
class PatchCandidate:
    top: int
    left: int
    category: str
    depth_bin: int
    score: float


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    vertical, horizontal = np.gradient(np.asarray(values, dtype=float))
    return np.hypot(vertical, horizontal)


def _score_maps(
    avo: np.ndarray,
    rgt: np.ndarray,
    segmentation: np.ndarray,
    representative_angles_degrees: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    labels = np.asarray(segmentation)
    boundary = np.zeros_like(labels, dtype=float)
    boundary[1:] += labels[1:] != labels[:-1]
    boundary[:, 1:] += labels[:, 1:] != labels[:, :-1]
    vertical_rgt, horizontal_rgt = np.gradient(np.asarray(rgt, dtype=float))
    dip = np.abs(horizontal_rgt) / (np.abs(vertical_rgt) + 1e-6)
    rgt_change = _gradient_magnitude(vertical_rgt)
    angles = np.sin(np.deg2rad(representative_angles_degrees)) ** 2
    centered_angles = angles - angles.mean()
    centered_avo = np.asarray(avo, dtype=float) - np.mean(avo, axis=0, keepdims=True)
    avo_gradient = np.sum(centered_angles[:, None, None] * centered_avo, axis=0) / (
        np.sum(centered_angles**2) + 1e-12
    )
    return {
        "facies_boundary": boundary,
        "high_dip": dip,
        "high_rgt_change": rgt_change,
        "high_avo_gradient_change": _gradient_magnitude(avo_gradient),
        "reservoir": (labels > 0).astype(float),
        "background": (labels == 0).astype(float),
    }


def diverse_patch_candidates(
    *,
    avo: np.ndarray,
    rgt: np.ndarray,
    segmentation: np.ndarray,
    valid_mask: np.ndarray,
    raw_shape: tuple[int, int],
    count: int,
    rng: np.random.Generator,
    config: PatchCandidateConfig = PatchCandidateConfig(),
    representative_angles_degrees: tuple[float, float, float] = (10.0, 24.0, 38.0),
    maximum_invalid_fraction: float = 0.15,
    excluded_coordinates: set[tuple[int, int]] | None = None,
) -> list[PatchCandidate]:
    """Select deduplicated, separated candidates across processes and depth bins."""
    if config.mode not in {"diverse", "uniform"}:
        raise ValueError("Candidate mode must be 'diverse' or 'uniform'")
    height, width = np.asarray(rgt).shape
    raw_height, raw_width = raw_shape
    if raw_height > height or raw_width > width or count < 1:
        raise ValueError("Patch shape/count is incompatible with the realization")
    valid = np.asarray(valid_mask, dtype=bool)
    top_count = height - raw_height + 1
    left_count = width - raw_width + 1
    tops, lefts = np.indices((top_count, left_count))
    centers_row = tops + raw_height // 2
    centers_col = lefts + raw_width // 2
    depth_bin = np.minimum(
        config.depth_bins - 1,
        (config.depth_bins * centers_row / max(height, 1)).astype(int),
    )
    valid_fraction = np.empty((top_count, left_count), dtype=float)
    for top in range(top_count):
        for left in range(left_count):
            valid_fraction[top, left] = valid[
                top : top + raw_height,
                left : left + raw_width,
            ].mean()
    allowed = valid_fraction >= 1.0 - maximum_invalid_fraction
    excluded = excluded_coordinates or set()
    for top, left in excluded:
        if 0 <= top < top_count and 0 <= left < left_count:
            allowed[top, left] = False
    if not allowed.any():
        raise ValueError("No valid candidate coordinates remain")

    if config.mode == "uniform":
        coordinates = np.argwhere(allowed)
        rng.shuffle(coordinates)
        uniform: list[PatchCandidate] = []
        for top, left in coordinates:
            if any(
                np.hypot(top - item.top, left - item.left)
                < config.minimum_separation_samples
                for item in uniform
            ) or any(
                np.hypot(top - prior_top, left - prior_left)
                < config.minimum_separation_samples
                for prior_top, prior_left in excluded
            ):
                continue
            uniform.append(
                PatchCandidate(
                    int(top),
                    int(left),
                    "uniform",
                    int(depth_bin[top, left]),
                    1.0,
                )
            )
            if len(uniform) == count:
                return uniform
        raise RuntimeError(
            f"Only {len(uniform)}/{count} separated uniform candidates could be selected"
        )

    maps = _score_maps(avo, rgt, segmentation, representative_angles_degrees)
    selected: list[PatchCandidate] = []
    used: set[tuple[int, int]] = set(excluded)

    def separated(top: int, left: int) -> bool:
        center = np.array(
            [top + raw_height / 2.0, left + raw_width / 2.0], dtype=float
        )
        separated_from_selected = all(
            np.linalg.norm(
                center
                - np.array(
                    [
                        item.top + raw_height / 2.0,
                        item.left + raw_width / 2.0,
                    ]
                )
            )
            >= config.minimum_separation_samples
            for item in selected
        )
        separated_from_prior_scales = all(
            np.hypot(top - prior_top, left - prior_left)
            >= config.minimum_separation_samples
            for prior_top, prior_left in excluded
        )
        return separated_from_selected and separated_from_prior_scales
    targets = [
        (category, bin_index)
        for bin_index in range(config.depth_bins)
        for category in config.categories
    ]
    attempts = 0
    maximum_attempts = max(count * config.maximum_attempt_multiplier, 1000)
    while len(selected) < count and attempts < maximum_attempts and targets:
        category, bin_index = targets[len(selected) % len(targets)]
        score_map = maps[category][centers_row, centers_col]
        eligible = allowed & (depth_bin == bin_index)
        values = score_map[eligible]
        attempts += 1
        if values.size == 0:
            targets.remove((category, bin_index))
            continue
        threshold = np.quantile(values, config.top_quantile)
        coordinates = np.argwhere(eligible & (score_map >= threshold))
        rng.shuffle(coordinates)
        accepted = False
        for top, left in coordinates:
            coordinate = (int(top), int(left))
            if coordinate in used:
                continue
            if not separated(int(top), int(left)):
                continue
            selected.append(
                PatchCandidate(
                    coordinate[0],
                    coordinate[1],
                    category,
                    bin_index,
                    float(score_map[coordinate]),
                )
            )
            used.add(coordinate)
            accepted = True
            break
        if not accepted:
            targets.remove((category, bin_index))

    if len(selected) < count:
        coordinates = np.argwhere(allowed)
        rng.shuffle(coordinates)
        for top, left in coordinates:
            coordinate = (int(top), int(left))
            if coordinate in used or not separated(*coordinate):
                continue
            selected.append(
                PatchCandidate(
                    coordinate[0],
                    coordinate[1],
                    "uniform_fill",
                    int(depth_bin[coordinate]),
                    1.0,
                )
            )
            used.add(coordinate)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)}/{count} unique candidates could be selected")
    return selected
