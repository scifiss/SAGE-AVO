"""Read-only accounting of frozen v00332z reflector-tracker decisions.

This module does not alter event detection, link scoring, or production tracking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from sage_avo.diagnostics.gap_tolerant_graph import _crosses, track_paths


def eligibility(
    events: list[dict[str, Any]],
    links: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Find events on any structurally eligible DAG path, ignoring path score.

    For each capped path length, retain the earliest possible start and latest
    possible end. Combining independent halves at an event is an exact
    existence test for the count/span constraints, not an objective optimum.
    """
    size = len(events)
    minimum_count = int(config["minimum_component_points"])
    minimum_span = int(config["minimum_component_span"])
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for row in links:
        if row["safe"]:
            source, target = int(row["source"]), int(row["target"])
            incoming[target].append(source)
            outgoing[source].append(target)
    order = sorted(range(size), key=lambda index: (events[index]["trace"], events[index]["time"]))
    forward = [{1: int(events[index]["trace"])} for index in range(size)]
    backward = [{1: int(events[index]["trace"])} for index in range(size)]
    for target in order:
        for source in incoming[target]:
            for count, start in forward[source].items():
                next_count = min(count + 1, minimum_count)
                forward[target][next_count] = min(forward[target].get(next_count, start), start)
    for source in reversed(order):
        for target in outgoing[source]:
            for count, end in backward[target].items():
                next_count = min(count + 1, minimum_count)
                backward[source][next_count] = max(backward[source].get(next_count, end), end)
    eligible: set[int] = set()
    max_count: list[int] = []
    max_span: list[int] = []
    for index in range(size):
        max_count.append(min(max(forward[index]) + max(backward[index]) - 1, minimum_count))
        max_span.append(max(backward[index].values()) - min(forward[index].values()))
        if any(
            left_count + right_count - 1 >= minimum_count
            and right_end - left_start >= minimum_span
            for left_count, left_start in forward[index].items()
            for right_count, right_end in backward[index].items()
        ):
            eligible.add(index)
    return {"events": eligible, "max_count": max_count, "max_span": max_span}


def masked_eligible_endpoints(
    events: list[dict[str, Any]], links: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Find frozen endpoints where one-best-predecessor masks a positive eligible path.

    The alternative DP retains the best score per (start trace, capped event
    count). It is an analysis-only upper bound on recoverable endpoints: an
    alternative may conflict with other selected paths.
    """
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in links:
        if row["safe"]:
            incoming[int(row["target"])].append(row)
    minimum_count = int(config["minimum_component_points"])
    minimum_span = int(config["minimum_component_span"])
    order = sorted(range(len(events)), key=lambda index: (events[index]["trace"], events[index]["time"]))
    best = np.zeros(len(events), float)
    predecessor: dict[int, dict[str, Any]] = {}
    states: list[dict[tuple[int, int], float]] = [
        {(int(event["trace"]), 1): 0.0} for event in events
    ]
    for target in order:
        for row in incoming[target]:
            source = int(row["source"])
            delta = float(config["path_step_reward"] * row["span"] - row["cost"])
            proposal = best[source] + delta
            if proposal > best[target] + 1e-10:
                best[target] = proposal
                predecessor[target] = row
            for (start, count), previous in states[source].items():
                key = (start, min(count + 1, minimum_count))
                states[target][key] = max(states[target].get(key, float("-inf")), previous + delta)
    masked = []
    for end in order:
        if best[end] <= 0:
            continue
        cursor = end
        count = 1
        while cursor in predecessor:
            cursor = int(predecessor[cursor]["source"])
            count += 1
        span = int(events[end]["trace"] - events[cursor]["trace"])
        if count >= minimum_count and span >= minimum_span:
            continue
        alternative = max(
            (
                score
                for (start, length), score in states[end].items()
                if length >= minimum_count
                and int(events[end]["trace"]) - start >= minimum_span
                and score > 0
            ),
            default=None,
        )
        if alternative is not None:
            masked.append(
                {
                    "endpoint_event": end,
                    "best_score": float(best[end]),
                    "best_path_event_count": count,
                    "best_path_span": span,
                    "eligible_alternative_score": float(alternative),
                }
            )
    return masked


def replay(
    events: list[dict[str, Any]], links: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the unmodified greedy tracker and expose its exclusion state."""
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_boundary: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in links:
        if row["safe"]:
            incoming[int(row["target"])].append(row)
            for boundary in range(events[row["source"]]["trace"], events[row["target"]]["trace"]):
                by_boundary[boundary].append(row)
    used: set[int] = set()
    blocked: set[int] = set()
    paths: list[list[int]] = []
    accepted_links: list[dict[str, Any]] = []
    initial_positive: set[int] = set()
    initial_best_eligible: set[int] = set()
    order = sorted(range(len(events)), key=lambda index: (events[index]["trace"], events[index]["time"]))
    iteration = 0
    while True:
        score = np.zeros(len(events), float)
        predecessor: dict[int, dict[str, Any]] = {}
        for target in order:
            if target in used:
                continue
            for row in incoming[target]:
                source = int(row["source"])
                if source in used or id(row) in blocked:
                    continue
                proposal = score[source] + config["path_step_reward"] * row["span"] - row["cost"]
                if proposal > score[target] + 1e-10:
                    score[target] = proposal
                    predecessor[target] = row
        if iteration == 0:
            for row in predecessor.values():
                initial_positive.update((int(row["source"]), int(row["target"])))
        candidates = []
        for end in sorted(range(len(events)), key=lambda index: (-score[index], index)):
            if score[end] <= 0:
                break
            chain = []
            cursor = end
            while cursor in predecessor:
                row = predecessor[cursor]
                chain.append(row)
                cursor = int(row["source"])
            chain.reverse()
            if not chain:
                continue
            indices = [int(chain[0]["source"])] + [int(row["target"]) for row in chain]
            if (
                len(indices) >= config["minimum_component_points"]
                and events[indices[-1]]["trace"] - events[indices[0]]["trace"]
                >= config["minimum_component_span"]
            ):
                if iteration == 0:
                    initial_best_eligible.update(indices)
                candidates.append((indices, chain))
                break
        if not candidates:
            break
        indices, chain = candidates[0]
        paths.append(indices)
        accepted_links.extend(chain)
        used.update(indices)
        for selected in chain:
            for boundary in range(events[selected["source"]]["trace"], events[selected["target"]]["trace"]):
                for candidate in by_boundary[boundary]:
                    if id(candidate) not in blocked and _crosses(candidate, selected, events):
                        blocked.add(id(candidate))
        iteration += 1
    original = track_paths(events, links, config)
    actual = [[(row["source"], row["target"]) for row in chain] for chain in original["path_links"]]
    reproduced = [
        [(path[index], path[index + 1]) for index in range(len(path) - 1)] for path in paths
    ]
    if actual != reproduced:
        raise AssertionError("Diagnostic replay diverged from frozen production tracker")
    return {
        "paths": paths,
        "accepted_links": accepted_links,
        "used": used,
        "blocked": {id(row) for row in links if id(row) in blocked},
        "initial_positive": initial_positive,
        "initial_best_eligible": initial_best_eligible,
    }


def classify_events(
    events: list[dict[str, Any]],
    links: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition detected events by first decisive exclusion; keep overlaps explicit."""
    replayed = replay(events, links, config)
    all_safe = eligibility(events, links, config)
    after_used = eligibility(
        events,
        [row for row in links if row["source"] not in replayed["used"] and row["target"] not in replayed["used"]],
        config,
    )
    after_crossing = eligibility(
        events,
        [
            row for row in links
            if row["source"] not in replayed["used"]
            and row["target"] not in replayed["used"]
            and id(row) not in replayed["blocked"]
        ],
        config,
    )
    tier = {name: set() for name in ("candidate", "plausible", "safe")}
    for row in links:
        pair = {int(row["source"]), int(row["target"])}
        tier["candidate"].update(pair)
        if row["plausible"]:
            tier["plausible"].update(pair)
        if row["safe"]:
            tier["safe"].update(pair)
    sparse = {int(node["event"]) for node in nodes}
    if not sparse <= replayed["used"]:
        raise AssertionError("Sparse node missing from accepted track")
    rows = []
    for index, event in enumerate(events):
        if index in replayed["used"]:
            reason = "accepted"
        elif index not in tier["candidate"]:
            reason = "no_association_candidate"
        elif index not in tier["plausible"]:
            reason = "cost_threshold_rejection"
        elif index not in tier["safe"]:
            reason = "fault_or_discontinuity_barrier"
        elif index not in all_safe["events"]:
            reason = (
                "inadequate_event_count"
                if all_safe["max_count"][index] < config["minimum_component_points"]
                else "inadequate_path_length"
                if all_safe["max_span"][index] < config["minimum_component_span"]
                else "joint_count_span_ineligibility"
            )
        elif index not in after_used["events"]:
            reason = "exclusive_node_conflict_or_greedy_selection"
        elif index not in after_crossing["events"]:
            reason = "noncrossing_constraint"
        else:
            reason = "objective_or_selection_unresolved"
        rows.append(
            {
                "event": index,
                "source_type": event["source_type"],
                "first_decisive_reason": reason,
                "has_candidate": index in tier["candidate"],
                "has_plausible": index in tier["plausible"],
                "has_safe": index in tier["safe"],
                "candidate_track": index in replayed["initial_positive"],
                "accepted_track": index in replayed["used"],
                "sparse_node": index in sparse,
                "structurally_eligible": index in all_safe["events"],
                "on_initial_best_eligible_path": index in replayed["initial_best_eligible"],
                "eligible_after_exclusive_nodes": index in after_used["events"],
                "eligible_after_noncrossing": index in after_crossing["events"],
            }
        )
    return rows, replayed
