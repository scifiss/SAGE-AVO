"""Focused, deterministic accounting and known-failure tests for v00332aa."""

import pytest

from sage_avo.diagnostics.gap_tolerant_graph import track_paths
from sage_avo.diagnostics.tracker_failure_audit import classify_events, eligibility, replay


def _event(trace, index):
    return {"trace": trace, "time": float(20 + index), "source_type": "strong"}


def _link(source, target, events, cost=1.0):
    return {
        "source": source,
        "target": target,
        "span": events[target]["trace"] - events[source]["trace"],
        "cost": cost,
        "safe": True,
        "plausible": True,
        "relation": "CANDIDATE",
    }


def _counterexample():
    # Two independent paths share a last event. Both span twelve traces,
    # but the six-link path has seven events and the seven-link path has eight.
    # Span reward therefore favors the invalid (fewer-link) path.
    traces = [0, 2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11, 12]
    events = [_event(trace, index) for index, trace in enumerate(traces)]
    short = [0, 1, 2, 3, 4, 5, 13]
    valid = [6, 7, 8, 9, 10, 11, 12, 13]
    links = [
        _link(source, target, events)
        for path in (short, valid)
        for source, target in zip(path, path[1:])
    ]
    config = {"path_step_reward": 8.0, "minimum_component_points": 8, "minimum_component_span": 12}
    return events, links, config, valid


def test_exact_structural_eligibility_finds_valid_alternative():
    events, links, config, valid = _counterexample()
    eligible = eligibility(events, links, config)
    assert set(valid) <= eligible["events"]
    assert eligible["max_count"][valid[-1]] == 8


def test_replay_matches_production_and_accounting_is_partitioned():
    events, links, config, _ = _counterexample()
    assert replay(events, links, config)["paths"] == track_paths(events, links, config)["components"]
    rows, _ = classify_events(events, links, [], config)
    assert len(rows) == len(events)
    assert all(row["has_candidate"] and row["has_safe"] for row in rows)
    assert not any(row["accepted_track"] for row in rows)


@pytest.mark.xfail(strict=True, reason="v00332z one-predecessor DP masks an eligible alternative")
def test_known_tracker_defect_preserves_eligible_alternative():
    """Failing regression target; do not modify the frozen production tracker here."""
    events, links, config, valid = _counterexample()
    paths = track_paths(events, links, config)["components"]
    assert valid in paths
