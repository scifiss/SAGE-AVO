"""Regression tests for v00332z physical-event paths and fault-safe gaps."""

import numpy as np

from sage_avo.diagnostics.gap_tolerant_graph import (
    _crosses,
    candidate_links,
    detect_events,
    score_links,
    sparse_graph,
    track_paths,
)
from sage_avo.diagnostics.native_rgt_graph import NativeRGT


def _config():
    return {
        "weak_local_window": 31,
        "weak_minimum_distance": 4,
        "weak_overlap_samples": 1.0,
        "weak_maximum_per_trace": 20,
        "gap_penalty": 1.0,
        "maximum_normalized_cost": 8.0,
        "minimum_waveform_cosine": 0.5,
        "minimum_phase_cosine": 0.2,
        "minimum_ava_cosine": 0.2,
        "minimum_component_points": 8,
        "minimum_component_span": 12,
        "path_step_reward": 8.0,
        "edge_spans_traces": [4, 8, 16],
    }


def _event(trace, time):
    return {
        "trace": trace,
        "time": float(time),
        "tau": float(time / 80),
        "waveform": np.ones(15),
        "phase_near": 0.0,
        "phase_mid": 0.0,
        "phase_far": 0.0,
        "a_near": 1.0,
        "a_mid": 0.8,
        "a_far": 0.6,
        "structural_curvature": 0.0,
    }


def _link(source, target, span, *, safe=True, barrier=False):
    return {
        "source": source,
        "target": target,
        "span": span,
        "safe": safe,
        "barrier": barrier,
        "cost": 1.0,
        "relation": "FAULT_OFFSET_CORRESPONDENCE" if barrier else "CANDIDATE",
        "waveform_cosine": 0.9,
        "phase_cosine": 0.9,
        "shift_continuity": 0.0,
    }


def test_weak_local_events_cover_dimmer_reflector():
    h, w = 96, 16
    t, x = np.indices((h, w))
    pulse = np.exp(-0.5 * ((t - 24) / 1.6) ** 2)
    weak = 0.25 * np.exp(-0.5 * ((t - 59) / 1.6) ** 2)
    avo = np.stack([pulse + weak, 0.8 * (pulse + weak), 0.6 * (pulse + weak)])
    tau = (t + 0.0 * x) / 80
    strong = {
        "event_quantile": 0.95,
        "minimum_distance": 5,
        "event_prominence_fraction": 0.06,
        "event_refinement_radius": 2,
        "event_waveform_radius": 2,
        "minimum_phase_concentration": 0.35,
        "maximum_events_per_trace": 28,
    }
    weak_settings = {
        "local_envelope_quantile": 0.55,
        "local_prominence_fraction": 0.1,
        "minimum_phase_concentration": 0.7,
    }
    events, counts = detect_events(
        avo, tau, np.ones((h, w), bool), strong, weak_settings, _config()
    )
    assert counts["weak"] > 0
    assert any(event["source_type"] == "weak" and abs(event["time"] - 59) < 1 for event in events)


def test_one_missing_trace_bridges_but_barrier_blocks_gap():
    h, w = 80, 20
    tau = np.broadcast_to(np.arange(h)[:, None] / 80, (h, w)).copy()
    events = [_event(x, 30) for x in range(w) if x != 8]
    rows = candidate_links(NativeRGT(tau), events, radius=2, max_gap=1)
    gap = next(row for row in rows if events[row["source"]]["trace"] == 7 and row["span"] == 2)
    scales = {
        key: 1.0
        for key in (
            "d_tau",
            "d_time",
            "waveform_penalty",
            "phase_penalty",
            "ava_penalty",
            "shift_continuity",
        )
    }
    scales.update(barrier_shift_jump=1.0, barrier_shift_second=1.0)
    scored = score_links(rows, scales, _config())
    tracking = track_paths(events, scored, _config())
    assert any(
        row["source"] == gap["source"] and row["target"] == gap["target"]
        for row in tracking["accepted_links"]
    )
    graph = sparse_graph(events, tracking, 2, _config(), 1.0)
    assert any(edge["gap_count"] == 1 for edge in graph["edges"])
    blocked = [
        dict(row, safe=False, barrier=True, relation="FAULT_OFFSET_CORRESPONDENCE")
        if row["span"] == 2
        else row
        for row in scored
    ]
    split = track_paths(events, blocked, _config())
    assert all(
        not (events[row["source"]]["trace"] < 8 < events[row["target"]]["trace"])
        for row in split["accepted_links"]
    )
    assert split["metadata"]


def test_crossing_segments_detected_and_not_selected():
    events = [_event(x, 30) for x in range(20)] + [_event(x, 45) for x in range(20)]
    a = _link(0, 1, 1)
    b = _link(20, 21, 1)
    assert not _crosses(a, b, events)
    crossing_events = [_event(0, 30), _event(1, 45), _event(0, 45), _event(1, 30)]
    assert _crosses(_link(0, 1, 1), _link(2, 3, 1), crossing_events)


def test_native_shift_discontinuity_rejects_plausible_skip_correspondence():
    h, w = 80, 20
    time, trace = np.indices((h, w))
    tau = (time - 6 * (trace >= 8)) / 80
    events = [_event(x, 30 if x < 8 else 36) for x in range(w) if x != 8]
    for event in events:
        event["tau"] = 30 / 80
    rows = candidate_links(NativeRGT(tau), events, radius=2, max_gap=1)
    gap = next(row for row in rows if events[row["source"]]["trace"] == 7 and row["span"] == 2)
    assert gap["shift_jump"] > 1
    scales = {
        key: 1.0
        for key in (
            "d_tau",
            "d_time",
            "waveform_penalty",
            "phase_penalty",
            "ava_penalty",
            "shift_continuity",
        )
    }
    scales.update(barrier_shift_jump=1.0, barrier_shift_second=1.0)
    scored = score_links([gap], scales, _config())[0]
    assert scored["plausible"]
    assert scored["barrier"]
    assert not scored["safe"]
    assert scored["relation"] == "FAULT_OFFSET_CORRESPONDENCE"
