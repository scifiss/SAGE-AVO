import numpy as np

from sage_avo.diagnostics.native_event_graph import (
    build_event_graph,
    candidate_event_pairs,
    detect_physical_events,
    event_repeatability,
    freeze_event_barriers,
    normal_event_contrasts,
    reciprocal_matches,
    robust_match_scales,
)
from sage_avo.diagnostics.native_rgt_graph import NativeRGT


def detector_config():
    return {
        "event_quantile": 0.55,
        "minimum_distance": 5,
        "event_prominence_fraction": 0.02,
        "event_refinement_radius": 2,
        "event_waveform_radius": 2,
        "minimum_phase_concentration": 0.2,
        "maximum_events_per_trace": 12,
    }


def match_config():
    return {
        "maximum_normalized_match_cost": 8.0,
        "minimum_match_waveform_cosine": 0.2,
        "minimum_match_phase_cosine": 0.0,
        "minimum_match_ava_cosine": 0.0,
        "barrier_robust_sigma": 6.0,
        "barrier_quantile_cap": 0.99,
        "minimum_barrier_waveform_cosine": 0.2,
        "minimum_barrier_phase_cosine": 0.0,
        "minimum_barrier_ava_cosine": 0.0,
        "minimum_component_points": 6,
        "minimum_component_trace_span": 5,
        "edge_spans_traces": [4, 8, 16],
    }


def synthetic():
    height, width = 100, 32
    time, trace = np.indices((height, width))
    event_time = 35 + 0.25 * trace + 2 * np.sin(trace / 6)
    signal = np.exp(-0.5 * ((time - event_time) / 1.5) ** 2)
    signal -= 0.8 * np.exp(-0.5 * ((time - event_time - 25) / 1.8) ** 2)
    avo = np.stack([signal, 0.8 * signal, 0.6 * signal]).astype(np.float32)
    tau = (time - 0.25 * trace - 2 * np.sin(trace / 6)) / 80
    valid = np.ones((height, width), dtype=bool)
    return tau, avo, valid, event_time[0]


def test_events_are_located_in_physical_time_then_mapped_to_native_rgt():
    tau, avo, valid, expected = synthetic()
    events = detect_physical_events(avo, tau, valid, detector_config())
    first = [event for event in events if event["time"] < 48]
    assert len(first) >= 0.8 * tau.shape[1]
    error = [abs(event["time"] - expected[event["trace"]]) for event in first]
    assert np.quantile(error, 0.9) < 1.0
    native = NativeRGT(tau)
    assert (
        max(
            abs(event["tau"] - native.forward_tau(event["trace"], event["time"]))
            for event in events
        )
        < 1e-12
    )


def test_angle_subset_event_localization_is_repeatable():
    tau, avo, valid, _ = synthetic()
    full = detect_physical_events(avo, tau, valid, detector_config())
    subset = detect_physical_events(avo, tau, valid, detector_config(), channels=(0, 2))
    result = event_repeatability(full, subset, tolerance=0.5)
    assert result["repeatability"] > 0.9
    assert result["error_p95"] < 0.5


def test_native_rgt_prediction_produces_reciprocal_event_association():
    tau, avo, valid, _ = synthetic()
    events = detect_physical_events(avo, tau, valid, detector_config())
    pairs = candidate_event_pairs(NativeRGT(tau), events, search_radius=3.0)
    scales = robust_match_scales(pairs)
    weights = {"tau": 1.0, "time": 1.0, "waveform": 1.0, "phase": 1.0, "ava": 1.0}
    matches = reciprocal_matches(pairs, scales, weights, match_config())
    assert sum(row["reflector_match"] for row in matches) >= tau.shape[1] - 2
    assert all(row["d_time"] <= 3.0 for row in matches)


def test_preunion_barrier_blocks_fault_and_safe_long_edges_cannot_jump_it():
    events = [
        {
            "event": trace,
            "trace": trace,
            "time": 20 + 0.4 * trace + (6 if trace >= 8 else 0),
            "tau": 0.5,
            "waveform": np.ones(15),
            "structural_curvature": 0.0,
        }
        for trace in range(18)
    ]
    matches = []
    for trace in range(17):
        matches.append(
            {
                "source": trace,
                "target": trace + 1,
                "reflector_match": True,
                "shift_jump": 8.0 if trace == 7 else 0.0,
                "shift_second_difference": 8.0 if trace == 7 else 0.0,
                "dip_difference": 0.0,
                "waveform_cosine": 0.95,
                "phase_cosine": 0.95,
                "ava_cosine": 0.95,
            }
        )
    thresholds = {
        "shift_jump": 2.0,
        "shift_second_difference": 2.0,
        "dip_difference": 2.0,
        "waveform_cosine": 0.5,
        "phase_cosine": 0.2,
        "ava_cosine": 0.2,
    }
    graph = build_event_graph(
        events,
        matches,
        thresholds,
        match_config(),
        node_spacing=2,
        curvature_threshold=1.0,
    )
    assert len(graph["components"]) == 2
    assert len(graph["metadata"]) == 1
    assert all(
        not (graph["nodes"][edge["source"]]["trace"] < 8 <= graph["nodes"][edge["target"]]["trace"])
        for edge in graph["edges"]
    )


def test_training_barriers_and_local_normal_contrasts_are_finite():
    tau, avo, valid, _ = synthetic()
    events = detect_physical_events(avo, tau, valid, detector_config())
    pairs = candidate_event_pairs(NativeRGT(tau), events, search_radius=3.0)
    scales = robust_match_scales(pairs)
    weights = {"tau": 1.0, "time": 1.0, "waveform": 1.0, "phase": 1.0, "ava": 1.0}
    matches = reciprocal_matches(pairs, scales, weights, match_config())
    thresholds = freeze_event_barriers(matches, match_config())
    for event in events:
        event["structural_curvature"] = 0.0
    graph = build_event_graph(
        events,
        matches,
        thresholds,
        match_config(),
        node_spacing=2,
        curvature_threshold=1.0,
    )
    contrasts = normal_event_contrasts(tau, avo, graph["nodes"], offset=1.0)
    values = [[row["delta_p"], row["delta_g"], row["delta_c"]] for row in contrasts]
    assert len(values) == len(graph["nodes"])
    assert np.isfinite(values).all()
