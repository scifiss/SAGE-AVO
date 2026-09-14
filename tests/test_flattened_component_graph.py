import numpy as np

from sage_avo.diagnostics.flattened_component_graph import (
    build_component_graph,
    detect_reflector_components,
    flatten_rgt,
    normal_contrasts,
)


def config():
    return {
        "ridge_quantile": 0.55,
        "ridge_minimum_distance": 4,
        "ridge_prominence_fraction": 0.02,
        "maximum_candidates_per_trace": 12,
        "maximum_flattened_jump_samples": 2.5,
        "minimum_waveform_cosine": 0.7,
        "minimum_phase_cosine": 0.2,
        "minimum_strength_ratio": 0.1,
        "waveform_radius": 2,
        "minimum_component_trace_span": 8,
        "minimum_component_points": 6,
        "node_spacing_traces": 4,
        "edge_spans_traces": [4, 8, 16],
    }


def synthetic():
    h, w = 80, 48
    t, x = np.indices((h, w))
    tau = t / 60 + 0.03 * np.sin(x / 9)
    signal = np.exp(-0.5 * ((tau - 0.45) / 0.025) ** 2)
    signal -= 0.8 * np.exp(-0.5 * ((tau - 0.85) / 0.03) ** 2)
    avo = np.stack([signal, 0.8 * signal, 0.6 * signal]).astype(np.float32)
    return tau, avo, np.ones((h, w), bool)


def test_flattening_round_trip_is_accurate_without_plateaus():
    tau, avo, valid = synthetic()
    mapping = flatten_rgt(tau, avo, valid, n_tau=80)
    assert np.quantile(mapping["roundtrip_tau_steps"], 0.99) < 1e-5
    assert np.quantile(mapping["roundtrip_time_samples"], 0.99) < 0.05
    assert mapping["avo"].shape == (3, 80, 48)


def test_plateaus_are_reported_not_reordered():
    tau, avo, valid = synthetic()
    tau[20:23] = tau[20]
    mapping = flatten_rgt(tau, avo, valid, n_tau=80)
    assert mapping["ambiguous_plateau_fraction"] > 0
    assert np.isfinite(mapping["inverse_t"]).all()


def test_components_are_observable_ridges_and_edges_never_cross_components():
    tau, avo, valid = synthetic()
    mapping = flatten_rgt(tau, avo, valid, n_tau=80)
    detection = detect_reflector_components(mapping, config())
    graph = build_component_graph(mapping, detection, config())
    assert len(detection["components"]) >= 2
    assert len(graph["nodes"]) > 0
    assert any(edge["span"] >= 16 for edge in graph["edges"])
    for edge in graph["edges"]:
        assert graph["nodes"][edge["source"]]["component"] == edge["component"]
        assert graph["nodes"][edge["target"]]["component"] == edge["component"]


def test_normal_contrast_is_local_and_finite():
    tau, avo, valid = synthetic()
    mapping = flatten_rgt(tau, avo, valid, n_tau=80)
    detection = detect_reflector_components(mapping, config())
    graph = build_component_graph(mapping, detection, config())
    rows = normal_contrasts(tau, avo, graph["nodes"], offset=1.0)
    assert len(rows) == len(graph["nodes"])
    assert np.isfinite([[r[f"delta_ava_{k}"] for k in range(3)] for r in rows]).all()
