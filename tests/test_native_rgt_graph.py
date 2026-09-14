import numpy as np

from sage_avo.diagnostics.native_rgt_graph import (
    NativeRGT,
    build_barrier_graph,
    freeze_thresholds,
    refine_candidates,
)


def graph_config():
    return {
        "minimum_component_points": 4,
        "minimum_component_trace_span": 3,
        "edge_spans_traces": [1, 2, 4],
    }


def test_native_inverse_is_exact_on_identifiable_samples_and_marks_plateaus():
    time, trace = np.indices((30, 8))
    tau = 0.2 * time + 0.03 * trace + 0.0004 * time**2
    native = NativeRGT(tau)
    time_error, tau_error = native.roundtrip()
    assert np.max(time_error) < 1e-10
    assert np.max(tau_error) < 1e-10

    tau[10:13, 2] = tau[10, 2]
    plateau_native = NativeRGT(tau)
    assert not plateau_native.identifiable[10:13, 2].any()
    assert np.isfinite(plateau_native.inverse_time(2, tau[10, 2]))


def test_continuous_refinement_uses_native_inverse_and_improves_strength():
    height, width = 80, 12
    time, trace = np.indices((height, width))
    tau = time / 60 + 0.013 * np.sin(trace / 3)
    center_tau = 0.607
    signal = np.exp(-0.5 * ((tau - center_tau) / 0.018) ** 2)
    avo = np.stack([signal, 0.8 * signal, 0.6 * signal])
    tau_grid = np.linspace(tau.max(axis=0).min() * 0, tau.max(axis=0).min(), 31)
    coarse_index = int(np.argmin(abs(tau_grid - center_tau)))
    detection = {
        "candidates": [
            {"candidate": trace_index, "trace": trace_index, "tau_index": coarse_index}
            for trace_index in range(width)
        ]
    }
    refined = refine_candidates(
        NativeRGT(tau),
        avo,
        {"tau_grid": tau_grid},
        detection,
        width_steps=1.0,
        samples=25,
    )
    assert np.median([row["refined_strength"] for row in refined]) >= np.median(
        [row["coarse_strength"] for row in refined]
    )
    assert (
        np.median([abs(row["refined_tau"] - center_tau) for row in refined])
        < np.diff(tau_grid).mean()
    )


def test_fault_barrier_precedes_union_and_long_edges_require_safe_path():
    refined = [
        {
            "candidate": trace,
            "trace": trace,
            "refined_tau": 0.5,
            "refined_time": 20.0 + 0.5 * trace + (6.0 if trace >= 4 else 0.0),
        }
        for trace in range(9)
    ]
    links = []
    for trace in range(8):
        links.append(
            {
                "source": trace,
                "target": trace + 1,
                "boundary": trace,
                "reciprocal": True,
                "shift_jump": 8.0 if trace == 3 else 0.0,
                "shift_second_difference": 8.0 if trace == 3 else 0.0,
                "original_waveform_cosine": 0.95,
                "original_phase_cosine": 0.95,
                "strength_ratio": 0.9,
            }
        )
    thresholds = {
        "shift_jump": 2.0,
        "shift_second_difference": 2.0,
        "original_waveform_cosine": 0.7,
        "original_phase_cosine": 0.5,
        "strength_ratio": 0.2,
    }
    graph = build_barrier_graph(refined, links, thresholds, graph_config())
    assert len(graph["components"]) == 2
    assert len(graph["metadata"]) == 1
    assert graph["metadata"][0]["relation"] == "FAULT_OFFSET_CORRESPONDENCE"
    assert all(
        not (graph["nodes"][edge["source"]]["trace"] < 4 <= graph["nodes"][edge["target"]]["trace"])
        for edge in graph["edges"]
    )


def test_threshold_freezing_uses_only_observable_reciprocal_links():
    rows = [
        {
            "reciprocal": True,
            "shift_jump": value,
            "shift_second_difference": value / 2,
            "original_waveform_cosine": 0.9,
            "original_phase_cosine": 0.8,
            "strength_ratio": 0.7,
            "physical_shift": 1 + value,
        }
        for value in np.linspace(0, 1, 20)
    ]
    config = {
        "barrier_robust_sigma": 6.0,
        "barrier_quantile_cap": 0.99,
        "minimum_original_waveform_cosine_floor": 0.5,
        "minimum_original_phase_cosine_floor": 0.2,
        "minimum_strength_ratio_floor": 0.2,
    }
    thresholds = freeze_thresholds(rows, config)
    assert thresholds["shift_jump"] <= 1.0
    assert thresholds["original_waveform_cosine"] >= 0.5
    assert thresholds["high_shift_magnitude"] > 1.0


def test_zero_inflated_shift_distribution_does_not_create_epsilon_barrier():
    values = [0.0] * 95 + [0.25, 0.5, 0.75, 1.0, 4.0]
    rows = [
        {
            "reciprocal": True,
            "shift_jump": value,
            "shift_second_difference": value,
            "original_waveform_cosine": 0.9,
            "original_phase_cosine": 0.8,
            "strength_ratio": 0.7,
            "physical_shift": 1.0,
        }
        for value in values
    ]
    config = {
        "barrier_robust_sigma": 6.0,
        "barrier_quantile_cap": 0.99,
        "minimum_original_waveform_cosine_floor": 0.5,
        "minimum_original_phase_cosine_floor": 0.2,
        "minimum_strength_ratio_floor": 0.2,
    }
    thresholds = freeze_thresholds(rows, config)
    assert thresholds["shift_jump"] >= np.quantile(values, 0.95)
    assert thresholds["shift_jump"] <= np.quantile(values, 0.99)
