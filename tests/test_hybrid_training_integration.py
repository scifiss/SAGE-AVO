"""Training/tiling alignment and component isolation on public synthetic arrays."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sage_avo.data.augmentation import AugmentationConfig, augment_patch
from sage_avo.config import load_config
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import build_patch_sampling_weights
from sage_avo.data.sparse_topology import (
    TopologyCache,
    canonical_hash,
    collate_sparse_patches,
    observable_hashes,
)
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.experiments.hybrid_integration import build_hybrid_condition, resolve_hybrid_config
from sage_avo.experiments.training import _validation_sample_metrics
from sage_avo.models import hybrid_sparse as hs
from sage_avo.models.sage_avo import SAGEAVO
from sage_avo.training.engine import _move_batch, sparse_forward_kwargs


ROOT = Path(__file__).resolve().parents[1]


def two_component_graph(shape=(12, 24)):
    nodes = [
        {"node": i, "component": i // 2, "time": t, "trace": x}
        for i, (t, x) in enumerate([(3.2, 2), (3.4, 5), (8.2, 18), (8.4, 21)])
    ]
    edges = [
        {
            "source": i,
            "target": i + 1,
            "component": i // 2,
            "delta_tau": 0.001,
            "delta_t": 0.2,
            "delta_x": 3,
            "geodesic_length": 3.1,
            "waveform_continuity": 0.9,
            "phase_continuity": 0.9,
            "shift_continuity": 0.01,
            "curvature": 0.01,
            "gap_count": 0,
        }
        for i in (0, 2)
    ]
    return hs.crop_accepted_graph(nodes, edges, top=0, left=0, raw_shape=shape, output_shape=shape)


@pytest.fixture
def fixture_dataset(tmp_path):
    rows, cols = np.indices((12, 24), dtype=np.float32)
    avo = np.stack((rows + 2 * cols, rows + 3 * cols, rows + 4 * cols)).astype(np.float32)
    norm = {
        "x_mean": [0.0] * 3,
        "x_std": [1.0] * 3,
        "y_mean": [3000.0, 1700.0, 2.4],
        "y_std": [100.0, 100.0, 0.1],
    }
    arrays = {
        "avo": avo,
        "rgt": rows.copy(),
        "valid_mask": np.ones((12, 24), np.float32),
        "elastic": np.broadcast_to(
            np.array([3000.0, 1700.0, 2.4], np.float32)[:, None, None], avo.shape
        ).copy(),
        "low": np.broadcast_to(
            np.array([2990.0, 1690.0, 2.39], np.float32)[:, None, None], avo.shape
        ).copy(),
        "segmentation": ((cols > 10) & (rows > 6)).astype(np.int64),
        "avo_clean": avo.copy(),
    }
    (tmp_path / "realizations").mkdir()
    np.savez(tmp_path / "realizations/realization_0007.npz", **arrays)
    (tmp_path / "normalization.json").write_text(json.dumps(norm))
    rows_index = []
    for split in ("train", "validation"):
        for top, left, height, width in [(0, 0, 12, 24), (0, 0, 8, 12), (0, 10, 6, 8)]:
            rows_index.append(
                dict(
                    split=split,
                    realization_id=7,
                    top=top,
                    left=left,
                    raw_height=height,
                    raw_width=width,
                    output_height=12,
                    output_width=24,
                    physics_eligible=int(height == 12),
                    convolution_halo_samples=0,
                    native_dt_seconds=0.004,
                    mute_origin_seconds=0.0,
                )
            )
    pd.DataFrame(rows_index).to_csv(tmp_path / "patch_index.csv", index=False)
    cache = TopologyCache(tmp_path / "cache", {}, {})
    graph = two_component_graph()
    nodes = [
        {
            "node": i,
            "component": int(graph.components[i]),
            "time": float(c[0]),
            "trace": float(c[1]),
        }
        for i, c in enumerate(graph.coordinates)
    ]
    edges = [
        {
            "source": i,
            "target": i + 1,
            "component": i // 2,
            "delta_tau": 0.001,
            "delta_t": 0.2,
            "delta_x": 3.0,
            "geodesic_length": 3.1,
            "waveform_continuity": 0.9,
            "phase_continuity": 0.9,
            "shift_continuity": 0.01,
            "curvature": 0.01,
            "gap_count": 0,
        }
        for i in (0, 2)
    ]
    record = dict(
        schema=1,
        realization_id=7,
        geometry={"shape": [12, 24], "coordinates": "native_time_sample_trace"},
        observable_hashes=observable_hashes(avo, arrays["rgt"], arrays["valid_mask"]),
        config_sha256=cache.config_hash,
        source_sha256=cache.source_hashes,
        nodes=nodes,
        edges=edges,
    )
    record["topology_sha256"] = canonical_hash(record)
    cache.directory.mkdir()
    (cache.directory / "realization_7.json").write_text(json.dumps(record))
    return tmp_path, cache, arrays, norm


def test_loader_cache_tiling_and_variable_empty_collation(fixture_dataset):
    path, cache, arrays, _ = fixture_dataset
    dataset = IndexedRealizationPatches(path, "train", topology_cache=cache)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_sparse_patches)))
    assert len(batch["sparse_graphs"]) == 3
    assert batch["sparse_graphs"][2].edge_index.numel() == 0
    provider = cache.tile_provider(
        7, (12, 24), avo=arrays["avo"], rgt=arrays["rgt"], support=arrays["valid_mask"]
    )
    expected = provider([(0, 0)])[0]
    actual = batch["sparse_graphs"][0]
    for name in ("coordinates", "components", "edge_index", "edge_attr"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    # Node sampling agrees with the resized observed linear ramp as well.
    patch = dataset[1]
    sampled = hs.sample_cnn_nodes(patch["avo"][None], patch["sparse_graph"])
    graph = patch["sparse_graph"]
    native_t = graph.coordinates[:, 0] * 7 / 11
    native_x = graph.coordinates[:, 1] * 11 / 23
    assert torch.allclose(sampled[:, 0], native_t + 2 * native_x, atol=1e-5)
    moved = _move_batch(batch, torch.device("cpu"))
    assert len(sparse_forward_kwargs(moved)["graphs"]) == 3


def test_loader_syncs_flip_physics_and_keeps_gain_noise_topology(fixture_dataset):
    path, cache, arrays, _ = fixture_dataset
    config = AugmentationConfig(
        horizontal_flip_probability=1, avo_gain_probability=0, avo_noise_probability=0
    )
    dense = IndexedRealizationPatches(
        path,
        "train",
        augment=True,
        augmentation_config=config,
        augmentation_generator=torch.Generator().manual_seed(9),
        matched_augmentation=True,
    )
    hybrid = IndexedRealizationPatches(
        path,
        "train",
        augment=True,
        augmentation_config=config,
        augmentation_generator=torch.Generator().manual_seed(9),
        matched_augmentation=True,
        topology_cache=cache,
    )
    a, c = dense[0], hybrid[0]
    assert c["augmentation_horizontal_flip"]
    for key in a:
        assert torch.equal(a[key], c[key])
    graph = c["sparse_graph"]
    base = cache.patch(7, top=0, left=0, raw_shape=(12, 24), output_shape=(12, 24))
    assert torch.equal(graph.components, base.components)
    assert torch.equal(graph.edge_index, base.edge_index)
    assert torch.equal(graph.edge_attr[:, 2], -base.edge_attr[:, 2])
    samples = hs.sample_cnn_nodes(c["avo"][None], graph)
    original_samples = hs.sample_cnn_nodes(torch.from_numpy(arrays["avo"])[None], base)
    assert torch.allclose(samples, original_samples, atol=2e-5)
    assert torch.equal(c["physics_avo"], c["avo"])
    changed = augment_patch(
        c,
        AugmentationConfig(
            horizontal_flip_probability=0,
            avo_gain_probability=1,
            avo_gain_minimum=2,
            avo_gain_maximum=2,
            avo_noise_probability=1,
        ),
        generator=torch.Generator().manual_seed(8),
    )
    assert changed["sparse_graph"] is graph
    assert not torch.equal(changed["avo"], c["avo"])


def test_original_sampler_distribution_and_rng_are_unchanged(fixture_dataset):
    path, cache, _, _ = fixture_dataset
    a = IndexedRealizationPatches(path, "train")
    c = IndexedRealizationPatches(path, "train", topology_cache=cache)
    wa, wc = build_patch_sampling_weights(a), build_patch_sampling_weights(c)
    assert torch.equal(wa, wc)
    orders = [
        list(
            WeightedRandomSampler(
                w, 30, replacement=True, generator=torch.Generator().manual_seed(91)
            )
        )
        for w in (wa, wc)
    ]
    assert orders[0] == orders[1]
    base = a[0]
    original = augment_patch(base, generator=torch.Generator().manual_seed(1))
    matched = augment_patch(base, generator=torch.Generator().manual_seed(1), record_geometry=True)
    assert all(torch.equal(original[k], matched[k]) for k in original)


def test_cache_refuses_mismatched_source_inputs_and_corruption(fixture_dataset):
    _, cache, arrays, _ = fixture_dataset
    with pytest.raises(ValueError, match="observations"):
        cache.validate_observations(
            7, avo=arrays["avo"] + 1, rgt=arrays["rgt"], support=arrays["valid_mask"]
        )
    with pytest.raises(ValueError, match="config/source"):
        TopologyCache(cache.directory, {"changed": True}, {}).load(7)
    path = cache.directory / "realization_7.json"
    record = json.loads(path.read_text())
    record["nodes"][0]["trace"] += 1
    path.write_text(json.dumps(record))
    cache.load.cache_clear()
    with pytest.raises(ValueError, match="content hash"):
        cache.load(7)


def test_observable_cache_prepares_once_with_real_frozen_algorithm(tmp_path, monkeypatch):
    from sage_avo.diagnostics import gap_tolerant_graph

    config = load_config(ROOT / "configs/development_diagnostics_v00332z.yaml")
    frozen = dict(
        weak_detector=config["weak_candidates"][0],
        search_radius=4,
        node_spacing=4,
        curvature_threshold=1.0,
        scales={
            "d_tau": 1.0,
            "d_time": 1.0,
            "waveform_penalty": 0.1,
            "phase_penalty": 0.2,
            "ava_penalty": 0.1,
            "shift_continuity": 0.1,
            "barrier_shift_jump": 1.0,
            "barrier_shift_second": 1.0,
        },
    )
    t, x = np.indices((96, 20))
    pulse = np.exp(-0.5 * ((t - 30) / 1.6) ** 2)
    avo = np.stack([pulse, 0.8 * pulse, 0.6 * pulse]).astype(np.float32)
    rgt = ((t + 0 * x) / 95).astype(np.float32)
    support = np.ones_like(rgt)
    calls = []
    detector = gap_tolerant_graph.detect_events

    def counted(*args, **kwargs):
        calls.append(True)
        return detector(*args, **kwargs)

    monkeypatch.setattr(gap_tolerant_graph, "detect_events", counted)
    cache = TopologyCache(tmp_path / "private_cache", frozen, config)
    first = cache.prepare(7, avo=avo, rgt=rgt, support=support)
    second = cache.prepare(7, avo=avo, rgt=rgt, support=support)
    assert first == second and len(calls) == 1
    assert first["nodes"] and first["edges"]
    assert set(first["observable_hashes"]) == {"avo", "rgt", "support"}
    assert not list(cache.directory.glob("*.tmp"))


def test_legacy_decoder_coupling_removed_by_post_decoder_projection(monkeypatch):
    torch.manual_seed(91)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2).eval()
    graph = two_component_graph()
    avo, low = torch.randn(1, 3, 12, 24), torch.randn(1, 3, 12, 24)
    rgt = torch.arange(12).float()[None, :, None].expand(1, 12, 24)
    args = (low, torch.tensor([0.4]), avo, low, rgt, [graph])
    original_sampling = hs.sample_cnn_nodes

    def perturb(cnn, item):
        features = original_sampling(cnn, item).clone()
        features[item.components == 0, 0] += 5
        return features

    for cls, legacy in [(hs.LegacyDecoderHybridSAGEAVO, True), (hs.HybridSparseSAGEAVO, False)]:
        wrapper = cls(dense, 8, gamma=0.1).eval()
        monkeypatch.setattr(hs, "sample_cnn_nodes", original_sampling)
        with torch.no_grad():
            before = wrapper(*args).velocity
        monkeypatch.setattr(hs, "sample_cnn_nodes", perturb)
        with torch.no_grad():
            after = wrapper(*args).velocity
        # Component B's footprint is far outside A's decoder convolution reach.
        difference = (after - before)[:, :, 8:10, 18:23].abs().max().item()
        assert difference > 1e-6 if legacy else difference == 0


def test_disrupted_payload_control_preserves_degrees_capacity_and_initialization():
    config = resolve_hybrid_config(ROOT / "configs/development_diagnostics_v00332ac.yaml")
    config["model"].update(hidden_channels=8, graph_layers=1, graph_heads=2)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    a, b, c = [build_hybrid_condition(config, name) for name in ("A", "B", "C")]
    assert torch.equal(cpu_state, torch.get_rng_state())
    if cuda_states:
        assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))
    assert sum(p.numel() for p in b.parameters()) == sum(p.numel() for p in c.parameters())
    assert all(torch.equal(a.state_dict()[k], c.dense.state_dict()[k]) for k in a.state_dict())
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in b.state_dict().items())
    graph = two_component_graph()
    perm = hs.component_source_permutation(graph, b.sparse.control_seed)
    assert torch.all(perm != torch.arange(4))
    assert torch.equal(graph.components[perm], graph.components)
    assert torch.equal(perm, hs.component_source_permutation(graph, b.sparse.control_seed))
    features = torch.randn(4, 8)
    assert not torch.allclose(
        b.sparse.messages(features, graph), c.sparse.messages(features, graph)
    )
    assert torch.equal(graph.edge_index, deepcopy(graph).edge_index)


def test_whole_flow_zero_scale_and_active_tiled_inference(fixture_dataset):
    _, cache, arrays, norm = fixture_dataset
    torch.manual_seed(18)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2)
    wrapper = hs.HybridSparseSAGEAVO(dense, 8, gamma=0)
    common = dict(
        avo=arrays["avo"],
        low=arrays["low"],
        rgt=arrays["rgt"],
        normalization=norm,
        patch_shape=(12, 24),
        stride=(6, 12),
        steps=4,
        batch_size=1,
        device=torch.device("cpu"),
    )
    provider = cache.tile_provider(
        7, (12, 24), avo=arrays["avo"], rgt=arrays["rgt"], support=arrays["valid_mask"]
    )
    baseline = infer_full_realization(dense, **common)
    matched = infer_full_realization(wrapper, graph_provider=provider, **common)
    assert all(np.array_equal(a, b) for a, b in zip(baseline, matched))
    wrapper.gamma = 0.1
    active, labels = infer_full_realization(wrapper, graph_provider=provider, **common)
    assert np.isfinite(active).all()
    assert labels.shape == (12, 24)


def test_actual_validation_sampling_handles_variable_and_empty_graph_batch(fixture_dataset):
    path, cache, _, _ = fixture_dataset
    dataset = IndexedRealizationPatches(path, "validation", topology_cache=cache)
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_sparse_patches)
    batch = next(iter(loader))
    torch.manual_seed(23)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2).eval()
    wrapper = hs.HybridSparseSAGEAVO(dense, 8, gamma=0.1).eval()
    args = (batch["low"], torch.full((3,), 0.4), batch["avo"], batch["low"], batch["rgt"])
    with torch.no_grad():
        reference = dense(*args)
        active = wrapper(*args, batch["sparse_graphs"])
    assert torch.equal(reference.velocity[2], active.velocity[2])
    assert torch.equal(reference.segmentation_logits, active.segmentation_logits)
    metrics = _validation_sample_metrics(
        wrapper, loader, torch.device("cpu"), steps=1, max_batches=1, guidance_scale=0.0
    )
    assert np.isfinite(metrics["criterion"])


def test_diagnostic_complete_objective_and_flow_uses_actual_collated_batch(
    fixture_dataset, tmp_path
):
    path, cache, _, norm = fixture_dataset
    module_spec = importlib.util.spec_from_file_location(
        "hybrid_integration_audit", ROOT / "scripts/check_hybrid_integration_v00332ac.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    dataset = IndexedRealizationPatches(path, "train", topology_cache=cache)
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_sparse_patches)))
    config = resolve_hybrid_config(ROOT / "configs/development_diagnostics_v00332ac.yaml")
    config["model"].update(hidden_channels=8, graph_layers=1, graph_heads=2)
    result = module.flow_qc(config, batch, norm, torch.device("cpu"), 4, tmp_path)
    assert result["zero_scale_whole_flow_max_error"] == 0
    assert result["active_direct_outside_support_max_error"] == 0
    assert result["all_gradients_finite"]
    assert result["exact_pp_endpoints_finite"]
    assert result["complete_objective_finite"]
    assert result["sparse_gradient_l1"] > 0
    assert result["parameter_state_unchanged"]
    assert result["B_C_initial_state_identical"]
    assert result["B_C_velocity_max_difference"] > 0


def test_report_and_test_exposure_audit_need_no_optional_formatter_or_test_data(
    tmp_path, monkeypatch
):
    module_spec = importlib.util.spec_from_file_location(
        "hybrid_report_audit", ROOT / "scripts/check_hybrid_integration_v00332ac.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    table = module.markdown_property_table([{"property": "Vp", "physical_rms": 0.125}])
    assert "| property | physical_rms |" in table and "| Vp | 0.125 |" in table
    baseline = tmp_path / "stage_artifacts/stage05/v00332d_epoch40_baseline/predictions/full"
    baseline.mkdir(parents=True)
    for rid in (7, 8):
        (baseline / f"realization_{rid}.npz").touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical exposure audit must never load test arrays")

    monkeypatch.setattr(np, "load", forbidden)
    exposure = module.historical_test_exposure(tmp_path, [7, 8])
    assert exposure["historically_evaluated_test_ids"] == [7, 8]
    assert exposure["status"] == "ALL_IMMUTABLE_TEST_CASES_PREVIOUSLY_EVALUATED"
    assert not exposure["independent_confirmation_available"]
    assert exposure["test_arrays_opened"] == 0
