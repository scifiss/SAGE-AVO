from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
augmentation = pytest.importorskip("sage_avo.data.augmentation")
sampling = pytest.importorskip("sage_avo.data.sampling")
checkpoints = pytest.importorskip("sage_avo.training.checkpoints")
losses = pytest.importorskip("sage_avo.training.losses")
schedules = pytest.importorskip("sage_avo.training.schedules")

AugmentationConfig = augmentation.AugmentationConfig
augment_patch = augmentation.augment_patch
PatchSamplingConfig = sampling.PatchSamplingConfig
build_patch_sampling_weights = sampling.build_patch_sampling_weights
load_checkpoint = checkpoints.load_checkpoint
save_checkpoint = checkpoints.save_checkpoint
migrate_original_sage_avo_state_dict = checkpoints.migrate_original_sage_avo_state_dict
LossWeights = losses.LossWeights
masked_ssim_loss = losses.masked_ssim_loss
physics_loss = losses.physics_loss
segmentation_loss = losses.segmentation_loss
legacy_instance_contrastive_loss = losses.legacy_instance_contrastive_loss
Curriculum = schedules.Curriculum


class _SamplingDataset:
    def __init__(self):
        rows = torch.arange(8).view(8, 1).expand(8, 6).float()
        self.items = [
            {
                "avo": torch.zeros(3, 8, 6),
                "rgt": rows,
                "segmentation": torch.zeros(8, 6, dtype=torch.long),
            },
            {
                "avo": torch.stack((rows, -rows, 2.0 * rows)),
                "rgt": rows + 0.3 * torch.arange(6).view(1, 6),
                "segmentation": torch.ones(8, 6, dtype=torch.long),
            },
        ]

    def __len__(self):
        return len(self.items)

    def sampling_fields(self, index):
        return self.items[index]


def test_weighted_patch_sampler_is_finite_normalized_and_content_sensitive():
    weights = build_patch_sampling_weights(
        _SamplingDataset(),
        PatchSamplingConfig(upper_quantile=1.0),
    )
    assert weights.dtype == torch.float64
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=torch.float64))
    assert weights[1] > weights[0]


def test_registered_augmentation_preserves_contract_and_flips_geometry():
    avo = torch.arange(3 * 4 * 5).reshape(3, 4, 5).float()
    item = {
        "avo": avo,
        "target": avo + 1,
        "low": avo + 2,
        "rgt": torch.arange(20).reshape(4, 5).float(),
        "mask": torch.ones(1, 4, 5),
        "segmentation": torch.zeros(4, 5, dtype=torch.long),
        "dip": torch.ones(4, 5),
    }
    config = AugmentationConfig(
        horizontal_flip_probability=1.0,
        avo_gain_probability=1.0,
        avo_gain_minimum=1.0,
        avo_gain_maximum=1.0,
        avo_noise_probability=0.0,
    )
    result = augment_patch(item, config, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(result["avo"], torch.flip(avo, dims=(-1,)))
    torch.testing.assert_close(result["dip"], -torch.flip(item["dip"], dims=(-1,)))
    for name in ("avo", "target", "low", "rgt", "mask", "segmentation"):
        assert result[name].shape == item[name].shape


def test_curriculum_reproduces_final_005_endpoints():
    base = LossWeights()
    schedule = Curriculum()
    first = schedule.weights_for_epoch(base, 0, 120)
    last = schedule.weights_for_epoch(base, 119, 120)
    assert first.density == pytest.approx(2.0)
    assert last.density == pytest.approx(3.5)
    assert first.ssim == pytest.approx(0.15)
    assert last.ssim == pytest.approx(0.05)
    assert last.physics == pytest.approx(0.35)
    assert last.structure == pytest.approx(0.375)


def test_segmentation_and_ssim_ignore_invalid_pixels():
    logits = torch.randn(1, 3, 8, 7)
    target = torch.randint(0, 3, (1, 8, 7))
    mask = torch.ones(1, 1, 8, 7)
    mask[:, :, :, -2:] = 0
    first, _, _ = segmentation_loss(logits, target, mask)
    changed_logits = logits.clone()
    changed_logits[:, :, :, -2:] += 1000.0
    changed_target = target.clone()
    changed_target[:, :, -2:] = (changed_target[:, :, -2:] + 1) % 3
    second, _, _ = segmentation_loss(changed_logits, changed_target, mask)
    torch.testing.assert_close(first, second)

    prediction = torch.randn(1, 3, 8, 7)
    reference = torch.randn_like(prediction)
    first_ssim = masked_ssim_loss(prediction, reference, mask)
    changed_prediction = prediction.clone()
    changed_prediction[:, :, :, -2:] += 1000.0
    # SSIM windows cross the validity boundary, so mask a wider collar for this check.
    interior_mask = torch.zeros_like(mask)
    interior_mask[:, :, 2:-2, 2:4] = 1
    first_ssim = masked_ssim_loss(prediction, reference, interior_mask, window_size=3)
    second_ssim = masked_ssim_loss(changed_prediction, reference, interior_mask, window_size=3)
    torch.testing.assert_close(first_ssim, second_ssim)


def test_physics_loss_has_finite_gradient_to_elastic_state():
    prediction = torch.zeros(1, 3, 12, 5, requires_grad=True)
    y_mean = torch.tensor([3000.0, 1600.0, 2.35]).view(1, 3, 1, 1)
    y_std = torch.tensor([250.0, 180.0, 0.04]).view(1, 3, 1, 1)
    x_mean = torch.zeros(1, 3, 1, 1)
    x_std = torch.tensor([0.003, 0.002, 0.0015]).view(1, 3, 1, 1)
    observed = torch.zeros(1, 3, 12, 5)
    mask = torch.ones(1, 1, 12, 5)
    prediction.data[:, 0, 6:] = 0.5
    loss = physics_loss(prediction, observed, y_mean, y_std, x_mean, x_std, mask=mask)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_optional_legacy_contrastive_term_is_differentiable():
    embeddings = torch.randn(1, 20, 8, requires_grad=True)
    mask = torch.ones(1, 1, 4, 5)
    loss = legacy_instance_contrastive_loss(
        embeddings,
        mask,
        max_samples=12,
        generator=torch.Generator().manual_seed(5),
    )
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_normalize_denormalize_round_trip():
    mean = torch.tensor([3000.0, 1600.0, 2.35]).view(1, 3, 1, 1)
    std = torch.tensor([250.0, 180.0, 0.04]).view(1, 3, 1, 1)
    physical = mean + std * torch.randn(2, 3, 5, 4)
    normalized = (physical - mean) / std
    torch.testing.assert_close(normalized * std + mean, physical)


def test_checkpoint_round_trip_includes_optimizer_scheduler_and_rng(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    generator = torch.Generator().manual_seed(9)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        2,
        {"validation": 1.0},
        {"name": "test"},
        scheduler=scheduler,
        generator_states={"time": generator.get_state()},
    )
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=4
    )
    checkpoint = load_checkpoint(
        path,
        restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
    assert checkpoint["epoch"] == 2
    assert "rng_state" in checkpoint


def test_checkpoint_rng_buffers_are_restored_as_cpu_byte_tensors(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(9)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        2,
        {},
        {},
        generator_states={"time": generator.get_state()},
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["rng_state"]["torch_cpu"] = checkpoint["rng_state"][
        "torch_cpu"
    ].to(torch.int64)
    checkpoint["rng_state"]["generators"]["time"] = checkpoint["rng_state"][
        "generators"
    ]["time"].to(torch.int64)
    torch.save(checkpoint, path)

    restored = torch.nn.Linear(3, 2)
    loaded = load_checkpoint(path, restored, restore_rng=True, map_location="cpu")
    cpu_state = loaded["rng_state"]["torch_cpu"]
    generator_state = loaded["rng_state"]["generators"]["time"]
    assert cpu_state.device.type == "cpu"
    assert generator_state.device.type == "cpu"
    assert cpu_state.dtype == torch.uint8
    assert generator_state.dtype == torch.uint8
    torch.Generator().set_state(generator_state)


def test_original_checkpoint_key_migration():
    original = {
        "module.time_embed.0.weight": torch.randn(4, 1),
        "module.gnn.node_proj.weight": torch.randn(4, 10),
        "module.gnn.seg_decoder.0.net.0.weight": torch.randn(4, 4, 3, 3),
        "module.dec.0.net.0.bias": torch.randn(4),
    }
    migrated = migrate_original_sage_avo_state_dict(original)
    assert set(migrated) == {
        "time_embedding.0.weight",
        "graph.node_projection.weight",
        "graph.segmentation.0.network.0.weight",
        "decoder.0.network.0.bias",
    }
