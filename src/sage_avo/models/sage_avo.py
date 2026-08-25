"""SAGE-AVO deterministic conditional residual-transport network."""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from sage_avo.forward.torch_forward import (
    CURRENT_ANGLE_BANDS,
    forward_avo_three_band_torch,
)
from sage_avo.training.flow import heun_integrate

from .graph import build_rgt_edges


class ModelOutput(NamedTuple):
    velocity: Tensor
    segmentation_logits: Tensor
    embeddings: Tensor
    edge_indices: list[Tensor]
    edge_weights: list[Tensor]
    attention_edge_indices: list[Tensor]
    attention_weights: list[Tensor]


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def angular_features(
    avo: Tensor,
    angles: tuple[float, float, float] = (10.0, 24.0, 38.0),
) -> tuple[Tensor, Tensor]:
    """Return near/mid/far, Shuey P/G, curvature, and G for edge weights."""
    if avo.shape[1] != 3:
        raise ValueError("avo must contain near/mid/far channels")
    angle_tensor = torch.tensor(angles, device=avo.device, dtype=avo.dtype)
    sin_squared = torch.sin(torch.deg2rad(angle_tensor)).square().view(1, 3, 1, 1)
    mean_x = sin_squared.mean(dim=1, keepdim=True)
    mean_y = avo.mean(dim=1, keepdim=True)
    gradient = ((sin_squared - mean_x) * (avo - mean_y)).sum(dim=1, keepdim=True)
    gradient /= (sin_squared - mean_x).square().sum(dim=1, keepdim=True) + 1e-6
    intercept = mean_y - gradient * mean_x
    curvature = avo[:, :1] - 2.0 * avo[:, 1:2] + avo[:, 2:3]
    return torch.cat((avo, intercept, gradient, curvature), dim=1), gradient


class StratigraphicGraphEncoder(nn.Module):
    """TransformerConv propagation along dynamically constructed RGT edges."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        classes: int = 3,
        layers: int = 2,
        heads: int = 4,
        max_shift: int = 3,
        graph_mode: str = "rgt",
        representative_angles: tuple[float, float, float] = (10.0, 24.0, 38.0),
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = hidden_channels
        self.max_shift = max_shift
        self.representative_angles = representative_angles
        if graph_mode not in {"rgt", "cartesian"}:
            raise ValueError("graph_mode must be 'rgt' or 'cartesian'")
        self.graph_mode = graph_mode
        self.node_projection = nn.Linear(input_channels + 6, hidden_channels)
        self.layers = nn.ModuleList(
            TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1)
            for _ in range(layers)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))
        self.segmentation = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8 if (hidden_channels // 2) % 8 == 0 else 1, hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, classes, 1),
        )

    def forward(
        self,
        tokens: Tensor,
        avo: Tensor,
        rgt: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        batch, nodes, _ = tokens.shape
        height, width = avo.shape[-2:]
        features, gradient = angular_features(avo, self.representative_angles)
        angle_tokens = features.flatten(2).transpose(1, 2)
        projected = self.node_projection(torch.cat((tokens, angle_tokens), dim=-1))
        edges = build_rgt_edges(rgt, self.max_shift, steered=self.graph_mode == "rgt")
        embeddings: list[Tensor] = []
        weights: list[Tensor] = []
        attention_edges: list[Tensor] = []
        attention_weights: list[Tensor] = []
        for item in range(batch):
            node_features = projected[item]
            edge_index = edges[item]
            flattened_gradient = gradient[item].reshape(nodes)
            contrast = torch.abs(flattened_gradient[edge_index[0]] - flattened_gradient[edge_index[1]])
            edge_weight = torch.exp(-contrast / (contrast.std(unbiased=False) + 1e-6))
            edge_attribute = edge_weight.unsqueeze(-1)
            final_attention_edges = edge_index
            final_attention = edge_weight
            for layer, normalization in zip(self.layers, self.normalizations):
                result, (layer_edges, alpha) = layer(
                    node_features,
                    edge_index,
                    edge_attr=edge_attribute,
                    return_attention_weights=True,
                )
                node_features = F.gelu(normalization(result))
                final_attention_edges = layer_edges
                final_attention = alpha.mean(dim=-1)
            embeddings.append(node_features)
            weights.append(edge_weight)
            attention_edges.append(final_attention_edges)
            attention_weights.append(final_attention)
        stacked = torch.stack(embeddings)
        spatial = stacked.reshape(batch, height, width, self.hidden_channels).permute(0, 3, 1, 2)
        return (
            stacked,
            self.segmentation(spatial),
            edges,
            weights,
            attention_edges,
            attention_weights,
        )


class SAGEAVO(nn.Module):
    """Refine normalized low-frequency elastic priors with AVO and RGT.

    The flow follows the deterministic straight path
    ``x_t = (1-t) * low + t * target``. It is not a probabilistic posterior.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        graph_layers: int = 2,
        graph_heads: int = 4,
        max_rgt_shift: int = 3,
        graph_mode: str = "rgt",
        classes: int = 3,
        representative_angles: tuple[float, float, float] = (10.0, 24.0, 38.0),
        physics_angles_degrees: Sequence[float] = tuple(float(value) for value in range(3, 46)),
        physics_bands_degrees: tuple[tuple[float, float], ...] = CURRENT_ANGLE_BANDS,
        physics_wavelet_hz: float = 14.0,
        physics_dt_seconds: float = 0.004,
        physics_wavelet_samples: int = 81,
        physics_apply_mute: bool = True,
        physics_mute_start: tuple[float, float] = (30.0, 0.0),
        physics_mute_end: tuple[float, float] = (45.0, 0.1),
        physics_taper_samples: int = 5,
        guidance_start_fraction: float = 1.0 / 3.0,
        guidance_interval_steps: int = 3,
        residual_trust_region_scales: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if graph_mode not in {"rgt", "cartesian", "none"}:
            raise ValueError("graph_mode must be 'rgt', 'cartesian', or 'none'")
        self.graph_mode = graph_mode
        if len(representative_angles) != 3:
            raise ValueError("representative_angles must contain near/mid/far values")
        if not 0.0 <= guidance_start_fraction <= 1.0:
            raise ValueError("guidance_start_fraction must lie in [0, 1]")
        if guidance_interval_steps < 1:
            raise ValueError("guidance_interval_steps must be positive")
        self.representative_angles = tuple(float(value) for value in representative_angles)
        self.physics_angles_degrees = tuple(float(value) for value in physics_angles_degrees)
        self.physics_bands_degrees = tuple(
            (float(minimum), float(maximum)) for minimum, maximum in physics_bands_degrees
        )
        self.physics_wavelet_hz = float(physics_wavelet_hz)
        self.physics_dt_seconds = float(physics_dt_seconds)
        self.physics_wavelet_samples = int(physics_wavelet_samples)
        self.physics_apply_mute = bool(physics_apply_mute)
        self.physics_mute_start = tuple(float(value) for value in physics_mute_start)
        self.physics_mute_end = tuple(float(value) for value in physics_mute_end)
        self.physics_taper_samples = int(physics_taper_samples)
        self.guidance_start_fraction = float(guidance_start_fraction)
        self.guidance_interval_steps = int(guidance_interval_steps)
        if residual_trust_region_scales is None:
            trust_scales = torch.ones(1, 3, 1, 1)
            trust_region_enabled = False
        else:
            trust_scales = torch.as_tensor(
                residual_trust_region_scales, dtype=torch.float32
            ).reshape(1, 3, 1, 1)
            if torch.any(~torch.isfinite(trust_scales)) or torch.any(trust_scales <= 0):
                raise ValueError("Residual trust-region scales must be finite and positive")
            trust_region_enabled = True
        self.register_buffer(
            "residual_trust_region_scales",
            trust_scales,
            persistent=False,
        )
        self.register_buffer(
            "residual_trust_region_enabled",
            torch.tensor(trust_region_enabled),
            persistent=False,
        )
        hidden = hidden_channels
        self.time_embedding = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.condition_embedding = nn.Conv2d(6, hidden, 1)
        self.encoder = nn.Sequential(ConvBlock(3 + 2 * hidden, hidden), ConvBlock(hidden, hidden))
        self.graph = (
            StratigraphicGraphEncoder(
                hidden,
                hidden,
                classes=classes,
                layers=graph_layers,
                heads=graph_heads,
                max_shift=max_rgt_shift,
                graph_mode=graph_mode,
                representative_angles=self.representative_angles,
            )
            if graph_mode != "none"
            else None
        )
        self.local_segmentation = (
            nn.Sequential(ConvBlock(hidden, hidden), nn.Conv2d(hidden, classes, 1))
            if graph_mode == "none"
            else None
        )
        self.decoder = nn.Sequential(ConvBlock(hidden, hidden), nn.Conv2d(hidden, 3, 1))
        # Preserve legacy state-dict keys so versioned checkpoints remain loadable.
        self.register_buffer("X_mean_buf", torch.zeros(1, 3, 1, 1))
        self.register_buffer("X_std_buf", torch.ones(1, 3, 1, 1))
        self.register_buffer("Y_mean_buf", torch.zeros(1, 3, 1, 1))
        self.register_buffer("Y_std_buf", torch.ones(1, 3, 1, 1))
        self.register_buffer("normalization_ready", torch.tensor(False), persistent=False)

    @staticmethod
    def _normalization_value(
        statistics: Mapping[str, Sequence[float]],
        lower_name: str,
        upper_name: str,
    ) -> Tensor:
        values = statistics.get(lower_name, statistics.get(upper_name))
        if values is None:
            raise KeyError(f"Normalization statistics require {lower_name!r}")
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.numel() < 3:
            raise ValueError(f"{lower_name} must contain at least three channels")
        return tensor[:3].reshape(1, 3, 1, 1)

    def set_norm_stats(self, statistics: Mapping[str, Sequence[float]]) -> None:
        """Install train-only statistics used by differentiable physics guidance."""
        values = {
            "X_mean_buf": self._normalization_value(statistics, "x_mean", "X_mean"),
            "X_std_buf": self._normalization_value(statistics, "x_std", "X_std"),
            "Y_mean_buf": self._normalization_value(statistics, "y_mean", "Y_mean"),
            "Y_std_buf": self._normalization_value(statistics, "y_std", "Y_std"),
        }
        if torch.any(values["X_std_buf"] <= 0) or torch.any(values["Y_std_buf"] <= 0):
            raise ValueError("Normalization standard deviations must be positive")
        for name, value in values.items():
            getattr(self, name).copy_(value.to(device=getattr(self, name).device))
        self.normalization_ready.fill_(True)

    def forward(self, state: Tensor, time: Tensor, avo: Tensor, low: Tensor, rgt: Tensor) -> ModelOutput:
        if rgt.ndim == 4 and rgt.shape[1] == 1:
            rgt = rgt[:, 0]
        if rgt.ndim != 3:
            raise ValueError("rgt must have shape [B,H,W] or [B,1,H,W]")
        batch, _, height, width = state.shape
        time_features = self.time_embedding(time[:, None]).unsqueeze(-1).unsqueeze(-1)
        time_features = time_features.expand(-1, -1, height, width)
        condition = self.condition_embedding(torch.cat((avo, low), dim=1))
        cnn = self.encoder(torch.cat((state, time_features, condition), dim=1))
        if self.graph is None:
            embeddings = cnn.flatten(2).transpose(1, 2)
            segmentation = self.local_segmentation(cnn)
            edges: list[Tensor] = []
            weights: list[Tensor] = []
            attention_edges: list[Tensor] = []
            attention_weights: list[Tensor] = []
            velocity = self._parameterize_velocity(self.decoder(cnn))
        else:
            tokens = cnn.flatten(2).transpose(1, 2)
            (
                embeddings,
                segmentation,
                edges,
                weights,
                attention_edges,
                attention_weights,
            ) = self.graph(tokens, avo, rgt)
            graph_spatial = embeddings.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
            velocity = self._parameterize_velocity(self.decoder(cnn + graph_spatial))
        return ModelOutput(
            velocity,
            segmentation,
            embeddings,
            edges,
            weights,
            attention_edges,
            attention_weights,
        )

    def _parameterize_velocity(self, raw_velocity: Tensor) -> Tensor:
        """Apply the optional training-derived smooth residual trust region."""
        if not bool(self.residual_trust_region_enabled.item()):
            return raw_velocity
        scales = self.residual_trust_region_scales.to(
            device=raw_velocity.device, dtype=raw_velocity.dtype
        )
        return scales * torch.tanh(raw_velocity / scales)

    def _physics_guided_correction(
        self,
        state: Tensor,
        avo: Tensor,
        *,
        scale: float,
        avo_mask: Tensor | None = None,
    ) -> Tensor:
        if not bool(self.normalization_ready.item()):
            raise RuntimeError("Call set_norm_stats() before enabling physics-guided sampling")
        with torch.enable_grad():
            differentiable_state = state.detach().clone().requires_grad_(True)
            physical = differentiable_state * self.Y_std_buf + self.Y_mean_buf
            angles = torch.as_tensor(
                self.physics_angles_degrees,
                device=state.device,
                dtype=state.dtype,
            )
            modeled = forward_avo_three_band_torch(
                physical[:, 0],
                physical[:, 1],
                physical[:, 2],
                angles_degrees=angles,
                wavelet_hz=self.physics_wavelet_hz,
                dt_seconds=self.physics_dt_seconds,
                wavelet_samples=self.physics_wavelet_samples,
                bands_degrees=self.physics_bands_degrees,
                apply_mute=self.physics_apply_mute,
                mute_start=self.physics_mute_start,
                mute_end=self.physics_mute_end,
                taper_samples=self.physics_taper_samples,
            )
            normalized_modeled = (modeled - self.X_mean_buf) / self.X_std_buf
            difference = (normalized_modeled - avo[:, :3]).square()
            if avo_mask is not None:
                mask = avo_mask
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask = mask.expand_as(difference).to(difference.dtype)
                physics_mismatch = (difference * mask).sum() / (mask.sum() + 1e-8)
            else:
                physics_mismatch = difference.mean()
            gradient = torch.autograd.grad(physics_mismatch, differentiable_state)[0]
            gradient = gradient / (
                gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
        return (state - float(scale) * gradient).detach()

    def sample(
        self,
        avo: Tensor,
        low: Tensor,
        rgt: Tensor,
        steps: int = 20,
        guidance_scale: float = 0.0,
        avo_mask: Tensor | None = None,
    ) -> Tensor:
        """Integrate deterministic residual transport with optional physics guidance."""

        def velocity(state: Tensor, time: Tensor) -> Tensor:
            with torch.no_grad():
                return self(state, time, avo, low, rgt).velocity

        correction = None
        if guidance_scale > 0.0:
            guidance_start = int(steps * self.guidance_start_fraction)

            def correction(state: Tensor, index: int) -> Tensor:
                active = index >= guidance_start and (index + 1) % self.guidance_interval_steps == 0
                if not active:
                    return state
                return self._physics_guided_correction(
                    state,
                    avo,
                    scale=guidance_scale,
                    avo_mask=avo_mask,
                )

        return heun_integrate(low.clone(), velocity, steps=steps, correction=correction)
