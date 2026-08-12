"""SAGE-AVO deterministic conditional residual-transport network."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from .graph import build_rgt_edges


class ModelOutput(NamedTuple):
    velocity: Tensor
    segmentation_logits: Tensor
    embeddings: Tensor
    edge_indices: list[Tensor]
    edge_weights: list[Tensor]


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


def angular_features(avo: Tensor, angles: tuple[float, float, float] = (10.0, 24.0, 38.0)) -> tuple[Tensor, Tensor]:
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
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = hidden_channels
        self.max_shift = max_shift
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
            nn.Conv2d(hidden_channels, classes, 1),
        )

    def forward(self, tokens: Tensor, avo: Tensor, rgt: Tensor) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
        batch, nodes, _ = tokens.shape
        height, width = avo.shape[-2:]
        features, gradient = angular_features(avo)
        angle_tokens = features.flatten(2).transpose(1, 2)
        projected = self.node_projection(torch.cat((tokens, angle_tokens), dim=-1))
        edges = build_rgt_edges(rgt, self.max_shift, steered=self.graph_mode == "rgt")
        embeddings: list[Tensor] = []
        weights: list[Tensor] = []
        for item in range(batch):
            node_features = projected[item]
            edge_index = edges[item]
            flattened_gradient = gradient[item].reshape(nodes)
            contrast = torch.abs(flattened_gradient[edge_index[0]] - flattened_gradient[edge_index[1]])
            edge_weight = torch.exp(-contrast / (contrast.std() + 1e-6))
            edge_attribute = edge_weight.unsqueeze(-1)
            for layer, normalization in zip(self.layers, self.normalizations):
                node_features = F.gelu(normalization(layer(node_features, edge_index, edge_attr=edge_attribute)))
            embeddings.append(node_features)
            weights.append(edge_weight)
        stacked = torch.stack(embeddings)
        spatial = stacked.reshape(batch, height, width, self.hidden_channels).permute(0, 3, 1, 2)
        return stacked, self.segmentation(spatial), edges, weights


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
    ) -> None:
        super().__init__()
        if graph_mode not in {"rgt", "cartesian", "none"}:
            raise ValueError("graph_mode must be 'rgt', 'cartesian', or 'none'")
        self.graph_mode = graph_mode
        hidden = hidden_channels
        self.time_embedding = nn.Sequential(nn.Linear(1, hidden), nn.GELU())
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

    def forward(self, state: Tensor, time: Tensor, avo: Tensor, low: Tensor, rgt: Tensor) -> ModelOutput:
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
            velocity = self.decoder(cnn)
        else:
            tokens = cnn.flatten(2).transpose(1, 2)
            embeddings, segmentation, edges, weights = self.graph(tokens, avo, rgt)
            graph_spatial = embeddings.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
            velocity = self.decoder(cnn + graph_spatial)
        return ModelOutput(velocity, segmentation, embeddings, edges, weights)

    @torch.no_grad()
    def sample(self, avo: Tensor, low: Tensor, rgt: Tensor, steps: int = 20) -> Tensor:
        """Integrate deterministic residual transport with Heun's method."""
        if steps < 1:
            raise ValueError("steps must be positive")
        state = low.clone()
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full((state.shape[0],), index * step_size, device=state.device)
            next_time = torch.full((state.shape[0],), (index + 1) * step_size, device=state.device)
            first = self(state, time, avo, low, rgt).velocity
            provisional = state + step_size * first
            second = self(provisional, next_time, avo, low, rgt).velocity
            state = state + 0.5 * step_size * (first + second)
        return state
