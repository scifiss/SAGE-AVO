"""SAGE-AVO deterministic conditional residual-transport network."""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax

from sage_avo.forward.torch_forward import (
    CURRENT_ANGLE_BANDS,
    forward_avo_three_band_torch,
)
from sage_avo.training.flow import heun_integrate

from .graph import (
    RGT_V1_LEGACY,
    build_horizon_candidate_edges,
    build_horizon_edges,
    build_normal_candidate_edges,
    build_normal_edges,
    build_experimental_rgt_relations,
)


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


class StructuralPriorTransformerConv(TransformerConv):
    """TransformerConv with an explicit per-edge additive attention bias."""

    _structural_attention_bias: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        attention_bias: Tensor | None = None,
        return_attention_weights: bool | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if attention_bias is not None and (
            attention_bias.ndim != 1 or attention_bias.shape[0] != edge_index.shape[1]
        ):
            raise ValueError("attention_bias must contain one scalar per edge")
        self._structural_attention_bias = attention_bias
        try:
            return super().forward(
                x,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=return_attention_weights,
            )
        finally:
            self._structural_attention_bias = None

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: Tensor | None,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        transformed_edge = None
        if self.lin_edge is not None:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_dim is configured")
            transformed_edge = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + transformed_edge
        alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5
        if self._structural_attention_bias is not None:
            alpha = alpha + self._structural_attention_bias[:, None]
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j if transformed_edge is None else value_j + transformed_edge
        return out * alpha.view(-1, self.heads, 1)


class InstrumentedTransformerConv(TransformerConv):
    """Behavior-identical TransformerConv with opt-in per-edge message scaling."""

    diagnostic_message_scale: Tensor | float | None = None
    attention_mode: str = "learned"

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: Tensor | None,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        transformed_edge = None
        if self.lin_edge is not None:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_dim is configured")
            transformed_edge = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + transformed_edge
        alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5
        if self.attention_mode == "learned":
            alpha = softmax(alpha, index, ptr, size_i)
        elif self.attention_mode == "uniform":
            # Preserve the value/edge-value/root paths. Query/key parameters
            # remain instantiated but do not receive gradients through alpha.
            degree = torch.bincount(index, minlength=size_i or 0).to(alpha.dtype)
            alpha = torch.ones_like(alpha) / degree[index, None]
        else:
            raise ValueError(f"unsupported attention_mode {self.attention_mode!r}")
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j if transformed_edge is None else value_j + transformed_edge
        out = out * alpha.view(-1, self.heads, 1)
        if self.diagnostic_message_scale is not None:
            scale = torch.as_tensor(
                self.diagnostic_message_scale, dtype=out.dtype, device=out.device
            )
            if scale.ndim == 1:
                scale = scale[:, None, None]
            out = out * scale
        return out


def relation_preserving_edge_attr_shuffle(
    attributes: Tensor, relation_edge_counts: Sequence[int]
) -> Tensor:
    """Permute connections within relations, carrying both directions together.

    Each relation is ordered as forward connections followed by their matching
    reverse connections, as returned by the frozen topology builders.
    """
    if sum(relation_edge_counts) != attributes.shape[0]:
        raise ValueError("relation counts must cover all edge attributes")
    chunks = []
    offset = 0
    for count in relation_edge_counts:
        if count < 0 or count % 2:
            raise ValueError("each relation must contain paired directed edges")
        half = count // 2
        block = attributes[offset : offset + count]
        order = torch.roll(torch.arange(half, device=attributes.device), half // 3 + 1)
        chunks.extend((block[:half][order], block[half:][order]))
        offset += count
    return torch.cat(chunks, dim=0) if chunks else attributes.clone()


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
        rgt_topology: str | None = None,
        graph_neighbor_scale: float = 1.0,
        attention_mode: str = "learned",
        segmentation_detach_graph: bool = False,
        confidence_normalized_mismatch_threshold: float | None = None,
        confidence_normalized_discontinuity_threshold: float | None = None,
        confidence_dip_residual_threshold: float | None = None,
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
        self.rgt_topology = rgt_topology or (
            RGT_V1_LEGACY if graph_mode == "rgt" else "cartesian"
        )
        self.graph_neighbor_scale = float(graph_neighbor_scale)
        if attention_mode not in {"learned", "uniform"}:
            raise ValueError("attention_mode must be learned or uniform")
        self.attention_mode = attention_mode
        self.segmentation_detach_graph = bool(segmentation_detach_graph)
        self.diagnostic_root_scale = 1.0
        self.diagnostic_neighbor_scale = 1.0
        self.diagnostic_edge_attr_mode = "current"
        self.diagnostic_capture_contributions = False
        self.diagnostic_tangential_message_scale: Tensor | float = 1.0
        self.diagnostic_normal_message_scale: Tensor | float = 1.0
        self.last_contribution_diagnostics: list[dict[str, float]] = []
        self.confidence_normalized_mismatch_threshold = (
            confidence_normalized_mismatch_threshold
        )
        self.confidence_normalized_discontinuity_threshold = (
            confidence_normalized_discontinuity_threshold
        )
        self.confidence_dip_residual_threshold = confidence_dip_residual_threshold
        self.node_projection = nn.Linear(input_channels + 6, hidden_channels)
        self.layers = nn.ModuleList(
            InstrumentedTransformerConv(
                hidden_channels, hidden_channels // heads, heads=heads, edge_dim=1
            )
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
        tangential_edges, normal_edges = build_experimental_rgt_relations(
            rgt,
            topology=self.rgt_topology,
            max_shift=self.max_shift,
            confidence_normalized_mismatch_threshold=(
                self.confidence_normalized_mismatch_threshold
            ),
            confidence_normalized_discontinuity_threshold=(
                self.confidence_normalized_discontinuity_threshold
            ),
            confidence_dip_residual_threshold=self.confidence_dip_residual_threshold,
        )
        edges = [
            torch.cat((tangential, normal), dim=1)
            for tangential, normal in zip(tangential_edges, normal_edges)
        ]
        embeddings: list[Tensor] = []
        weights: list[Tensor] = []
        attention_edges: list[Tensor] = []
        attention_weights: list[Tensor] = []
        for item in range(batch):
            node_features = projected[item]
            edge_index = edges[item]
            flattened_gradient = gradient[item].reshape(nodes)
            contrast = torch.abs(
                flattened_gradient[edge_index[0]] - flattened_gradient[edge_index[1]]
            )
            edge_weight = torch.exp(-contrast / (contrast.std(unbiased=False) + 1e-6))
            edge_attribute = edge_weight.unsqueeze(-1)
            if self.diagnostic_edge_attr_mode == "zero":
                edge_attribute = torch.zeros_like(edge_attribute)
            elif self.diagnostic_edge_attr_mode == "shuffled":
                edge_attribute = edge_attribute.roll(edge_attribute.shape[0] // 3 + 1, dims=0)
            elif self.diagnostic_edge_attr_mode == "relation_preserving_shuffled":
                edge_attribute = relation_preserving_edge_attr_shuffle(
                    edge_attribute,
                    (tangential_edges[item].shape[1], normal_edges[item].shape[1]),
                )
            elif self.diagnostic_edge_attr_mode != "current":
                raise ValueError(
                    f"unsupported diagnostic edge_attr mode {self.diagnostic_edge_attr_mode!r}"
                )
            final_attention_edges = edge_index
            final_attention = edge_weight
            item_diagnostics: list[dict[str, float]] = []
            for layer_index, (layer, normalization) in enumerate(
                zip(self.layers, self.normalizations), start=1
            ):
                layer.attention_mode = self.attention_mode
                tangential_scale = self.diagnostic_tangential_message_scale
                normal_scale = self.diagnostic_normal_message_scale
                if (
                    not isinstance(tangential_scale, float)
                    or not isinstance(normal_scale, float)
                    or tangential_scale != 1.0
                    or normal_scale != 1.0
                ):
                    layer.diagnostic_message_scale = torch.cat(
                        (
                            torch.ones(
                                tangential_edges[item].shape[1],
                                dtype=node_features.dtype,
                                device=node_features.device,
                            )
                            * tangential_scale,
                            torch.ones(
                                normal_edges[item].shape[1],
                                dtype=node_features.dtype,
                                device=node_features.device,
                            )
                            * normal_scale,
                        )
                    )
                else:
                    layer.diagnostic_message_scale = None
                result, (layer_edges, alpha) = layer(
                    node_features,
                    edge_index,
                    edge_attr=edge_attribute,
                    return_attention_weights=True,
                )
                controlled = (
                    self.diagnostic_root_scale != 1.0
                    or self.graph_neighbor_scale * self.diagnostic_neighbor_scale != 1.0
                )
                if controlled or self.diagnostic_capture_contributions:
                    root = layer.lin_skip(node_features)
                    neighbor = result - root
                    if self.diagnostic_capture_contributions:
                        root_norm = root.square().mean().sqrt()
                        neighbor_norm = neighbor.square().mean().sqrt()
                        cosine = F.cosine_similarity(
                            root.flatten(), neighbor.flatten(), dim=0, eps=1e-12
                        )
                        item_diagnostics.append(
                            {
                                "layer": float(layer_index),
                                "root_rms": float(root_norm.detach().cpu()),
                                "neighbor_rms": float(neighbor_norm.detach().cpu()),
                                "neighbor_root_ratio": float(
                                    (neighbor_norm / root_norm.clamp_min(1e-12)).detach().cpu()
                                ),
                                "root_neighbor_cosine": float(cosine.detach().cpu()),
                                "tangential_directed_edge_count": float(
                                    tangential_edges[item].shape[1]
                                ),
                                "normal_directed_edge_count": float(
                                    normal_edges[item].shape[1]
                                ),
                            }
                        )
                    if controlled:
                        result = (
                            self.diagnostic_root_scale * root
                            + self.graph_neighbor_scale
                            * self.diagnostic_neighbor_scale
                            * neighbor
                        )
                node_features = F.gelu(normalization(result))
                final_attention_edges = layer_edges
                final_attention = alpha.mean(dim=-1)
                layer.diagnostic_message_scale = None
            embeddings.append(node_features)
            weights.append(edge_weight)
            attention_edges.append(final_attention_edges)
            attention_weights.append(final_attention)
            self.last_contribution_diagnostics = item_diagnostics
        stacked = torch.stack(embeddings)
        spatial = stacked.reshape(batch, height, width, self.hidden_channels).permute(0, 3, 1, 2)
        return (
            stacked,
            self.segmentation(spatial.detach() if self.segmentation_detach_graph else spatial),
            edges,
            weights,
            attention_edges,
            attention_weights,
        )


class RelationalStratigraphicGraphEncoder(nn.Module):
    """Experimental dual-relation RGT encoder for v00332h diagnostics.

    Tangential and normal messages have independent TransformerConv weights.
    A learned per-node softmax gate fuses the local, tangential, and normal
    streams. This class is opt-in and does not alter the production encoder.
    """

    relation_names = ("tangential", "normal")

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        classes: int = 3,
        layers: int = 2,
        heads: int = 4,
        max_shift: int = 3,
        normal_lateral_shift: int = 1,
        relation_candidates: int = 1,
        structural_prior_initial_strengths: tuple[tuple[float, float], tuple[float, float]] = (
            (0.5, 0.5),
            (0.5, 0.5),
        ),
        elastic_structural_prior_initial_strengths: tuple[tuple[float, float], tuple[float, float]]
        | None = None,
        elastic_structural_prior_component_mask: tuple[
            tuple[float, float], tuple[float, float]
        ] = ((1.0, 1.0), (1.0, 1.0)),
        graph_mode: str = "relational_rgt",
        representative_angles: tuple[float, float, float] = (10.0, 24.0, 38.0),
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        if graph_mode not in {
            "relational_rgt",
            "relational_candidate_rgt",
            "relational_candidate_cartesian",
            "relational_structural_prior_rgt",
            "relational_structural_prior_cartesian",
            "relational_task_decoupled_rgt",
            "relational_task_decoupled_cartesian",
            "relational_task_specific_rgt",
            "relational_task_specific_cartesian",
            "cartesian",
        }:
            raise ValueError("unsupported relational graph_mode")
        if relation_candidates < 1:
            raise ValueError("relation_candidates must be positive")
        self.hidden_channels = hidden_channels
        self.max_shift = max_shift
        self.normal_lateral_shift = normal_lateral_shift
        self.relation_candidates = relation_candidates
        self.representative_angles = representative_angles
        self.graph_mode = graph_mode
        self.node_projection = nn.Linear(input_channels + 6, hidden_channels)
        layer_type = (
            StructuralPriorTransformerConv
            if graph_mode.startswith(
                (
                    "relational_structural_prior",
                    "relational_task_decoupled",
                    "relational_task_specific",
                )
            )
            else TransformerConv
        )
        self.tangential_layers = nn.ModuleList(
            layer_type(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=4)
            for _ in range(layers)
        )
        self.normal_layers = nn.ModuleList(
            layer_type(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=4)
            for _ in range(layers)
        )
        self.tangential_normalizations = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(layers)
        )
        self.normal_normalizations = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(layers)
        )
        self.relation_gate = nn.Linear(3 * hidden_channels, 3)
        if graph_mode.startswith(
            (
                "relational_structural_prior",
                "relational_task_decoupled",
                "relational_task_specific",
            )
        ):
            strengths = torch.as_tensor(structural_prior_initial_strengths, dtype=torch.float32)
            if strengths.shape != (2, 2) or torch.any(strengths <= 0):
                raise ValueError(
                    "structural_prior_initial_strengths must be positive with shape [2,2]"
                )
            self.structural_attention_raw_strengths = nn.Parameter(
                torch.log(torch.expm1(strengths))
            )
        else:
            self.register_parameter("structural_attention_raw_strengths", None)
        if graph_mode.startswith("relational_task_specific"):
            elastic_strengths = torch.as_tensor(
                elastic_structural_prior_initial_strengths
                if elastic_structural_prior_initial_strengths is not None
                else structural_prior_initial_strengths,
                dtype=torch.float32,
            )
            if elastic_strengths.shape != (2, 2) or torch.any(elastic_strengths <= 0):
                raise ValueError(
                    "elastic_structural_prior_initial_strengths must be positive with shape [2,2]"
                )
            self.elastic_attention_raw_strengths = nn.Parameter(
                torch.log(torch.expm1(elastic_strengths))
            )
        else:
            self.register_parameter("elastic_attention_raw_strengths", None)
        component_mask = torch.as_tensor(
            elastic_structural_prior_component_mask, dtype=torch.float32
        )
        if component_mask.shape != (2, 2) or torch.any(
            (component_mask != 0.0) & (component_mask != 1.0)
        ):
            raise ValueError(
                "elastic_structural_prior_component_mask must be binary with shape [2,2]"
            )
        self.register_buffer(
            "elastic_structural_prior_component_mask",
            component_mask,
            persistent=False,
        )
        self.segmentation = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8 if (hidden_channels // 2) % 8 == 0 else 1, hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, classes, 1),
        )

    @staticmethod
    def _edge_attributes(
        edge_index: Tensor,
        gradient: Tensor,
        rgt: Tensor,
        width: int,
        spatial_scale: int,
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        gradient_flat = gradient.reshape(-1)
        rgt_flat = rgt.reshape(-1)
        gradient_contrast = torch.abs(gradient_flat[source] - gradient_flat[destination])
        gradient_scale = gradient_contrast.std(unbiased=False) + 1e-6
        gradient_affinity = torch.exp(-gradient_contrast / gradient_scale)
        rgt_contrast = torch.abs(rgt_flat[source] - rgt_flat[destination])
        rgt_scale = rgt_contrast.std(unbiased=False) + rgt_contrast.mean() + 1e-6
        rgt_affinity = torch.exp(-rgt_contrast / rgt_scale)
        row_offset = (destination // width - source // width).to(gradient.dtype)
        column_offset = (destination % width - source % width).to(gradient.dtype)
        offset_scale = float(max(spatial_scale, 1))
        attributes = torch.stack(
            (
                gradient_affinity,
                rgt_affinity,
                row_offset / offset_scale,
                column_offset / offset_scale,
            ),
            dim=-1,
        )
        return attributes, gradient_affinity

    @staticmethod
    def _apply_relation(
        projected: Tensor,
        edge_index: Tensor,
        edge_attribute: Tensor,
        layers: nn.ModuleList,
        normalizations: nn.ModuleList,
        attention_bias: Tensor | None = None,
        structural_strengths: Tensor | None = None,
        stream: str = "shared",
    ) -> tuple[Tensor, list[dict[str, Tensor | str | None]]]:
        node_features = projected
        details: list[dict[str, Tensor | str | None]] = []
        for layer_index, (layer, normalization) in enumerate(zip(layers, normalizations), start=1):
            if isinstance(layer, StructuralPriorTransformerConv):
                result, (layer_edges, alpha) = layer(
                    node_features,
                    edge_index,
                    edge_attr=edge_attribute,
                    attention_bias=attention_bias,
                    return_attention_weights=True,
                )
            else:
                result, (layer_edges, alpha) = layer(
                    node_features,
                    edge_index,
                    edge_attr=edge_attribute,
                    return_attention_weights=True,
                )
            node_features = F.gelu(normalization(result))
            details.append(
                {
                    "stream": stream,
                    "layer": torch.tensor(layer_index, device=projected.device),
                    "edge_index": layer_edges,
                    "attention": alpha.mean(dim=-1),
                    "attention_prior": attention_bias,
                    "structural_strengths": structural_strengths,
                }
            )
        return node_features, details

    def _forward_impl(
        self,
        tokens: Tensor,
        avo: Tensor,
        rgt: Tensor,
    ) -> tuple[
        tuple[Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
        list[list[dict[str, Tensor | str]]],
        dict[str, Tensor],
    ]:
        batch, _, _ = tokens.shape
        height, width = avo.shape[-2:]
        features, gradient = angular_features(avo, self.representative_angles)
        projected = self.node_projection(
            torch.cat((tokens, features.flatten(2).transpose(1, 2)), dim=-1)
        )
        steered = self.graph_mode in {
            "relational_rgt",
            "relational_candidate_rgt",
            "relational_structural_prior_rgt",
            "relational_task_decoupled_rgt",
            "relational_task_specific_rgt",
        }
        candidate_mode = self.graph_mode in {
            "relational_candidate_rgt",
            "relational_candidate_cartesian",
            "relational_structural_prior_rgt",
            "relational_structural_prior_cartesian",
            "relational_task_decoupled_rgt",
            "relational_task_decoupled_cartesian",
            "relational_task_specific_rgt",
            "relational_task_specific_cartesian",
        }
        task_decoupled = self.graph_mode.startswith(
            ("relational_task_decoupled", "relational_task_specific")
        )
        task_specific = self.graph_mode.startswith("relational_task_specific")
        if candidate_mode:
            tangential_edges = build_horizon_candidate_edges(
                rgt,
                self.max_shift,
                candidates=self.relation_candidates,
                steered=steered,
            )
            normal_edges = build_normal_candidate_edges(
                rgt,
                self.normal_lateral_shift,
                candidates=self.relation_candidates,
                steered=steered,
            )
        else:
            tangential_edges = build_horizon_edges(rgt, self.max_shift, steered=steered)
            normal_edges = build_normal_edges(
                rgt,
                self.normal_lateral_shift if steered else 0,
                steered=steered,
            )
        embeddings: list[Tensor] = []
        segmentation_embeddings: list[Tensor] = []
        combined_edges: list[Tensor] = []
        combined_weights: list[Tensor] = []
        combined_attention_edges: list[Tensor] = []
        combined_attention_weights: list[Tensor] = []
        detail_batches: list[list[dict[str, Tensor | str]]] = []
        gates: dict[str, list[Tensor]] = (
            {
                "segmentation": [],
                "elastic": [],
            }
            if task_decoupled
            else {"shared": []}
        )
        for item in range(batch):
            relation_outputs: list[Tensor] = []
            segmentation_relation_outputs: list[Tensor] = []
            relation_details: list[dict[str, Tensor | str]] = []
            relation_weights: list[Tensor] = []
            for relation_index, (relation, edge_index, layers, normalizations) in enumerate(
                (
                    (
                        "tangential",
                        tangential_edges[item],
                        self.tangential_layers,
                        self.tangential_normalizations,
                    ),
                    (
                        "normal",
                        normal_edges[item],
                        self.normal_layers,
                        self.normal_normalizations,
                    ),
                )
            ):
                edge_attribute, edge_weight = self._edge_attributes(
                    edge_index,
                    gradient[item],
                    rgt[item],
                    width,
                    max(self.max_shift, self.normal_lateral_shift),
                )
                structural_strengths = (
                    F.softplus(self.structural_attention_raw_strengths[relation_index])
                    if self.structural_attention_raw_strengths is not None
                    else None
                )
                attention_bias = None
                elastic_attention_bias = None
                elastic_strengths = None
                if structural_strengths is not None and steered:
                    avo_contrast = -torch.log(edge_attribute[:, 0].clamp_min(1e-6))
                    rgt_contrast = -torch.log(edge_attribute[:, 1].clamp_min(1e-6))
                    direction = -1.0 if relation == "tangential" else 1.0
                    attention_bias = direction * (
                        structural_strengths[0] * rgt_contrast
                        + structural_strengths[1] * avo_contrast
                    )
                    if task_specific:
                        elastic_strengths = F.softplus(
                            self.elastic_attention_raw_strengths[relation_index]
                        ) * self.elastic_structural_prior_component_mask[relation_index]
                        elastic_attention_bias = direction * (
                            elastic_strengths[0] * rgt_contrast
                            + elastic_strengths[1] * avo_contrast
                        )
                if task_decoupled:
                    segmentation_output, segmentation_details = self._apply_relation(
                        projected[item],
                        edge_index,
                        edge_attribute,
                        layers,
                        normalizations,
                        attention_bias=attention_bias,
                        structural_strengths=structural_strengths,
                        stream="segmentation",
                    )
                    elastic_output, elastic_details = self._apply_relation(
                        projected[item],
                        edge_index,
                        edge_attribute,
                        layers,
                        normalizations,
                        attention_bias=elastic_attention_bias,
                        structural_strengths=elastic_strengths,
                        stream="elastic",
                    )
                    segmentation_relation_outputs.append(segmentation_output)
                    relation_outputs.append(elastic_output)
                    layer_details = [*segmentation_details, *elastic_details]
                else:
                    relation_output, layer_details = self._apply_relation(
                        projected[item],
                        edge_index,
                        edge_attribute,
                        layers,
                        normalizations,
                        attention_bias=attention_bias,
                        structural_strengths=structural_strengths,
                    )
                    relation_outputs.append(relation_output)
                relation_weights.append(edge_weight)
                for detail in layer_details:
                    relation_details.append({"relation": relation, **detail})
            elastic_gate = torch.softmax(
                self.relation_gate(
                    torch.cat((projected[item], relation_outputs[0], relation_outputs[1]), dim=-1)
                ),
                dim=-1,
            )
            fused = (
                elastic_gate[:, :1] * projected[item]
                + elastic_gate[:, 1:2] * relation_outputs[0]
                + elastic_gate[:, 2:3] * relation_outputs[1]
            )
            embeddings.append(fused)
            if task_decoupled:
                segmentation_gate = torch.softmax(
                    self.relation_gate(
                        torch.cat(
                            (
                                projected[item],
                                segmentation_relation_outputs[0],
                                segmentation_relation_outputs[1],
                            ),
                            dim=-1,
                        )
                    ),
                    dim=-1,
                )
                segmentation_fused = (
                    segmentation_gate[:, :1] * projected[item]
                    + segmentation_gate[:, 1:2] * segmentation_relation_outputs[0]
                    + segmentation_gate[:, 2:3] * segmentation_relation_outputs[1]
                )
                segmentation_embeddings.append(segmentation_fused)
                gates["segmentation"].append(segmentation_gate)
                gates["elastic"].append(elastic_gate)
            else:
                segmentation_embeddings.append(fused)
                gates["shared"].append(elastic_gate)
            detail_batches.append(relation_details)
            combined_edges.append(torch.cat((tangential_edges[item], normal_edges[item]), dim=1))
            combined_weights.append(torch.cat(relation_weights))
            final_details = [
                detail
                for detail in relation_details
                if int(detail["layer"].item()) == len(self.tangential_layers)
                and detail["stream"] == ("elastic" if task_decoupled else "shared")
            ]
            combined_attention_edges.append(
                torch.cat([detail["edge_index"] for detail in final_details], dim=1)
            )
            combined_attention_weights.append(
                torch.cat([detail["attention"] for detail in final_details])
            )
        stacked = torch.stack(embeddings)
        segmentation_stacked = torch.stack(segmentation_embeddings)
        spatial = segmentation_stacked.reshape(batch, height, width, self.hidden_channels).permute(
            0, 3, 1, 2
        )
        output = (
            stacked,
            self.segmentation(spatial),
            combined_edges,
            combined_weights,
            combined_attention_edges,
            combined_attention_weights,
        )
        return (
            output,
            detail_batches,
            {stream: torch.stack(stream_gates) for stream, stream_gates in gates.items()},
        )

    def forward(
        self, tokens: Tensor, avo: Tensor, rgt: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        output, _, _ = self._forward_impl(tokens, avo, rgt)
        return output

    def diagnostic_forward(
        self, tokens: Tensor, avo: Tensor, rgt: Tensor
    ) -> tuple[
        tuple[Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
        list[list[dict[str, Tensor | str]]],
        dict[str, Tensor],
    ]:
        """Return the normal forward result plus per-relation attention and gates."""
        return self._forward_impl(tokens, avo, rgt)


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
        normal_rgt_lateral_shift: int = 1,
        graph_relation_candidates: int = 1,
        graph_structural_prior_initial_strengths: tuple[
            tuple[float, float], tuple[float, float]
        ] = ((0.5, 0.5), (0.5, 0.5)),
        graph_elastic_structural_prior_initial_strengths: tuple[
            tuple[float, float], tuple[float, float]
        ]
        | None = None,
        graph_elastic_structural_prior_component_mask: tuple[
            tuple[float, float], tuple[float, float]
        ] = ((1.0, 1.0), (1.0, 1.0)),
        graph_mode: str = "rgt",
        rgt_topology: str | None = None,
        graph_neighbor_scale: float = 1.0,
        graph_attention_mode: str = "learned",
        segmentation_detach_graph: bool = False,
        confidence_normalized_mismatch_threshold: float | None = None,
        confidence_normalized_discontinuity_threshold: float | None = None,
        confidence_dip_residual_threshold: float | None = None,
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
        if graph_mode not in {
            "rgt",
            "relational_rgt",
            "relational_candidate_rgt",
            "relational_structural_prior_rgt",
            "relational_task_decoupled_rgt",
            "relational_task_specific_rgt",
            "cartesian",
            "none",
        }:
            raise ValueError("unsupported graph_mode")
        self.graph_mode = graph_mode
        if graph_mode not in {"rgt", "cartesian"} and (
            graph_attention_mode != "learned" or segmentation_detach_graph
        ):
            raise ValueError("v00332r interventions require the single-stream graph encoder")
        self.diagnostic_graph_reinjection_scale = 1.0
        self.diagnostic_capture_fusion = False
        self.last_fusion_diagnostics: dict[str, float] = {}
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
        if graph_mode in {
            "relational_rgt",
            "relational_candidate_rgt",
            "relational_structural_prior_rgt",
            "relational_task_decoupled_rgt",
            "relational_task_specific_rgt",
        }:
            self.graph = RelationalStratigraphicGraphEncoder(
                hidden,
                hidden,
                classes=classes,
                layers=graph_layers,
                heads=graph_heads,
                max_shift=max_rgt_shift,
                normal_lateral_shift=normal_rgt_lateral_shift,
                relation_candidates=graph_relation_candidates,
                structural_prior_initial_strengths=graph_structural_prior_initial_strengths,
                elastic_structural_prior_initial_strengths=(
                    graph_elastic_structural_prior_initial_strengths
                ),
                elastic_structural_prior_component_mask=(
                    graph_elastic_structural_prior_component_mask
                ),
                graph_mode=graph_mode,
                representative_angles=self.representative_angles,
            )
        elif graph_mode != "none":
            self.graph = StratigraphicGraphEncoder(
                hidden,
                hidden,
                classes=classes,
                layers=graph_layers,
                heads=graph_heads,
                max_shift=max_rgt_shift,
                graph_mode=graph_mode,
                rgt_topology=rgt_topology,
                graph_neighbor_scale=graph_neighbor_scale,
                attention_mode=graph_attention_mode,
                segmentation_detach_graph=segmentation_detach_graph,
                confidence_normalized_mismatch_threshold=(
                    confidence_normalized_mismatch_threshold
                ),
                confidence_normalized_discontinuity_threshold=(
                    confidence_normalized_discontinuity_threshold
                ),
                confidence_dip_residual_threshold=confidence_dip_residual_threshold,
                representative_angles=self.representative_angles,
            )
        else:
            self.graph = None
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

    def forward(
        self, state: Tensor, time: Tensor, avo: Tensor, low: Tensor, rgt: Tensor
    ) -> ModelOutput:
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
            if self.diagnostic_capture_fusion:
                cnn_norm = cnn.square().mean().sqrt()
                graph_norm = graph_spatial.square().mean().sqrt()
                cnn_graph_cosine = F.cosine_similarity(
                    cnn.flatten(), graph_spatial.flatten(), dim=0, eps=1e-12
                )
                with torch.no_grad():
                    decoder_delta = (
                        self.decoder(cnn + graph_spatial) - self.decoder(cnn)
                    ).square().mean().sqrt()
                self.last_fusion_diagnostics = {
                    "cnn_rms": float(cnn_norm.detach().cpu()),
                    "graph_spatial_rms": float(graph_norm.detach().cpu()),
                    "graph_cnn_ratio": float(
                        (graph_norm / cnn_norm.clamp_min(1e-12)).detach().cpu()
                    ),
                    "cnn_graph_cosine": float(cnn_graph_cosine.detach().cpu()),
                    "decoder_graph_effect_rms": float(decoder_delta.detach().cpu()),
                }
            velocity = self._parameterize_velocity(
                self.decoder(cnn + self.diagnostic_graph_reinjection_scale * graph_spatial)
            )
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
            gradient = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
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
