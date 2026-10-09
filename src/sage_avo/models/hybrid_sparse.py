"""Optional research-only sparse reflector messages beside unchanged dense SAGE-AVO.

Topology is the frozen v00332z accepted graph. Only observable node positions
and accepted within-component edges enter this module; truth masks are QC only.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from sage_avo.training.flow import heun_integrate

from .sage_avo import ModelOutput, SAGEAVO


@dataclass(frozen=True)
class SparsePatchGraph:
    """One crop's physical-node graph in resized model-patch coordinates."""

    coordinates: Tensor  # [N,2] in (time,row; trace,column) order
    components: Tensor  # [N], frozen accepted-component IDs
    edge_index: Tensor  # [2,2E], both directions
    edge_attr: Tensor  # [2E,9], observable descriptors only
    shape: tuple[int, int]

    def to(self, device: torch.device | str) -> SparsePatchGraph:
        return SparsePatchGraph(
            self.coordinates.to(device),
            self.components.to(device),
            self.edge_index.to(device),
            self.edge_attr.to(device),
            self.shape,
        )


def flip_sparse_graph_horizontal(graph: SparsePatchGraph) -> SparsePatchGraph:
    """Mirror topology with the existing horizontal-flip augmentation draw.

    The caller must use the *same* augmentation decision for AVO/RGT and this
    graph. No detector or tracker rerun is performed on perturbed AVO.
    """
    coords = graph.coordinates.clone()
    coords[:, 1] = graph.shape[1] - 1 - coords[:, 1]
    attributes = graph.edge_attr.clone()
    attributes[:, 2] *= -1  # signed delta-x descriptor only
    return SparsePatchGraph(
        coords, graph.components.clone(), graph.edge_index.clone(), attributes, graph.shape
    )


def crop_accepted_graph(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    *,
    top: int,
    left: int,
    raw_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> SparsePatchGraph:
    """Induce a graph on available CNN input; never keep an outside endpoint.

    scipy.ndimage.zoom's default non-grid mode aligns input/output endpoints,
    hence the (output-1)/(raw-1) coordinate mapping used here.
    """
    raw_h, raw_w = raw_shape
    out_h, out_w = output_shape
    if min(raw_h, raw_w, out_h, out_w) < 2:
        raise ValueError("Graph crop axes require at least two samples")
    selected: dict[int, int] = {}
    coordinates: list[tuple[float, float]] = []
    components: list[int] = []
    for node in nodes:
        time, trace = float(node["time"]), float(node["trace"])
        if not (top <= time <= top + raw_h - 1 and left <= trace <= left + raw_w - 1):
            continue
        selected[int(node["node"])] = len(coordinates)
        coordinates.append(
            (
                (time - top) * (out_h - 1) / (raw_h - 1),
                (trace - left) * (out_w - 1) / (raw_w - 1),
            )
        )
        components.append(int(node["component"]))
    pairs: list[tuple[int, int]] = []
    attrs: list[tuple[float, ...]] = []
    for edge in edges:
        if edge.get("relation") == "FAULT_OFFSET_CORRESPONDENCE":
            continue
        source, target = int(edge["source"]), int(edge["target"])
        if source not in selected or target not in selected:
            continue
        i, j = selected[source], selected[target]
        if components[i] != components[j] or components[i] != int(edge["component"]):
            raise ValueError("Sparse tangential edge crosses frozen reflector components")
        if i == j:
            continue
        base = (
            float(edge["delta_tau"]),
            float(edge["delta_t"]) / 50.0,
            float(edge["delta_x"]) / 100.0,
            float(edge["geodesic_length"]) / 100.0,
            float(edge["waveform_continuity"]),
            float(edge["phase_continuity"]),
            float(edge["shift_continuity"]),
            float(edge["curvature"]),
            float(edge["gap_count"]) / 10.0,
        )
        pairs.extend(((i, j), (j, i)))
        attrs.extend((base, (-base[0], -base[1], -base[2], *base[3:])))
    return SparsePatchGraph(
        torch.tensor(coordinates, dtype=torch.float32).reshape(-1, 2),
        torch.tensor(components, dtype=torch.long),
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        torch.tensor(attrs, dtype=torch.float32).reshape(-1, 9),
        output_shape,
    )


def sample_cnn_nodes(cnn: Tensor, graph: SparsePatchGraph) -> Tensor:
    """Differentiably sample the *current* flow-time CNN features at sparse nodes."""
    if cnn.ndim != 4 or cnn.shape[0] != 1 or cnn.shape[-2:] != graph.shape:
        raise ValueError("Require one CNN feature map matching graph patch shape")
    if graph.coordinates.shape[0] == 0:
        return cnn.new_zeros((0, cnn.shape[1]))
    coords = graph.coordinates.to(device=cnn.device, dtype=cnn.dtype)
    height, width = graph.shape
    grid = torch.stack(
        (2.0 * coords[:, 1] / (width - 1) - 1.0, 2.0 * coords[:, 0] / (height - 1) - 1.0),
        dim=-1,
    ).reshape(1, -1, 1, 2)
    return F.grid_sample(cnn, grid, mode="bilinear", align_corners=True)[0, :, :, 0].T


def scatter_component_local(
    values: Tensor, graph: SparsePatchGraph, active: Tensor
) -> tuple[Tensor, Tensor]:
    """Deposit to four bilinear pixels; discard any cross-component collision."""
    height, width = graph.shape
    channels = values.shape[1]
    flat = values.new_zeros((height * width, channels))
    weights = values.new_zeros((height * width, 1))
    if values.shape[0] == 0 or not bool(active.any()):
        return flat.T.reshape(channels, height, width), weights.T.reshape(1, height, width)
    coords = graph.coordinates.to(device=values.device, dtype=values.dtype)[active]
    owners = graph.components.to(values.device)[active]
    selected = values[active]
    row0 = torch.floor(coords[:, 0]).long()
    col0 = torch.floor(coords[:, 1]).long()
    dr = coords[:, 0] - row0
    dc = coords[:, 1] - col0
    indices, masses, labels, features = [], [], [], []
    for row_offset, row_weight in ((0, 1 - dr), (1, dr)):
        for col_offset, col_weight in ((0, 1 - dc), (1, dc)):
            row = (row0 + row_offset).clamp(0, height - 1)
            col = (col0 + col_offset).clamp(0, width - 1)
            mass = row_weight * col_weight
            keep = mass > 0
            indices.append((row[keep] * width + col[keep]).long())
            masses.append(mass[keep])
            labels.append(owners[keep])
            features.append(selected[keep])
    index = torch.cat(indices)
    weight = torch.cat(masses)
    component = torch.cat(labels)
    feature = torch.cat(features)
    minimum = torch.full((height * width,), torch.iinfo(torch.long).max, device=values.device)
    maximum = torch.full((height * width,), -1, device=values.device, dtype=torch.long)
    minimum.scatter_reduce_(0, index, component, reduce="amin", include_self=True)
    maximum.scatter_reduce_(0, index, component, reduce="amax", include_self=True)
    uncontested = minimum[index] == maximum[index]
    index = index[uncontested]
    weight = weight[uncontested]
    feature = feature[uncontested]
    flat.index_add_(0, index, feature * weight[:, None])
    weights.index_add_(0, index, weight[:, None])
    supported = weights > 0
    flat = torch.where(supported, flat / weights.clamp_min(1e-12), flat)
    return flat.T.reshape(channels, height, width), supported.T.reshape(1, height, width).to(values.dtype)


class SparseReflectorBranch(nn.Module):
    """Two-layer TransformerConv on safe accepted within-component long edges."""

    def __init__(
        self, channels: int, heads: int = 2, *,
        message_control: str = "genuine", control_seed: int = 12345,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if message_control not in {"genuine", "source_permuted"}:
            raise ValueError("Unknown sparse message control")
        self.message_control = message_control
        self.control_seed = int(control_seed)
        self.layers = nn.ModuleList(
            TransformerConv(channels, channels // heads, heads=heads, edge_dim=9)
            for _ in range(2)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(2))
        self.projection = nn.Linear(channels, channels)

    def messages(self, features: Tensor, graph: SparsePatchGraph) -> Tensor:
        """Component-local messages; B disrupts only the source payload identity.

        Queries/root features, endpoints, degrees, edge attributes, component
        restrictions and parameters are identical to C. Each layer permutes
        source keys/values within active nodes of the same safe component.
        """
        permutation = component_source_permutation(graph, self.control_seed)
        hidden = features
        for layer, norm in zip(self.layers, self.norms):
            source = hidden[permutation] if self.message_control == "source_permuted" else hidden
            hidden = norm(hidden + F.gelu(layer(
                (source, hidden), graph.edge_index, graph.edge_attr
            )))
        return self.projection(hidden)

    def forward(
        self, cnn: Tensor, graphs: Sequence[SparsePatchGraph]
    ) -> tuple[Tensor, Tensor]:
        if cnn.ndim != 4 or len(graphs) != cnn.shape[0]:
            raise ValueError("One sparse graph is required per CNN batch item")
        deltas, masks = [], []
        for item, original in enumerate(graphs):
            graph = original.to(cnn.device)
            if graph.shape != cnn.shape[-2:]:
                raise ValueError("Sparse graph shape does not match CNN feature map")
            if graph.edge_index.shape[1] == 0:
                deltas.append(torch.zeros_like(cnn[item]))
                masks.append(torch.zeros_like(cnn[item, :1]))
                continue
            features = sample_cnn_nodes(cnn[item : item + 1], graph)
            if not torch.isfinite(graph.edge_attr).all():
                raise ValueError("Sparse graph edge descriptors must be finite")
            hidden = self.messages(features, graph)
            active = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.bool)
            active[graph.edge_index.reshape(-1)] = True
            dense, mask = scatter_component_local(hidden, graph, active)
            deltas.append(dense)
            masks.append(mask)
        return torch.stack(deltas), torch.stack(masks)


def component_source_permutation(graph: SparsePatchGraph, seed: int) -> Tensor:
    """Deterministically reassign active source payloads, preserving graph degrees.

    This is a message-source control, not a claim to rewire the frozen graph.
    Isolated nodes are unchanged. No payload is imported across a component.
    """
    result = torch.arange(len(graph.coordinates), device=graph.components.device)
    active = torch.unique(graph.edge_index)
    for component in torch.unique(graph.components[active]).tolist():
        nodes = active[graph.components[active] == component]
        if len(nodes) < 2:
            continue
        generator = torch.Generator().manual_seed((int(seed) + 104729 * component) % (2**63 - 1))
        order = torch.randperm(len(nodes), generator=generator).to(nodes.device)
        # A fixed-point-free cyclic permutation of a randomized ordering.
        shuffled = nodes[order]
        result[shuffled] = torch.roll(shuffled, 1)
    return result


class LegacyDecoderHybridSAGEAVO(nn.Module):
    """Optional research wrapper; gamma=0 delegates exactly to the dense model.

    Sparse features are captured from the existing CNN at every forward call.
    The temporary decoder pre-hook adds component-local sparse information to
    velocity decoding only; the dense graph and segmentation path are unchanged.
    This wrapper is a feasibility prototype, not a thread-safe serving API.
    """

    def __init__(self, dense: SAGEAVO, channels: int, gamma: float = 0.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.dense = dense
        self.sparse = SparseReflectorBranch(channels)
        self.gamma = float(gamma)
        self.last_support: Tensor | None = None

    def set_norm_stats(self, statistics: Any) -> None:
        self.dense.set_norm_stats(statistics)

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        avo: Tensor,
        low: Tensor,
        rgt: Tensor,
        graphs: Sequence[SparsePatchGraph] | None = None,
    ) -> ModelOutput:
        if self.gamma == 0:
            self.last_support = None
            return self.dense(state, time, avo, low, rgt)
        if graphs is None or len(graphs) != state.shape[0]:
            raise ValueError("Active sparse branch requires one graph per batch item")
        if self.dense.diagnostic_capture_fusion:
            raise ValueError("Sparse prototype does not combine with decoder-fusion instrumentation")
        captured: list[Tensor] = []
        original_decoder_input: list[Tensor] = []

        def save_cnn(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            captured.append(output)

        def fuse(_module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one current-time CNN feature map")
            sparse, support = self.sparse(captured[0], graphs)
            self.last_support = support
            original_decoder_input.append(inputs[0])
            return (inputs[0] + self.gamma * support * sparse,)

        def confine(
            _module: nn.Module, _inputs: tuple[Tensor, ...], active_output: Tensor
        ) -> Tensor:
            if len(original_decoder_input) != 1 or self.last_support is None:
                raise RuntimeError("Expected one decoder input and one sparse support mask")
            # Call the Sequential's children directly to bypass only its
            # temporary outer hooks. This reproduces the identical dense
            # decoder, including spatial GroupNorm and convolutions.
            baseline_output = original_decoder_input[0]
            for layer in self.dense.decoder:
                baseline_output = layer(baseline_output)
            return baseline_output + self.last_support * (active_output - baseline_output)

        encoder_hook = self.dense.encoder.register_forward_hook(save_cnn)
        decoder_hook = self.dense.decoder.register_forward_pre_hook(fuse)
        output_hook = self.dense.decoder.register_forward_hook(confine)
        try:
            return self.dense(state, time, avo, low, rgt)
        finally:
            output_hook.remove()
            decoder_hook.remove()
            encoder_hook.remove()

    def sample(
        self,
        avo: Tensor,
        low: Tensor,
        rgt: Tensor,
        graphs: Sequence[SparsePatchGraph] | None = None,
        *,
        steps: int = 20,
        guidance_scale: float = 0.0,
        avo_mask: Tensor | None = None,
    ) -> Tensor:
        if self.gamma == 0:
            return self.dense.sample(
                avo, low, rgt, steps=steps, guidance_scale=guidance_scale, avo_mask=avo_mask
            )
        if graphs is None:
            raise ValueError("Active sparse sampling requires patch-local graphs")

        def velocity(state: Tensor, time: Tensor) -> Tensor:
            with torch.no_grad():
                return self(state, time, avo, low, rgt, graphs).velocity

        correction = None
        if guidance_scale > 0:
            start = int(steps * self.dense.guidance_start_fraction)

            def correction(state: Tensor, index: int) -> Tensor:
                active = index >= start and (index + 1) % self.dense.guidance_interval_steps == 0
                return (
                    self.dense._physics_guided_correction(
                        state, avo, scale=guidance_scale, avo_mask=avo_mask
                    )
                    if active else state
                )

        return heun_integrate(low.clone(), velocity, steps=steps, correction=correction)


class HybridSparseSAGEAVO(LegacyDecoderHybridSAGEAVO):
    """v00332ac pointwise post-decoder residual, with no spatial normalization.

    The dense model is evaluated once on the same state. The sparse branch
    samples that call's pre-GNN CNN features and projects them directly to three
    velocity channels. Direct sparse contributions cannot mix components via
    decoder convolutions or spatial GroupNorm. Subsequent conditional-flow
    evaluations can still spread changed states through the existing dense CNN.
    """

    def __init__(
        self, dense: SAGEAVO, channels: int, gamma: float = 0.0, *,
        message_control: str = "genuine", control_seed: int = 12345,
    ) -> None:
        super().__init__(dense, channels, gamma)
        self.sparse.message_control = message_control
        if message_control not in {"genuine", "source_permuted"}:
            raise ValueError("Unknown sparse message control")
        self.sparse.control_seed = int(control_seed)
        self.velocity_projection = nn.Conv2d(channels, 3, 1, bias=False)

    def forward(
        self, state: Tensor, time: Tensor, avo: Tensor, low: Tensor, rgt: Tensor,
        graphs: Sequence[SparsePatchGraph] | None = None,
    ) -> ModelOutput:
        if self.gamma == 0:
            self.last_support = None
            return self.dense(state, time, avo, low, rgt)
        if graphs is None or len(graphs) != len(state):
            raise ValueError("Active sparse branch requires one graph per batch item")
        captured: list[Tensor] = []

        def save(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            captured.append(output)

        handle = self.dense.encoder.register_forward_hook(save)
        try:
            output = self.dense(state, time, avo, low, rgt)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one current-state CNN feature map")
        sparse, mask = self.sparse(captured[0], graphs)
        self.last_support = mask
        residual = self.gamma * mask * self.velocity_projection(sparse)
        return output._replace(velocity=output.velocity + residual)
