"""Multimodal nanopore RNA modification model (PyTorch only).

Inputs
------
signal:
    Normalized raw-current windows, shape [batch, signal_samples, signal_features].
sequence_tokens:
    Basecalled/aligned RNA token IDs, shape [batch, sequence_length].
structure_adjacency:
    RNA backbone + base-pair probability matrix, shape [batch, sequence_length,
    sequence_length]. It should include self loops (the model also adds them).

Outputs
-------
read_logits:
    Per-read logits for [m6A, m5C].
read_probabilities:
    Per-read modification probabilities for [m6A, m5C].
site_ratios:
    Site/sample stoichiometry estimates in [0, 1] for [m6A, m5C].

The model has three modalities:
1. raw current: parallel Mamba-style SSM and BiLSTM encoders;
2. basecalled sequence: a local Transformer or an optional Hugging Face model;
3. RNA secondary structure: an edge-weighted graph-attention encoder.

This file intentionally does not read FAST5/POD5. Signal-to-base alignment and
window extraction belong in the data pipeline; a neural network should receive
already aligned, normalized local windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    signal_features: int = 4  # e.g. current, delta-current, dwell, move
    d_model: int = 128
    signal_layers: int = 2
    lstm_layers: int = 2
    sequence_layers: int = 3
    graph_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.15
    vocab_size: int = 8
    padding_idx: int = 0
    max_sequence_length: int = 256
    pretrained_sequence_model: Optional[str] = None
    freeze_pretrained_sequence_model: bool = True


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Mean pool valid positions; mask is True for valid values."""
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class SelectiveSSMBlock(nn.Module):
    """A compact, pure-PyTorch Mamba-style selective state-space block.

    It implements input-dependent decay and input gates with a recurrent scan.
    For production speed, this block can be replaced by mamba_ssm.Mamba without
    changing the surrounding interface.
    """

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.depthwise = nn.Conv1d(
            inner, inner, kernel_size=3, padding=2, groups=inner
        )
        self.decay_proj = nn.Linear(inner, inner)
        self.input_proj = nn.Linear(inner, inner)
        self.output_gate = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        # Causal depthwise convolution: padding=2 then remove the last 2 steps.
        u = self.depthwise(u.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        u = F.silu(u)
        decay = torch.sigmoid(self.decay_proj(u))
        proposal = torch.tanh(self.input_proj(u))

        state = torch.zeros_like(proposal[:, 0])
        outputs = []
        for t in range(x.size(1)):
            candidate = decay[:, t] * state + (1.0 - decay[:, t]) * proposal[:, t]
            if mask is not None:
                valid = mask[:, t].unsqueeze(-1)
                state = torch.where(valid, candidate, state)
                out_t = torch.where(valid, state, torch.zeros_like(state))
            else:
                state, out_t = candidate, candidate
            outputs.append(out_t)
        y = torch.stack(outputs, dim=1)
        y = y * torch.sigmoid(self.output_gate(gate))
        return residual + self.dropout(self.out_proj(y))


class SignalEncoder(nn.Module):
    """Parallel Mamba-style and BiLSTM raw-current encoders with gated fusion."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.input_projection = nn.Sequential(
            nn.Conv1d(cfg.signal_features, d, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.mamba = nn.ModuleList(
            [SelectiveSSMBlock(d, dropout=cfg.dropout) for _ in range(cfg.signal_layers)]
        )
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(d)

    def forward(self, signal: Tensor, signal_mask: Tensor) -> Tensor:
        x = self.input_projection(signal.transpose(1, 2)).transpose(1, 2)
        # Two stride-2 convolutions: valid length is ceil(ceil(length/2)/2).
        lengths = signal_mask.sum(1)
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        t = x.size(1)
        mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]

        mamba_x = x
        for block in self.mamba:
            mamba_x = block(mamba_x, mask)
        lstm_x, _ = self.lstm(x)

        mamba_feature = masked_mean(mamba_x, mask)
        lstm_feature = masked_mean(lstm_x, mask)
        both = torch.cat([mamba_feature, lstm_feature], dim=-1)
        gate = self.fusion_gate(both)
        return self.out_norm(gate * mamba_feature + (1.0 - gate) * lstm_feature)


class SequenceEncoder(nn.Module):
    """RNA encoder backed by a local Transformer or a Hugging Face backbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pretrained = None
        self.local_embedding = None
        self.local_encoder = None

        if cfg.pretrained_sequence_model:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "Install `transformers` to use pretrained_sequence_model, "
                    "or set it to None for the built-in sequence encoder."
                ) from exc
            self.pretrained = AutoModel.from_pretrained(cfg.pretrained_sequence_model)
            hidden = self.pretrained.config.hidden_size
            if cfg.freeze_pretrained_sequence_model:
                for parameter in self.pretrained.parameters():
                    parameter.requires_grad = False
            self.output_projection = nn.Linear(hidden, cfg.d_model)
        else:
            self.local_embedding = nn.Embedding(
                cfg.vocab_size, cfg.d_model, padding_idx=cfg.padding_idx
            )
            self.position_embedding = nn.Embedding(cfg.max_sequence_length, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.attention_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.local_encoder = nn.TransformerEncoder(
                layer, num_layers=cfg.sequence_layers, norm=nn.LayerNorm(cfg.d_model)
            )
            self.output_projection = nn.Identity()

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.pretrained is not None:
            context = torch.no_grad() if not any(
                p.requires_grad for p in self.pretrained.parameters()
            ) else torch.enable_grad()
            with context:
                x = self.pretrained(
                    input_ids=tokens, attention_mask=mask.long()
                ).last_hidden_state
            x = self.output_projection(x)
        else:
            if tokens.size(1) > self.position_embedding.num_embeddings:
                raise ValueError("sequence exceeds max_sequence_length")
            positions = torch.arange(tokens.size(1), device=tokens.device)[None, :]
            x = self.local_embedding(tokens) + self.position_embedding(positions)
            x = self.local_encoder(x, src_key_padding_mask=~mask)
        x = x * mask.unsqueeze(-1)
        return x, masked_mean(x, mask)


class WeightedGraphAttentionLayer(nn.Module):
    """Dense edge-weighted multi-head graph attention for short RNA windows."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        reshape = lambda z: z.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5

        eye = torch.eye(n, device=x.device, dtype=adjacency.dtype)[None]
        adjacency = torch.maximum(adjacency, eye)
        valid_edges = (adjacency > 0) & mask[:, :, None] & mask[:, None, :]
        scores = scores + torch.log(adjacency.clamp_min(1e-6))[:, None]
        scores = scores.masked_fill(~valid_edges[:, None], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * mask[:, None, :, None]
        y = torch.matmul(self.dropout(attention), v)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        x = x + self.dropout(self.out(y))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * mask.unsqueeze(-1)


class StructureEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            WeightedGraphAttentionLayer(cfg.d_model, cfg.attention_heads, cfg.dropout)
            for _ in range(cfg.graph_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, node_features: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        x = node_features
        for layer in self.layers:
            x = layer(x, adjacency, mask)
        return masked_mean(self.norm(x), mask)


class MultimodalFusion(nn.Module):
    """Learns sample-specific weights over signal, sequence, and structure."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 3)
        )
        self.projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.post = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, signal: Tensor, sequence: Tensor, structure: Tensor):
        modalities = [signal, sequence, structure]
        weights = torch.softmax(self.gate(torch.cat(modalities, dim=-1)), dim=-1)
        stacked = torch.stack(
            [projection(value) for projection, value in zip(self.projections, modalities)],
            dim=1,
        )
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return self.post(fused), weights


class RNAModificationModel(nn.Module):
    """End-to-end three-modal model for joint m6A/m5C prediction."""

    modification_order = ("m6A", "m5C")

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.signal_encoder = SignalEncoder(cfg)
        self.sequence_encoder = SequenceEncoder(cfg)
        self.structure_encoder = StructureEncoder(cfg)
        self.fusion = MultimodalFusion(cfg.d_model, cfg.dropout)
        self.read_classifier = nn.Linear(cfg.d_model, 2)
        # Direct ratio head is useful with experimentally known mixture ratios.
        self.ratio_regressor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model // 2, 2)
        )

    def forward(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        sequence_tokens: Tensor,
        sequence_mask: Tensor,
        structure_adjacency: Tensor,
    ) -> Dict[str, Tensor]:
        signal_feature = self.signal_encoder(signal, signal_mask)
        sequence_nodes, sequence_feature = self.sequence_encoder(
            sequence_tokens, sequence_mask
        )
        structure_feature = self.structure_encoder(
            sequence_nodes, structure_adjacency, sequence_mask
        )
        fused, modality_weights = self.fusion(
            signal_feature, sequence_feature, structure_feature
        )
        read_logits = self.read_classifier(fused)
        return {
            "read_logits": read_logits,
            "read_probabilities": torch.sigmoid(read_logits),
            "site_ratios": torch.sigmoid(self.ratio_regressor(fused)),
            "modality_weights": modality_weights,
            "fused_features": fused,
        }


def multitask_loss(
    outputs: Dict[str, Tensor],
    read_labels: Tensor,
    ratio_targets: Tensor,
    ratio_mask: Optional[Tensor] = None,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Joint BCE classification and Smooth-L1 stoichiometry loss.

    Labels and targets use column order [m6A, m5C]. A ratio_mask can suppress
    unavailable assay labels. For site-only weak labels, replace read BCE with a
    multiple-instance learning loss that aggregates reads belonging to a site.
    """
    classification = F.binary_cross_entropy_with_logits(
        outputs["read_logits"], read_labels.float()
    )
    regression_elementwise = F.smooth_l1_loss(
        outputs["site_ratios"], ratio_targets.float(), reduction="none"
    )
    if ratio_mask is None:
        regression = regression_elementwise.mean()
    else:
        ratio_mask = ratio_mask.to(regression_elementwise.dtype)
        regression = (regression_elementwise * ratio_mask).sum() / ratio_mask.sum().clamp_min(1)
    total = classification_weight * classification + regression_weight * regression
    return {"loss": total, "classification_loss": classification, "regression_loss": regression}


def build_backbone_adjacency(pair_probabilities: Tensor, mask: Tensor) -> Tensor:
    """Combine RNA backbone edges with a secondary-structure probability matrix."""
    b, n, _ = pair_probabilities.shape
    adjacency = pair_probabilities.clamp(0, 1).clone()
    indices = torch.arange(n - 1, device=adjacency.device)
    adjacency[:, indices, indices + 1] = 1.0
    adjacency[:, indices + 1, indices] = 1.0
    adjacency = adjacency * mask[:, :, None] * mask[:, None, :]
    return adjacency


def demo() -> None:
    """CPU smoke test with variable-length synthetic inputs."""
    torch.manual_seed(7)
    cfg = ModelConfig(d_model=64, signal_layers=1, sequence_layers=1, graph_layers=1)
    model = RNAModificationModel(cfg)

    batch, signal_length, sequence_length = 3, 400, 41
    signal = torch.randn(batch, signal_length, cfg.signal_features)
    signal_mask = torch.arange(signal_length)[None] < torch.tensor([400, 350, 290])[:, None]
    tokens = torch.randint(1, 5, (batch, sequence_length))
    sequence_mask = torch.arange(sequence_length)[None] < torch.tensor([41, 39, 35])[:, None]
    tokens = tokens.masked_fill(~sequence_mask, cfg.padding_idx)

    pair_probabilities = torch.zeros(batch, sequence_length, sequence_length)
    # Illustrative stem edges; in real use pass RNAfold/RNAstructure probabilities.
    for i in range(8):
        pair_probabilities[:, i + 4, 32 - i] = 0.8
        pair_probabilities[:, 32 - i, i + 4] = 0.8
    adjacency = build_backbone_adjacency(pair_probabilities, sequence_mask)

    outputs = model(signal, signal_mask, tokens, sequence_mask, adjacency)
    losses = multitask_loss(
        outputs,
        read_labels=torch.randint(0, 2, (batch, 2)).float(),
        ratio_targets=torch.rand(batch, 2),
    )
    print("read probabilities [m6A, m5C]:\n", outputs["read_probabilities"])
    print("site ratios [m6A, m5C]:\n", outputs["site_ratios"])
    print("modality weights [signal, sequence, structure]:\n", outputs["modality_weights"])
    print("loss:", float(losses["loss"]))


if __name__ == "__main__":
    demo()
