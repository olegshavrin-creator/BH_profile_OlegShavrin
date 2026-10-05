"""MCDM fusion model for the personality task only: port of MM-PSYCHE MultiModalFusionModel_v1
(src/models/models.py) with the emotion / ambivalence heads removed.

Pipeline for a batch of pooled modality vectors {mod: [B, D_mod]}:
  modality projectors (Projector + AdapterFusion) -> [B, N, H]
  graph attention over modalities (C_mods)
  task projector: per-modality personality logits [B, N, 5] -> back to hidden [B, N, H] -> graph attention (C_preds)
  cross-attention: queries = C_preds, keys/values = C_mods -> mean over modalities -> task representation [B, H]
  head: Projector(H -> out) -> Linear(out -> 5) -> sigmoid
  guide bank: cosine similarity to 5 learned prototypes, sigmoid, averaged with the head output
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.proj(x)


class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(hidden_dim, max(1, hidden_dim // 2)), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(max(1, hidden_dim // 2), hidden_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.layernorm(x + self.adapter(x))


class GraphAttentionLayer(nn.Module):
    """Dense GAT over the N modality nodes (MM-PSYCHE v1 layer, pure PyTorch)."""

    def __init__(self, in_dim, out_dim=None, dropout=0.1, alpha=0.2):
        super().__init__()
        out_dim = out_dim or in_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        B, N, _ = h.size()
        Wh = self.W(h)
        a_input = torch.cat([Wh.unsqueeze(2).expand(-1, -1, N, -1), Wh.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        attention = torch.where(adj > 0, e, torch.full_like(e, -9e15))
        attention = self.dropout(F.softmax(attention, dim=-1))
        return torch.matmul(attention, Wh)


class Identity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


@dataclass
class ModelConfig:
    modality_dims: Dict[str, int] = field(default_factory=lambda: {"face": 1024, "audio": 1536, "text": 1536, "behavior": 1536})
    hidden_dim: int = 512
    num_heads: int = 8
    out_dim: int = 512
    dropout: float = 0.15
    n_traits: int = 5
    use_graph: bool = True
    use_attention: bool = True
    use_guidebank: bool = True
    use_task_projectors: bool = True


class PersonalityFusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.hidden_dim, cfg.dropout
        self.modalities = list(cfg.modality_dims.keys())
        self.modality_projectors = nn.ModuleDict({
            m: nn.Sequential(Projector(d, H, D), AdapterFusion(H, D)) for m, d in cfg.modality_dims.items()
        })
        Graph = (lambda: GraphAttentionLayer(H, dropout=D)) if cfg.use_graph else Identity
        self.graph_features = Graph()
        if cfg.use_task_projectors:
            self.predictor = nn.Sequential(Projector(H, cfg.n_traits, D), AdapterFusion(cfg.n_traits, D))
            self.prediction_projector = nn.Sequential(Projector(cfg.n_traits, H, D), AdapterFusion(H, D))
            self.graph_task = Graph()
        else:
            self.graph_task = Graph()
        self.cross_attn = nn.MultiheadAttention(H, cfg.num_heads, dropout=D, batch_first=True) if cfg.use_attention else None
        self.head = nn.Sequential(Projector(H, cfg.out_dim, D), nn.Linear(cfg.out_dim, cfg.n_traits), nn.Sigmoid())
        self.guide_bank = nn.Parameter(torch.randn(cfg.n_traits, H)) if cfg.use_guidebank else None

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        xs = []
        for m in self.modalities:
            if m in features and features[m] is not None:
                xs.append(self.modality_projectors[m](features[m]))
        if not xs:
            raise ValueError("no modality features given")
        x = torch.stack(xs, dim=1)                                    # [B, N, H]
        B, N, _ = x.shape
        adj = torch.ones(B, N, N, device=x.device)
        ctx_mods = self.graph_features(x, adj)
        if self.cfg.use_task_projectors:
            per_mod_logits = self.predictor(x)                        # [B, N, 5]
            ctx_preds = self.graph_task(self.prediction_projector(per_mod_logits), adj)
        else:
            ctx_preds = self.graph_task(ctx_mods, adj)
        if self.cross_attn is not None:
            rep, _ = self.cross_attn(ctx_preds, ctx_mods, ctx_mods)
            rep = rep.mean(dim=1)
        else:
            rep = ctx_preds.mean(dim=1)
        scores = self.head(rep)                                        # [B, 5] in (0, 1)
        if self.guide_bank is not None:
            sim = F.cosine_similarity(rep.unsqueeze(1), self.guide_bank.unsqueeze(0), dim=-1)
            scores = (scores + torch.sigmoid(sim)) / 2.0
        return scores
