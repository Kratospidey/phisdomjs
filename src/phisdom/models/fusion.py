from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ConvBlock1D, DomGCN


class CheapEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((0, self.out_dim), device=x.device, dtype=x.dtype)
        return self.net(x)


class DomEncoder(nn.Module):
    """Light DOM encoder that reuses DomGCN up to projection layer and returns a per-graph embedding."""

    def __init__(self, hidden: int = 128, layers: int = 2, out_dim: int = 128, num_embeddings: Optional[dict] = None):
        super().__init__()
        # Reuse DomGCN internals by composing and tapping into its projected node features + mean pool.
        self.gcn = DomGCN(hidden=hidden, layers=layers, out_dim=out_dim, num_embeddings=num_embeddings)

    def forward(self, node_feats_raw: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        # Hack: call DomGCN forward to get logits, but we need the penultimate representation.
        # We'll mirror its computation here to get the pooled feature before classifier.
        if node_feats_raw.numel() == 0:
            B = int(batch_index.max().item() + 1) if batch_index.numel() else 1
            return torch.zeros((max(1, B), self.gcn.proj.out_features), dtype=torch.float32, device=batch_index.device if batch_index.is_cuda else None)
        # Replicate internals (keep in sync with DomGCN)
        t = (node_feats_raw[:, 0] % self.gcn.t_buckets).clamp_min(0)
        c = (node_feats_raw[:, 1] % self.gcn.c_buckets).clamp_min(0)
        d = node_feats_raw[:, 2].clamp(0, 255)
        x = node_feats_raw[:, 3].clamp(0, 7)
        x0 = torch.cat([self.gcn.t_emb(t), self.gcn.c_emb(c), self.gcn.d_emb(d), self.gcn.x_emb(x)], dim=-1)
        h = x0
        for g in self.gcn.gcns:
            h = g(h, edge_index)
        h = self.gcn.proj(h)
        B = int(batch_index.max().item() + 1) if batch_index.numel() else 1
        out = torch.zeros((B, h.size(-1)), device=h.device, dtype=h.dtype)
        counts = torch.bincount(batch_index, minlength=B).clamp(min=1).unsqueeze(-1).float()
        out.index_add_(0, batch_index, h)
        out = out / counts
        return out


class CrossModalTransformerFusion(nn.Module):
    """Cross-modal fusion via Transformer encoder over modality tokens.

    Modalities supported (any subset can be provided at runtime):
      - url:      charseq -> ConvBlock1D -> proj(d_model)
      - js:       charseq -> ConvBlock1D -> proj(d_model)
      - text:     charseq -> ConvBlock1D -> proj(d_model)
      - dom:      graph   -> DomEncoder    -> proj(d_model)
      - cheap:    vector  -> MLP           -> proj(d_model)

    We add a learned [CLS] token, modality type embeddings, and run L layers of MHSA.
    Output is [CLS] passed to a binary classifier head.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        url_vocab: int = 2 + 26 + 26 + 10 + 28,
        js_vocab: int = 2 + 26 + 26 + 10 + 42,
        text_vocab: int = 2 + 26 + 26 + 10 + 2 + 32,
        cheap_dim: int = 32,
    ):
        super().__init__()
        # Encoders
        self.enc_url = ConvBlock1D(url_vocab, emb_dim=48, out_dim=d_model, dropout=dropout)
        self.enc_js = ConvBlock1D(js_vocab, emb_dim=48, out_dim=d_model, dropout=dropout)
        self.enc_text = ConvBlock1D(text_vocab, emb_dim=64, out_dim=d_model, dropout=dropout)
        self.enc_dom = DomEncoder(hidden=d_model, layers=2, out_dim=d_model)
        self.enc_cheap = CheapEncoder(in_dim=cheap_dim, out_dim=d_model, hidden=64, dropout=dropout)

        # Token/type embeddings
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mod_types = nn.Embedding(6, d_model)  # 0=CLS, 1=URL,2=JS,3=TEXT,4=DOM,5=CHEAP

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        types: List[int] = []
        B = None

        def _ensure_batch_dim(x: torch.Tensor) -> int:
            nonlocal B
            if B is None:
                B = x.size(0)
            return B or x.size(0)

        # URL
        if (ids := batch.get("url_input_ids")) is not None:
            _ensure_batch_dim(ids)
            h = self.enc_url(ids)
            tokens.append(h)
            types.append(1)
        # JS
        if (ids := batch.get("js_input_ids")) is not None:
            _ensure_batch_dim(ids)
            h = self.enc_js(ids)
            tokens.append(h)
            types.append(2)
        # TEXT
        if (ids := batch.get("text_input_ids")) is not None:
            _ensure_batch_dim(ids)
            h = self.enc_text(ids)
            tokens.append(h)
            types.append(3)
        # DOM
        if (nf := batch.get("dom_node_feats_raw")) is not None:
            ei = batch.get("dom_edge_index")
            bidx = batch.get("dom_batch_index")
            assert ei is not None and bidx is not None
            _ensure_batch_dim(bidx)
            h = self.enc_dom(nf, ei, bidx)
            tokens.append(h)
            types.append(4)
        # CHEAP
        if (cf := batch.get("cheap_features")) is not None:
            _ensure_batch_dim(cf)
            h = self.enc_cheap(cf)
            tokens.append(h)
            types.append(5)

        if not tokens:
            raise ValueError("No modalities provided in batch")

        X = torch.stack(tokens, dim=1) if len(tokens) > 1 else tokens[0].unsqueeze(1)  # [B, M, d]
        B = X.size(0)
        # prepend CLS
        cls = self.cls.expand(B, -1, -1)
        X = torch.cat([cls, X], dim=1)  # [B, 1+M, d]
        # type embeddings
        type_ids = torch.tensor([0] + types, device=X.device).unsqueeze(0).expand(B, -1)
        X = X + self.mod_types(type_ids)
        H = self.tr(X)  # [B, 1+M, d]
        H0 = self.norm(H[:, 0])
        logits = self.head(H0).squeeze(-1)
        return logits
