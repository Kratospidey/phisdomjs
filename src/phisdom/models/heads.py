from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, out_dim: int = 128, kernels=(2, 4, 6, 8), dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, out_dim, k, padding=0) for k in kernels])
        self.proj = nn.Linear(out_dim * len(kernels), out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.emb(input_ids).transpose(1, 2)  # [B, E, T]
        feats = []
        T = x.shape[-1]
        for conv in self.convs:
            ks = getattr(conv, "kernel_size", (1,))
            k = int(ks[0] if isinstance(ks, (list, tuple)) else ks)
            xt = x
            if T < k:
                # left-pad to ensure at least one valid conv position
                pad = k - T
                xt = F.pad(x, (pad, 0))
            h = conv(xt)  # [B, C, T']
            h = F.relu(h)
            h = F.max_pool1d(h, kernel_size=h.shape[-1]).squeeze(-1)  # [B, C]
            feats.append(h)
        z = torch.cat(feats, dim=-1)
        z = self.drop(z)
        z = self.proj(z)
        return z  # [B, out_dim]


class UrlCharCNN(nn.Module):
    def __init__(self, vocab_size: int = 2 + 26 + 26 + 10 + 28, emb_dim: int = 48, out_dim: int = 128, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.block = ConvBlock1D(vocab_size=vocab_size, emb_dim=emb_dim, out_dim=out_dim, dropout=dropout, pad_idx=pad_idx)
        self.cls = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_logits: bool = True):
        z = self.block(input_ids, attention_mask)
        logits = self.cls(z).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class JsCharCNN(nn.Module):
    def __init__(self, vocab_size: int = 2 + 26 + 26 + 10 + 42, emb_dim: int = 48, out_dim: int = 128, dropout: float = 0.15, pad_idx: int = 0):
        super().__init__()
        self.block = ConvBlock1D(vocab_size=vocab_size, emb_dim=emb_dim, out_dim=out_dim, dropout=dropout, pad_idx=pad_idx)
        self.cls = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_logits: bool = True):
        z = self.block(input_ids, attention_mask)
        logits = self.cls(z).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class TextCharCNN(nn.Module):
    def __init__(self, vocab_size: int = 2 + 26 + 26 + 10 + 2 + 32, emb_dim: int = 64, out_dim: int = 128, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        # Reuse ConvBlock1D with a slightly larger vocab (letters+digits+space+punct)
        self.block = ConvBlock1D(vocab_size=vocab_size, emb_dim=emb_dim, out_dim=out_dim, dropout=dropout, pad_idx=pad_idx)
        self.cls = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_logits: bool = True):
        z = self.block(input_ids, attention_mask)
        logits = self.cls(z).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class CheapMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        z = self.net(x)
        logits = self.cls(z).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, F], edge_index: [2, E]
        N = x.size(0)
        E = edge_index.size(1)
        # Simple mean aggregation: add self-loops
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=N).clamp(min=1).float()
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        agg = agg / deg.unsqueeze(-1)
        h = self.lin(agg)
        return F.relu(h)


class DomGCN(nn.Module):
    def __init__(self, node_feat_dim: int = 4, hidden: int = 128, layers: int = 2, out_dim: int = 128, num_embeddings: Optional[dict] = None):
        super().__init__()
        # Embed hashed categorical (t,c) and numeric (d,x)
        # Map:
        #  - t_hash, c_hash -> embed via modulo buckets
        #  - d (depth), x (text bin) -> small embeddings
        self.t_buckets = (num_embeddings or {}).get("t", 4096)
        self.c_buckets = (num_embeddings or {}).get("c", 4096)
        self.t_emb = nn.Embedding(self.t_buckets, 32)
        self.c_emb = nn.Embedding(self.c_buckets, 32)
        self.d_emb = nn.Embedding(256, 8)
        self.x_emb = nn.Embedding(8, 8)
        in_dim = 32 + 32 + 8 + 8
        self.gcns = nn.ModuleList([GraphConv(in_dim if i == 0 else hidden, hidden) for i in range(layers)])
        self.proj = nn.Linear(hidden, out_dim)
        self.cls = nn.Linear(out_dim, 1)

    def forward(self, node_feats_raw: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor, return_logits: bool = True):
        # node_feats_raw: [N,4] (t,c,d,x)
        if node_feats_raw.numel() == 0:
            # no nodes: return zeros for a single-item batch
            B = int(batch_index.max().item() + 1) if batch_index.numel() else 1
            z = torch.zeros((max(1, B), self.proj.out_features), dtype=torch.float32, device=batch_index.device if batch_index.is_cuda else None)
            logits = self.cls(z).squeeze(-1)
            return logits if return_logits else torch.sigmoid(logits)
        t = (node_feats_raw[:, 0] % self.t_buckets).clamp_min(0)
        c = (node_feats_raw[:, 1] % self.c_buckets).clamp_min(0)
        d = node_feats_raw[:, 2].clamp(0, 255)
        x = node_feats_raw[:, 3].clamp(0, 7)
        x0 = torch.cat([self.t_emb(t), self.c_emb(c), self.d_emb(d), self.x_emb(x)], dim=-1)
        h = x0
        for g in self.gcns:
            h = g(h, edge_index)
        h = self.proj(h)
        # mean-pool per graph in batch
        B = int(batch_index.max().item() + 1) if batch_index.numel() else 1
        out = torch.zeros((B, h.size(-1)), device=h.device, dtype=h.dtype)
        counts = torch.bincount(batch_index, minlength=B).clamp(min=1).unsqueeze(-1).float()
        out.index_add_(0, batch_index, h)
        out = out / counts
        logits = self.cls(out).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
