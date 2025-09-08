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
        *,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        # turn modalities on/off (useful for ablations)
        use_url: bool = True,
        use_js: bool = True,
        use_text: bool = True,
        use_dom: bool = True,
        use_cheap: bool = True,
        # cheap features (set to None to activate lazy detection)
        cheap_dim: int | None = None,
        # token embedding assumptions for raw (byte/char/id) inputs
        token_vocab_size: int = 512,   # safe for bytes or small char vocabs
        token_embed_dim: int = 64,     # local token emb size before projecting
        # Legacy compatibility
        nhead: int | None = None,
        num_layers: int | None = None,
        url_vocab: int | None = None,
        js_vocab: int | None = None,
        text_vocab: int | None = None,
    ):
        """
        Minimal but sturdy init for cross-modal fusion:
          - URL/JS/TEXT encoders: accept either Long (B,L) or Float (B,D) and emit (B, d_model)
          - DOM encoder: mean-pools nodes per-graph → MLP → (B, d_model)
          - CHEAP encoder: robust to unknown in_dim (uses LazyLinear if cheap_dim is None)
          - Transformer encoder over tokens [CLS, url?, js?, text?, dom?, cheap?]
        """
        super().__init__()
        
        # Handle legacy parameter names for compatibility
        if nhead is not None:
            n_heads = nhead
        if num_layers is not None:
            n_layers = num_layers
            
        self.d_model = d_model
        self.use_url  = use_url
        self.use_js   = use_js
        self.use_text = use_text
        self.use_dom  = use_dom
        self.use_cheap= use_cheap

        # ---------------------------
        # Small helper encoders
        # ---------------------------
        class TokenOrFeatEncoder(nn.Module):
            """Accepts Long (B,L) → Embedding+mean, or Float (B,D) → Linear; outputs (B, d_model)."""
            def __init__(self, d_model: int, vocab_size: int, emb_dim: int):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.lin = nn.Linear(emb_dim, d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.dtype in (torch.long, torch.int64, torch.int32):     # (B, L) ids
                    emb = self.emb(x)                                     # (B, L, E)
                    mask = (x != 0).unsqueeze(-1).float()                 # (B, L, 1)
                    s = (emb * mask).sum(dim=1)                           # (B, E)
                    n = mask.sum(dim=1).clamp_min(1.0)                    # (B, 1)
                    h = s / n                                             # (B, E)
                    return self.lin(h)                                    # (B, d)
                # float features (B, D) → map to d_model
                if x.dim() == 2:
                    return nn.Linear(x.size(-1), self.lin.out_features, device=x.device, dtype=x.dtype)(x)
                # fallback: flatten then linear
                x = x.view(x.size(0), -1)
                return nn.Linear(x.size(-1), self.lin.out_features, device=x.device, dtype=x.dtype)(x)

        class CheapEncoder(nn.Module):
            """Robust cheap-feature MLP: LazyLinear avoids matmul shape errors if width changes."""
            def __init__(self, in_dim: int | None, out_dim: int, hidden: int = 64, dropout: float = 0.1):
                super().__init__()
                first = nn.Linear(in_dim, hidden) if in_dim is not None else nn.LazyLinear(hidden)
                self.net = nn.Sequential(
                    first, nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, out_dim)
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # ensure 2D
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                return self.net(x)

        class DomMeanPoolEncoder(nn.Module):
            """Mean-pool node features per graph using dom_batch; no external deps."""
            def __init__(self, out_dim: int, hidden: int = 128, dropout: float = 0.1):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.LazyLinear(hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, out_dim)
                )
            def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor | None, dom_batch: torch.Tensor) -> torch.Tensor:
                # node_x: (N, F), dom_batch: (N,) with graph ids in [0..B-1]
                B = int(dom_batch.max().item()) + 1 if dom_batch.numel() else 1
                F = node_x.size(-1)
                sums = node_x.new_zeros((B, F))
                counts = node_x.new_zeros((B, 1))
                sums.index_add_(0, dom_batch, node_x)
                one = torch.ones_like(dom_batch, dtype=node_x.dtype).unsqueeze(-1)
                counts.index_add_(0, dom_batch, one)
                pooled = sums / counts.clamp_min(1.0)
                return self.mlp(pooled)  # (B, out_dim)

        # ---------------------------
        # Per-modality encoders (→ d_model)
        # ---------------------------
        if use_url:
            self.enc_url  = TokenOrFeatEncoder(d_model, token_vocab_size, token_embed_dim)
        if use_js:
            self.enc_js   = TokenOrFeatEncoder(d_model, token_vocab_size, token_embed_dim)
        if use_text:
            self.enc_text = TokenOrFeatEncoder(d_model, token_vocab_size, token_embed_dim)
        if use_dom:
            self.enc_dom  = DomMeanPoolEncoder(d_model)
        if use_cheap:
            self.enc_cheap= CheapEncoder(cheap_dim, d_model)

        # Optional projections (kept for API symmetry)
        self.proj_url   = nn.Identity()
        self.proj_js    = nn.Identity()
        self.proj_text  = nn.Identity()
        self.proj_dom   = nn.Identity()
        self.proj_cheap = nn.Identity()

        # ---------------------------
        # Token type + positional embeddings
        #   0:CLS, 1:url, 2:js, 3:text, 4:dom, 5:cheap
        # ---------------------------
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.type_emb = nn.Embedding(6, d_model)
        # max tokens we ever place: 1 (CLS) + up to 5 modalities
        self.pos_emb = nn.Embedding(1 + 5, d_model)

        # ---------------------------
        # Transformer encoder (batch_first=True)
        # ---------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ---------------------------
        # Classifier head
        # ---------------------------
        self.cls_head = nn.Linear(d_model, 1)

        # ---------------------------
        # Init
        # ---------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights. Skip LazyLinear - it will initialize itself on first forward."""
        if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0., std=0.02)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Assemble a variable-length token sequence [CLS, (url)?, (js)?, (text)?, (dom)?, (cheap)?]
        then run a Transformer encoder and return a single logit per item.

        Expected (robust) batch keys:
          URL  : 'url_input_ids' | 'url_ids' | 'url_bytes' | 'url_tokens'
          JS   : 'js_input_ids' | 'js_ids'  | 'js_bytes'  | 'js_tokens'  | 'js_text'
          TEXT : 'text_input_ids'| 'text_ids'| 'text_bytes'| 'text_tokens'| 'text'
          DOM  : node feats: 'dom_node_feats_raw'|'dom_node_feats'|'dom_nodes'
                 edges    : 'dom_edge_index'|'dom_edges'
                 graph ids: 'dom_batch_index'|'dom_batch'|'dom_graph_index'
          CHEAP: 'cheap_features'|'cheap'
          IDS  : 'ids'|'id'  (optional; not needed here)
        """
        device = next(self.parameters()).device
        d = self.d_model
        tokens: list[torch.Tensor] = []
        type_ids: list[int] = []

        # ---------- helpers ----------
        def _first(*names):
            for n in names:
                x = batch.get(n, None)
                if x is not None:
                    return x
            return None

        def _batch_size() -> int:
            # Try to infer B from any present tensor
            for k in (
                "cheap_features","cheap",
                "url_input_ids","url_ids","url_bytes","url_tokens",
                "js_input_ids","js_ids","js_bytes","js_tokens",
                "text_input_ids","text_ids","text_bytes","text_tokens",
            ):
                x = batch.get(k, None)
                if isinstance(x, torch.Tensor) and x.ndim >= 1:
                    return x.size(0)
            # DOM-only case
            dom_b = _first("dom_batch_index","dom_batch","dom_graph_index")
            if isinstance(dom_b, torch.Tensor):
                return int(dom_b.max().item()) + 1 if dom_b.numel() else 1
            # Last resort
            y = batch.get("labels", None)
            if isinstance(y, torch.Tensor) and y.ndim >= 1:
                return y.size(0)
            return 1

        B = _batch_size()

        # ---------- [CLS] ----------
        cls = self.cls.view(1, d).expand(B, d).unsqueeze(1)  # (B,1,d)
        tokens.append(cls)
        type_ids.append(0)  # CLS

        # ---------- URL ----------
        if self.use_url:
            url_ids = _first("url_input_ids", "url_ids", "url_bytes", "url_tokens")
            if isinstance(url_ids, torch.Tensor) and url_ids.numel() > 0:
                url_h = self.enc_url(url_ids)  # -> (B, d_*)
                if url_h.size(-1) != d:
                    url_h = self.proj_url(url_h)
                tokens.append(url_h.unsqueeze(1))  # (B,1,d)
                type_ids.append(1)

        # ---------- JS ----------
        if self.use_js:
            js_ids = _first("js_input_ids", "js_ids", "js_bytes", "js_tokens")
            if js_ids is None:
                # sometimes collator may pass JS as a pre-concatenated text field
                js_ids = batch.get("js_text", None)
            if isinstance(js_ids, torch.Tensor) and js_ids.numel() > 0:
                js_h = self.enc_js(js_ids)
                if js_h.size(-1) != d:
                    js_h = self.proj_js(js_h)
                tokens.append(js_h.unsqueeze(1))
                type_ids.append(2)

        # ---------- TEXT ----------
        if self.use_text:
            text_ids = _first("text_input_ids", "text_ids", "text_bytes", "text_tokens", "text")
            if isinstance(text_ids, torch.Tensor) and text_ids.numel() > 0:
                tx_h = self.enc_text(text_ids)
                if tx_h.size(-1) != d:
                    tx_h = self.proj_text(tx_h)
                tokens.append(tx_h.unsqueeze(1))
                type_ids.append(3)

        # ---------- DOM ----------
        if self.use_dom:
            dom_x = _first("dom_node_feats_raw", "dom_node_feats", "dom_nodes")
            dom_e = _first("dom_edge_index", "dom_edges")
            dom_b = _first("dom_batch_index", "dom_batch", "dom_graph_index")
            if (dom_x is not None and isinstance(dom_x, torch.Tensor) and dom_x.numel() > 0 and 
                dom_e is not None and isinstance(dom_e, torch.Tensor) and 
                dom_b is not None and isinstance(dom_b, torch.Tensor)):
                dom_h = self.enc_dom(dom_x, dom_e, dom_b)  # -> (B, d_*)
                if dom_h.size(-1) != d:
                    dom_h = self.proj_dom(dom_h)
                tokens.append(dom_h.unsqueeze(1))
                type_ids.append(4)

        # ---------- CHEAP ----------
        if self.use_cheap:
            cheap = _first("cheap_features", "cheap")
            if isinstance(cheap, torch.Tensor) and cheap.numel() > 0:
                ch_h = self.enc_cheap(cheap)  # -> (B, d_*)
                if ch_h.size(-1) != d:
                    ch_h = self.proj_cheap(ch_h)
                tokens.append(ch_h.unsqueeze(1))
                type_ids.append(5)

        # If no modality made it (extreme edge case), return zeros
        if len(tokens) == 1:  # only CLS present
            logits = self.cls_head(tokens[0].squeeze(1)).squeeze(-1)
            return logits

        # ---------- stack + add type/pos ----------
        X = torch.cat(tokens, dim=1)  # (B, T, d)
        T = X.size(1)

        # type embeddings
        type_ids_t = torch.tensor(type_ids, device=device).view(1, T)
        type_emb = self.type_emb(type_ids_t)  # (1, T, d)

        # pos embeddings (0..T-1)
        pos = torch.arange(T, device=device).view(1, T)
        pos_emb = self.pos_emb(pos)  # (1, T, d)

        X = X + type_emb + pos_emb  # broadcast over batch

        # ---------- transformer ----------
        H = self.encoder(X)  # (B, T, d) because batch_first=True
        cls_out = H[:, 0, :]  # (B, d)

        # ---------- classifier ----------
        logits = self.cls_head(cls_out).squeeze(-1)  # (B,)
        return logits
