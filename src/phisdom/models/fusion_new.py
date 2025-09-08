from __future__ import annotations
import math
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional encodings."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    i = torch.arange(d_model, device=device).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * (i // 2)) / d_model)
    pe = pos / denom
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe  # (T, D)


class CrossModalTransformerFusion(nn.Module):
    """
    Minimal, robust fusion:
      - per-modality token -> embed -> project to d_model
      - optional CHEAP features -> [CHEAP] token via (Lazy)Linear
      - concat + type emb + sinusoidal pos emb
      - TransformerEncoder + masked mean pool -> logit
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        *,
        use_url: bool = True,
        use_js: bool = True,
        use_text: bool = True,
        use_dom: bool = True,
        use_cheap: bool = True,
        # if None -> LazyLinear for cheap width auto-adapt
        cheap_dim: Optional[int] = None,
        # shared token embed settings (per-stream embedders will be created lazily)
        token_vocab_size: int = 512,
        token_embed_dim: int = 48,
    ):
        super().__init__()
        self.d_model = d_model
        self.use = dict(url=use_url, js=use_js, text=use_text, dom=use_dom, cheap=use_cheap)
        self.token_vocab_size = token_vocab_size
        self.token_embed_dim = token_embed_dim

        # modality registries (lazy-populated the first time we see that stream)
        self._tok_emb: nn.ModuleDict = nn.ModuleDict()   # name -> nn.Embedding
        self._tok_proj: nn.ModuleDict = nn.ModuleDict()  # name -> nn.Linear(token_embed_dim, d_model)

        # cheap projector (fixed or lazy)
        if use_cheap:
            if cheap_dim is None:
                self.enc_cheap = nn.LazyLinear(d_model)
            else:
                self.enc_cheap = nn.Linear(int(cheap_dim), d_model)
        else:
            self.enc_cheap = None

        # type embeddings (URL/JS/TEXT/DOM/CHEAP)
        self.type_index = {name: i for i, name in enumerate(["url", "js", "text", "dom", "cheap"])}
        self.type_emb = nn.Embedding(len(self.type_index), d_model)

        # core encoder + head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=4 * d_model, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Linear(d_model, 1)

        # Initialize weights (skip LazyLinear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights. Skip LazyLinear - it will initialize itself on first forward."""
        if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0., std=0.02)

    # ---- helpers ----------------------------------------------------------

    def _ensure_stream_modules(self, name: str):
        """Lazily create embedding and projection for a modality."""
        if name not in self._tok_emb:
            self._tok_emb[name] = nn.Embedding(self.token_vocab_size, self.token_embed_dim)
            # Initialize the new embedding
            nn.init.normal_(self._tok_emb[name].weight, mean=0., std=0.02)
        if name not in self._tok_proj:
            self._tok_proj[name] = nn.Linear(self.token_embed_dim, self.d_model)
            # Initialize the new projection
            nn.init.xavier_uniform_(self._tok_proj[name].weight)
            if self._tok_proj[name].bias is not None:
                nn.init.constant_(self._tok_proj[name].bias, 0.0)

    @staticmethod
    def _pick(batch: Dict[str, torch.Tensor], *keys: str) -> Optional[torch.Tensor]:
        """Pick the first available key from batch."""
        for k in keys:
            v = batch.get(k, None)
            if v is not None:
                return v
        return None

    def _encode_stream(
        self,
        name: str,
        tokens: torch.Tensor,           # (B, L) ints
        mask: Optional[torch.Tensor],   # (B, L) bool or 0/1
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token stream: ids -> (B,L,D), (B,L)"""
        self._ensure_stream_modules(name)
        emb = self._tok_emb[name](tokens.to(device))
        x = self._tok_proj[name](emb)
        if mask is None:
            mask = torch.ones(tokens.shape, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device).bool()
        # add type emb
        x = x + self.type_emb.weight[self.type_index[name]].view(1, 1, -1)
        return x, mask

    # ---- forward ----------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Accepts any subset of {url, js, text, dom, cheap}, using these aliases:
          tokens:  {name}_tokens | {name}_ids | {name}_input_ids
          masks:   {name}_mask   | {name}_attention_mask
          cheap:   cheap_features | cheap
        Returns logits of shape (B, 1).
        """
        device = next(self.parameters()).device
        streams: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        # order is stable (URL, JS, TEXT, DOM) so downstream pooling is deterministic
        for name in ("url", "js", "text", "dom"):
            if not self.use.get(name, False):
                continue
            tok = self._pick(batch,
                f"{name}_tokens", f"{name}_ids", f"{name}_input_ids", name  # last fallback: raw key
            )
            if tok is None:
                continue
            msk = self._pick(batch, f"{name}_mask", f"{name}_attention_mask")
            x, m = self._encode_stream(name, tok, msk, device)
            streams.append(x)
            masks.append(m)

        # optional CHEAP -> [CHEAP] token
        if self.use.get("cheap", False) and self.enc_cheap is not None:
            cf = self._pick(batch, "cheap_features", "cheap")
            if cf is not None:
                cf = cf.to(device)
                cheap_tok = self.enc_cheap(cf).unsqueeze(1)  # (B,1,D)
                cheap_tok = cheap_tok + self.type_emb.weight[self.type_index["cheap"]].view(1, 1, -1)
                streams.append(cheap_tok)
                masks.append(torch.ones(cf.size(0), 1, dtype=torch.bool, device=device))

        # if nothing present, return zeros safely
        if not streams:
            B = batch.get("labels", torch.zeros(1, device=device)).shape[0] if isinstance(batch.get("labels"), torch.Tensor) else 1
            return torch.zeros((B, 1), device=device)

        X = torch.cat(streams, dim=1)       # (B, T, D)
        M = torch.cat(masks, dim=1)         # (B, T)

        # positions
        pe = _sinusoidal_pe(X.size(1), X.size(2), device)  # (T,D)
        X = X + pe.unsqueeze(0)

        # transformer + masked mean pool
        out = self.encoder(X, src_key_padding_mask=~M)     # encoder expects True=pad in mask
        denom = M.sum(dim=1, keepdim=1).clamp_min(1)
        pooled = (out * M.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.cls(pooled)                           # (B,1)
        return logits
