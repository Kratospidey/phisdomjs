"""Cross-modal fusion model with optional reversible layers and diagnostics.

This file was fully rewritten to remove corruption introduced during previous
patch attempts. Provides:
 - Separate attention vs feed-forward dropout (attn_dropout / ff_dropout)
 - Optional reversible blocks (memory saving, no attn entropy capture)
 - Gradient checkpointing (disabled when diagnostics capturing attention)
 - Attention entropy (non-reversible path) + modality correlation
 - Simple modality pruning diagnostics based on post-encoder norm contribution
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


def _sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    i = torch.arange(d_model, device=device).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * (i // 2)) / d_model)
    pe = pos / denom
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe


class CrossModalTransformerFusion(nn.Module):
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
        cheap_dim: Optional[int] = None,
        token_vocab_size: int = 512,
        token_embed_dim: int = 48,
        ff_mult: int = 4,
        grad_checkpoint: bool = False,
        attn_dropout: Optional[float] = None,
        ff_dropout: Optional[float] = None,
        reversible: bool = False,
        record_diagnostics: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.ff_mult = ff_mult if ff_mult > 0 else 4
        self.grad_checkpoint = bool(grad_checkpoint)
        self.reversible = bool(reversible)
        self.use = dict(url=use_url, js=use_js, text=use_text, dom=use_dom, cheap=use_cheap)
        self.token_vocab_size = token_vocab_size
        self.token_embed_dim = token_embed_dim
        self.record_diagnostics = record_diagnostics
        self.attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.ff_dropout = ff_dropout if ff_dropout is not None else dropout

        # Diagnostics buffers
        self._last_stream_stats: List[Dict[str, Any]] = []
        self._last_corr = None
        self._last_corr_modalities: List[str] = []
        self._last_attn_entropies = None
        self._last_pooled_stats = None

        # Per-modality embedding + projection
        self._tok_emb = nn.ModuleDict()
        self._tok_proj = nn.ModuleDict()

        # Cheap feature encoder: allow dynamic input dim across calls
        self._cheap_in_dim: Optional[int] = None
        self._cheap_fixed: bool = False
        if use_cheap:
            if cheap_dim is None:
                # Construct on first use based on incoming feature dimension
                self.enc_cheap = None
            else:
                if cheap_dim <= 0:
                    raise ValueError("cheap_dim must be positive")
                self.enc_cheap = nn.Linear(int(cheap_dim), d_model)
                self._cheap_in_dim = int(cheap_dim)
                self._cheap_fixed = True
        else:
            self.enc_cheap = None

        # Type / modality embedding
        self.type_index = {n: i for i, n in enumerate(["url", "js", "text", "dom", "cheap"])}
        self.type_emb = nn.Embedding(len(self.type_index), d_model)

        # Standard encoder layer (non-reversible)
        class _EncLayer(nn.Module):
            def __init__(self, d_model: int, n_heads: int, attn_do: float, ff_do: float, ff_mult: int):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_do, batch_first=True)
                self.linear1 = nn.Linear(d_model, ff_mult * d_model)
                self.linear2 = nn.Linear(ff_mult * d_model, d_model)
                self.ln1 = nn.LayerNorm(d_model)
                self.ln2 = nn.LayerNorm(d_model)
                self.do_attn = nn.Dropout(attn_do)
                self.do_ff = nn.Dropout(ff_do)
            def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None):
                attn_out, w = self.self_attn(x, x, x, key_padding_mask=key_padding, need_weights=True, average_attn_weights=False)
                x = self.ln1(x + self.do_attn(attn_out))
                ff = self.linear2(self.do_ff(F.gelu(self.linear1(x))))
                x = self.ln2(x + self.do_ff(ff))
                return x, w

        if not self.reversible:
            self.layers = nn.ModuleList([
                _EncLayer(d_model, n_heads, self.attn_dropout, self.ff_dropout, self.ff_mult)
                for _ in range(n_layers)
            ])
        else:
            if d_model % 2 != 0:
                raise ValueError("d_model must be even for reversible")
            if (d_model // 2) % n_heads != 0:
                raise ValueError("(d_model/2) must be divisible by n_heads for reversible")
            class _RevBlock(nn.Module):
                def __init__(self, d: int, n_heads: int, attn_do: float, ff_do: float, ff_mult: int):
                    super().__init__()
                    self.attn = nn.MultiheadAttention(d//2, n_heads, dropout=attn_do, batch_first=True)
                    self.ln1 = nn.LayerNorm(d//2)
                    self.ln2 = nn.LayerNorm(d//2)
                    self.ff1 = nn.Linear(d//2, ff_mult * (d//2))
                    self.ff2 = nn.Linear(ff_mult * (d//2), d//2)
                    self.do_attn = nn.Dropout(attn_do)
                    self.do_ff = nn.Dropout(ff_do)
                def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None):
                    a, b = torch.chunk(x, 2, dim=-1)
                    attn_out, _ = self.attn(self.ln1(a), self.ln1(a), self.ln1(a), key_padding_mask=key_padding, need_weights=False)
                    b2 = b + self.do_attn(attn_out)
                    ff = self.ff2(self.do_ff(F.gelu(self.ff1(self.ln2(b2)))))
                    a2 = a + ff
                    return torch.cat([a2, b2], dim=-1)
            self.rev_layers = nn.ModuleList([
                _RevBlock(d_model, n_heads, self.attn_dropout, self.ff_dropout, self.ff_mult)
                for _ in range(n_layers)
            ])

        self.cls = nn.Linear(d_model, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0., std=0.02)

    def _ensure_stream_modules(self, name: str, device: torch.device):
        if name not in self._tok_emb:
            emb = nn.Embedding(self.token_vocab_size, self.token_embed_dim).to(device)
            nn.init.normal_(emb.weight, mean=0., std=0.02)
            self._tok_emb[name] = emb
        if name not in self._tok_proj:
            proj = nn.Linear(self.token_embed_dim, self.d_model).to(device)
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)
            self._tok_proj[name] = proj

    @staticmethod
    def _pick(batch: Dict[str, torch.Tensor], *keys: str) -> Optional[torch.Tensor]:
        for k in keys:
            v = batch.get(k)
            if v is not None:
                return v
        return None

    def _encode_stream(self, name: str, tokens: torch.Tensor, mask: Optional[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_stream_modules(name, device)
        x = self._tok_proj[name](self._tok_emb[name](tokens.to(device)))
        m = torch.ones(tokens.shape, dtype=torch.bool, device=device) if mask is None else mask.to(device).bool()
        x = x + self.type_emb.weight[self.type_index[name]].view(1,1,-1)
        return x, m

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        streams: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        stream_names: List[str] = []
        per_stream_means: List[torch.Tensor] = []
        if self.record_diagnostics:
            self._last_stream_stats = []
            self._last_corr = None
        for name in ("url", "js", "text", "dom"):
            if not self.use.get(name, False):
                continue
            tok = self._pick(batch, f"{name}_tokens", f"{name}_ids", f"{name}_input_ids", name)
            if tok is None:
                continue
            msk = self._pick(batch, f"{name}_mask", f"{name}_attention_mask")
            x, m = self._encode_stream(name, tok, msk, device)
            streams.append(x)
            masks.append(m)
            stream_names.append(name)
            if self.record_diagnostics:
                with torch.no_grad():
                    tok_norms = x.detach().norm(dim=-1)
                    self._last_stream_stats.append({
                        "modality": name,
                        "tokens_total": int(m.sum().item()),
                        "mean_token_norm": float(tok_norms.mean().cpu()) if tok_norms.numel() else 0.0,
                        "std_token_norm": float(tok_norms.std(unbiased=False).cpu()) if tok_norms.numel() else 0.0,
                        "seq_len_mean": float(m.sum(dim=1).float().mean().cpu()) if m.ndim==2 else 0.0,
                    })
                    per_stream_means.append(x.detach().mean(dim=1).mean(dim=0))
        if self.use.get("cheap", False):
            cheap_tok = None  # ensure defined for diagnostics guards
            cf = self._pick(batch, "cheap_features", "cheap")
            if cf is not None:
                # (Re)initialize cheap encoder if needed to match incoming dim
                in_dim = int(cf.size(1)) if isinstance(cf, torch.Tensor) and cf.ndim == 2 else None
                if in_dim is not None:
                    # Determine policy: dynamic adaptation when other streams enabled; strict otherwise
                    other_streams_enabled = any(self.use.get(n, False) for n in ("url","js","text","dom"))
                    if self.enc_cheap is None:
                        if self._cheap_in_dim is None:
                            # No preset width: if cheap_dim was provided at init it set _cheap_in_dim; otherwise set on first use
                            self.enc_cheap = nn.Linear(in_dim, self.d_model).to(device)
                            nn.init.xavier_uniform_(self.enc_cheap.weight)
                            if self.enc_cheap.bias is not None:
                                nn.init.constant_(self.enc_cheap.bias, 0.0)
                            self._cheap_in_dim = in_dim
                        else:
                            # Preset width exists (fixed cheap_dim); create with that and enforce match
                            if in_dim != self._cheap_in_dim:
                                raise RuntimeError(f"cheap_features width {in_dim} != expected {self._cheap_in_dim}")
                            self.enc_cheap = nn.Linear(self._cheap_in_dim, self.d_model).to(device)
                            nn.init.xavier_uniform_(self.enc_cheap.weight)
                            if self.enc_cheap.bias is not None:
                                nn.init.constant_(self.enc_cheap.bias, 0.0)
                    else:
                        # Encoder exists: allow dynamic reinit only if other streams are enabled; otherwise enforce strict
                        if in_dim != self._cheap_in_dim:
                            if self._cheap_fixed:
                                raise RuntimeError(f"cheap_features width changed from {self._cheap_in_dim} to {in_dim}")
                            if other_streams_enabled:
                                # dynamically adapt by reinitializing
                                self.enc_cheap = nn.Linear(in_dim, self.d_model).to(device)
                                nn.init.xavier_uniform_(self.enc_cheap.weight)
                                if self.enc_cheap.bias is not None:
                                    nn.init.constant_(self.enc_cheap.bias, 0.0)
                                self._cheap_in_dim = in_dim
                            else:
                                raise RuntimeError(f"cheap_features width changed from {self._cheap_in_dim} to {in_dim}")
                if self.enc_cheap is not None:
                    cheap_tok = self.enc_cheap(cf.to(device)).unsqueeze(1)
                    cheap_tok = cheap_tok + self.type_emb.weight[self.type_index["cheap"]].view(1,1,-1)
                    streams.append(cheap_tok)
                    masks.append(torch.ones(cheap_tok.size(0),1,dtype=torch.bool,device=device))
                    stream_names.append("cheap")
                if self.record_diagnostics and ('cheap' in stream_names) and (cheap_tok is not None):
                    with torch.no_grad():
                        tn = cheap_tok.detach().norm(dim=-1)
                        self._last_stream_stats.append({
                            "modality": "cheap",
                            "tokens_total": int(tn.numel()),
                            "mean_token_norm": float(tn.mean().cpu()) if tn.numel() else 0.0,
                            "std_token_norm": float(tn.std(unbiased=False).cpu()) if tn.numel() else 0.0,
                            "seq_len_mean": 1.0,
                        })
                        per_stream_means.append(cheap_tok.detach().mean(dim=1).mean(dim=0))
        if not streams:
            B = batch.get("labels", torch.zeros(1, device=device)).shape[0] if isinstance(batch.get("labels"), torch.Tensor) else 1
            return torch.zeros((B,1), device=device)
        X = torch.cat(streams, dim=1)
        M = torch.cat(masks, dim=1)
        if self.record_diagnostics and per_stream_means:
            with torch.no_grad():
                means = torch.stack(per_stream_means, dim=0)
                eps = 1e-8
                norms = means.norm(dim=-1, keepdim=True).clamp_min(eps)
                sim = (means @ means.t()) / (norms * norms.t())
                self._last_corr = sim.detach().cpu()
                for s in self._last_stream_stats:
                    s["corr_index"] = stream_names.index(s["modality"]) if s["modality"] in stream_names else -1
                self._last_corr_modalities = list(stream_names)
        X = X + _sinusoidal_pe(X.size(1), X.size(2), device).unsqueeze(0)
        key_padding = ~M
        attn_entropy_layers: List[Dict[str, Any]] = []
        out = X
        capture = self.record_diagnostics
        if not self.reversible:
            for li, layer in enumerate(self.layers):  # type: ignore[attr-defined]
                if self.grad_checkpoint and self.training and not capture:
                    def _f(x):
                        y, _ = layer(x, key_padding)
                        return y
                    out = _checkpoint(_f, out)
                    aw = None
                else:
                    out, aw = layer(out, key_padding)
                if capture and aw is not None:
                    with torch.no_grad():
                        B, H, Tq, Tk = aw.shape
                        if Tq <= 512 and Tk <= 512:
                            if key_padding is not None:
                                aw = aw.masked_fill(key_padding.view(B,1,1,Tk), 0.0)
                            p = aw.clamp_min(1e-12)
                            Z = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                            p = p / Z
                            ent = -(p * p.log()).sum(dim=-1)
                            attn_entropy_layers.append({
                                "layer": li,
                                "mean_entropy": float(ent.mean().cpu()) if ent.numel() else 0.0,
                                "head_mean_entropy": ent.mean(dim=(0,2)).cpu().tolist() if ent.numel() else [],
                                "T": Tq,
                            })
                        else:
                            attn_entropy_layers.append({"layer": li, "skipped": True, "T": int(Tq)})
        else:
            for blk in self.rev_layers:  # type: ignore[attr-defined]
                if self.grad_checkpoint and self.training and not capture:
                    out = _checkpoint(lambda x: blk(x, key_padding), out)
                else:
                    out = blk(out, key_padding)
        if capture:
            self._last_attn_entropies = attn_entropy_layers
        if out is None or not isinstance(out, torch.Tensor):  # safety guard
            raise RuntimeError("Encoder produced no tensor output")
        pooled = (out * M.unsqueeze(-1)).sum(dim=1) / M.sum(dim=1, keepdim=True).clamp_min(1)
        logits = self.cls(pooled)
        if self.record_diagnostics:
            with torch.no_grad():
                spans = []
                offset = 0
                for name, m in zip(stream_names, masks):
                    L = m.size(1)
                    spans.append((name, offset, offset + L))
                    offset += L
                token_norms = out.detach().norm(dim=-1)
                contrib = []
                for name, a, b in spans:
                    seg = token_norms[:, a:b]
                    if seg.numel():
                        contrib.append({
                            'modality': name,
                            'mean_token_norm_post': float(seg.mean().cpu()),
                            'std_token_norm_post': float(seg.std(unbiased=False).cpu()),
                        })
                post_map = {c['modality']: c for c in contrib}
                for s in self._last_stream_stats:
                    if s['modality'] in post_map:
                        s.update(post_map[s['modality']])
                if contrib:
                    global_mean = sum(c['mean_token_norm_post'] for c in contrib) / max(1,len(contrib))
                    prune = [c['modality'] for c in contrib if c['mean_token_norm_post'] < 0.25 * global_mean]
                else:
                    prune = []
                pooled_norm = pooled.detach().norm(dim=-1)
                self._last_pooled_stats = {
                    'mean_pooled_norm': float(pooled_norm.mean().cpu()) if pooled_norm.numel() else 0.0,
                    'std_pooled_norm': float(pooled_norm.std(unbiased=False).cpu()) if pooled_norm.numel() else 0.0,
                    'attn_entropy_available': bool(self._last_attn_entropies),
                    'prune_candidate_modalities': prune,
                    'reversible': self.reversible,
                }
        return logits
