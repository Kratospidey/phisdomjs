# tests/test_xfusion_smoke.py
from __future__ import annotations
import math
import os
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import pytest

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Your fusion module
from phisdom.models.fusion import CrossModalTransformerFusion


def _rand_token_ids(B: int, Lmax: int, vocab: int = 512, pad_id: int = 0) -> torch.Tensor:
    """Variable-length token ids with 0 padding (shape: B, Lmax)."""
    ids = torch.zeros((B, Lmax), dtype=torch.long)
    for b in range(B):
        L = random.randint(max(2, Lmax // 3), Lmax)  # ensure at least a couple tokens
        if L > 0:
            ids[b, :L] = torch.randint(1, vocab, (L,))  # 1..vocab-1, 0 stays padding
    return ids


def _make_dom_batch(B: int, nodes_per: int = 7, feat_dim: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DOM as a mini-graph pack:
      - node features: (N, F)
      - edge_index: (2, E) (unused by the mean-pool encoder; keep empty)
      - dom_batch: (N,) graph ids [0..B-1]
    """
    N = B * nodes_per
    node_x = torch.randn(N, feat_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    dom_batch = torch.arange(B, dtype=torch.long).repeat_interleave(nodes_per)
    return node_x, edge_index, dom_batch


def _fake_batch(B: int = 16, L: int = 64, cheap_dim: int = 74) -> Dict[str, Any]:
    """Batch shaped like MultiModalCollator output."""
    url_tokens  = _rand_token_ids(B, L)        # (B, L)
    js_tokens   = _rand_token_ids(B, L)        # (B, L)
    text_tokens = _rand_token_ids(B, L)        # (B, L)
    dom_x, dom_ei, dom_b = _make_dom_batch(B, nodes_per=5, feat_dim=48)
    cheap = torch.randn(B, cheap_dim)          # (B, C) ← this is where 74 previously crashed

    labels = (torch.rand(B) > 0.8).float()     # heavily imbalanced like your data

    return {
        "ids": [f"fake_{i}" for i in range(B)],
        "labels": labels,
        # token modalities (IDs → Embedding path in the model)
        "url_input_ids": url_tokens,
        "js_input_ids": js_tokens,
        "text_input_ids": text_tokens,
        # DOM graph pack
        "dom_node_feats_raw": dom_x,
        "dom_edge_index": dom_ei,
        "dom_batch_index": dom_b,
        # cheap features
        "cheap_features": cheap,
    }


def _train_step(model: nn.Module, batch: Dict[str, Any], device: torch.device) -> float:
    model.train()
    crit = nn.BCEWithLogitsLoss()

    labels = batch["labels"].to(device)  # shape: (B,)
    # push rest to device
    feed = {}
    for k, v in batch.items():
        if k == "labels" or k == "ids":
            continue
        feed[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    logits = model(feed)  # expected (B,) or (B,1)
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    assert logits.shape == labels.shape, f"logits {logits.shape} vs labels {labels.shape}"

    loss = crit(logits, labels)
    loss.backward()  # smoke gradients
    return float(loss.item())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_full_modalities(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    random.seed(0)

    B, L, CHEAP = 16, 64, 74  # CHEAP=74 reproduces the previously failing width
    batch = _fake_batch(B=B, L=L, cheap_dim=CHEAP)

    # IMPORTANT:
    #   Pass cheap_dim=None so enc_cheap uses LazyLinear and adapts to CHEAP=74.
    model = CrossModalTransformerFusion(
        d_model=256, n_heads=8, n_layers=2, dropout=0.1,
        use_url=True, use_js=True, use_text=True, use_dom=True, use_cheap=True,
        cheap_dim=None,                 # ← robust to width changes
        token_vocab_size=512, token_embed_dim=64,
    ).to(torch.device(device))

    loss = _train_step(model, batch, torch.device(device))
    assert math.isfinite(loss), "loss should be finite"


def test_forward_no_dom_cpu():
    torch.manual_seed(1)
    random.seed(1)

    B, L, CHEAP = 8, 32, 64
    batch = _fake_batch(B=B, L=L, cheap_dim=CHEAP)

    # Drop DOM keys to mimic no-DOM usage
    for k in ["dom_node_feats_raw", "dom_edge_index", "dom_batch_index"]:
        batch.pop(k, None)

    model = CrossModalTransformerFusion(
        d_model=128, n_heads=4, n_layers=1, dropout=0.1,
        use_url=True, use_js=True, use_text=True, use_dom=False, use_cheap=True,
        cheap_dim=None,  # still robust
        token_vocab_size=512, token_embed_dim=48,
    )

    loss = _train_step(model, batch, torch.device("cpu"))
    assert math.isfinite(loss), "loss should be finite"


def _enc_cheap_is_lazy(model: nn.Module) -> bool:
    """Best-effort introspection: True if the first layer of enc_cheap is LazyLinear."""
    enc = getattr(model, "enc_cheap", None)
    if enc is None:
        return False
    net = getattr(enc, "net", None)
    if isinstance(net, nn.Sequential) and len(net) > 0:
        return isinstance(net[0], nn.LazyLinear)
    # Some implementations may store the first layer directly
    if isinstance(net, nn.LazyLinear):
        return True
    first = getattr(enc, "first", None)
    return isinstance(first, nn.LazyLinear)


def test_width_mismatch_strict_encoder_raises_or_skip():
    """
    Intentional mismatch: model expects cheap_dim=32 but batch has 74 features.
    - If encoder is strict (Linear with fixed in_features), we assert a RuntimeError.
    - If encoder is adaptive (LazyLinear), we SKIP because this failure mode no longer exists.
    """
    B, L, BATCH_CHEAP = 8, 32, 74
    batch = _fake_batch(B=B, L=L, cheap_dim=BATCH_CHEAP)

    model = CrossModalTransformerFusion(
        d_model=128, n_heads=4, n_layers=1, dropout=0.1,
        use_url=True, use_js=True, use_text=True, use_dom=True, use_cheap=True,
        cheap_dim=32,               # <-- deliberately wrong input width
        token_vocab_size=512, token_embed_dim=48,
    )

    if _enc_cheap_is_lazy(model):
        pytest.skip("enc_cheap is LazyLinear; old shape-mismatch failure no longer applies.")

    # Strict Linear path should blow up on matmul with mismatched dims.
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        _train_step(model, batch, torch.device("cpu"))


def test_width_mismatch_adapts_with_lazy_ok():
    """
    Same batch (74 cheap features) but with adaptive encoder (cheap_dim=None).
    This must run end-to-end without errors and produce a finite loss.
    """
    torch.manual_seed(7)

    B, L, BATCH_CHEAP = 8, 32, 74
    batch = _fake_batch(B=B, L=L, cheap_dim=BATCH_CHEAP)

    model = CrossModalTransformerFusion(
        d_model=128, n_heads=4, n_layers=1, dropout=0.1,
        use_url=True, use_js=True, use_text=True, use_dom=True, use_cheap=True,
        cheap_dim=None,             # <-- Lazy/adaptive: learns input width on first forward
        token_vocab_size=512, token_embed_dim=48,
    )

    loss = _train_step(model, batch, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss)), "loss should be finite with LazyLinear adaptation"


if __name__ == "__main__":
    # Allow quick ad-hoc run without pytest
    torch.manual_seed(123)
    random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    B, L, CHEAP = 12, 48, 74
    batch = _fake_batch(B=B, L=L, cheap_dim=CHEAP)
    model = CrossModalTransformerFusion(
        d_model=192, n_heads=6, n_layers=2,
        use_url=True, use_js=True, use_text=True, use_dom=True, use_cheap=True,
        cheap_dim=None,  # adapt to CHEAP width at first forward
    ).to(device)
    loss = _train_step(model, batch, device)
    print(f"[smoke] loss={loss:.4f}")
