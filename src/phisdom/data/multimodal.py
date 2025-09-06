from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json

from .schema import load_jsonl
from .new_heads import PaddedSeqCollator, DomGraphCollator
from .normalize import normalize_dom
from ..features.extractors import (
    extract_js_charseq,
    js_minify_whitespace,
    extract_dom_graph,
    extract_text_title,
    extract_text_visible,
)


@dataclass
class MultiModalDataset:
    path: str
    use_url: bool = True
    use_js: bool = True
    use_text: bool = False
    use_dom: bool = True
    use_cheap: bool = True
    # Adversarial/canonicalization knobs
    js_raw_field: Optional[str] = None  # if set and js_charseq missing, encode from raw
    js_canonicalize: bool = True       # apply minify whitespace before encoding
    html_canonicalize: bool = False    # if raw HTML present, recompute features from normalized HTML
    html_field: str = "html"
    id_field: str = "id"
    label_field: str = "label"

    def __post_init__(self):
        self.rows: List[Dict[str, Any]] = load_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        out: Dict[str, Any] = {"id": r.get(self.id_field, str(idx)), "label": int(r.get(self.label_field, 0))}
        if self.use_url:
            out["url_seq"] = list(r.get("url_charseq") or [])
        if self.use_js:
            seq = r.get("js_charseq")
            if (not seq) and self.js_raw_field is not None:
                raw = r.get(self.js_raw_field)
                if isinstance(raw, list):
                    raw = "\n".join(str(x) for x in raw)
                s = str(raw or "")
                if self.js_canonicalize:
                    s = js_minify_whitespace(s)
                seq = extract_js_charseq(s, max_len=2048)
            out["js_seq"] = list(seq or [])
        html_norm: Optional[str] = None
        if self.use_text or self.use_dom:
            html = r.get(self.html_field)
            if self.html_canonicalize and isinstance(html, str) and html:
                try:
                    html_norm = normalize_dom(html)
                except Exception:
                    html_norm = html
        if self.use_text:
            if html_norm is not None:
                title = extract_text_title(html_norm)
                vis = extract_text_visible(html_norm)
            else:
                title = r.get("text_title") or ""
                vis = r.get("text_visible") or ""
            msg = str(title) + "\n[SEP]\n" + str(vis)
            out["text_seq"] = [ord(c) % 256 for c in msg[:1024]]
        if self.use_dom:
            if html_norm is not None:
                try:
                    g = extract_dom_graph(html_norm)
                except Exception:
                    g = r.get("dom_graph") or {"n": 0, "nodes": [], "edges": []}
            else:
                g = r.get("dom_graph") or {"n": 0, "nodes": [], "edges": []}
            out["dom_graph"] = g
        if self.use_cheap:
            # cheap features are computed in CheapFeaturesDataset; here keep raw row for collator to compute
            out["row"] = r
        return out


class MultiModalCollator:
    def __init__(self, pad_idx: int = 0, cheap_dim: Optional[int] = None):
        self.pad_idx = pad_idx
        self.seq = PaddedSeqCollator(pad_idx=pad_idx)
        self.graph = DomGraphCollator()
        self.cheap_dim = cheap_dim

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch
        from .cheap_features import row_to_features, CHEAP_FEATURES
        out: Dict[str, Any] = {}
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        out["ids"], out["labels"] = ids, labels
        # URL
        if "url_seq" in batch[0]:
            url_batch = [{"id": b["id"], "seq": b.get("url_seq") or [], "label": 0} for b in batch]
            url = self.seq(url_batch)
            out["url_input_ids"] = url["input_ids"]
        # JS
        if "js_seq" in batch[0]:
            js_batch = [{"id": b["id"], "seq": b.get("js_seq") or [], "label": 0} for b in batch]
            js = self.seq(js_batch)
            out["js_input_ids"] = js["input_ids"]
        # TEXT (optional)
        if "text_seq" in batch[0]:
            tx_batch = [{"id": b["id"], "seq": b.get("text_seq") or [], "label": 0} for b in batch]
            tx = self.seq(tx_batch)
            out["text_input_ids"] = tx["input_ids"]
        # DOM
        if "dom_graph" in batch[0]:
            g_batch = [{"id": b["id"], "graph": b.get("dom_graph"), "label": 0} for b in batch]
            g = self.graph(g_batch)
            out["dom_node_feats_raw"] = g["node_feats_raw"]
            out["dom_edge_index"] = g["edge_index"]
            out["dom_batch_index"] = g["batch_index"]
        # CHEAP
        if "row" in batch[0]:
            feats = [row_to_features(b.get("row") or {}, use_features=True) for b in batch]
            X = torch.tensor(feats, dtype=torch.float32)
            out["cheap_features"] = X
        return out
