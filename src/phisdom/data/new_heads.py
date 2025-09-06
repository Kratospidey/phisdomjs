from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from phisdom.data.schema import load_jsonl
from phisdom.features.extractors import extract_js_charseq
from phisdom.data.cheap_features import row_to_features, CHEAP_FEATURES


# ---- Datasets ----


@dataclass
class UrlSeqDataset:
    path: str
    seq_field: str = "url_charseq"
    label_field: str = "label"
    id_field: str = "id"

    def __post_init__(self):
        rows = load_jsonl(self.path)
        # keep rows with non-empty sequences
        self.rows: List[Dict[str, Any]] = [r for r in rows if isinstance(r.get(self.seq_field), list) and len(r[self.seq_field]) > 0]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "id": r.get(self.id_field, str(idx)),
            "seq": r.get(self.seq_field) or [],
            "label": int(r.get(self.label_field, 0)),
        }


@dataclass
class JsSeqDataset:
    path: str
    seq_field: str = "js_charseq"
    label_field: str = "label"
    id_field: str = "id"
    raw_field: Optional[str] = None  # e.g., "js_augmented" or "js_raw"; if provided and seq missing, encode on-the-fly

    def __post_init__(self):
        rows = load_jsonl(self.path)
        def ok_row(r: Dict[str, Any]) -> bool:
            if isinstance(r.get(self.seq_field), list) and len(r[self.seq_field]) > 0:
                return True
            if self.raw_field is not None and isinstance(r.get(self.raw_field), (str, list)):
                return True
            return False
        self.rows: List[Dict[str, Any]] = [r for r in rows if ok_row(r)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        seq: List[int]
        if isinstance(r.get(self.seq_field), list) and len(r[self.seq_field]) > 0:
            seq = list(r.get(self.seq_field) or [])
        elif self.raw_field is not None:
            raw = r.get(self.raw_field)
            if isinstance(raw, list):
                raw = "\n".join(str(x) for x in raw)
            seq = extract_js_charseq(str(raw or ""), max_len=2048)
        else:
            seq = []
        return {
            "id": r.get(self.id_field, str(idx)),
            "seq": seq,
            "label": int(r.get(self.label_field, 0)),
        }


@dataclass
class DomGraphDataset:
    path: str
    graph_field: str = "dom_graph"
    label_field: str = "label"
    id_field: str = "id"

    def __post_init__(self):
        rows = load_jsonl(self.path)
        def ok(g: Any) -> bool:
            return isinstance(g, dict) and isinstance(g.get("nodes"), list) and len(g.get("nodes", [])) > 0
        self.rows: List[Dict[str, Any]] = [r for r in rows if ok(r.get(self.graph_field))]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "id": r.get(self.id_field, str(idx)),
            "graph": r.get(self.graph_field) or {"n": 0, "nodes": [], "edges": []},
            "label": int(r.get(self.label_field, 0)),
        }


# ---- Collators ----


class PaddedSeqCollator:
    def __init__(self, pad_idx: int = 0, max_len: Optional[int] = None):
        self.pad_idx = int(pad_idx)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch  # late import
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        seqs: List[List[int]] = [list(b.get("seq") or []) for b in batch]
        if self.max_len is not None:
            seqs = [s[: self.max_len] for s in seqs]
        maxL = max(1, max((len(s) for s in seqs), default=1))
        out = torch.full((len(seqs), maxL), fill_value=self.pad_idx, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxL), dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = len(s)
            if L > 0:
                out[i, :L] = torch.tensor(s, dtype=torch.long)
                mask[i, :L] = True
        return {"ids": ids, "input_ids": out, "attention_mask": mask, "labels": labels}


class DomGraphCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch  # late import
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        # Build batched graph tensors with index offsets
        node_features: List[List[int]] = []  # list of [t_hash, c_hash, depth, xbin]
        edge_src: List[int] = []
        edge_dst: List[int] = []
        batch_index: List[int] = []
        node_offset = 0
        graph_ptr: List[Tuple[int, int]] = []  # (start_idx, n_nodes)

        for gi, b in enumerate(batch):
            g = b.get("graph") or {}
            nodes = g.get("nodes") or []
            edges = g.get("edges") or []
            n = len(nodes)
            graph_ptr.append((node_offset, n))
            for i in range(n):
                nd = nodes[i] or {}
                t = int(nd.get("t", 0))
                c = int(nd.get("c", 0))
                d = int(nd.get("d", 0))
                x = int(nd.get("x", 0))
                node_features.append([t, c, d, x])
                batch_index.append(gi)
            for (u, v) in edges:
                try:
                    edge_src.append(int(u) + node_offset)
                    edge_dst.append(int(v) + node_offset)
                except Exception:
                    continue
            node_offset += n

        if node_features:
            nf = torch.tensor(node_features, dtype=torch.long)
        else:
            nf = torch.zeros((0, 4), dtype=torch.long)
        ei = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long)
        bidx = torch.tensor(batch_index, dtype=torch.long) if batch_index else torch.zeros((0,), dtype=torch.long)
        return {"ids": ids, "node_feats_raw": nf, "edge_index": ei, "batch_index": bidx, "labels": labels}


# ---- Text dataset (title + visible) ----


def _default_text_alphabet() -> Dict[str, int]:
    # 0=PAD, 1=UNK; letters (case-sensitive), digits, space, basic punctuation
    alphabet = {}
    idx = 2
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ":
        alphabet[ch] = idx; idx += 1
    for ch in "-_.:,;!?/'\"()[]{}@#%&*+=<>|\\\n\t":
        if ch not in alphabet:
            alphabet[ch] = idx; idx += 1
    return alphabet


_TEXT_ALPHA = _default_text_alphabet()


def _encode_text_to_charseq(s: str, max_len: Optional[int] = None) -> List[int]:
    if not isinstance(s, str):
        s = ""
    ids: List[int] = []
    for ch in s:
        ids.append(_TEXT_ALPHA.get(ch, 1))  # 1 = UNK
        if max_len is not None and len(ids) >= max_len:
            break
    return ids


@dataclass
class TextSeqDataset:
    path: str
    title_field: str = "text_title"
    visible_field: str = "text_visible"
    label_field: str = "label"
    id_field: str = "id"
    max_len: Optional[int] = None

    def __post_init__(self):
        rows = load_jsonl(self.path)
        self.rows: List[Dict[str, Any]] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        title = r.get(self.title_field) or ""
        visible = r.get(self.visible_field) or ""
        # simple concat with separator
        text = str(title) + "\n[SEP]\n" + str(visible)
        seq = _encode_text_to_charseq(text, self.max_len)
        return {
            "id": r.get(self.id_field, str(idx)),
            "seq": seq,
            "label": int(r.get(self.label_field, 0)),
        }


# ---- Cheap features dataset ----


@dataclass
class CheapFeaturesDataset:
    path: str
    label_field: str = "label"
    id_field: str = "id"

    def __post_init__(self):
        self.rows: List[Dict[str, Any]] = load_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        feats = row_to_features(r, use_features=True)
        return {
            "id": r.get(self.id_field, str(idx)),
            "feats": feats,
            "label": int(r.get(self.label_field, 0)),
        }


class CheapFeaturesCollator:
    def __call__(self, batch: List[Dict[str, Any]]):
        import torch
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        feats = [b.get("feats") or [0.0] * len(CHEAP_FEATURES) for b in batch]
        X = torch.tensor(feats, dtype=torch.float32)
        return {"ids": ids, "features": X, "labels": labels}
