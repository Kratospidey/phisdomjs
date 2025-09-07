from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

from phisdom.features.extractors import extract_js_charseq, extract_url_charseq
from phisdom.data.cheap_features import row_to_features, CHEAP_FEATURES


# ---- Shared lazy JSONL index ----


class _LazyIndex:
    def __init__(self, path: str):
        self.path = path
        self.offsets: List[int] = []
        off = 0
        with open(path, "rb") as f:
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def read_row(self, i: int) -> Dict[str, Any]:
        import json as _json
        try:
            import orjson as _orjson  # type: ignore
        except Exception:
            _orjson = None  # type: ignore
        with open(self.path, "rb") as f:
            f.seek(self.offsets[i])
            raw = f.readline()
        if _orjson is not None:
            return _orjson.loads(raw)
        return _json.loads(raw)


# ---- URL char-seq dataset ----


@dataclass
class UrlSeqDataset:
    path: str
    seq_field: str = "url_charseq"
    label_field: str = "label"
    id_field: str = "id"
    raw_fields: Tuple[str, ...] = ("url_final", "url_raw", "url")
    max_len: Optional[int] = 256

    def __post_init__(self):
        base = _LazyIndex(self.path)
        keep: List[int] = []
        for i in range(len(base)):
            r = base.read_row(i)
            seq = r.get(self.seq_field)
            if isinstance(seq, list) and len(seq) > 0:
                keep.append(i)
                continue
            # If any raw URL field exists
            for k in self.raw_fields:
                v = r.get(k)
                if isinstance(v, str) and v:
                    keep.append(i)
                    break
        self._base = base
        self._index = keep

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
        seq = r.get(self.seq_field)
        if not (isinstance(seq, list) and len(seq) > 0):
            s: Optional[str] = None
            for k in self.raw_fields:
                v = r.get(k)
                if isinstance(v, str) and v:
                    s = v
                    break
            if s is None:
                enc: List[int] = []
            else:
                ml = self.max_len if self.max_len is not None else len(s)
                enc = extract_url_charseq(s, max_len=int(ml))
        else:
            enc = seq[: self.max_len] if (self.max_len is not None) else list(seq)
        return {"id": r.get(self.id_field, str(idx)), "seq": enc, "label": int(r.get(self.label_field, 0))}


# ---- JS char-seq dataset ----


@dataclass
class JsSeqDataset:
    path: str
    seq_field: str = "js_charseq"
    label_field: str = "label"
    id_field: str = "id"
    raw_field: Optional[str] = None
    raw_candidates: Tuple[str, ...] = ("js_augmented", "js_raw")
    scripts_budget: int = 8192

    def __post_init__(self):
        base = _LazyIndex(self.path)

        def _has_raw(r: Dict[str, Any]) -> bool:
            if self.raw_field:
                raw = r.get(self.raw_field)
                if isinstance(raw, str) and raw:
                    return True
                if isinstance(raw, list) and any(str(x) for x in raw):
                    return True
            for k in self.raw_candidates:
                v = r.get(k)
                if isinstance(v, str) and v:
                    return True
                if isinstance(v, list) and any(str(x) for x in v):
                    return True
            scr = r.get("scripts")
            if isinstance(scr, list):
                for it in scr:
                    try:
                        t = (it.get("text") if isinstance(it, dict) else None)
                        if isinstance(t, str) and t:
                            return True
                    except Exception:
                        continue
            return False

        def ok_row(r: Dict[str, Any]) -> bool:
            seq = r.get(self.seq_field)
            if isinstance(seq, list) and len(seq) > 0:
                return True
            return _has_raw(r)

        keep: List[int] = []
        for i in range(len(base)):
            r = base.read_row(i)
            if ok_row(r):
                keep.append(i)
        self._base = base
        self._index = keep

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
        seq: List[int]
        if isinstance(r.get(self.seq_field), list) and len(r[self.seq_field]) > 0:
            seq = list(r[self.seq_field])
        else:
            s: Optional[str] = None
            if self.raw_field is not None:
                raw = r.get(self.raw_field)
                if isinstance(raw, list):
                    s = "\n".join(str(x) for x in raw)
                elif isinstance(raw, str):
                    s = raw
            if not s:
                for k in self.raw_candidates:
                    v = r.get(k)
                    if isinstance(v, list) and v:
                        s = "\n".join(str(x) for x in v)
                        break
                    if isinstance(v, str) and v:
                        s = v
                        break
            if not s:
                scr = r.get("scripts")
                if isinstance(scr, list):
                    parts: List[str] = []
                    used = 0
                    for it in scr:
                        try:
                            t = (it.get("text") if isinstance(it, dict) else None)
                        except Exception:
                            t = None
                        if not isinstance(t, str) or not t:
                            continue
                        remaining = self.scripts_budget - used
                        if remaining <= 0:
                            break
                        chunk = t[:remaining]
                        parts.append(chunk)
                        used += len(chunk)
                    if parts:
                        s = "\n".join(parts)
            if s:
                try:
                    seq = extract_js_charseq(s)
                except Exception:
                    seq = []
            else:
                seq = []
        return {
            "id": r.get(self.id_field, str(idx)),
            "seq": seq,
            "label": int(r.get(self.label_field, 0)),
        }


# ---- DOM graph dataset ----


@dataclass
class DomGraphDataset:
    path: str
    graph_field: str = "dom_graph"
    label_field: str = "label"
    id_field: str = "id"

    def __post_init__(self):
        base = _LazyIndex(self.path)

        def ok(r: Dict[str, Any]) -> bool:
            g = r.get(self.graph_field)
            if not isinstance(g, dict):
                return False
            nodes = g.get("nodes") or []
            return isinstance(nodes, list) and len(nodes) > 0

        keep: List[int] = []
        for i in range(len(base)):
            r = base.read_row(i)
            if ok(r):
                keep.append(i)
        self._base = base
        self._index = keep

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
        return {
            "id": r.get(self.id_field, str(idx)),
            "graph": r.get(self.graph_field, {}),
            "label": int(r.get(self.label_field, 0)),
        }


# ---- Collators ----


class PaddedSeqCollator:
    def __init__(self, pad_idx: int = 0, max_len: Optional[int] = None):
        self.pad_idx = int(pad_idx)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        seqs: List[List[int]] = [list(b.get("seq") or []) for b in batch]
        if self.max_len is not None:
            seqs = [s[: self.max_len] for s in seqs]
        maxL = max(1, max((len(s) for s in seqs), default=1))
        out = torch.full((len(seqs), maxL), fill_value=self.pad_idx, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxL), dtype=torch.bool)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, : len(s)] = True
        return {"ids": ids, "input_ids": out, "attention_mask": mask, "labels": labels}


class DomGraphCollator:
    def __call__(self, batch: List[Dict[str, Any]]):
        import torch
        ids = [b["id"] for b in batch]
        labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        node_features: List[List[int]] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []
        batch_index: List[int] = []
        node_offset = 0

        for gi, b in enumerate(batch):
            g = b.get("graph") or b.get("dom_graph") or {}
            nodes = g.get("nodes") or []
            edges = g.get("edges") or g.get("edge_index") or []
            n_nodes = len(nodes)
            if n_nodes == 0:
                continue
            for _n in nodes:
                if isinstance(_n, dict):
                    t = int(_n.get("t_hash", 0))
                    c = int(_n.get("c_hash", 0))
                    d = int(_n.get("depth", 0))
                    xb = int(_n.get("xbin", 0))
                elif isinstance(_n, (list, tuple)) and len(_n) >= 4:
                    t, c, d, xb = int(_n[0]), int(_n[1]), int(_n[2]), int(_n[3])
                else:
                    t = c = d = xb = 0
                node_features.append([t, c, d, xb])
                batch_index.append(gi)
            if isinstance(edges, list):
                for e in edges:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        edge_src.append(int(e[0]) + node_offset)
                        edge_dst.append(int(e[1]) + node_offset)
            elif isinstance(edges, dict):
                src = edges.get("src") or []
                dst = edges.get("dst") or []
                for s, d in zip(src, dst):
                    edge_src.append(int(s) + node_offset)
                    edge_dst.append(int(d) + node_offset)
            elif hasattr(edges, "__len__") and len(edges) == 2:
                srcs, dsts = edges[0], edges[1]
                for s, d in zip(srcs, dsts):
                    edge_src.append(int(s) + node_offset)
                    edge_dst.append(int(d) + node_offset)
            node_offset += n_nodes

        nf = torch.tensor(node_features, dtype=torch.long) if node_features else torch.zeros((0, 4), dtype=torch.long)
        ei = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long)
        bidx = torch.tensor(batch_index, dtype=torch.long) if batch_index else torch.zeros((0,), dtype=torch.long)
        return {"ids": ids, "node_feats_raw": nf, "edge_index": ei, "batch_index": bidx, "labels": labels}


# ---- Text dataset (title + visible) ----


def _default_text_alphabet() -> Dict[str, int]:
    raw = os.getenv("PHISDOM_TEXT_ALPHABET")
    if not raw:
        fp = os.getenv("PHISDOM_TEXT_ALPHA_FILE")
        if fp and os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    raw = fh.read()
            except Exception:
                raw = None
    extra = os.getenv("PHISDOM_TEXT_ALPHABET_EXTRA")
    extra_file = os.getenv("PHISDOM_TEXT_ALPHA_EXTRA_FILE")
    if not raw:
        raw = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "-_.:,;!?/'\"()[]{}@#%&*+=<>|\\\n\t"
        )
    if extra:
        raw += extra
    if extra_file and os.path.exists(extra_file):
        try:
            with open(extra_file, "r", encoding="utf-8") as fh:
                raw += fh.read()
        except Exception:
            pass
    seen = set(); chars: List[str] = []
    for ch in raw.replace("\\n", "\n").replace("\\t", "\t"):
        if ch not in seen:
            seen.add(ch); chars.append(ch)
    alpha: Dict[str, int] = {}; i = 2
    for ch in chars:
        alpha[ch] = i; i += 1
    return alpha


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
        base = _LazyIndex(self.path)
        keep: List[int] = []
        for i in range(len(base)):
            r = base.read_row(i)
            title = r.get(self.title_field) or ""
            visible = r.get(self.visible_field) or ""
            if title or visible:
                keep.append(i)
        self._base = base
        self._index = keep

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
        title = r.get(self.title_field) or ""
        visible = r.get(self.visible_field) or ""
        seq = _encode_text_to_charseq((title + "\n" + visible), self.max_len)
        return {"id": r.get(self.id_field, str(idx)), "seq": seq, "label": int(r.get(self.label_field, 0))}


# ---- Cheap features dataset ----


@dataclass
class CheapFeaturesDataset:
    path: str
    label_field: str = "label"
    id_field: str = "id"

    def __post_init__(self):
        self._base = _LazyIndex(self.path)
        self._index = list(range(len(self._base)))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
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
