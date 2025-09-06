from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from phisdom.data.schema import load_jsonl


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

    def __post_init__(self):
        rows = load_jsonl(self.path)
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
