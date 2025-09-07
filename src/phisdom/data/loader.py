from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from phisdom.data.schema import load_jsonl

try:  # soft import for typing/runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


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


@dataclass
class JsonlPhishDataset:
    path: str
    id_field: str = "id"
    html_field: str = "html"
    label_field: str = "label"

    def __post_init__(self):
        # Build lazy index; filter by presence of html field by scanning once to collect eligible offsets
        base = _LazyIndex(self.path)
        self._index: List[int] = []
        for i in range(len(base)):
            r = base.read_row(i)
            if self.html_field in r and r[self.html_field]:
                self._index.append(i)
        self._base = base

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._base.read_row(self._index[idx])
        return {
            "id": r.get(self.id_field, str(idx)),
            "html": r.get(self.html_field, ""),
            "label": int(r.get(self.label_field, 0)),
        }


class MarkupLMDataCollator:
    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        # Late import to avoid hard dependency in tests
        import torch  # type: ignore

        html_list = [b["html"] for b in batch]
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        # MarkupLMProcessor accepts html=[...]
        enc = self.processor(
            html_strings=html_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = labels
        return enc
