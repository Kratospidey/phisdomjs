from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# NOTE: We intentionally avoid a hard torch dependency at import time so that
# modules performing lightweight tasks (e.g. data inspection) do not fail.
# For static type checkers (Pylance/mypy), we still provide a proper base class.
try:  # pragma: no cover - runtime soft import
    import torch  # type: ignore
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:  # Fallback shim when torch isn't installed
    torch = None  # type: ignore
    class _TorchDataset:  # type: ignore
        """Minimal shim replicating torch.utils.data.Dataset interface.
        We deliberately use broad return types so subclasses remain type-compatible.
        """
        def __len__(self) -> int:  # pragma: no cover - trivial
            return 0
        def __getitem__(self, idx: int) -> Any:  # pragma: no cover - trivial
            raise IndexError(idx)

if TYPE_CHECKING:  # Provide a stable symbol for type checkers
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore

DatasetBase = _TorchDataset  # alias used for inheritance (avoids Pylance error)


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
class JsonlPhishDataset(DatasetBase):
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
            self.html_field: r.get(self.html_field, ""),
            self.label_field: int(r.get(self.label_field, 0)),
        }


class MarkupLMDataCollator:
    def __init__(self, processor, max_length: int = 512, html_field: str = "html", label_field: str = "label", max_html_chars: int | None = 800000):
        self.processor = processor
        self.max_length = max_length
        self.html_field = html_field
        self.label_field = label_field
        self.max_html_chars = max_html_chars
        self._seen = 0

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch  # type: ignore
        html_list = []
        for b in batch:
            h = b.get(self.html_field, "")
            # Truncate extremely large HTML blobs if a limit is set (None disables)
            if self.max_html_chars is not None and self.max_html_chars >= 0 and len(h) > self.max_html_chars:
                h = h[: self.max_html_chars]
            html_list.append(h)
        labels = torch.tensor([int(b.get(self.label_field, 0)) for b in batch], dtype=torch.long)
        enc = self.processor(
            html_strings=html_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = labels
        self._seen += len(batch)
        if self._seen % 512 == 0:  # periodic lightweight progress
            print(f"[markup-collator] processed {self._seen} examples", flush=True)
        return enc
