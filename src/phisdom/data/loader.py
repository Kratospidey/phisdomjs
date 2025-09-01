from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from phisdom.data.schema import load_jsonl

try:  # soft import for typing/runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class JsonlPhishDataset:
    path: str
    id_field: str = "id"
    html_field: str = "html"
    label_field: str = "label"

    def __post_init__(self):
        rows = load_jsonl(self.path)
        self.rows: List[Dict[str, Any]] = [r for r in rows if self.html_field in r and r[self.html_field]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
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
