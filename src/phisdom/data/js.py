from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from phisdom.data.schema import load_jsonl


def concat_scripts(row: Dict[str, Any], max_scripts: int = 10, max_chars: int = 4000) -> str:
    scripts = row.get("scripts") or []
    texts: List[str] = []
    for s in scripts:
        if len(texts) >= max_scripts:
            break
        t = s.get("text") or ""
        t = t.strip()
        if not t:
            continue
        texts.append(t)
    if not texts:
        return ""
    joined = "\n\n<sep>\n\n".join(texts)
    if len(joined) > max_chars:
        return joined[:max_chars]
    return joined


class JsonlJsDataset:
    def __init__(self, jsonl_path: str, drop_empty: bool = True, max_scripts: int = 10, max_chars: int = 4000):
        self.rows: List[Dict[str, Any]] = []
        for r in load_jsonl(jsonl_path):
            txt = concat_scripts(r, max_scripts=max_scripts, max_chars=max_chars)
            if drop_empty and not txt:
                continue
            self.rows.append({
                "id": r.get("id"),
                "label": int(r.get("label", 0)),
                "text": txt,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]
