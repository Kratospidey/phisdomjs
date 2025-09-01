from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


@dataclass
class ScriptItem:
    # Either inline or external JS payload
    src: Optional[str]  # URL if external, else None
    inline: bool
    text: Optional[str]  # script source text (may be None if not captured)
    attrs: Dict[str, Any]


@dataclass
class PageRecord:
    id: str
    url: str
    etld1: str
    timestamp: float
    source: str
    label: int  # 1=phish, 0=benign
    html: str
    scripts: List[ScriptItem]
    headers: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure floats/ints are JSON-serializable consistently
        return d


def dumps(obj: Any) -> str:
    if orjson is not None:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False)


def dump_jsonl(records: List[PageRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(dumps(r.to_dict()))
            f.write("\n")


def append_jsonl(record: PageRecord, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(dumps(record.to_dict()))
        f.write("\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if orjson is not None:
                rows.append(orjson.loads(line))
            else:
                rows.append(json.loads(line))
    return rows
