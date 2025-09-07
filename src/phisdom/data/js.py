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
        # Build a small index and read lazily to avoid loading entire files
        self._path = jsonl_path
        self._offsets: List[int] = []
        off = 0
        with open(jsonl_path, "rb") as f:
            for line in f:
                self._offsets.append(off)
                off += len(line)
        # Precompute a list of valid indices to skip empty JS quickly
        self._valid: List[int] = []
        for i in range(len(self._offsets)):
            import json as _json
            try:
                import orjson as _orjson  # type: ignore
            except Exception:
                _orjson = None  # type: ignore
            with open(jsonl_path, "rb") as f:
                f.seek(self._offsets[i])
                raw = f.readline()
            r = _orjson.loads(raw) if _orjson is not None else _json.loads(raw)
            txt = concat_scripts(r, max_scripts=max_scripts, max_chars=max_chars)
            if drop_empty and not txt:
                continue
            self._valid.append(i)
        self._max_scripts = max_scripts
        self._max_chars = max_chars

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import json as _json
        try:
            import orjson as _orjson  # type: ignore
        except Exception:
            _orjson = None  # type: ignore
        with open(self._path, "rb") as f:
            f.seek(self._offsets[self._valid[idx]])
            raw = f.readline()
        r = _orjson.loads(raw) if _orjson is not None else _json.loads(raw)
        txt = concat_scripts(r, max_scripts=self._max_scripts, max_chars=self._max_chars)
        return {"id": r.get("id"), "label": int(r.get("label", 0)), "text": txt}
