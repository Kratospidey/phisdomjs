#!/usr/bin/env python3
"""
Repair/dedupe IDs in JSONL datasets.

For each input JSONL:
- Ensure each record has an 'id' field; if missing/empty, derive from url_final/url.
- Deduplicate by 'id' (keep first by default).
- Write back in-place unless --out is provided.

Usage:
  python scripts/repair_ids.py --in data/pages.jsonl [data/pages_*.jsonl] [--out-suffix .fixed]
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys
from typing import Iterable, Tuple


def norm_url(obj: dict) -> str | None:
    for k in ("url_final", "url", "href", "u"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v.strip()
    return None


def derive_id(obj: dict) -> str | None:
    url = norm_url(obj)
    if not url:
        return None
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return h


def iter_jsonl(path: str) -> Iterable[Tuple[int, dict]]:
    with io.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                sys.stderr.write(f"[repair_ids][WARN] Bad JSON at {path}:{i}\n")
                continue
            yield i, obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="Input JSONL file(s)")
    ap.add_argument("--out-suffix", default="", help="If set, write to <in><suffix> instead of in-place")
    ap.add_argument("--keep", choices=["first", "last"], default="first",
                    help="On duplicate IDs, keep first or last record (default: first)")
    args = ap.parse_args()

    for path in args.inputs:
        ids_seen = set()
        out_records = []
        kept = 0
        deduped = 0
        added = 0
        missing_id = 0
        for _, obj in iter_jsonl(path):
            rid = obj.get("id")
            if not rid:
                rid = derive_id(obj)
                if rid:
                    obj["id"] = rid
                    added += 1
                else:
                    missing_id += 1
                    # skip records without any way to derive id
                    continue
            if rid in ids_seen:
                deduped += 1
                if args.keep == "last":
                    # replace previous occurrence
                    for i in range(len(out_records) - 1, -1, -1):
                        if out_records[i].get("id") == rid:
                            out_records[i] = obj
                            break
                continue
            ids_seen.add(rid)
            out_records.append(obj)
            kept += 1

        out_path = path + args.out_suffix if args.out_suffix else path
        with io.open(out_path, "w", encoding="utf-8") as w:
            for obj in out_records:
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.write(
            json.dumps({
                "file": path,
                "wrote": out_path,
                "kept": kept,
                "deduped": deduped,
                "id_added": added,
                "missing_id_skipped": missing_id,
            }) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
