#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import List
from phisdom.data.schema import dumps
from phisdom.features.extractors import js_minify_whitespace, js_hex_escape_subset, js_split_string_concat


def main():
    ap = argparse.ArgumentParser(description="Phase 6: JS augmentation (minify/obfuscate) and write augmented dataset")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--prob-hex", type=float, default=0.05)
    ap.add_argument("--prob-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Stream read input and stream write output to avoid large memory usage
    import json as _json
    try:
        import orjson as _orjson  # type: ignore
    except Exception:
        _orjson = None  # type: ignore

    with open(args.in_jsonl, "rb") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw:
                continue
            try:
                row = _orjson.loads(raw) if _orjson is not None else _json.loads(raw)
            except Exception:
                continue
            js_raw = row.get("js_raw") or ""
            if isinstance(js_raw, list):
                js_raw = "\n".join(str(x) for x in js_raw)
            s = str(js_raw)
            s = js_minify_whitespace(s)
            s = js_hex_escape_subset(s, prob=args.prob_hex, seed=args.seed)
            s = js_split_string_concat(s, prob=args.prob_split, seed=args.seed)
            row["js_augmented"] = s[:8192]
            fout.write(dumps(row))
            fout.write("\n")


if __name__ == "__main__":
    main()
