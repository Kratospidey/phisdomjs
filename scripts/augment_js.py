#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import List
from phisdom.data.schema import load_jsonl, dumps
from phisdom.features.extractors import js_minify_whitespace, js_hex_escape_subset, js_split_string_concat


def main():
    ap = argparse.ArgumentParser(description="Phase 6: JS augmentation (minify/obfuscate) and write augmented dataset")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--prob-hex", type=float, default=0.05)
    ap.add_argument("--prob-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.in_jsonl)
    out = []
    for r in rows:
        js_raw = r.get("js_raw") or ""
        if isinstance(js_raw, list):
            js_raw = "\n".join(str(x) for x in js_raw)
        s = str(js_raw)
        s = js_minify_whitespace(s)
        s = js_hex_escape_subset(s, prob=args.prob_hex, seed=args.seed)
        s = js_split_string_concat(s, prob=args.prob_split, seed=args.seed)
        r2 = dict(r)
        r2["js_augmented"] = s[:8192]
        out.append(r2)
    # Write generic dicts using the JSON dumper (not PageRecord-aware)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in out:
            f.write(dumps(row))
            f.write("\n")


if __name__ == "__main__":
    main()
