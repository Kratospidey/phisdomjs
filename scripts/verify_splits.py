#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple


def counts(path: str) -> Tuple[int, int, int]:
    n = pos = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            try:
                y = int(obj.get("label", 0))
            except Exception:
                y = 0
            if y == 1:
                pos += 1
    neg = n - pos
    return n, pos, neg


def main():
    ap = argparse.ArgumentParser(description="Fail if any split is one-class or empty")
    ap.add_argument("--train", default="data/pages_train.jsonl")
    ap.add_argument("--val", default="data/pages_val.jsonl")
    ap.add_argument("--test", default="data/pages_test.jsonl")
    ap.add_argument("--min-pos", type=int, default=5, help="Minimum positives required in val/test")
    args = ap.parse_args()

    ok = True
    for split, p in [("train", args.train), ("val", args.val), ("test", args.test)]:
        n, pos, neg = counts(p)
        print(f"[VERIFY] {split}: total={n} pos={pos} neg={neg}")
        if n == 0:
            print(f"[VERIFY][ERROR] {split} is empty: {p}")
            ok = False
        if pos == 0 or neg == 0:
            # Require at least some positives in val/test; train may be imbalanced but warn
            if split in ("val", "test"):
                print(f"[VERIFY][ERROR] {split} has one class only (pos={pos}, neg={neg})")
                ok = False
            else:
                print(f"[VERIFY][WARN] {split} is one-class (pos={pos}, neg={neg})")
        if split in ("val", "test") and pos < args.min_pos:
            print(f"[VERIFY][ERROR] {split} has too few positives (pos={pos} < {args.min_pos})")
            ok = False
    if not ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
