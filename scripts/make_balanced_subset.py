#!/usr/bin/env python3
"""
Create a balanced subset JSONL by sampling N positives and N negatives.

Labels are inferred via fields:
- y, label, is_phish, target, class -> consider values {1,"1",True,"phish","phishing"} as positive.

Usage:
  python scripts/make_balanced_subset.py \
    --in data/pages.jsonl --out data/pages_balanced.jsonl --n 6000 --seed 42
"""
from __future__ import annotations
import argparse, io, json, os, random, sys
from typing import Iterable, Tuple


POS_STRINGS = {"1", "phish", "phishing", "malicious", "bad"}


def is_pos(obj: dict) -> bool:
    for k in ("y", "label", "is_phish", "target", "class"):
        if k not in obj:
            continue
        v = obj[k]
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return v == 1
        if isinstance(v, str):
            return v.strip().lower() in POS_STRINGS
    # fallback: look for simple heuristics
    tag = obj.get("tag") or obj.get("source") or ""
    if isinstance(tag, str) and tag.lower().startswith("phish"):
        return True
    return False


def iter_jsonl(path: str) -> Iterable[dict]:
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True, help="Input JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--n", type=int, default=6000, help="Count per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    pos, neg = [], []
    for obj in iter_jsonl(args.input):
        (pos if is_pos(obj) else neg).append(obj)

    if len(pos) < args.n or len(neg) < args.n:
        sys.stderr.write(f"[balanced] Not enough samples: pos={len(pos)} neg={len(neg)} need {args.n}\n")
        sys.exit(2)

    pos_sample = random.sample(pos, args.n)
    neg_sample = random.sample(neg, args.n)
    out = pos_sample + neg_sample
    if args.shuffle:
        random.shuffle(out)
    with io.open(args.out, "w", encoding="utf-8") as w:
        for obj in out:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input": args.input,
        "output": args.out,
        "pos": len(pos),
        "neg": len(neg),
        "sampled_per_class": args.n,
        "total_written": len(out)
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
