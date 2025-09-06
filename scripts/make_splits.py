#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import List
import tldextract

from phisdom.data.schema import iter_jsonl
from phisdom.utils.splits import time_group_split, export_split_indices


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    pct = max(0.0, min(100.0, pct))
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def main():
    parser = argparse.ArgumentParser(description="Create time-aware, group-disjoint splits from JSONL dataset")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", required=True, help="Path to write splits JSON")
    parser.add_argument("--test-after", type=float, default=None, help="Unix timestamp; test has timestamps strictly greater")
    parser.add_argument("--auto-cutoff-percentile", type=float, default=80.0, help="If --test-after not provided, use this percentile of timestamps as cutoff (0-100)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    groups: List[str] = []
    times: List[float] = []
    # Stream over the dataset to avoid loading everything into RAM
    for r in iter_jsonl(args.dataset):
        if "etld1" in r and r["etld1"]:
            g = r["etld1"]
        else:
            tx = tldextract.extract(r.get("url", ""))
            g = ".".join([p for p in [tx.domain, tx.suffix] if p])
        groups.append(g)
        try:
            t = float(r.get("timestamp", 0.0))
        except Exception:
            t = 0.0
        times.append(t)

    cutoff = args.test_after if args.test_after is not None else _percentile(sorted(times), args.auto_cutoff_percentile)

    splits = time_group_split(groups, times, test_after=cutoff, val_frac=args.val_frac, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cutoff": cutoff, **export_split_indices(splits)}, f)


if __name__ == "__main__":
    main()
