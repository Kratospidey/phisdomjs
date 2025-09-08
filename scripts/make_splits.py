#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import List, Dict, Tuple
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


def _class_counts(indices: List[int], labels: List[int]) -> Tuple[int, int, int]:
    n = len(indices)
    pos = sum(1 for i in indices if int(labels[i]) == 1)
    neg = n - pos
    return n, pos, neg


def _find_cutoff_with_both_classes(times: List[float], labels: List[int], init_cutoff: float) -> float:
    """Search nearby cutoffs to ensure post-cutoff split has both classes.

    Strategy: consider unique sorted timestamps; locate the index nearest the initial cutoff,
    then expand a window outward to find a cutoff that yields post>cutoff containing at least one
    positive and one negative.
    """
    # Prepare unique sorted timestamps
    uniq = sorted(set(float(t) for t in times))
    if not uniq:
        return init_cutoff
    # Find nearest position to init_cutoff
    import bisect
    j = bisect.bisect_right(uniq, init_cutoff)
    # Generate candidate indices around j (prefer slightly earlier cutoffs to enlarge post split)
    order: List[int] = []
    L = len(uniq)
    lo = j - 1
    hi = j
    while lo >= 0 or hi < L:
        if lo >= 0:
            order.append(lo)
            lo -= 1
        if hi < L:
            order.append(hi)
            hi += 1
        if len(order) > 2000:
            break
    # Fast index lists by cutoff test
    for idx in order:
        c = uniq[idx]
        # Post-split indices are t > c
        any_pos = False
        any_neg = False
        for t, y in zip(times, labels):
            if t > c:
                if int(y) == 1:
                    any_pos = True
                else:
                    any_neg = True
                if any_pos and any_neg:
                    return float(c)
    return init_cutoff


def _label_aware_val_groups(pre_groups: Dict[str, List[int]], labels: List[int], val_frac: float, seed: int | None) -> Tuple[List[int], List[int]]:
    """Choose validation groups from pre-cutoff groups ensuring both classes if possible.

    Returns (val_indices, train_indices) from pre_groups.
    """
    import random as _random
    rng = _random.Random(seed)
    # Compute class mix per group
    group_keys = list(pre_groups.keys())
    rng.shuffle(group_keys)
    info: List[Tuple[str, int, int]] = []  # (group, pos, neg)
    for g in group_keys:
        idxs = pre_groups[g]
        pos = sum(1 for i in idxs if int(labels[i]) == 1)
        neg = len(idxs) - pos
        info.append((g, pos, neg))
    # Partition groups
    both = [g for g, p, n in info if p > 0 and n > 0]
    pos_only = [g for g, p, n in info if p > 0 and n == 0]
    neg_only = [g for g, p, n in info if p == 0 and n > 0]
    target = max(1, int(round(val_frac * len(group_keys))))
    val_groups: List[str] = []
    # Prefer mixed groups (contain both classes)
    for g in both:
        if len(val_groups) >= target:
            break
        val_groups.append(g)
    # Ensure at least one pos and one neg by adding from pos_only/neg_only if needed
    def has_pos(groups: List[str]) -> bool:
        return any(any(int(labels[i]) == 1 for i in pre_groups[g]) for g in groups)
    def has_neg(groups: List[str]) -> bool:
        return any(any(int(labels[i]) == 0 for i in pre_groups[g]) for g in groups)
    rng.shuffle(pos_only)
    rng.shuffle(neg_only)
    while len(val_groups) < target and (pos_only or neg_only):
        # Alternate selection to keep balance
        if not has_pos(val_groups) and pos_only:
            val_groups.append(pos_only.pop())
            continue
        if not has_neg(val_groups) and neg_only:
            val_groups.append(neg_only.pop())
            continue
        # Otherwise fill remaining arbitrarily
        if pos_only:
            val_groups.append(pos_only.pop())
        elif neg_only:
            val_groups.append(neg_only.pop())
    # Build indices
    val_idx: List[int] = []
    train_idx: List[int] = []
    val_set = set(val_groups)
    for g, idxs in pre_groups.items():
        (val_idx if g in val_set else train_idx).extend(idxs)
    return sorted(val_idx), sorted(train_idx)


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
    labels: List[int] = []
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
        try:
            y = int(r.get("label", 0))
        except Exception:
            y = 0
        labels.append(y)

    # Initial cutoff
    cutoff = args.test_after if args.test_after is not None else _percentile(sorted(times), args.auto_cutoff_percentile)
    # If post-cutoff is one-class, try to adjust cutoff nearby
    adj_cutoff = _find_cutoff_with_both_classes(times, labels, cutoff)
    if adj_cutoff != cutoff:
        cutoff = adj_cutoff

    # Build pre/post partitions by group, then pick label-aware val groups
    pre_idx = [i for i, t in enumerate(times) if t <= cutoff]
    post_idx = [i for i, t in enumerate(times) if t > cutoff]
    # Group indices for pre-cutoff
    pre_groups: Dict[str, List[int]] = {}
    for i in pre_idx:
        pre_groups.setdefault(groups[i], []).append(i)
    val_idx, train_idx = _label_aware_val_groups(pre_groups, labels, args.val_frac, args.seed)
    test_idx = sorted(post_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    # Sanity summary
    tr_n, tr_p, tr_n0 = _class_counts(train_idx, labels)
    va_n, va_p, va_n0 = _class_counts(val_idx, labels)
    te_n, te_p, te_n0 = _class_counts(test_idx, labels)
    print(f"[SPLITS] cutoff={cutoff:.1f}")
    print(f"[SPLITS] train: n={tr_n} pos={tr_p} neg={tr_n0}")
    print(f"[SPLITS] val  : n={va_n} pos={va_p} neg={va_n0}")
    print(f"[SPLITS] test : n={te_n} pos={te_p} neg={te_n0}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cutoff": cutoff, **export_split_indices(splits)}, f)


if __name__ == "__main__":
    main()
