#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import List, Dict, Tuple
import tldextract

from phisdom.data.schema import iter_jsonl
import numpy as np
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


def _find_cutoff_with_both_classes(times: List[float], labels: List[int], init_cutoff: float, min_pos_test: int = 1) -> float:
    """Search nearby cutoffs to ensure post-cutoff split has both classes and >= min_pos_test positives.

    Strategy: consider unique sorted timestamps; locate the index nearest the initial cutoff,
    then expand a window outward to find a cutoff that yields post>cutoff containing at least
    one negative and at least ``min_pos_test`` positives.
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
        pos_cnt = 0
        neg_found = False
        for t, y in zip(times, labels):
            if t > c:
                if int(y) == 1:
                    pos_cnt += 1
                else:
                    neg_found = True
                if pos_cnt >= max(1, int(min_pos_test)) and neg_found:
                    return float(c)
    return init_cutoff


def _label_aware_val_groups(pre_groups: Dict[str, List[int]], labels: List[int], val_frac: float, seed: int | None, min_pos_val: int = 1) -> Tuple[List[int], List[int]]:
    """Choose validation groups from pre-cutoff groups ensuring both classes if possible, and
    try to include at least ``min_pos_val`` positive examples in validation (if available).

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
    # If we don't yet have enough positives in the provisional selection, add more groups
    def pos_in_groups(groups: List[str]) -> int:
        return sum(1 for g in groups for i in pre_groups.get(g, []) if int(labels[i]) == 1)
    # Prefer groups that add positives first
    if min_pos_val > 0:
        # Candidates not already chosen
        remaining = [g for g in group_keys if g not in val_groups]
        # Sort by descending positive count
        remaining.sort(key=lambda g: sum(1 for i in pre_groups[g] if int(labels[i]) == 1), reverse=True)
        while pos_in_groups(val_groups) < min_pos_val and remaining:
            g = remaining.pop(0)
            # Avoid overshooting target too much, but allow if needed to satisfy min_pos_val
            val_groups.append(g)

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
    parser.add_argument("--min-pos-val", type=int, default=1, help="Minimum number of phish (positives) to include in validation if available")
    parser.add_argument("--min-pos-test", type=int, default=1, help="Minimum number of phish (positives) required in test (post-cutoff)")
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
    # If post-cutoff is one-class or lacks enough positives, try to adjust cutoff nearby
    adj_cutoff = _find_cutoff_with_both_classes(times, labels, cutoff, min_pos_test=max(1, int(args.min_pos_test)))
    if adj_cutoff != cutoff:
        cutoff = adj_cutoff

    # Build pre/post partitions by group, then pick label-aware val groups
    pre_idx = [i for i, t in enumerate(times) if t <= cutoff]
    post_idx = [i for i, t in enumerate(times) if t > cutoff]
    # Group indices for pre-cutoff
    pre_groups: Dict[str, List[int]] = {}
    for i in pre_idx:
        pre_groups.setdefault(groups[i], []).append(i)
    val_idx, train_idx = _label_aware_val_groups(pre_groups, labels, args.val_frac, args.seed, min_pos_val=max(1, int(args.min_pos_val)))
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

    # Final guard: if either val or test has zero positives, warn and fall back to a simple stratified allocation (time-agnostic)
    if (va_p < max(1, int(args.min_pos_val))) or (te_p < max(1, int(args.min_pos_test))):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            print("[SPLITS][WARN] One-class split detected; falling back to stratified shuffle to ensure positives in val/test")
            X = np.arange(len(labels)).reshape(-1, 1)
            y = np.array([int(v) for v in labels])
            # Use test size equal to fraction of post-cutoff to preserve rough ratio
            post_frac = max(0.05, min(0.8, len(post_idx) / max(1, len(labels))))
            val_frac = max(0.01, min(0.49, float(args.val_frac)))
            # First, split off test
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=post_frac, random_state=args.seed)
            train_val_idx, test_idx2 = next(sss1.split(X, y))
            # Then split train_val into train/val
            y_tv = y[train_val_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac / max(1e-9, (1 - post_frac)), random_state=args.seed)
            train_idx2_rel, val_idx2_rel = next(sss2.split(np.arange(len(train_val_idx)).reshape(-1, 1), y_tv))
            train_idx = sorted([train_val_idx[i] for i in train_idx2_rel])
            val_idx = sorted([train_val_idx[i] for i in val_idx2_rel])
            test_idx = sorted(test_idx2)
            tr_n, tr_p, tr_n0 = _class_counts(train_idx, labels)
            va_n, va_p, va_n0 = _class_counts(val_idx, labels)
            te_n, te_p, te_n0 = _class_counts(test_idx, labels)
            print(f"[SPLITS][STRAT] train: n={tr_n} pos={tr_p} neg={tr_n0}")
            print(f"[SPLITS][STRAT] val  : n={va_n} pos={va_p} neg={va_n0}")
            print(f"[SPLITS][STRAT] test : n={te_n} pos={te_p} neg={te_n0}")
        except Exception as e:
            print(f"[SPLITS][WARN] Stratified fallback unavailable ({e}); proceeding with current splits")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cutoff": cutoff, **export_split_indices(splits)}, f)


if __name__ == "__main__":
    main()
