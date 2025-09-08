#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import List, Dict, Tuple
import tldextract

from phisdom.data.schema import iter_jsonl
import numpy as np
from phisdom.utils.splits import time_group_split, export_split_indices

# Offline tldextract initialization to avoid network PSL refresh stalls
try:  # pragma: no cover - defensive init
    # Empty tuple disables network fetch (uses bundled snapshot)
    _EXT = tldextract.TLDExtract(suffix_list_urls=())  # type: ignore[arg-type]
except Exception:  # pragma: no cover
    _EXT = None


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


def _counts(idxs: List[int], labels: List[int]) -> Tuple[int, int, int, float]:
    """Return (n, pos, neg, pos_ratio)."""
    n = len(idxs)
    p = sum(1 for i in idxs if int(labels[i]) == 1)
    n0 = n - p
    r = (p / n) if n > 0 else 0.0
    return n, p, n0, r


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


def _search_cutoff_for_target_ratio(
    times: List[float],
    labels: List[int],
    init_cutoff: float,
    target_ratio: float | None,
    tol: float,
    min_pos: int,
    min_total: int,
) -> float:
    """Scan unique time cutoffs near init_cutoff to make test (t>cutoff) pos-ratio close to target.

    Respects min_pos and min_total in test. Returns best cutoff (or init_cutoff if none better).
    If target_ratio is None, returns init_cutoff.
    """
    if target_ratio is None:
        return init_cutoff

    uniq = sorted(set(float(t) for t in times))
    if not uniq:
        return init_cutoff

    import bisect

    # Indices sorted by time
    order = sorted(range(len(times)), key=lambda i: times[i])
    sorted_times = [times[i] for i in order]
    ys = [int(labels[i]) for i in order]

    # Precompute suffix totals and positives
    L = len(order)
    suf_tot = [0] * (L + 1)
    suf_pos = [0] * (L + 1)
    for k in range(L - 1, -1, -1):
        suf_tot[k] = suf_tot[k + 1] + 1
        suf_pos[k] = suf_pos[k + 1] + ys[k]

    def test_counts_for_cut(c: float) -> Tuple[int, int, int, float]:
        # first index strictly greater than c
        k = bisect.bisect_right(sorted_times, c)
        n = suf_tot[k]
        p = suf_pos[k]
        n0 = n - p
        r = (p / n) if n > 0 else 0.0
        return n, p, n0, r

    # Find nearest position to init_cutoff
    j = bisect.bisect_right(uniq, init_cutoff)
    cand_idx: List[int] = []
    lo, hi = j - 1, j
    U = len(uniq)
    while lo >= 0 or hi < U:
        if lo >= 0:
            cand_idx.append(lo)
            lo -= 1
        if hi < U:
            cand_idx.append(hi)
            hi += 1
        if len(cand_idx) > 2000:  # tightened for faster search
            break

    best_c = init_cutoff
    _, _, _, r0 = test_counts_for_cut(init_cutoff)
    best_err = abs(r0 - target_ratio)

    for idx in cand_idx:
        c = uniq[idx]
        n, p, _, r = test_counts_for_cut(c)
        if n < max(1, int(min_total)) or p < max(1, int(min_pos)):
            continue
        err = abs(r - target_ratio)
        # prefer meeting tol; if both within tol, prefer closer to init_cutoff; else smaller error
        if (best_err > tol and err < best_err) or (err <= tol and best_err > tol) or (err < best_err):
            best_err = err
            best_c = c
            if err <= tol:
                break

    return best_c


def _label_aware_val_groups(
    pre_groups: Dict[str, List[int]],
    labels: List[int],
    val_frac: float,
    seed: int | None,
    min_pos_val: int = 1,
    min_pos_train: int = 1,
) -> Tuple[List[int], List[int]]:
    """Choose validation groups from pre-cutoff groups ensuring both classes if possible.
    Also try to include at least ``min_pos_val`` positives in validation while reserving
    at least ``min_pos_train`` positives for the remaining train split when available.

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
    pre_pos_total = sum(p for _, p, _ in info)
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
    # Reserve some positives for train: desired_val_pos is capped so that train keeps min_pos_train if available
    desired_val_pos = min(max(0, int(min_pos_val)), max(0, pre_pos_total - max(1, int(min_pos_train))))
    if desired_val_pos > 0:
        # Candidates not already chosen
        remaining = [g for g in group_keys if g not in val_groups]
        # Sort by descending positive count
        remaining.sort(key=lambda g: sum(1 for i in pre_groups[g] if int(labels[i]) == 1), reverse=True)
        while pos_in_groups(val_groups) < desired_val_pos and remaining:
            g = remaining.pop(0)
            # Skip groups with zero positives (won't help)
            if sum(1 for i in pre_groups[g] if int(labels[i]) == 1) == 0:
                continue
            val_groups.append(g)

    # Build indices
    val_idx: List[int] = []
    train_idx: List[int] = []
    val_set = set(val_groups)
    for g, idxs in pre_groups.items():
        (val_idx if g in val_set else train_idx).extend(idxs)
    # Last-ditch: if train has zero positives but pre has some, move the smallest-positive val group back to train
    def count_pos(idxs: List[int]) -> int:
        return sum(1 for i in idxs if int(labels[i]) == 1)

    if pre_pos_total > 0 and count_pos(train_idx) == 0 and count_pos(val_idx) > 0:
        val_pos_groups: List[Tuple[str, int]] = []
        for g in val_groups:
            p = sum(1 for i in pre_groups[g] if int(labels[i]) == 1)
            if p > 0:
                val_pos_groups.append((g, p))
        if val_pos_groups:
            val_pos_groups.sort(key=lambda x: x[1])
            g_move = val_pos_groups[0][0]
            val_groups.remove(g_move)
            # rebuild indices
            val_idx = []
            train_idx = []
            val_set = set(val_groups)
            for g, idxs in pre_groups.items():
                (val_idx if g in val_set else train_idx).extend(idxs)

    return sorted(val_idx), sorted(train_idx)


def _label_aware_val_groups_ratio(
    pre_groups: Dict[str, List[int]],
    labels: List[int],
    val_frac: float,
    seed: int | None,
    target_ratio_val: float | None,
    ratio_tol: float,
    min_pos_val: int,
    refine_topk: int = 0,
) -> Tuple[List[int], List[int]]:
    """Fast O(G log G) greedy validation group selection.

    - If target_ratio_val is provided: greedily add groups minimizing |ratio-target| (+ small size penalty).
    - If no target: prefer mixed groups -> positives-only -> negatives-only.
    Ensures at least min_pos_val positives if available by optionally adding more positive-rich groups.
    Returns (val_indices, train_indices).
    """
    import random as _random
    rng = _random.Random(seed)

    groups = list(pre_groups.keys())
    rng.shuffle(groups)

    # Precompute stats
    gN: Dict[str, int] = {}
    gP: Dict[str, int] = {}
    gR: Dict[str, float] = {}
    totalN = 0
    totalP = 0
    for g in groups:
        idxs = pre_groups[g]
        n = len(idxs)
        p = sum(1 for i in idxs if int(labels[i]) == 1)
        r = (p / n) if n > 0 else 0.0
        gN[g] = n
        gP[g] = p
        gR[g] = r
        totalN += n
        totalP += p

    target_val_N = max(1, int(round(val_frac * max(1, totalN))))

    if target_ratio_val is None:
        mixed = [g for g in groups if 0 < gP[g] < gN[g]]
        pos_only = [g for g in groups if gP[g] == gN[g] and gP[g] > 0]
        neg_only = [g for g in groups if gP[g] == 0]
        rng.shuffle(mixed)
        rng.shuffle(pos_only)
        rng.shuffle(neg_only)
        sel: List[str] = []
        curN = 0
        for bucket in (mixed, pos_only, neg_only):
            for g in bucket:
                if curN >= target_val_N:
                    break
                sel.append(g)
                curN += gN[g]
            if curN >= target_val_N:
                break
        # Ensure min positives if we can (add positive-rich remaining groups)
        if sum(gP[g] for g in sel) < min_pos_val and totalP >= min_pos_val:
            remaining = [g for g in groups if g not in sel]
            remaining.sort(key=lambda g: gP[g], reverse=True)
            curP = sum(gP[g] for g in sel)
            for g in remaining:
                if curP >= min_pos_val or curN >= target_val_N:
                    break
                if gP[g] == 0:
                    continue
                sel.append(g)
                curN += gN[g]
                curP += gP[g]
    else:
        remaining = set(groups)
        sel = []  # type: ignore[assignment]
        curN = 0
        curP = 0

        if remaining:
            g0 = min(remaining, key=lambda g: abs(gR[g] - target_ratio_val))  # type: ignore[arg-type]
            sel.append(g0)
            remaining.remove(g0)
            curN += gN[g0]
            curP += gP[g0]

        while curN < target_val_N and remaining:
            def score(g: str) -> float:
                newN = curN + gN[g]
                newP = curP + gP[g]
                newR = (newP / newN) if newN > 0 else 0.0
                size_pen = abs(newN - target_val_N) / max(1, target_val_N)
                return abs(newR - target_ratio_val) + 0.05 * size_pen

            gbest = min(remaining, key=score)
            sel.append(gbest)
            remaining.remove(gbest)
            curN += gN[gbest]
            curP += gP[gbest]

        # Ensure minimum positives if possible
        if curP < min_pos_val and totalP >= min_pos_val:
            pos_sorted = sorted(list(remaining), key=lambda g: gP[g], reverse=True)
            for g in pos_sorted:
                if curP >= min_pos_val or curN >= target_val_N:
                    break
                if gP[g] == 0:
                    break
                sel.append(g)
                curN += gN[g]
                curP += gP[g]

        # Optional bounded refinement (swap among top-K groups) to reduce ratio error
        if refine_topk > 0 and target_ratio_val is not None and abs((curP / curN) - target_ratio_val) > ratio_tol:
            # Prepare candidate lists
            val_groups = sel
            train_groups = [g for g in groups if g not in sel]
            # Rank by contribution to ratio error
            def contrib_err(g: str, in_val: bool) -> float:
                if in_val:
                    # Error if removed
                    newN = curN - gN[g]
                    newP = curP - gP[g]
                else:
                    newN = curN + gN[g]
                    newP = curP + gP[g]
                if newN <= 0:
                    return float('inf')
                return abs((newP / newN) - target_ratio_val)

            top_val = sorted(val_groups, key=lambda g: contrib_err(g, True), reverse=True)[:refine_topk]
            top_train = sorted(train_groups, key=lambda g: contrib_err(g, False), reverse=True)[:refine_topk]
            improved = True
            iters = 0
            while improved and iters < refine_topk:
                improved = False
                iters += 1
                base_err = abs((curP / curN) - target_ratio_val)
                best_swap: Tuple[str, str, float] | None = None
                for gv in top_val:
                    for gt in top_train:
                        newN = curN - gN[gv] + gN[gt]
                        newP = curP - gP[gv] + gP[gt]
                        if newP < min_pos_val:
                            continue
                        newR = (newP / newN) if newN > 0 else 0.0
                        new_err = abs(newR - target_ratio_val)
                        if new_err + 1e-12 < base_err:
                            if best_swap is None or new_err < best_swap[2]:
                                best_swap = (gv, gt, new_err)
                if best_swap is not None:
                    gv, gt, _ = best_swap
                    sel.remove(gv)
                    sel.append(gt)
                    curN = curN - gN[gv] + gN[gt]
                    curP = curP - gP[gv] + gP[gt]
                    # Update candidate lists (simple approach)
                    val_groups = sel
                    train_groups = [g for g in groups if g not in sel]
                    top_val = sorted(val_groups, key=lambda g: contrib_err(g, True), reverse=True)[:refine_topk]
                    top_train = sorted(train_groups, key=lambda g: contrib_err(g, False), reverse=True)[:refine_topk]
                    improved = True
                else:
                    break

    vset = set(sel)
    val_idx: List[int] = []
    train_idx: List[int] = []
    for g, idxs in pre_groups.items():
        (val_idx if g in vset else train_idx).extend(idxs)
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
    # Ratio-targeted options
    parser.add_argument("--target-pos-ratio-train", type=float, default=None, help="Desired positive ratio in TRAIN (0-1). If None, don't enforce.")
    parser.add_argument("--target-pos-ratio-val", type=float, default=None, help="Desired positive ratio in VAL (0-1). If None, don't enforce.")
    parser.add_argument("--target-pos-ratio-test", type=float, default=None, help="Desired positive ratio in TEST (0-1). If None, don't enforce.")
    parser.add_argument("--ratio-tol", type=float, default=0.05, help="Allowed absolute deviation from target ratios (0-0.5).")
    parser.add_argument("--balance-splits", action="store_true", help="Convenience: set all three targets to 0.5 unless explicitly provided.")
    parser.add_argument("--min-total-test", type=int, default=500, help="Minimum total examples required in TEST when searching cutoffs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--refine-topk", type=int, default=0, help="If >0, attempt up to K limited group swaps to reduce val ratio error (bounded refinement)")
    parser.add_argument("--fast", action="store_true", help="(Deprecated) Fast mode is now default; kept for CLI compatibility")
    args = parser.parse_args()

    # Normalize ratio flags
    def _clamp_ratio(x: float | None) -> float | None:
        if x is None:
            return None
        try:
            return float(max(0.0, min(1.0, x)))
        except Exception:
            return None

    if args.balance_splits:
        if args.target_pos_ratio_train is None:
            args.target_pos_ratio_train = 0.5
        if args.target_pos_ratio_val is None:
            args.target_pos_ratio_val = 0.5
        if args.target_pos_ratio_test is None:
            args.target_pos_ratio_test = 0.5

    args.target_pos_ratio_train = _clamp_ratio(args.target_pos_ratio_train)
    args.target_pos_ratio_val = _clamp_ratio(args.target_pos_ratio_val)
    args.target_pos_ratio_test = _clamp_ratio(args.target_pos_ratio_test)
    try:
        args.ratio_tol = float(max(0.0, min(0.5, args.ratio_tol)))
    except Exception:
        args.ratio_tol = 0.05

    groups: List[str] = []
    times: List[float] = []
    labels: List[int] = []
    # Stream over the dataset to avoid loading everything into RAM
    for r in iter_jsonl(args.dataset):
        if "etld1" in r and r["etld1"]:
            g = r["etld1"]
        else:
            # Use offline extractor if available
            if _EXT is not None:
                tx = _EXT(r.get("url", ""))
            else:  # fallback
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

    # Check if 50/50 balance is possible given the data
    total_examples = len(labels)
    total_positives = sum(1 for y in labels if int(y) == 1)
    overall_pos_ratio = total_positives / max(1, total_examples)
    
    # If balance-splits requested but impossible, adjust targets and warn
    if args.balance_splits and overall_pos_ratio < 0.4:
        print(f"[SPLITS][WARN] Dataset has only {overall_pos_ratio:.1%} positives ({total_positives}/{total_examples})")
        print(f"[SPLITS][WARN] 50/50 balance impossible; adjusting targets to {overall_pos_ratio:.1%}")
        args.target_pos_ratio_train = overall_pos_ratio
        args.target_pos_ratio_val = overall_pos_ratio  
        args.target_pos_ratio_test = overall_pos_ratio

    # Initial cutoff
    cutoff = args.test_after if args.test_after is not None else _percentile(sorted(times), args.auto_cutoff_percentile)
    # If a target test ratio is provided, search for a nearby cutoff that reaches it within tolerance
    if args.target_pos_ratio_test is not None:
        cutoff2 = _search_cutoff_for_target_ratio(
            times,
            labels,
            cutoff,
            target_ratio=args.target_pos_ratio_test,
            tol=args.ratio_tol,
            min_pos=max(1, int(args.min_pos_test)),
            min_total=max(1, int(args.min_total_test)),
        )
        cutoff = cutoff2
    else:
        # Otherwise, try to ensure post-cutoff has both classes and min positives
        adj_cutoff = _find_cutoff_with_both_classes(
            times, labels, cutoff, min_pos_test=max(1, int(args.min_pos_test))
        )
        if adj_cutoff != cutoff:
            cutoff = adj_cutoff

    # Build pre/post partitions by group, then pick label-aware val groups
    pre_idx = [i for i, t in enumerate(times) if t <= cutoff]
    post_idx = [i for i, t in enumerate(times) if t > cutoff]
    
    # Always build pre_groups for group-aware logic
    pre_groups: Dict[str, List[int]] = {}
    for i in pre_idx:
        pre_groups.setdefault(groups[i], []).append(i)
    
    # Check if time-based splitting makes target ratios impossible
    pre_pos = sum(1 for i in pre_idx if int(labels[i]) == 1)
    post_pos = sum(1 for i in post_idx if int(labels[i]) == 1)
    total_pos = pre_pos + post_pos
    
    # If balance-splits is requested, check if we have enough positives in each split
    balance_impossible = False
    if args.balance_splits and total_pos > 0:
        # Need roughly 50% positives in each split
        train_target_pos = int(0.5 * len(pre_idx) * (1 - args.val_frac))
        val_target_pos = int(0.5 * len(pre_idx) * args.val_frac)
        test_target_pos = int(0.5 * len(post_idx))
        
        if (pre_pos < train_target_pos + val_target_pos * 0.5 or  # Not enough for train+val
            post_pos < test_target_pos * 0.5):  # Not enough for test
            balance_impossible = True
            print(f"[SPLITS][WARN] Time cutoff makes 50/50 balance impossible (pre_pos={pre_pos}, post_pos={post_pos})")
            print(f"[SPLITS][WARN] Need ~{train_target_pos + val_target_pos} pre-cutoff pos, ~{test_target_pos} post-cutoff pos")
    
    if balance_impossible:
        # Fall back to stratified splitting that can achieve target ratios
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            print("[SPLITS][INFO] Using stratified splits to achieve target ratios")
            X = np.arange(len(labels)).reshape(-1, 1)
            y = np.array([int(v) for v in labels])
            
            # Preserve approximate test size from time-based split
            test_frac = max(0.1, min(0.4, len(post_idx) / max(1, len(labels))))
            val_frac = max(0.01, min(0.3, float(args.val_frac)))
            
            # First split: train+val vs test
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=args.seed)
            train_val_idx, test_idx = next(sss1.split(X, y))
            
            # Second split: train vs val
            y_tv = y[train_val_idx]
            actual_val_frac = val_frac / (1 - test_frac)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=actual_val_frac, random_state=args.seed)
            train_idx_rel, val_idx_rel = next(sss2.split(np.arange(len(train_val_idx)).reshape(-1, 1), y_tv))
            
            train_idx = sorted([train_val_idx[i] for i in train_idx_rel])
            val_idx = sorted([train_val_idx[i] for i in val_idx_rel])
            test_idx = sorted(test_idx)
            
        except Exception as e:
            print(f"[SPLITS][WARN] Stratified fallback failed ({e}); using time-based splits")
            val_idx, train_idx = _label_aware_val_groups_ratio(
                pre_groups,
                labels,
                val_frac=args.val_frac,
                seed=args.seed,
                target_ratio_val=args.target_pos_ratio_val,
                ratio_tol=args.ratio_tol,
                min_pos_val=max(1, int(args.min_pos_val)),
                refine_topk=max(0, int(getattr(args, "refine_topk", 0))),
            )
            test_idx = sorted(post_idx)
    else:
        val_idx, train_idx = _label_aware_val_groups_ratio(
            pre_groups,
            labels,
            val_frac=args.val_frac,
            seed=args.seed,
            target_ratio_val=args.target_pos_ratio_val,
            ratio_tol=args.ratio_tol,
            min_pos_val=max(1, int(args.min_pos_val)),
            refine_topk=max(0, int(getattr(args, "refine_topk", 0))),
        )
        test_idx = sorted(post_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    # Sanity summary
    tr_n, tr_p, tr_n0, tr_r = _counts(train_idx, labels)
    va_n, va_p, va_n0, va_r = _counts(val_idx, labels)
    te_n, te_p, te_n0, te_r = _counts(test_idx, labels)
    # Last-ditch: if train has zero positives but pre had positives, move a smallest-positive val group back to train
    if tr_p == 0 and any(int(labels[i]) == 1 for i in pre_idx) and va_p > 0:
        # Build val group list and their positive counts
        val_group_pos: List[Tuple[str, int]] = []
        for g, idxs in pre_groups.items():
            # group is in val if any of its indices are in val_idx
            if any(i in set(val_idx) for i in idxs):
                p = sum(1 for i in idxs if int(labels[i]) == 1)
                if p > 0:
                    val_group_pos.append((g, p))
        if val_group_pos:
            val_group_pos.sort(key=lambda x: x[1])
            g_move = val_group_pos[0][0]
            g_set = set(pre_groups[g_move])
            train_idx = sorted(train_idx + pre_groups[g_move])
            val_idx = sorted([i for i in val_idx if i not in g_set])
            splits = {"train": train_idx, "val": val_idx, "test": test_idx}
            tr_n, tr_p, tr_n0, tr_r = _counts(train_idx, labels)
            va_n, va_p, va_n0, va_r = _counts(val_idx, labels)
            print(f"[SPLITS][WARN] Moved group '{g_move}' from val->train to ensure train has positives.")

    print(f"[SPLITS] cutoff={cutoff:.1f}")
    print(f"[SPLITS] train: n={tr_n} pos={tr_p} neg={tr_n0} ratio_pos={tr_r:.3f}")
    print(f"[SPLITS] val  : n={va_n} pos={va_p} neg={va_n0} ratio_pos={va_r:.3f}")
    print(f"[SPLITS] test : n={te_n} pos={te_p} neg={te_n0} ratio_pos={te_r:.3f}")
    # Warn if ratios miss targets
    def _check_ratio(name: str, r: float, target: float | None):
        if target is None:
            return
        if abs(r - target) > args.ratio_tol:
            print(
                f"[SPLITS][WARN] {name} positive ratio {r:.3f} missed target {target:.3f} by > tol={args.ratio_tol:.3f}"
            )

    _check_ratio("TRAIN", tr_r, args.target_pos_ratio_train)
    _check_ratio("VAL", va_r, args.target_pos_ratio_val)
    _check_ratio("TEST", te_r, args.target_pos_ratio_test)
    # Warn if train ended up one-class despite positives existing pre-cutoff
    if tr_p == 0 and any(int(labels[i]) == 1 for i in pre_idx):
        print("[SPLITS][WARN] Train split has zero positives while pre-cutoff had positives; reserved positives for train.")

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
            # Ensure min positives in val/test if possible by swapping from train
            def ensure_min_pos(idxs: List[int], min_pos: int) -> List[int]:
                if min_pos <= 0:
                    return idxs
                pos = sum(1 for i in idxs if int(labels[i]) == 1)
                if pos >= min_pos:
                    return idxs
                needed = min_pos - pos
                train_pos = [i for i in train_idx if int(labels[i]) == 1]
                swap_ids = train_pos[:needed]
                if swap_ids:
                    updated = sorted(idxs + swap_ids)
                    keep = set(swap_ids)
                    train_idx[:] = [i for i in train_idx if i not in keep]
                    return updated
                return idxs
            # Apply to val/test (best-effort)
            val_idx = ensure_min_pos(val_idx, max(1, int(args.min_pos_val)))
            test_idx = ensure_min_pos(test_idx, max(1, int(args.min_pos_test)))
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
