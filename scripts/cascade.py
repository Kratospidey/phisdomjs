#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def read_preds(path: str) -> Dict[str, Tuple[int, float]]:
    m: Dict[str, Tuple[int, float]] = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            id_ = str(obj.get("id"))
            y = int(obj.get("label", 0))
            p = float(obj.get("prob", 0.0))
            m[id_] = (y, p)
    return m


def find_threshold_for_precision(y: np.ndarray, p: np.ndarray, target_precision: float, greater_is_positive: bool = True) -> float:
    """Find threshold to achieve target precision, with deterministic tie handling."""
    order = np.argsort(-p) if greater_is_positive else np.argsort(p)
    tp = fp = 0
    best_thr = float("inf") if greater_is_positive else -float("inf")
    for idx in order:
        thr = p[idx]
        is_pos = (y[idx] == 1)
        if is_pos:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        if prec >= target_precision:
            best_thr = thr
            break
    # if never reached, pick extreme
    if math.isinf(best_thr):
        best_thr = 1.0 if greater_is_positive else 0.0
    return float(best_thr)


def main():
    ap = argparse.ArgumentParser(description="Two-tier cascade: Stage1 (URL+cheap) short-circuit, Stage2 (fused) fallback")
    ap.add_argument("--url-dir", default="artifacts/url_head")
    ap.add_argument("--cheap-dir", default="artifacts/cheap_mlp")
    ap.add_argument("--fusion-dir", default="artifacts/fusion")
    ap.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    ap.add_argument("--test-jsonl", default="data/pages_test.jsonl")
    ap.add_argument("--out-dir", default="artifacts/cascade")
    ap.add_argument("--target-precision", type=float, default=0.99, help="Precision target for stage1 phish acceptance")
    ap.add_argument("--target-benign-precision", type=float, default=0.995, help="Precision target for stage1 benign acceptance")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check if fusion directory has predictions, fall back to logistic fusion if needed
    def _has_preds(d: str) -> bool:
        return os.path.exists(os.path.join(d, "preds_val.jsonl")) and os.path.exists(os.path.join(d, "preds_test.jsonl"))
    fusion_dir = args.fusion_dir
    if not _has_preds(fusion_dir):
        alt = "artifacts/fusion"
        if _has_preds(alt):
            print(f"[CASCADE][WARN] '{fusion_dir}' has no predictions; falling back to '{alt}'.")
            fusion_dir = alt
        else:
            print(f"[CASCADE][WARN] No usable fusion preds found in '{fusion_dir}' or '{alt}'. Cascade may fail.")
    
    def load_split(split: str):
        url = read_preds(os.path.join(args.url_dir, f"preds_{split}.jsonl"))
        cheap = read_preds(os.path.join(args.cheap_dir, f"preds_{split}.jsonl"))
        fused = read_preds(os.path.join(fusion_dir, f"preds_{split}.jsonl"))
        ids = [i for i in fused.keys() if i in url and i in cheap]
        y = np.array([fused[i][0] for i in ids], dtype=int)
        p_url = np.array([url[i][1] for i in ids], dtype=float)
        p_cheap = np.array([cheap[i][1] for i in ids], dtype=float)
        p_fused = np.array([fused[i][1] for i in ids], dtype=float)
        return ids, y, p_url, p_cheap, p_fused

    # Prepare val for thresholds
    ids_v, yv, pu_v, pc_v, pf_v = load_split("val")
    # simple stage1 score: average of URL and Cheap; could be replaced with a trained LR if desired
    if yv.size == 0:
        thr_hi = 1.0
        thr_lo = 0.0
    else:
        # stage-1 is simple average; clamp to [0,1] defensively
        s1_v = np.clip(0.5 * pu_v + 0.5 * pc_v, 0.0, 1.0)
        # Guard against one-class validation
        if len(set(yv.tolist())) < 2:
            thr_hi = 1.0
            thr_lo = 0.0
        else:
            thr_hi = find_threshold_for_precision(yv, s1_v, args.target_precision, greater_is_positive=True)
            # For benign acceptance, use (1 - score) precision on negatives
            inv_scores = 1.0 - s1_v
            thr_lo = find_threshold_for_precision(1 - yv, inv_scores, args.target_benign_precision, greater_is_positive=True)
            # convert back to score threshold
            thr_lo = 1.0 - thr_lo
            # Ensure non-overlapping thresholds so fusion still handles the middle band
            if thr_lo > thr_hi:
                eps = 1e-6
                mid = 0.5 * (thr_lo + thr_hi)
                thr_lo = max(0.0, mid - eps)
                thr_hi = min(1.0, mid + eps)

    # Apply cascade on a split
    def apply(split: str):
        ids, y, pu, pc, pf = load_split(split)
        if len(ids) == 0:
            # Empty alignment across heads; return NaNs to avoid runtime warnings
            return ids, y, np.array([], dtype=float), float("nan"), float("nan"), float("nan")
        s1 = np.clip(0.5 * pu + 0.5 * pc, 0.0, 1.0)
        accept_phish = s1 >= thr_hi
        accept_benign = s1 <= thr_lo
        # final probabilities: use s1 where accepted, else fused
        p_final = np.where(accept_phish | accept_benign, s1, pf)
        
        # Robust coverage computation (avoid NaN from 0/0)
        total = len(accept_phish) if accept_phish.size > 0 else 0
        if total > 0:
            coverage = float(np.sum(accept_phish | accept_benign)) / total
            cov_hi = float(np.sum(accept_phish)) / total
            cov_lo = float(np.sum(accept_benign)) / total
        else:
            coverage = float("nan")
            cov_hi = float("nan") 
            cov_lo = float("nan")
            
        return ids, y, p_final, coverage, cov_hi, cov_lo

    # Save preds and report
    out = {
        "stage1": {"thr_hi": thr_hi, "thr_lo": thr_lo},
        "coverage": {},
    }
    for split in ["val", "test"]:
        ids, y, p, cov, cov_hi, cov_lo = apply(split)
        # Avoid numpy NaNs in JSON by converting with float(); if empty, use NaN explicitly
        if len(ids) == 0:
            out["coverage"][split] = {"overall": float("nan"), "phish": float("nan"), "benign": float("nan")}
        else:
            out["coverage"][split] = {"overall": float(cov), "phish": float(cov_hi), "benign": float(cov_lo)}
        # dump preds
        with open(os.path.join(args.out_dir, f"preds_{split}.jsonl"), "w", encoding="utf-8") as f:
            for i, yy, pp in zip(ids, y.tolist(), p.tolist()):
                f.write(json.dumps({"id": i, "label": int(yy), "prob": float(pp)}))
                f.write("\n")

    with open(os.path.join(args.out_dir, "cascade.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
