#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
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
    # Sort by score descending for positives, ascending for negatives if needed
    order = np.argsort(-p) if greater_is_positive else np.argsort(p)
    tp = 0
    fp = 0
    thr = 1.0 if greater_is_positive else 0.0
    best_thr = thr
    for i, idx in enumerate(order):
        pred_pos = 1
        if greater_is_positive:
            thr = p[idx]
            pred_pos = 1
        else:
            thr = p[idx]
            pred_pos = 1  # we're thresholding on low scores as positives for benign class
        if y[idx] == (1 if greater_is_positive else 0):
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        if prec >= target_precision:
            best_thr = thr
            break
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
    def split_from_path(p: str) -> str:
        base = os.path.splitext(os.path.basename(p))[0]
        return base.split('_')[-1]

    def load_split(split: str):
        url = read_preds(os.path.join(args.url_dir, f"preds_{split}.jsonl"))
        cheap = read_preds(os.path.join(args.cheap_dir, f"preds_{split}.jsonl"))
        fused = read_preds(os.path.join(args.fusion_dir, f"preds_{split}.jsonl"))
        ids = [i for i in fused.keys() if i in url and i in cheap]
        y = np.array([fused[i][0] for i in ids], dtype=int)
        p_url = np.array([url[i][1] for i in ids], dtype=float)
        p_cheap = np.array([cheap[i][1] for i in ids], dtype=float)
        p_fused = np.array([fused[i][1] for i in ids], dtype=float)
        return ids, y, p_url, p_cheap, p_fused

    # Prepare val for thresholds
    ids_v, yv, pu_v, pc_v, pf_v = load_split("val")
    # simple stage1 score: average of URL and Cheap; could be replaced with a trained LR if desired
    s1_v = 0.5 * pu_v + 0.5 * pc_v
    thr_hi = find_threshold_for_precision(yv, s1_v, args.target_precision, greater_is_positive=True)
    # For benign acceptance, use (1 - score) precision on negatives
    inv_scores = 1.0 - s1_v
    thr_lo = find_threshold_for_precision(1 - yv, inv_scores, args.target_benign_precision, greater_is_positive=True)
    # convert back to score threshold
    thr_lo = 1.0 - thr_lo

    # Apply cascade on a split
    def apply(split: str):
        ids, y, pu, pc, pf = load_split(split)
        s1 = 0.5 * pu + 0.5 * pc
        accept_phish = s1 >= thr_hi
        accept_benign = s1 <= thr_lo
        fall_back = ~(accept_phish | accept_benign)
        # final probabilities: use s1 where accepted, else fused
        p_final = np.where(accept_phish | accept_benign, s1, pf)
        coverage = float(np.mean(accept_phish | accept_benign))
        return ids, y, p_final, coverage, float(np.mean(accept_phish)), float(np.mean(accept_benign))

    # Save preds and report
    out = {
        "stage1": {"thr_hi": thr_hi, "thr_lo": thr_lo},
        "coverage": {},
    }
    for split in ["val", "test"]:
        ids, y, p, cov, cov_hi, cov_lo = apply(split)
        out["coverage"][split] = {"overall": cov, "phish": cov_hi, "benign": cov_lo}
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
