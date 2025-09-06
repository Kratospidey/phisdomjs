#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from phisdom.data.schema import load_jsonl
from phisdom.data.cheap_features import CHEAP_FEATURES, row_to_features
from phisdom.metrics import pr_auc, roc_auc, fpr_at_tpr


def read_preds(path: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            id_ = str(obj.get("id"))
            p = float(obj.get("prob", 0.0))
            m[id_] = p
    return m


def _row_features(r: Dict[str, object], use_features: bool) -> List[float]:
    return row_to_features(r, use_features)


def align(jsonl_path: str, use_features: bool, **head_dirs: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # head_dirs: mapping from head name -> directory containing preds_{split}.jsonl
    split = os.path.splitext(os.path.basename(jsonl_path))[0].split('_')[-1]
    preds_by_head: Dict[str, Dict[str, float]] = {}
    for name, d in head_dirs.items():
        preds_by_head[name] = read_preds(os.path.join(d, f"preds_{split}.jsonl"))
    rows = load_jsonl(jsonl_path)
    xs: List[List[float]] = []
    ys: List[int] = []
    ids: List[str] = []
    for r in rows:
        id_ = str(r.get("id"))
        # Require presence across all provided heads
        if not all(id_ in preds for preds in preds_by_head.values()):
            continue
        base = [preds_by_head[name][id_] for name in head_dirs.keys()]
        base.extend(_row_features(r, use_features))
        xs.append(base)
        ys.append(int(r.get("label", 0)))
        ids.append(id_)
    X = np.array(xs, dtype=float)
    y = np.array(ys, dtype=int)
    return X, y, np.array(ids), list(head_dirs.keys())


def fuse_average(p_dom: np.ndarray, p_js: np.ndarray, w_dom: float = 0.5) -> np.ndarray:
    w_js = 1.0 - w_dom
    return np.clip(w_dom * p_dom + w_js * p_js, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Fuse one or more calibrated heads (DOM/JS/URL/light) and optional cheap features")
    # Default to baselines; allow overriding with lightweight heads
    ap.add_argument("--dom-dir", default="artifacts/markup_run")
    ap.add_argument("--js-dir", default="artifacts/js_codet5p")
    ap.add_argument("--url-dir", default="artifacts/url_head")
    ap.add_argument("--dom-light-dir", default="artifacts/dom_gcn")
    ap.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    ap.add_argument("--test-jsonl", default="data/pages_test.jsonl")
    ap.add_argument("--out-dir", default="artifacts/fusion")
    ap.add_argument("--method", choices=["average", "logistic"], default="logistic")
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        bool_action = None  # type: ignore
    if bool_action is not None:
        ap.add_argument("--use-cheap-features", action=bool_action, default=True, help="Include lightweight crawler features in the stacker (default: True)")
    else:
        ap.add_argument("--use-cheap-features", action="store_true", help="Include lightweight crawler features in the stacker")
    ap.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the heads to include: only keep ones that have preds files
    heads = {}
    for name, d in {
        "p_dom": args.dom_dir,
        "p_js": args.js_dir,
        "p_url": args.url_dir,
        "p_dom_light": args.dom_light_dir,
    }.items():
        if os.path.exists(os.path.join(d, "preds_val.jsonl")) and os.path.exists(os.path.join(d, "preds_test.jsonl")):
            heads[name] = d

    # Align data with available heads
    Xv, yv, ids_v, feature_names = align(args.val_jsonl, bool(getattr(args, "use_cheap_features", True)), **heads)
    Xt, yt, ids_t, _ = align(args.test_jsonl, bool(getattr(args, "use_cheap_features", True)), **heads)
    if Xv.size == 0 or Xt.size == 0:
        print("WARN: No overlap between DOM and JS predictions; ensure eval scripts were run and IDs match.")
    pv: np.ndarray
    pt: np.ndarray

    if args.method == "average":
        # Simple equal-weight average across all heads
        pv = np.clip(np.mean(Xv[:, : len(heads)] or 0.0, axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
        pt = np.clip(np.mean(Xt[:, : len(heads)] or 0.0, axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
    else:
        # Logistic regression fusion
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            print("WARN: scikit-learn not installed; falling back to average fusion")
            pv = fuse_average(Xv[:, 0], Xv[:, 1])
            pt = fuse_average(Xt[:, 0], Xt[:, 1])
        else:
            # L2 by default; use class_weight balanced to be robust
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(Xv, yv)
            pv = clf.predict_proba(Xv)[:, 1]
            pt = clf.predict_proba(Xt)[:, 1]
            # Save fusion weights
            w = {
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "feature_names": list(heads.keys()) + (CHEAP_FEATURES if bool(getattr(args, "use_cheap_features", True)) else []),
            }
            with open(os.path.join(args.out_dir, "fusion_weights.json"), "w", encoding="utf-8") as f:
                json.dump(w, f, indent=2)

    # Metrics and thresholds on test
    pr = pr_auc(yt.tolist(), pt.tolist())
    roc = roc_auc(yt.tolist(), pt.tolist())
    thresholds = {}
    for tpr in args.tpr:
        fpr, thr = fpr_at_tpr(yt.tolist(), pt.tolist(), tpr)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    cal = {"metrics": {"pr_auc": pr, "roc_auc": roc}, "thresholds": thresholds}
    with open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    def dump(path: str, ids: np.ndarray, labels: np.ndarray, probs: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for id_, y, p in zip(ids.tolist(), labels.tolist(), probs.tolist()):
                f.write(json.dumps({"id": id_, "label": int(y), "prob": float(p)}))
                f.write("\n")

    dump(os.path.join(args.out_dir, "preds_val.jsonl"), ids_v, yv, pv)
    dump(os.path.join(args.out_dir, "preds_test.jsonl"), ids_t, yt, pt)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
