#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from phisdom.data.schema import load_jsonl
from phisdom.data.cheap_features import CHEAP_FEATURES, row_to_features
from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr


def read_preds(path: str) -> Dict[str, float]:
    """Read predictions from a JSONL file, returning a mapping from stable ID to probability."""
    m: Dict[str, float] = {}
    if not os.path.exists(path):
        print(f"Warning: Prediction file missing: {path}")
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Look for various ID fields that could serve as stable keys
            id_ = obj.get("id") or obj.get("uid") or obj.get("sha1") or obj.get("url_hash")
            if id_ is None:
                raise KeyError(f"Prediction row missing stable key (id/uid/sha1/url_hash): {list(obj.keys())}")
            # Look for score in various field names
            score = obj.get("prob")
            if score is None:
                score = obj.get("score") if obj.get("score") is not None else obj.get("logit")
            if score is None:
                raise KeyError(f"Prediction row missing score field (prob/score/logit): {list(obj.keys())}")
            m[str(id_)] = float(score)
    return m


def _row_features(r: Dict[str, object], use_features: bool) -> List[float]:
    return row_to_features(r, use_features)


def align(jsonl_path: str, use_features: bool, **head_dirs: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Align predictions from multiple heads using stable ID joining."""
    # head_dirs: mapping from head name -> directory containing preds_{split}.jsonl
    split = os.path.splitext(os.path.basename(jsonl_path))[0].split('_')[-1]
    preds_by_head: Dict[str, Dict[str, float]] = {}
    
    # Load predictions from each head
    for name, d in head_dirs.items():
        preds_by_head[name] = read_preds(os.path.join(d, f"preds_{split}.jsonl"))
        print(f"Loaded {len(preds_by_head[name])} predictions for {name}")
    
    # Load ground truth labels
    rows = load_jsonl(jsonl_path)
    gold_labels: Dict[str, int] = {}
    for r in rows:
        id_ = r.get("id") or r.get("uid") or r.get("sha1") or r.get("url_hash")
        if id_ is None:
            raise KeyError(f"Gold row missing stable key (id/uid/sha1/url_hash): {list(r.keys())}")
        gold_labels[str(id_)] = int(r.get("label", 0))
    
    print(f"Loaded {len(gold_labels)} gold labels")
    
    # Find common IDs across all heads and gold labels
    common_ids = set(gold_labels.keys())
    for name, preds in preds_by_head.items():
        common_ids &= set(preds.keys())
        print(f"After intersecting with {name}: {len(common_ids)} common IDs")
    
    if not common_ids:
        print("Warning: No common IDs found across all heads and gold labels!")
        return np.array([]), np.array([]), np.array([]), list(head_dirs.keys())
    
    # Build aligned feature matrix and labels
    xs: List[List[float]] = []
    ys: List[int] = []
    ids: List[str] = []
    
    for id_ in sorted(common_ids):  # sort for deterministic order
        # Base features: predictions from each head
        base = [preds_by_head[name][id_] for name in head_dirs.keys()]
        
        # Optionally add cheap features
        if use_features:
            # Find the original row to get cheap features
            row = next((r for r in rows if str(r.get("id") or r.get("uid") or r.get("sha1") or r.get("url_hash")) == id_), None)
            if row is not None:
                base.extend(_row_features(row, use_features))
            else:
                # If we can't find the original row, add zeros for cheap features
                from phisdom.data.cheap_features import CHEAP_FEATURES
                base.extend([0.0] * len(CHEAP_FEATURES))
        
        xs.append(base)
        ys.append(gold_labels[id_])
        ids.append(id_)
    
    print(f"Final aligned dataset: {len(ids)} examples with {len(xs[0]) if xs else 0} features")
    
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
    ap.add_argument("--text-dir", default="artifacts/text_head")
    ap.add_argument("--cheap-mlp-dir", default="artifacts/cheap_mlp")
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
        "p_text": args.text_dir,
        "p_cheap": args.cheap_mlp_dir,
    }.items():
        if os.path.exists(os.path.join(d, "preds_val.jsonl")) and os.path.exists(os.path.join(d, "preds_test.jsonl")):
            heads[name] = d

    # Align data with available heads
    Xv, yv, ids_v, feature_names = align(args.val_jsonl, bool(getattr(args, "use_cheap_features", True)), **heads)
    Xt, yt, ids_t, _ = align(args.test_jsonl, bool(getattr(args, "use_cheap_features", True)), **heads)
    if Xv.size == 0 or Xt.size == 0:
        print("WARN: No aligned predictions for fusion. Did you run eval scripts for all included heads?")
    pv: np.ndarray
    pt: np.ndarray

    if args.method == "average":
        # Simple equal-weight average across all heads
        pv = np.clip(np.mean(Xv[:, : len(heads)], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
        pt = np.clip(np.mean(Xt[:, : len(heads)], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
    else:
        # Logistic regression fusion
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            print("WARN: scikit-learn not installed; falling back to average fusion")
            pv = np.clip(np.mean(Xv[:, : len(heads)], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
            pt = np.clip(np.mean(Xt[:, : len(heads)], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
        else:
            # Guard: if yv has a single class, LR is ill-posed; fall back to simple average
            if len(set(yv.tolist())) < 2:
                print("WARN: Validation split is one-class; using average fusion.")
                pv = np.clip(np.mean(Xv[:, : len(heads)], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
                pt = np.clip(np.mean(Xt[:, : len(heads)], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
            else:
                # L2 by default; use class_weight balanced to be robust
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Xv, yv)
                pv = clf.predict_proba(Xv)[:, 1]
                pt = clf.predict_proba(Xt)[:, 1]
                # Save fusion weights
                try:
                    w = {
                        "coef": clf.coef_.tolist(),
                        "intercept": clf.intercept_.tolist(),
                        "feature_names": list(heads.keys()) + (CHEAP_FEATURES if bool(getattr(args, "use_cheap_features", True)) else []),
                    }
                    with open(os.path.join(args.out_dir, "fusion_weights.json"), "w", encoding="utf-8") as f:
                        json.dump(w, f, indent=2)
                except Exception:
                    pass

    # Metrics and thresholds on test
    pr = pr_auc_safe(yt.tolist(), pt.tolist())
    roc = roc_auc_safe(yt.tolist(), pt.tolist())
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
