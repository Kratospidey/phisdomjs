#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

from phisdom.data.schema import load_jsonl
from phisdom.data.cheap_features import CHEAP_FEATURES, row_to_features
from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr
from phisdom.utils.alignment import get_alignment_strategy
from phisdom.utils.prediction_standardizer import (
    validate_prediction_format,
    standardize_prediction_format,
    save_standardized_predictions
)


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
    ap.add_argument("--alignment-strategy", choices=["inner_join", "coverage_max"], default="inner_join",
                    help="Alignment strategy: inner_join (strict) or coverage_max (with imputation)")
    ap.add_argument("--min-heads", type=int, default=2,
                    help="Minimum number of heads required per sample (for coverage_max strategy)")
    ap.add_argument("--exclude-heads", nargs="*", default=[],
                    help="Head names to exclude from fusion (e.g., 'p_dom_light' 'p_cheap')")
    ap.add_argument("--require-heads", nargs="*", default=[],
                    help="Head names that must be present (empty = use all available)")
    ap.add_argument("--include-fusion-head", action="store_true", help="Include existing first-level fusion (artifacts/fusion) as meta head p_fusion")
    ap.add_argument("--fusion-dir", default="artifacts/fusion", help="Directory of existing fusion predictions (preds_*.jsonl)")
    ap.add_argument("--export-unified-json", action="store_true", help="Export a unified JSONL with per-head and fused probabilities")
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        bool_action = None  # type: ignore
    if bool_action is not None:
        ap.add_argument("--use-cheap-features", action=bool_action, default=True, help="Include lightweight crawler features in the stacker (default: True)")
        ap.add_argument("--validate-predictions", action=bool_action, default=True, help="Validate prediction formats before fusion")
    else:
        ap.add_argument("--use-cheap-features", action="store_true", help="Include lightweight crawler features in the stacker")
        ap.add_argument("--validate-predictions", action="store_true", help="Validate prediction formats before fusion")
    ap.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the heads to include: only keep ones that have preds files
    all_heads = {
        "p_dom": args.dom_dir,
        "p_js": args.js_dir,
        "p_url": args.url_dir,
        "p_dom_light": args.dom_light_dir,
        "p_text": args.text_dir,
        "p_cheap": args.cheap_mlp_dir,
    }

    if args.include_fusion_head:
        fval = os.path.join(args.fusion_dir, "preds_val.jsonl")
        ftest = os.path.join(args.fusion_dir, "preds_test.jsonl")
        if os.path.exists(fval) and os.path.exists(ftest):
            all_heads["p_fusion"] = args.fusion_dir
        else:
            print(f"WARNING: --include-fusion-head requested but predictions missing in {args.fusion_dir}")
    
    # Filter out excluded heads
    excluded = set(args.exclude_heads)
    heads = {name: d for name, d in all_heads.items() if name not in excluded}
    
    # Filter to only heads with prediction files
    available_heads = {}
    for name, d in heads.items():
        val_preds = os.path.join(d, "preds_val.jsonl")
        test_preds = os.path.join(d, "preds_test.jsonl")
        if os.path.exists(val_preds) and os.path.exists(test_preds):
            available_heads[name] = d
        else:
            print(f"WARNING: Skipping {name} - missing prediction files in {d}")
    
    if not available_heads:
        print("ERROR: No heads with valid prediction files found!")
        return
    
    print(f"Using heads: {list(available_heads.keys())}")
    
    # Validate prediction formats if requested
    if getattr(args, "validate_predictions", True):
        print("Validating prediction formats...")
        for name, d in available_heads.items():
            for split in ["val", "test"]:
                pred_path = os.path.join(d, f"preds_{split}.jsonl")
                diagnostics = validate_prediction_format(pred_path)
                if not diagnostics["valid"]:
                    print(f"ERROR: Invalid prediction format in {pred_path}")
                    for error in diagnostics["errors"]:
                        print(f"  - {error}")
                    return
                if diagnostics["warnings"]:
                    print(f"WARNING: Issues in {pred_path}")
                    for warning in diagnostics["warnings"]:
                        print(f"  - {warning}")
    
    # Set up alignment strategy
    alignment_kwargs = {}
    if args.alignment_strategy == "coverage_max":
        alignment_kwargs["min_heads"] = args.min_heads
    
    aligner = get_alignment_strategy(args.alignment_strategy, **alignment_kwargs)
    
    # Determine required heads
    required_heads = None
    if args.require_heads:
        required_heads = set(args.require_heads)
        # Validate that required heads are available
        missing = required_heads - set(available_heads.keys())
        if missing:
            print(f"ERROR: Required heads not available: {missing}")
            return

    # Align data with available heads
    Xv, yv, ids_v, feature_names = aligner.align(
        args.val_jsonl, 
        available_heads, 
        required_heads=required_heads,
        use_cheap_features=bool(getattr(args, "use_cheap_features", True))
    )
    Xt, yt, ids_t, _ = aligner.align(
        args.test_jsonl, 
        available_heads, 
        required_heads=required_heads,
        use_cheap_features=bool(getattr(args, "use_cheap_features", True))
    )
    
    if Xv.size == 0 or Xt.size == 0:
        print("ERROR: No aligned predictions for fusion. Check that eval scripts have run for all included heads.")
        return
    
    # Perform fusion
    pv: np.ndarray
    pt: np.ndarray

    if args.method == "average":
        # Simple equal-weight average across all head predictions (not cheap features)
        n_heads = len(available_heads)
        pv = np.clip(np.mean(Xv[:, :n_heads], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
        pt = np.clip(np.mean(Xt[:, :n_heads], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
    else:
        # Logistic regression fusion
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("ERROR: scikit-learn not installed; falling back to average fusion")
            n_heads = len(available_heads)
            pv = np.clip(np.mean(Xv[:, :n_heads], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
            pt = np.clip(np.mean(Xt[:, :n_heads], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
        else:
            # Guard: if yv has a single class, LR is ill-posed; fall back to simple average
            if len(set(yv.tolist())) < 2:
                print("WARNING: Validation split is one-class; using average fusion.")
                n_heads = len(available_heads)
                pv = np.clip(np.mean(Xv[:, :n_heads], axis=1), 0.0, 1.0) if Xv.size else np.zeros((0,))
                pt = np.clip(np.mean(Xt[:, :n_heads], axis=1), 0.0, 1.0) if Xt.size else np.zeros((0,))
            else:
                # Scale features for better logistic regression performance
                scaler = StandardScaler()
                Xv_scaled = scaler.fit_transform(Xv)
                Xt_scaled = scaler.transform(Xt)
                
                # Use cross-validation and balanced class weights for robustness
                clf = LogisticRegression(
                    max_iter=1000, 
                    class_weight="balanced",
                    random_state=42,
                    solver='liblinear'  # Robust for small datasets
                )
                clf.fit(Xv_scaled, yv)
                pv = clf.predict_proba(Xv_scaled)[:, 1]
                pt = clf.predict_proba(Xt_scaled)[:, 1]
                
                # Save fusion weights for interpretability
                try:
                    weights_info = {
                        "coef": clf.coef_.tolist(),
                        "intercept": clf.intercept_.tolist(),
                        "feature_names": feature_names,
                        "scaling_mean": scaler.mean_.tolist() if scaler.mean_ is not None else [],
                        "scaling_std": scaler.scale_.tolist() if scaler.scale_ is not None else [],
                        "method": "logistic_regression",
                        "class_weight": "balanced"
                    }
                    with open(os.path.join(args.out_dir, "fusion_weights.json"), "w", encoding="utf-8") as f:
                        json.dump(weights_info, f, indent=2)
                except Exception as e:
                    print(f"WARNING: Could not save fusion weights: {e}")

    # Sanitize any non-finite values (rare: corrupted preds)
    def _sanitize(arr: np.ndarray, name: str) -> np.ndarray:
        if not np.isfinite(arr).all():
            print(f"WARNING: Non-finite values detected in {name}; replacing with 0.5")
            arr = arr.copy()
            bad = ~np.isfinite(arr)
            arr[bad] = 0.5
        return arr

    pv = _sanitize(pv, "val fused probs")
    pt = _sanitize(pt, "test fused probs")

    # Metrics and thresholds on test
    pr = pr_auc_safe(yt.tolist(), pt.tolist())
    roc = roc_auc_safe(yt.tolist(), pt.tolist())
    thresholds = {}
    for tpr in args.tpr:
        fpr, thr = fpr_at_tpr(yt.tolist(), pt.tolist(), tpr)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    # Standardize and save predictions
    val_preds, val_meta = standardize_prediction_format(
        ids_v, yv, pv, "fusion", "val", auto_flip=True
    )
    test_preds, test_meta = standardize_prediction_format(
        ids_t, yt, pt, "fusion", "test", auto_flip=True
    )
    
    save_standardized_predictions(val_preds, val_meta, args.out_dir, "val")
    save_standardized_predictions(test_preds, test_meta, args.out_dir, "test")

    # Save calibration results
    cal = {"metrics": {"pr_auc": pr, "roc_auc": roc}, "thresholds": thresholds}
    with open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    print(json.dumps(cal, indent=2))

    # Optional unified export (test + val)
    if args.export_unified_json:
        try:
            unified_path_val = os.path.join(args.out_dir, "unified_val.jsonl")
            unified_path_test = os.path.join(args.out_dir, "unified_test.jsonl")

            # Reconstruct per-head probabilities for each split for heads only (exclude cheap feature columns)
            # We re-run alignment (already have Xv, Xt). Feature ordering = heads (+ cheap features if present at end)
            n_feature_names = len(feature_names)
            head_names = [h for h in feature_names if h.startswith("p_")]
            # Determine index mapping
            head_indices = [feature_names.index(h) for h in head_names]

            def _export(path: str, ids, labels, X, fused_probs):
                with open(path, "w", encoding="utf-8") as f:
                    for sid, lab, row, fp in zip(ids, labels.tolist(), X, fused_probs.tolist()):
                        obj = {
                            "id": sid,
                            "label": int(lab),
                            "fused_prob": float(fp),
                            "heads": {hn: float(row[idx]) for hn, idx in zip(head_names, head_indices)}
                        }
                        f.write(json.dumps(obj))
                        f.write("\n")

            _export(unified_path_val, ids_v, yv, Xv, pv)
            _export(unified_path_test, ids_t, yt, Xt, pt)
            print(f"Unified probability exports written: {unified_path_val}, {unified_path_test}")
        except Exception as e:
            print(f"WARNING: Failed unified export: {e}")


if __name__ == "__main__":
    main()
