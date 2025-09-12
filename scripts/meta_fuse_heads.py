#!/usr/bin/env python
"""Meta-fusion weight search over first-level head probabilities.

Goal: Find non-negative weights (sum=1) over a set of calibrated head probabilities
that maximize validation PR-AUC (primary) and break ties with ROC-AUC.

Outputs:
  - calibration.json  (metrics on test with chosen weights)
  - weights.json      (selected weights + search diagnostics)
  - preds_val.jsonl / preds_test.jsonl (standard format with auto flip detection)

Simplifying assumptions:
  * Uses existing standardized prediction files for each head directory
    (preds_val.jsonl / preds_test.jsonl) optionally with --head-tag.
  * Alignment is performed via existing alignment strategy utilities (inner join or coverage_max).
  * Cheap features are optional but by default excluded from weight search (can include via flag).
  * Weight search strategies: grid over simplex for <=5 heads (coarse), random Dirichlet sampling otherwise.

Future extensions (not yet implemented):
  * Per-threshold optimization (e.g., optimize FPR@TPR target directly)
  * Two-stage stacking (logistic after weight search)
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr
from phisdom.utils.alignment import get_alignment_strategy
from phisdom.utils.prediction_standardizer import (
    validate_prediction_format,
    standardize_prediction_format,
    save_standardized_predictions,
)


def enumerate_simplex(n: int, steps: int) -> np.ndarray:
    """Enumerate discrete simplex with given number of steps per dimension.
    Returns array shape (K, n) of weight vectors (sum=1).
    For n>5 and/or large steps this grows combinatorially; caller should gate usage.
    """
    if n == 1:
        return np.ones((1, 1), dtype=float)
    grid = np.linspace(0.0, 1.0, steps)
    results: List[List[float]] = []
    cur: List[float] = []

    def backtrack(idx: int, remaining: float):
        if idx == n - 1:
            cur.append(remaining)
            results.append(cur.copy())
            cur.pop()
            return
        for v in grid:
            if v > remaining:
                break
            cur.append(v)
            backtrack(idx + 1, remaining - v)
            cur.pop()
    backtrack(0, 1.0)
    arr = np.array(results, dtype=float)
    # Filter numerical drift & normalize
    arr = arr[np.isclose(arr.sum(axis=1), 1.0, atol=1e-6)]
    if arr.size == 0:
        return np.ones((1, n), dtype=float) / n
    return arr


def random_dirichlet(n: int, samples: int, alpha: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    return rng.dirichlet(np.full(n, alpha, dtype=float), size=samples)


def apply_weights(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.clip(X @ w, 0.0, 1.0)


def _tolist_safe(x) -> List[float]:
    try:
        if x is None:
            return []
        # numpy arrays expose tolist(); fall back to list()
        return x.tolist() if hasattr(x, "tolist") else list(x)
    except Exception:
        return []


def weight_search(
    Xv: np.ndarray,
    yv: np.ndarray,
    head_names: List[str],
    strategy: str,
    grid_steps: int,
    random_samples: int,
    dirichlet_alpha: float,
    # Regularization/constraints (optional)
    l2_reg: float = 0.0,
    prefer_heads: Optional[set] = None,
    prefer_weight: float = 0.0,
    cap_weights: Optional[Dict[str, float]] = None,
    oof_folds: int = 0,
    oof_seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    n = len(head_names)
    candidates: np.ndarray
    meta: Dict[str, object] = {"strategy": strategy, "head_names": head_names}
    if strategy == "grid" and n <= 5:
        candidates = enumerate_simplex(n, grid_steps)
        meta["candidate_mode"] = f"grid:{grid_steps}"; meta["num_candidates"] = int(candidates.shape[0])
    else:
        candidates = random_dirichlet(n, random_samples, alpha=dirichlet_alpha)
        meta["candidate_mode"] = f"dirichlet:{random_samples}"; meta["num_candidates"] = int(candidates.shape[0])

    best_w = None
    best_key = (-1.0, -1.0)  # (pr_auc, roc_auc)
    # Store per-candidate scores (index, pr, roc) - explicitly cast to float for type checkers
    scores: List[Dict[str, float]] = []
    # Helper: apply caps and compute objective
    name_to_idx = {n: i for i, n in enumerate(head_names)}
    rng = np.random.default_rng(oof_seed)

    def _score_candidate(w: np.ndarray) -> Tuple[float, float]:
        # Enforce per-head caps (skip if violated)
        if cap_weights:
            for h, m in cap_weights.items():
                idx = name_to_idx.get(h)
                if idx is not None and w[idx] > m + 1e-9:
                    return (-1.0, -1.0)  # invalid
        # Evaluate PR/ROC with optional OOF folds (for weight-search robustness)
        if oof_folds and oof_folds > 1 and len(np.unique(yv)) > 1:
            try:
                from sklearn.model_selection import StratifiedKFold
            except Exception:
                # Fallback to single split if sklearn missing
                pv = apply_weights(Xv, w)
                pr_val = pr_auc_safe(yv.tolist(), pv.tolist())
                roc_val = roc_auc_safe(yv.tolist(), pv.tolist())
                pr = float(pr_val if pr_val is not None else 0.0)
                roc = float(roc_val if roc_val is not None else 0.0)
            else:
                skf = StratifiedKFold(n_splits=oof_folds, shuffle=True, random_state=oof_seed)
                oof_preds = np.zeros_like(yv, dtype=float)
                for tr, te in skf.split(Xv, yv):
                    # weights are not trained; still evaluate strictly on held-out
                    oof_preds[te] = np.clip(Xv[te] @ w, 0.0, 1.0)
                pr_val = pr_auc_safe(yv.tolist(), oof_preds.tolist())
                roc_val = roc_auc_safe(yv.tolist(), oof_preds.tolist())
                pr = float(pr_val if pr_val is not None else 0.0)
                roc = float(roc_val if roc_val is not None else 0.0)
        else:
            pv = apply_weights(Xv, w)
            pr_val = pr_auc_safe(yv.tolist(), pv.tolist())
            roc_val = roc_auc_safe(yv.tolist(), pv.tolist())
            pr = float(pr_val if pr_val is not None else 0.0)
            roc = float(roc_val if roc_val is not None else 0.0)

        # Regularization and preferences: maximize (pr - l2*||w||^2 + prefer_weight*sum(w_preferred))
        if l2_reg:
            pr -= float(l2_reg) * float(np.dot(w, w))
        if prefer_heads and prefer_weight:
            s = 0.0
            for h in prefer_heads:
                idx = name_to_idx.get(h)
                if idx is not None:
                    s += float(w[idx])
            pr += float(prefer_weight) * s
        return (pr, roc)

    for i, w in enumerate(candidates):
        pr, roc = _score_candidate(w)
        scores.append({"i": i, "pr": pr, "roc": roc})
        key = (pr, roc)
        if key > best_key:
            best_key = key
            best_w = w
    meta["best_score"] = {"pr": best_key[0], "roc": best_key[1]}
    meta["scores_evaluated"] = len(scores)
    meta["top5"] = sorted(scores, key=lambda d: (d["pr"], d["roc"]), reverse=True)[:5]
    return best_w if best_w is not None else np.ones(n) / n, meta


def main():
    ap = argparse.ArgumentParser(description="Meta-fusion weight search (simplex) for calibrated heads")
    ap.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    ap.add_argument("--test-jsonl", default="data/pages_test.jsonl")
    ap.add_argument("--out-dir", default="artifacts/fusion_meta")
    # Head directories
    ap.add_argument("--dom-dir", default="artifacts/markup_run")
    ap.add_argument("--js-dir", default="artifacts/js_codet5p")
    ap.add_argument("--url-dir", default="artifacts/url_head")
    ap.add_argument("--text-dir", default="artifacts/text_head")
    ap.add_argument("--include-fusion-head", action="store_true", help="Include existing first-level fusion head as additional input")
    ap.add_argument("--fusion-dir", default="artifacts/fusion", help="Directory of existing fusion predictions (if included)")
    ap.add_argument("--head-tag", default="", help="Suffix tag for input prediction files (e.g. _full)")
    ap.add_argument("--tag", default="", help="Suffix tag for output prediction files")
    ap.add_argument("--alignment-strategy", choices=["inner_join", "coverage_max"], default="inner_join")
    ap.add_argument("--min-heads", type=int, default=2)
    ap.add_argument("--strategy", choices=["grid", "random"], default="random")
    ap.add_argument("--grid-steps", type=int, default=6, help="Steps per dimension for grid (<=5 heads)")
    ap.add_argument("--random-samples", type=int, default=5000)
    ap.add_argument("--dirichlet-alpha", type=float, default=1.0)
    ap.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
    ap.add_argument("--exclude-heads", nargs="*", default=[])
    ap.add_argument("--require-heads", nargs="*", default=[])
    ap.add_argument("--validate-predictions", action="store_true")
    ap.add_argument("--include-cheap-features", action="store_true", help="If set, treat cheap feature probabilities as separate weight inputs (rare)")
    # Unified export path (to align exactly with coverage_max fused data)
    ap.add_argument("--use-unified", default="", help="Directory with unified_val.jsonl/unified_test.jsonl produced by fuse_heads --export-unified-json")
    ap.add_argument("--include-fusion-prob", action="store_true", help="When using --use-unified, include fused_prob as p_fusion input")
    # OOF stacking / robustness for weight selection
    ap.add_argument("--oof-folds", type=int, default=0, help="Use K-fold OOF evaluation during weight search (>=2 enables)")
    ap.add_argument("--oof-seed", type=int, default=42)
    # Regularization and constraints on weights
    ap.add_argument("--l2", type=float, default=0.0, help="L2 penalty on weight vector magnitude during search")
    ap.add_argument("--cap-weight", action="append", default=[], help="Per-head cap like p_js:0.5 (can be repeated)")
    ap.add_argument("--prefer-heads", nargs="*", default=[], help="Heads to softly prefer (adds prefer-weight * sum(w_i) to objective)")
    ap.add_argument("--prefer-weight", type=float, default=0.0)
    # Optional alternative stacker (logistic) with OOF training
    ap.add_argument("--stacker", choices=["weights", "logistic"], default="weights")
    ap.add_argument("--logreg-C", type=float, default=1.0, help="Inverse regularization strength for logistic stacker (if used)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build head mapping
    all_heads = {
        "p_dom": args.dom_dir,
        "p_js": args.js_dir,
        "p_url": args.url_dir,
        "p_text": args.text_dir,
    }
    if args.include_fusion_head:
        fval = os.path.join(args.fusion_dir, "preds_val.jsonl")
        ftest = os.path.join(args.fusion_dir, "preds_test.jsonl")
        if os.path.exists(fval) and os.path.exists(ftest):
            all_heads["p_fusion"] = args.fusion_dir
        else:
            print("WARNING: Requested --include-fusion-head but predictions missing; skipping.")

    all_heads = {k: v for k, v in all_heads.items() if k not in set(args.exclude_heads)}

    # Auto-infer head tag if not provided: if corresponding preds_*_full.jsonl exist
    # for at least one head directory, prefer "_full". If the split filename already
    # includes _full (e.g., pages_val_full.jsonl), do not add a tag to avoid duplication.
    in_tag = args.head_tag
    base_val = os.path.basename(args.val_jsonl)
    base_test = os.path.basename(args.test_jsonl)
    split_has_full = base_val.endswith("_full.jsonl") or base_test.endswith("_full.jsonl")
    # If user passed --head-tag _full but splits already *_full, drop it to prevent preds_*_full_full.jsonl
    if split_has_full and in_tag == "_full":
        print("[fuse][INFO] Split filenames already include _full; ignoring --head-tag _full to avoid duplication")
        in_tag = ""
    if not in_tag:
        if not split_has_full:
            # Probe heads for _full files
            any_full = False
            for _, d in {**all_heads}.items():
                if os.path.exists(os.path.join(d, "preds_val_full.jsonl")) and os.path.exists(os.path.join(d, "preds_test_full.jsonl")):
                    any_full = True
                    break
            if any_full:
                in_tag = "_full"
                print(f"[fuse][INFO] Auto-inferred --head-tag {in_tag} based on *_full prediction files")
    def _with_in_tag(base: str) -> str:
        if not in_tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}{in_tag}{ext}" if ext else f"{base}{in_tag}"  # safety

    # Filter to directories with preds
    available: Dict[str, str] = {}
    for name, d in all_heads.items():
        vp = os.path.join(d, _with_in_tag("preds_val.jsonl"))
        tp = os.path.join(d, _with_in_tag("preds_test.jsonl"))
        if os.path.exists(vp) and os.path.exists(tp):
            available[name] = d
        else:
            print(f"WARNING: {name} missing prediction files; skipping.")
    if not available:
        print("ERROR: No available heads")
        return
    print(f"Meta-fusion using heads: {list(available.keys())}")

    if args.require_heads:
        missing = set(args.require_heads) - set(available.keys())
        if missing:
            print(f"ERROR: Required heads missing: {missing}")
            return

    # Optional prediction validation
    if args.validate_predictions:
        from phisdom.utils.prediction_standardizer import validate_prediction_format
        for name, d in available.items():
            for split in ["val", "test"]:
                path = os.path.join(d, _with_in_tag(f"preds_{split}.jsonl"))
                diag = validate_prediction_format(path)
                if not diag["valid"]:
                    print(f"ERROR: Invalid prediction file {path}")
                    return

    # Option A: Use unified export (exact same coverage as fused)
    if args.use_unified:
        uni_dir = args.use_unified
        uv = os.path.join(uni_dir, "unified_val.jsonl")
        ut = os.path.join(uni_dir, "unified_test.jsonl")
        if not (os.path.exists(uv) and os.path.exists(ut)):
            print(f"ERROR: --use-unified specified but files missing in {uni_dir}")
            return
        def _load_unified(path: str):
            ids: List[str] = []
            labels: List[int] = []
            rows: List[List[float]] = []
            # Determine head names from first row
            head_keys: Optional[List[str]] = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if head_keys is None:
                        head_keys = sorted(list(obj.get("heads", {}).keys()))
                        # Apply require/exclude filters
                        if args.require_heads:
                            head_keys = [h for h in head_keys if h in set(args.require_heads)]
                        if args.exclude_heads:
                            head_keys = [h for h in head_keys if h not in set(args.exclude_heads)]
                        # Optionally include fused_prob as p_fusion
                        if args.include_fusion_prob:
                            head_keys = head_keys + ["p_fusion"]
                    ids.append(obj["id"])
                    labels.append(int(obj["label"]))
                    row = [float(obj["heads"].get(h, 0.0)) for h in head_keys if h != "p_fusion"]
                    if args.include_fusion_prob:
                        row.append(float(obj.get("fused_prob", 0.0)))
                    rows.append(row)
            if head_keys is None:
                return np.zeros((0,0)), np.array([]), [], []
            return np.array(rows, dtype=float), np.array(labels, dtype=int), ids, head_keys
        Xv, yv, ids_v, feature_names = _load_unified(uv)
        Xt, yt, ids_t, feature_names_t = _load_unified(ut)
        # Guard consistent feature ordering
        if feature_names != feature_names_t:
            print("WARNING: Unified val/test head order mismatch; aligning by intersection")
            common = [h for h in feature_names if h in set(feature_names_t)]
            idx_v = [feature_names.index(h) for h in common]
            idx_t = [feature_names_t.index(h) for h in common]
            Xv = Xv[:, idx_v]
            Xt = Xt[:, idx_t]
            feature_names = common
    else:
        # Align predictions across heads (legacy path)
        alignment_kwargs = {}
        if args.alignment_strategy == "coverage_max":
            alignment_kwargs["min_heads"] = args.min_heads
        aligner = get_alignment_strategy(args.alignment_strategy, **alignment_kwargs)

        Xv, yv, ids_v, feature_names = aligner.align(
            args.val_jsonl,
            {k: v for k, v in available.items()},
            required_heads=set(args.require_heads) if args.require_heads else None,
            use_cheap_features=bool(args.include_cheap_features),
            head_tag=in_tag
        )
        Xt, yt, ids_t, _ = aligner.align(
            args.test_jsonl,
            {k: v for k, v in available.items()},
            required_heads=set(args.require_heads) if args.require_heads else None,
            use_cheap_features=bool(args.include_cheap_features),
            head_tag=in_tag
        )
    if Xv.size == 0 or Xt.size == 0:
        print("ERROR: Alignment produced empty matrices")
        return

    # Restrict to probability heads (prefixed with p_) for downstream
    prob_head_indices = [i for i, n in enumerate(feature_names) if n.startswith("p_")]
    head_names = [feature_names[i] for i in prob_head_indices]
    Xv_heads = Xv[:, prob_head_indices]
    Xt_heads = Xt[:, prob_head_indices]

    # Parse caps map
    caps: Dict[str, float] = {}
    for item in (args.cap_weight or []):
        try:
            if ":" in item:
                k, v = item.split(":", 1)
            elif "=" in item:
                k, v = item.split("=", 1)
            else:
                continue
            caps[k.strip()] = float(v)
        except Exception:
            continue

    if args.stacker == "logistic":
        # Optional logistic stacker with OOF training and final fit
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold
        except Exception:
            print("ERROR: scikit-learn required for --stacker logistic")
            return
        # Scale features
        scaler = StandardScaler()
        Xv_s = scaler.fit_transform(Xv_heads)
        Xt_s = scaler.transform(Xt_heads)
        # OOF preds if requested
        if args.oof_folds and args.oof_folds > 1 and len(np.unique(yv)) > 1:
            skf = StratifiedKFold(n_splits=args.oof_folds, shuffle=True, random_state=args.oof_seed)
            oof = np.zeros_like(yv, dtype=float)
            for tr, te in skf.split(Xv_s, yv):
                clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=args.logreg_C, solver="liblinear", random_state=42)
                clf.fit(Xv_s[tr], yv[tr])
                oof[te] = clf.predict_proba(Xv_s[te])[:, 1]
            # Fit final model on full val
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=args.logreg_C, solver="liblinear", random_state=42)
            clf.fit(Xv_s, yv)
            pv = clf.predict_proba(Xv_s)[:, 1]
            pt = clf.predict_proba(Xt_s)[:, 1]
            # Save weights
            try:
                weights_info = {
                    "coef": clf.coef_.tolist(),
                    "intercept": clf.intercept_.tolist(),
                    "head_names": head_names,
                    "scaler_mean": _tolist_safe(getattr(scaler, "mean_", None)),
                    "scaler_scale": _tolist_safe(getattr(scaler, "scale_", None)),
                    "stacker": "logistic",
                    "oof_folds": args.oof_folds,
                    "C": args.logreg_C,
                }
                with open(os.path.join(args.out_dir, "meta_logistic_weights.json"), "w", encoding="utf-8") as f:
                    json.dump(weights_info, f, indent=2)
            except Exception:
                pass
        else:
            # Simple fit without OOF
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=args.logreg_C, solver="liblinear", random_state=42)
            clf.fit(Xv_s, yv)
            pv = clf.predict_proba(Xv_s)[:, 1]
            pt = clf.predict_proba(Xt_s)[:, 1]
            try:
                weights_info = {
                    "coef": clf.coef_.tolist(),
                    "intercept": clf.intercept_.tolist(),
                    "head_names": head_names,
                    "scaler_mean": _tolist_safe(getattr(scaler, "mean_", None)),
                    "scaler_scale": _tolist_safe(getattr(scaler, "scale_", None)),
                    "stacker": "logistic",
                    "oof_folds": 0,
                    "C": args.logreg_C,
                }
                with open(os.path.join(args.out_dir, "meta_logistic_weights.json"), "w", encoding="utf-8") as f:
                    json.dump(weights_info, f, indent=2)
            except Exception:
                pass
        # For reporting consistency, synthesize a "weights" vector via normalized positive coefs (if available)
        w = None
        try:
            coefs = np.maximum(0.0, clf.coef_.reshape(-1))
            w = coefs / coefs.sum() if coefs.sum() > 0 else np.ones_like(coefs) / len(coefs)
        except Exception:
            w = np.ones(len(head_names)) / max(1, len(head_names))
        meta = {"strategy": "logistic", "head_names": head_names, "best_score": {}}
        print(f"Selected weights (approx from logistic coefs): {dict(zip(head_names, map(lambda x: round(float(x),4), w)))}")
    else:
        # Weight search with optional constraints/OOF
        w, meta = weight_search(
            Xv_heads,
            yv,
            head_names,
            strategy=args.strategy,
            grid_steps=args.grid_steps,
            random_samples=args.random_samples,
            dirichlet_alpha=args.dirichlet_alpha,
            l2_reg=args.l2,
            prefer_heads=set(args.prefer_heads) if args.prefer_heads else None,
            prefer_weight=args.prefer_weight,
            cap_weights=caps,
            oof_folds=args.oof_folds,
            oof_seed=args.oof_seed,
        )
        print(f"Selected weights: {dict(zip(head_names, map(lambda x: round(float(x),4), w)))}")
        pv = apply_weights(Xv_heads, w)
        pt = apply_weights(Xt_heads, w)

    # Metrics & thresholds on test
    pr = pr_auc_safe(yt.tolist(), pt.tolist())
    roc = roc_auc_safe(yt.tolist(), pt.tolist())
    import math
    thresholds = {}
    for tpr in args.tpr:
        fpr, thr = fpr_at_tpr(yt.tolist(), pt.tolist(), tpr)
        if isinstance(thr, float) and math.isinf(thr):
            thresholds[str(tpr)] = {"fpr": float("nan"), "threshold": 0.5}
        else:
            thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    # Standardize/save preds
    val_preds, val_meta = standardize_prediction_format(ids_v, yv, pv, "fusion_meta", "val", auto_flip=True)
    test_preds, test_meta = standardize_prediction_format(ids_t, yt, pt, "fusion_meta", "test", auto_flip=True)

    tag = args.tag
    def _with_tag(base: str) -> str:
        if not tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}{tag}{ext}" if ext else f"{base}{tag}"

    if tag:
        for split, preds, meta_obj in [("val", val_preds, val_meta), ("test", test_preds, test_meta)]:
            pred_path = os.path.join(args.out_dir, _with_tag(f"preds_{split}.jsonl"))
            with open(pred_path, "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p)); f.write("\n")
            meta_path = os.path.join(args.out_dir, _with_tag(f"preds_{split}_metadata.json"))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, indent=2)
    else:
        save_standardized_predictions(val_preds, val_meta, args.out_dir, "val")
        save_standardized_predictions(test_preds, test_meta, args.out_dir, "test")

    cal = {"metrics": {"pr_auc": pr, "roc_auc": roc}, "thresholds": thresholds}
    with open(os.path.join(args.out_dir, _with_tag("calibration.json")), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    weights_dict = {"weights": {h: float(wi) for h, wi in zip(head_names, w)}, "search_meta": meta}
    with open(os.path.join(args.out_dir, _with_tag("weights.json")), "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
