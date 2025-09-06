#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from phisdom.data.schema import load_jsonl
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


CHEAP_FEATURES: List[str] = [
    # URL/redirect
    "redirect_hops", "redirect_max_ms", "url_len", "num_dots", "num_pct", "has_at", "host_is_ip",
    # New tiny lexicals
    "host_hyphens", "has_punycode",
    # Redirect sketch
    "redir_hops", "redir_cross_host", "has_meta_refresh", "has_js_loc_replace",
    # DNS/RDAP
    "dns_created_days_ago", "dns_updated_days_ago", "ns_count", "mx_present", "ttl_min", "ttl_mean",
    # New RDAP compact
    "rdap_age_days", "rdap_registrar_hash64", "rdap_ns_count", "rdap_has_privacy",
    # TLS
    "cert_age_days", "san_count",
    # New TLS compact
    "tls_not_before_days", "tls_san_count", "tls_issuer_spki_hash64",
    # Headers snapshot: skip string values; optionally presence as bools
    # Request graph
    "req_unique_etld1", "req_thirdparty_ratio", "req_counts_script", "req_counts_css", "req_counts_xhr", "req_counts_img",
    # Form semantics
    "form_pw_count", "form_cross_site", "form_login_tokens", "form_hidden_count", "form_autocomplete_off", "onsubmit_handlers",
    # New compact form/action
    "form_fp_hash64", "num_pw", "num_email", "num_hidden", "form_method_get", "action_cross_origin", "action_proto_mismatch", "iframe_login", "top_form_count", "iframe_form_count", "form_css_sig_hash64",
    # JS heuristics
    "js_entropy", "js_eval_ct", "js_atob_ct", "js_b64_blob_ct", "js_keylog_listeners",
    # New micro-counters
    "js_eval_like", "js_hex_ratio", "js_fromcharcode", "js_hi_entropy_ratio", "js_atob", "key_listener_pw", "key_listeners_total",
    # Fingerprinting
    "fp_canvas", "fp_webgl", "fp_audio", "fp_font_enum", "fp_webrtc",
    # Visual-lite (skip raw hashes by default)
    "favicon_dhash64", "fav_rel_count", "fav_cross_origin", "logo_phash64", "logo_from_alt_or_name",
    # Title↔host
    "title_host_jaccard_q8",
]


def _row_features(r: Dict[str, object], use_features: bool) -> List[float]:
    feats: List[float] = []
    if not use_features:
        return feats
    for name in CHEAP_FEATURES:
        v = r.get(name)
        if isinstance(v, bool):
            feats.append(1.0 if v else 0.0)
        elif v is None:
            feats.append(0.0)
        else:
            try:
                feats.append(float(str(v)))
            except Exception:
                feats.append(0.0)
    return feats


def align(dom_dir: str, js_dir: str, jsonl_path: str, use_features: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dom_preds = read_preds(os.path.join(dom_dir, f"preds_{os.path.splitext(os.path.basename(jsonl_path))[0].split('_')[-1]}.jsonl"))
    js_preds = read_preds(os.path.join(js_dir, f"preds_{os.path.splitext(os.path.basename(jsonl_path))[0].split('_')[-1]}.jsonl"))
    rows = load_jsonl(jsonl_path)
    xs: List[List[float]] = []
    ys: List[int] = []
    ids: List[str] = []
    for r in rows:
        id_ = str(r.get("id"))
        if id_ in dom_preds and id_ in js_preds:
            base = [dom_preds[id_], js_preds[id_]]
            base.extend(_row_features(r, use_features))
            xs.append(base)
            ys.append(int(r.get("label", 0)))
            ids.append(id_)
    X = np.array(xs, dtype=float)
    y = np.array(ys, dtype=int)
    return X, y, np.array(ids)


def fuse_average(p_dom: np.ndarray, p_js: np.ndarray, w_dom: float = 0.5) -> np.ndarray:
    w_js = 1.0 - w_dom
    return np.clip(w_dom * p_dom + w_js * p_js, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Fuse DOM and JS heads into a single probability")
    ap.add_argument("--dom-dir", default="artifacts/markup_run")
    ap.add_argument("--js-dir", default="artifacts/js_codet5p")
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

    # Align data
    Xv, yv, ids_v = align(args.dom_dir, args.js_dir, args.val_jsonl, bool(getattr(args, "use_cheap_features", True)))
    Xt, yt, ids_t = align(args.dom_dir, args.js_dir, args.test_jsonl, bool(getattr(args, "use_cheap_features", True)))
    if Xv.size == 0 or Xt.size == 0:
        print("WARN: No overlap between DOM and JS predictions; ensure eval scripts were run and IDs match.")
    pv: np.ndarray
    pt: np.ndarray

    if args.method == "average":
        pv = fuse_average(Xv[:, 0], Xv[:, 1])
        pt = fuse_average(Xt[:, 0], Xt[:, 1])
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
            w = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "feature_names": ["p_dom", "p_js"] + (CHEAP_FEATURES if bool(getattr(args, "use_cheap_features", True)) else [])}
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
