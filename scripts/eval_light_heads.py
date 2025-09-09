#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phisdom.data.new_heads import (
    UrlSeqDataset,
    JsSeqDataset,
    DomGraphDataset,
    PaddedSeqCollator,
    DomGraphCollator,
)
from phisdom.models.heads import UrlCharCNN, JsCharCNN, DomGCN
from phisdom.models.calibration import fit_temperature, TemperatureScaler
from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr
from phisdom.utils.prediction_standardizer import (
    standardize_prediction_format,
    save_standardized_predictions
)


@torch.no_grad()
def eval_logits_seq(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    if not logits_list:
        return torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def eval_logits_graph(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        x_nodes = batch["node_feats_raw"].to(device)
        x_edges = batch["edge_index"].to(device)
        x_bidx = batch["batch_index"].to(device)
        y = batch["labels"].to(device)
        logits = model(x_nodes, x_edges, x_bidx)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    if not logits_list:
        return torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def save_preds_standardized(path: str, ids: List[str], labels: np.ndarray, probs: np.ndarray, model_name: str, split: str) -> None:
    """Save predictions using standardized format with auto-flip detection."""
    preds, metadata = standardize_prediction_format(
        ids, labels, probs, model_name, split, auto_flip=True
    )
    
    # Save in both old format (for backwards compatibility) and new format
    with open(path, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps({"id": pred["id"], "label": pred["label"], "prob": pred["prob"]}))
            f.write("\n")
    
    # Save metadata
    dir_path = os.path.dirname(path)
    meta_path = os.path.join(dir_path, f"preds_{split}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Evaluate calibrated lightweight heads (URL/JS CharCNN, DOM GCN)")
    ap.add_argument("--head", choices=["url", "js", "dom", "text", "cheap"], required=True)
    ap.add_argument("--model-dir", required=True, help="Directory with model.pt and optional temp_scale.pt")
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-jsonl", default=None, help="Optional: also dump preds_train.jsonl for this split")
    ap.add_argument("--tag", default="", help="Optional suffix tag for output files (e.g. _full) to avoid overwriting base preds/calibration")
    # Lightweight XAI flags
    ap.add_argument("--lime", action="store_true", help="Generate LIME explanations (slow)")
    ap.add_argument("--shap", action="store_true", help="Generate SHAP explanations (slow)")
    ap.add_argument("--xai-samples", type=int, default=3, help="Number of positive + negative samples each (total ~2*xai-samples) from test for explanations")
    ap.add_argument("--xai-num-features", type=int, default=12, help="Top features/tokens for LIME")
    ap.add_argument("--xai-num-perturb", type=int, default=200, help="Perturbation samples for LIME")
    ap.add_argument("--xai-num-evals", type=int, default=120, help="Approx max_evals for SHAP text explainer")
    ap.add_argument("--xai-max-len", type=int, default=256, help="Max decoded characters/tokens for explanation context (truncation for speed)")
    ap.add_argument("--xai-device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--xai-cache", action="store_true", help="Cache per-sample explanations to avoid recomputation")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets and model
    if args.head == "url":
        tr_ds = UrlSeqDataset(args.val_jsonl)  # used only for count sanity
        va_ds = UrlSeqDataset(args.val_jsonl)
        te_ds = UrlSeqDataset(args.test_jsonl)
        coll = PaddedSeqCollator(pad_idx=0)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = UrlCharCNN().to(device)
        eval_fn = eval_logits_seq
        get_ids = lambda ds: [ds[i].get("id", str(i)) for i in range(len(ds))]
    elif args.head == "js":
        tr_ds = JsSeqDataset(args.val_jsonl)
        va_ds = JsSeqDataset(args.val_jsonl)
        te_ds = JsSeqDataset(args.test_jsonl)
        coll = PaddedSeqCollator(pad_idx=0)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = JsCharCNN().to(device)
        eval_fn = eval_logits_seq
        get_ids = lambda ds: [ds[i].get("id", str(i)) for i in range(len(ds))]
    elif args.head == "dom":
        tr_ds = DomGraphDataset(args.val_jsonl)
        va_ds = DomGraphDataset(args.val_jsonl)
        te_ds = DomGraphDataset(args.test_jsonl)
        coll = DomGraphCollator()
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = DomGCN().to(device)
        eval_fn = eval_logits_graph
        get_ids = lambda ds: [ds[i].get("id", str(i)) for i in range(len(ds))]
    elif args.head == "text":
        from phisdom.data.new_heads import TextSeqDataset
        from phisdom.models.heads import TextCharCNN
        tr_ds = TextSeqDataset(args.val_jsonl)
        va_ds = TextSeqDataset(args.val_jsonl)
        te_ds = TextSeqDataset(args.test_jsonl)
        coll = PaddedSeqCollator(pad_idx=0)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = TextCharCNN().to(device)
        eval_fn = eval_logits_seq
        get_ids = lambda ds: [ds[i].get("id", str(i)) for i in range(len(ds))]
    else:
        from phisdom.data.new_heads import CheapFeaturesDataset, CheapFeaturesCollator
        from phisdom.data.cheap_features import CHEAP_FEATURES
        from phisdom.models.heads import CheapMLP
        tr_ds = CheapFeaturesDataset(args.val_jsonl)
        va_ds = CheapFeaturesDataset(args.val_jsonl)
        te_ds = CheapFeaturesDataset(args.test_jsonl)
        coll = CheapFeaturesCollator()
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
    # Build model with hidden size inferred from config or checkpoint
        hidden = None
        cfg_path = os.path.join(args.model_dir, "model_config.json")
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
                hidden = int(cfg.get("hidden"))
            except Exception:
                hidden = None
        # If still unknown, peek at state dict shape (net.0.weight)
        st_path = os.path.join(args.model_dir, "model.pt")
        if hidden is None and os.path.exists(st_path):
            try:
                st_cpu = torch.load(st_path, map_location="cpu")
                w0 = st_cpu.get("net.0.weight")
                if isinstance(w0, torch.Tensor) and w0.ndim == 2:
                    hidden = int(w0.shape[0])
            except Exception:
                hidden = None
        if hidden is None:
            hidden = 64  # default fallback
        model = CheapMLP(in_dim=len(CHEAP_FEATURES), hidden=hidden).to(device)
        # redefine eval_fn for MLP
        @torch.no_grad()
        def eval_logits_mlp(model, loader, device):
            model.eval()
            logits_list: List[torch.Tensor] = []
            labels_list: List[torch.Tensor] = []
            for batch in loader:
                x = batch["features"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                logits_list.append(logits.detach().cpu())
                labels_list.append(y.detach().cpu())
            if not logits_list:
                return torch.empty((0,)), torch.empty((0,), dtype=torch.long)
            return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
        eval_fn = eval_logits_mlp
        get_ids = lambda ds: [ds[i].get("id", str(i)) for i in range(len(ds))]

    # Load weights
    state_p = os.path.join(args.model_dir, "model.pt")
    if not os.path.exists(state_p):
        raise SystemExit(f"[ERROR] Missing model weights at {state_p}. Train the head first.")
    model.load_state_dict(torch.load(state_p, map_location=device))

    # Val logits for calibration
    val_logits, val_labels_t = eval_fn(model, va_dl, device)
    val_labels = val_labels_t.numpy().astype(int) if val_labels_t.numel() else np.zeros((0,), dtype=int)

    # Load or fit temperature
    scaler_path = os.path.join(args.model_dir, "temp_scale.pt")
    if os.path.exists(scaler_path):
        ckpt = torch.load(scaler_path, map_location="cpu")
        ts = TemperatureScaler()
        # Backward compat: we saved {log_T: tensor}
        ts.log_T = torch.nn.Parameter(ckpt["log_T"].float().clone())
    else:
        ts, _ = fit_temperature(val_logits, torch.from_numpy(val_labels)) if val_logits.numel() else (TemperatureScaler(), None)
        torch.save({"log_T": ts.log_T.detach().cpu()}, scaler_path)

    # Calibrated probabilities
    with torch.no_grad():
        p_val = torch.sigmoid(val_logits / torch.exp(ts.log_T)).numpy() if val_logits.numel() else np.zeros((0,), dtype=float)
    # Test logits
    test_logits, test_labels_t = eval_fn(model, te_dl, device)
    test_labels = test_labels_t.numpy().astype(int) if test_labels_t.numel() else np.zeros((0,), dtype=int)
    with torch.no_grad():
        p_test = torch.sigmoid(test_logits / torch.exp(ts.log_T)).numpy() if test_logits.numel() else np.zeros((0,), dtype=float)

    # Metrics on test
    pr = pr_auc_safe(test_labels.tolist(), p_test.tolist()) if p_test.size else None
    roc = roc_auc_safe(test_labels.tolist(), p_test.tolist()) if p_test.size else None
    thresholds = {}
    for tpr in (0.95, 0.90):
        fpr, thr = fpr_at_tpr(test_labels.tolist(), p_test.tolist(), tpr) if p_test.size else (1.0, 1.0)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    # Save predictions using standardized format
    ids_val = get_ids(va_ds)
    ids_test = get_ids(te_ds)
    tag = args.tag
    def _with_tag(base: str) -> str:
        # Insert tag before extension if present, else append
        if not tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}{tag}{ext}" if ext else f"{base}{tag}"

    save_preds_standardized(os.path.join(args.model_dir, _with_tag("preds_val.jsonl")), ids_val, val_labels, p_val, args.head, "val")
    save_preds_standardized(os.path.join(args.model_dir, _with_tag("preds_test.jsonl")), ids_test, test_labels, p_test, args.head, "test")

    # Optional: compute and save train preds
    if args.train_jsonl:
        # Build train loader matching head type
        if args.head == "url":
            tr_ds = UrlSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [tr_ds[i].get("id", str(i)) for i in range(len(tr_ds))]
        elif args.head == "js":
            tr_ds = JsSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [tr_ds[i].get("id", str(i)) for i in range(len(tr_ds))]
        elif args.head == "dom":
            tr_ds = DomGraphDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=DomGraphCollator())  # type: ignore[arg-type]
            tr_eval = eval_logits_graph
            tr_ids = [tr_ds[i].get("id", str(i)) for i in range(len(tr_ds))]
        elif args.head == "text":
            from phisdom.data.new_heads import TextSeqDataset
            tr_ds = TextSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [tr_ds[i].get("id", str(i)) for i in range(len(tr_ds))]
        else:
            from phisdom.data.new_heads import CheapFeaturesDataset, CheapFeaturesCollator
            tr_ds = CheapFeaturesDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=CheapFeaturesCollator())  # type: ignore[arg-type]
            @torch.no_grad()
            def tr_eval(model, loader, device):
                model.eval()
                logits_list: List[torch.Tensor] = []
                labels_list: List[torch.Tensor] = []
                for batch in loader:
                    x = batch["features"].to(device)
                    y = batch["labels"].to(device)
                    logits = model(x)
                    logits_list.append(logits.detach().cpu())
                    labels_list.append(y.detach().cpu())
                if not logits_list:
                    return torch.empty((0,)), torch.empty((0,), dtype=torch.long)
                return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
            tr_ids = [tr_ds[i].get("id", str(i)) for i in range(len(tr_ds))]
        tr_logits, tr_labels_t = tr_eval(model, tr_dl, device)
        with torch.no_grad():
            p_train = torch.sigmoid(tr_logits / torch.exp(ts.log_T)).numpy() if tr_logits.numel() else np.zeros((0,), dtype=float)
        save_preds_standardized(os.path.join(args.model_dir, _with_tag("preds_train.jsonl")), tr_ids, tr_labels_t.numpy().astype(int) if tr_labels_t.numel() else np.zeros((0,), dtype=int), p_train, args.head, "train")

    # Save calibration/metrics
    cal = {
        "T": float(torch.exp(ts.log_T).item()),
        "metrics": {"pr_auc": pr, "roc_auc": roc},
        "thresholds": thresholds,
    }
    cal_path = os.path.join(args.model_dir, _with_tag("calibration_eval.json"))
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    print(json.dumps(cal, indent=2))

    # ----------------------------
    # Lightweight Explanations
    # ----------------------------
    if (args.lime or args.shap) and test_logits.numel():
        try:
            from lime.lime_text import LimeTextExplainer  # noqa: F401
        except Exception:
            if args.lime:
                print("[INFO] LIME not installed; skipping LIME")
                args.lime = False  # type: ignore
        try:
            import shap  # noqa: F401
        except Exception:
            if args.shap:
                print("[INFO] SHAP not installed; skipping SHAP")
                args.shap = False  # type: ignore

    def _extract_raw(ds, idx: int) -> str:
        # Attempt to reconstruct raw sequence for char CNN / cheap features
        row = ds[idx]
        if args.head in ("url","js","text"):
            seq = row.get("seq") or []
            # In datasets char ids are ints; we can't invert reliably without alphabet.
            # For explanations we treat each int as a pseudo-token string idN
            return " ".join(f"id{int(t)}" for t in seq[: args.xai_max_len])
        if args.head == "dom":
            # Represent each node as type-depth-xbin token
            g = row.get("graph") or {}
            nodes = g.get("nodes") or []
            toks = []
            for n in nodes[: args.xai_max_len]:
                if isinstance(n, dict):
                    toks.append(f"t{n.get('t_hash',0)}d{n.get('depth',0)}x{n.get('xbin',0)}")
                elif isinstance(n, (list, tuple)) and len(n) >= 4:
                    toks.append(f"t{int(n[0])}d{int(n[2])}x{int(n[3])}")
            return " ".join(toks)
        if args.head == "cheap":
            feats = row.get("feats") or row.get("features") or []
            return " ".join(f"f{i}:{float(v):.3g}" for i,v in enumerate(feats[: args.xai_max_len]))
        return ""

    if (args.lime or args.shap) and test_logits.numel():
        out_dir = os.path.join(args.model_dir, "xai")
        os.makedirs(out_dir, exist_ok=True)
        # Build sample pools
        # We need original test dataset object: te_ds
        # Collect label & prob info
        probs_test = p_test
        labels_test = test_labels
        idx_pos = [i for i,l in enumerate(labels_test) if l==1]
        idx_neg = [i for i,l in enumerate(labels_test) if l==0]
        # Prefer high-confidence examples in middle probability band to avoid degenerate (very close to 0/1) which reduce explanation variety
        def _pick(idx_list, n):
            if not idx_list:
                return []
            # sort by distance from 0.5 descending (prefer ambiguous) or fallback
            scored = sorted(idx_list, key=lambda i: abs(0.5 - float(probs_test[i])))
            return [scored[i] for i in range(min(n, len(scored)))]
        sel_pos = _pick(idx_pos, args.xai_samples)
        sel_neg = _pick(idx_neg, args.xai_samples)
        sel = sel_pos + sel_neg
        if not sel:
            print("[INFO] No samples selected for XAI")
        else:
            # Build prediction function mapping list[str] -> proba matrix (binary)
            device_xai = torch.device("cuda" if (args.xai_device=="cuda" and torch.cuda.is_available()) else "cpu")
            model_xai = model.to(device_xai)
            model_xai.eval()
            def predict_proba(texts: List[str]):
                # Our pseudo-tokenization: tokens are space separated already
                feats_batches: List[List[int]] = []
                for t in texts:
                    toks = t.split()
                    # map idN -> int(N), f#:v tokens -> treat value sign
                    enc: List[int] = []
                    for tok in toks[: args.xai_max_len]:
                        if tok.startswith("id") and tok[2:].isdigit():
                            enc.append(int(tok[2:]))
                        elif tok.startswith("t") and 'd' in tok:  # dom composite token -> hash
                            # crude hash
                            enc.append(abs(hash(tok)) % 5000)
                        elif tok.startswith("f") and ":" in tok:
                            try:
                                # cheap feature; we bucket by index sign (index * sign)
                                idx, val = tok[1:].split(":",1)
                                idx_i = int(idx)
                                val_f = float(val)
                                enc.append(idx_i if val_f >=0 else (idx_i+1000))
                            except Exception:
                                enc.append(0)
                        else:
                            # fallback hash
                            enc.append(abs(hash(tok)) % 5000)
                    feats_batches.append(enc)
                # pad
                maxL = max(1, max((len(s) for s in feats_batches), default=1))
                X = torch.zeros((len(feats_batches), maxL), dtype=torch.long)
                for i,s in enumerate(feats_batches):
                    if s:
                        X[i,:len(s)] = torch.tensor(s[:maxL], dtype=torch.long)
                with torch.no_grad():
                    out = model_xai(X.to(device_xai))
                    logits = out.detach().cpu()
                    probs = torch.sigmoid(logits).numpy().reshape(-1)
                p0 = 1.0 - probs
                return np.vstack([p0, probs]).T
            # LIME -----------------------------------------------------------------
            if args.lime:
                try:
                    from lime.lime_text import LimeTextExplainer
                    explainer = LimeTextExplainer(class_names=["benign","phish"])  # type: ignore
                    lime_dir = os.path.join(out_dir, "lime")
                    os.makedirs(lime_dir, exist_ok=True)
                    for i in sel:
                        raw = _extract_raw(te_ds, i)
                        rid = getattr(te_ds, '_index', list(range(len(te_ds))))[i] if hasattr(te_ds,'_index') else i
                        cache_path = os.path.join(lime_dir, f"lime_{rid}.json")
                        if args.xai_cache and os.path.exists(cache_path):
                            continue
                        try:
                            exp = explainer.explain_instance(raw, predict_proba, num_features=args.xai_num_features, num_samples=args.xai_num_perturb)
                            toks_weights = exp.as_list(label=1)
                            out_record = {"id": str(rid), "tokens": [t for t,_ in toks_weights], "weights": [float(w) for _,w in toks_weights]}
                            with open(cache_path, "w", encoding="utf-8") as f:
                                json.dump(out_record, f)
                            print(f"[INFO] LIME saved {cache_path}")
                        except Exception as e:
                            print(f"[WARN] LIME failed for id={rid}: {e}")
                except Exception as e:
                    print(f"[WARN] LIME initialization failed: {e}")
            # SHAP -----------------------------------------------------------------
            if args.shap:
                try:
                    import shap
                    # Use partition explainer on text tokens (treat each token as independent)
                    explainer = shap.Explainer(predict_proba)  # type: ignore
                    shap_dir = os.path.join(out_dir, "shap")
                    os.makedirs(shap_dir, exist_ok=True)
                    for i in sel:
                        raw = _extract_raw(te_ds, i)
                        rid = getattr(te_ds, '_index', list(range(len(te_ds))))[i] if hasattr(te_ds,'_index') else i
                        cache_path = os.path.join(shap_dir, f"shap_{rid}.json")
                        if args.xai_cache and os.path.exists(cache_path):
                            continue
                        try:
                            sv = explainer([raw], max_evals=args.xai_num_evals)  # type: ignore
                            # Extract per-token values
                            try:
                                vals = sv.values[0]
                                toks = raw.split()
                                # Align lengths
                                L = min(len(toks), len(vals))
                                record = {"id": str(rid), "tokens": toks[:L], "weights": [float(v) for v in vals[:L]]}
                            except Exception:
                                record = {"id": str(rid), "tokens": raw.split(), "weights": []}
                            with open(cache_path, "w", encoding="utf-8") as f:
                                json.dump(record, f)
                            print(f"[INFO] SHAP saved {cache_path}")
                        except Exception as e:
                            print(f"[WARN] SHAP failed for id={rid}: {e}")
                except Exception as e:
                    print(f"[WARN] SHAP initialization failed: {e}")


if __name__ == "__main__":
    main()
