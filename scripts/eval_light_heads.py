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
from phisdom.metrics import pr_auc, roc_auc, fpr_at_tpr


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


def save_preds(path: str, ids: List[str], labels: np.ndarray, probs: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for id_, y, p in zip(ids, labels.tolist(), probs.tolist()):
            f.write(json.dumps({"id": id_, "label": int(y), "prob": float(p)}))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Evaluate calibrated lightweight heads (URL/JS CharCNN, DOM GCN)")
    ap.add_argument("--head", choices=["url", "js", "dom", "text", "cheap"], required=True)
    ap.add_argument("--model-dir", required=True, help="Directory with model.pt and optional temp_scale.pt")
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-jsonl", default=None, help="Optional: also dump preds_train.jsonl for this split")
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
        get_ids = lambda ds: [r.get("id", str(i)) for i, r in enumerate(ds.rows)]
    elif args.head == "js":
        tr_ds = JsSeqDataset(args.val_jsonl)
        va_ds = JsSeqDataset(args.val_jsonl)
        te_ds = JsSeqDataset(args.test_jsonl)
        coll = PaddedSeqCollator(pad_idx=0)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = JsCharCNN().to(device)
        eval_fn = eval_logits_seq
        get_ids = lambda ds: [r.get("id", str(i)) for i, r in enumerate(ds.rows)]
    elif args.head == "dom":
        tr_ds = DomGraphDataset(args.val_jsonl)
        va_ds = DomGraphDataset(args.val_jsonl)
        te_ds = DomGraphDataset(args.test_jsonl)
        coll = DomGraphCollator()
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
        model = DomGCN().to(device)
        eval_fn = eval_logits_graph
        get_ids = lambda ds: [r.get("id", str(i)) for i, r in enumerate(ds.rows)]
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
        get_ids = lambda ds: [r.get("id", str(i)) for i, r in enumerate(ds.rows)]
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
        model = CheapMLP(in_dim=len(CHEAP_FEATURES)).to(device)
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
        get_ids = lambda ds: [r.get("id", str(i)) for i, r in enumerate(ds.rows)]

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
    pr = pr_auc(test_labels.tolist(), p_test.tolist()) if p_test.size else 0.0
    roc = roc_auc(test_labels.tolist(), p_test.tolist()) if p_test.size else 0.0
    thresholds = {}
    for tpr in (0.95, 0.90):
        fpr, thr = fpr_at_tpr(test_labels.tolist(), p_test.tolist(), tpr) if p_test.size else (1.0, 1.0)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    # Save preds
    ids_val = get_ids(va_ds)
    ids_test = get_ids(te_ds)
    save_preds(os.path.join(args.model_dir, "preds_val.jsonl"), ids_val, val_labels, p_val)
    save_preds(os.path.join(args.model_dir, "preds_test.jsonl"), ids_test, test_labels, p_test)

    # Optional: compute and save train preds
    if args.train_jsonl:
        # Build train loader matching head type
        if args.head == "url":
            tr_ds = UrlSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [r.get("id", str(i)) for i, r in enumerate(tr_ds.rows)]
        elif args.head == "js":
            tr_ds = JsSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [r.get("id", str(i)) for i, r in enumerate(tr_ds.rows)]
        elif args.head == "dom":
            tr_ds = DomGraphDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=DomGraphCollator())  # type: ignore[arg-type]
            tr_eval = eval_logits_graph
            tr_ids = [r.get("id", str(i)) for i, r in enumerate(tr_ds.rows)]
        elif args.head == "text":
            from phisdom.data.new_heads import TextSeqDataset
            tr_ds = TextSeqDataset(args.train_jsonl)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedSeqCollator(pad_idx=0))  # type: ignore[arg-type]
            tr_eval = eval_logits_seq
            tr_ids = [r.get("id", str(i)) for i, r in enumerate(tr_ds.rows)]
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
            tr_ids = [r.get("id", str(i)) for i, r in enumerate(tr_ds.rows)]
        tr_logits, tr_labels_t = tr_eval(model, tr_dl, device)
        with torch.no_grad():
            p_train = torch.sigmoid(tr_logits / torch.exp(ts.log_T)).numpy() if tr_logits.numel() else np.zeros((0,), dtype=float)
        save_preds(os.path.join(args.model_dir, "preds_train.jsonl"), tr_ids, tr_labels_t.numpy().astype(int) if tr_labels_t.numel() else np.zeros((0,), dtype=int), p_train)

    # Save calibration/metrics
    cal = {
        "T": float(torch.exp(ts.log_T).item()),
        "metrics": {"pr_auc": pr, "roc_auc": roc},
        "thresholds": thresholds,
    }
    with open(os.path.join(args.model_dir, "calibration_eval.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
