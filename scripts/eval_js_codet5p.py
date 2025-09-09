#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from phisdom.data.js import JsonlJsDataset
from phisdom.calibration import TemperatureScaler
from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr


def predict(model: T5EncoderModel, tokenizer, ds: JsonlJsDataset, max_length: int, clf_path: str) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to device
    model.to(device)  # type: ignore[misc]
    model.eval()
    clf_w = None
    if os.path.exists(clf_path):
        state = torch.load(clf_path, map_location=device)
        w = state.get("weight")
        b = state.get("bias")
        if w is not None:
            clf_w = (w.to(device), b.to(device) if b is not None else None)
    probs: List[float] = []
    bs = 8
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch = [ds[j] for j in range(i, min(len(ds), i + bs))]
            texts = [r.get("text", "") for r in batch]
            enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            if clf_w is not None:
                W, B = clf_w
                logits = pooled @ W.T
                if B is not None:
                    logits = logits + B
            else:
                logits = pooled @ torch.zeros((pooled.size(1), 2), device=device)
            logits = logits.detach().cpu().numpy()
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits - m)
            p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
            probs.extend(p1.tolist())
    return np.array(probs, dtype=float)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5EncoderModel.from_pretrained(args.model_dir)

    val_ds = JsonlJsDataset(args.val_jsonl)
    test_ds = JsonlJsDataset(args.test_jsonl)

    # Predict
    clf_path = os.path.join(args.model_dir, "classifier.pt")
    p_val = predict(model, tokenizer, val_ds, args.max_length, clf_path)
    p_test = predict(model, tokenizer, test_ds, args.max_length, clf_path)
    y_val = np.array([int(val_ds[i].get("label", 0)) for i in range(len(val_ds))], dtype=int)
    y_test = np.array([int(test_ds[i].get("label", 0)) for i in range(len(test_ds))], dtype=int)

    # Calibrate on val
    # Convert probabilities to logits for temperature scaling stability
    eps = 1e-6
    z_val = np.log(np.clip(p_val, eps, 1 - eps)) - np.log(1 - np.clip(p_val, eps, 1 - eps))
    ts = TemperatureScaler(is_logit=True)
    T = ts.fit(y_val.tolist(), z_val.tolist())
    p_val_cal = np.array(ts.transform(z_val.tolist()))

    # Calibrate test
    z_test = np.log(np.clip(p_test, eps, 1 - eps)) - np.log(1 - np.clip(p_test, eps, 1 - eps))
    p_test_cal = np.array(ts.transform(z_test.tolist()))

    # Metrics
    pr = pr_auc_safe(y_test.tolist(), p_test_cal.tolist())
    roc = roc_auc_safe(y_test.tolist(), p_test_cal.tolist())

    thresholds = {}
    for tpr in args.tpr:
        fpr, thr = fpr_at_tpr(y_test.tolist(), p_test_cal.tolist(), tpr)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    cal = {"temperature": T, "metrics": {"pr_auc": pr, "roc_auc": roc}, "thresholds": thresholds}
    os.makedirs(args.model_dir, exist_ok=True)
    tag = getattr(args, "tag", "")
    def _with_tag(base: str) -> str:
        if not tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}{tag}{ext}" if ext else f"{base}{tag}"
    with open(os.path.join(args.model_dir, _with_tag("calibration.json")), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    def dump_preds(path: str, ds: JsonlJsDataset, probs: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for i, p in enumerate(probs.tolist()):
                r = ds[i]
                obj = {"id": r.get("id", str(i)), "label": int(r.get("label", 0)), "prob": float(p)}
                f.write(json.dumps(obj))
                f.write("\n")

    # reuse _with_tag defined above

    dump_preds(os.path.join(args.model_dir, _with_tag("preds_val.jsonl")), val_ds, p_val_cal)
    dump_preds(os.path.join(args.model_dir, _with_tag("preds_test.jsonl")), test_ds, p_test_cal)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate CodeT5+ JS classifier")
    ap.add_argument("--model-dir", default="artifacts/js_codet5p")
    ap.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    ap.add_argument("--test-jsonl", default="data/pages_test.jsonl")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
    ap.add_argument("--tag", default="", help="Optional suffix tag for output prediction/calibration files (e.g. _full)")
    args = ap.parse_args()
    main(args)
