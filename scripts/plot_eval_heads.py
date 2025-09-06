#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from phisdom.metrics import pr_auc


def read_preds(path: str) -> Tuple[List[int], List[float]]:
    ys: List[int] = []
    ps: List[float] = []
    if not os.path.exists(path):
        return ys, ps
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ys.append(int(obj.get("label", 0)))
            ps.append(float(obj.get("prob", 0.0)))
    return ys, ps


def precision_recall_curve(y: List[int], p: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # Simple PR curve; relies on sorting scores
    arr = sorted(zip(p, y), key=lambda t: -t[0])
    tp = 0
    fp = 0
    P: List[float] = []
    R: List[float] = []
    pos = sum(y)
    for score, yt in arr:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, pos)
        P.append(prec)
        R.append(rec)
    return np.array(R), np.array(P)


def main():
    ap = argparse.ArgumentParser(description="Plot PR curves for calibrated heads and fusion")
    ap.add_argument("--url-dir", default="artifacts/url_head")
    ap.add_argument("--js-dir", default="artifacts/js_charcnn")
    ap.add_argument("--dom-dir", default="artifacts/dom_gcn")
    ap.add_argument("--fusion-dir", default="artifacts/fusion")
    ap.add_argument("--text-dir", default="artifacts/text_head")
    ap.add_argument("--cheap-dir", default="artifacts/cheap_mlp")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", default="artifacts/fusion/heads_pr_curves.png")
    args = ap.parse_args()

    paths = {
        "URL": os.path.join(args.url_dir, f"preds_{args.split}.jsonl"),
        "JS": os.path.join(args.js_dir, f"preds_{args.split}.jsonl"),
        "DOM (MarkupLM)": os.path.join(args.dom_dir, f"preds_{args.split}.jsonl"),
        "DOM (Light GCN)": os.path.join(os.path.dirname(args.dom_dir) + "/dom_gcn", f"preds_{args.split}.jsonl"),
        "Text": os.path.join(args.text_dir, f"preds_{args.split}.jsonl"),
        "Cheap MLP": os.path.join(args.cheap_dir, f"preds_{args.split}.jsonl"),
        "Fused": os.path.join(args.fusion_dir, f"preds_{args.split}.jsonl"),
    }

    plt.figure(figsize=(7, 5))
    for name, path in paths.items():
        y, p = read_preds(path)
        if not p:
            continue
        R, P = precision_recall_curve(y, p)
        auc = pr_auc(y, p)
        plt.plot(R, P, label=f"{name} (AP={auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Calibrated)")
    plt.legend(loc="best")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
