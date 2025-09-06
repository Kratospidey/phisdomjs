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

from phisdom.data.multimodal import MultiModalDataset, MultiModalCollator
from phisdom.models.fusion import CrossModalTransformerFusion
from phisdom.models.calibration import fit_temperature, TemperatureScaler
from phisdom.metrics import pr_auc, roc_auc, fpr_at_tpr


def train_one_epoch(model, loader, opt, sched, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for batch in loader:
        labels = batch.pop("labels").float().to(device)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    ids: List[str] = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        ids.extend(batch.get("ids", []))
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        logits = model(batch)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    if not logits_list:
        return ids, torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    return ids, torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Train cross-modal transformer fusion with cross-attention")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--out-dir", default="artifacts/fusion_xattn")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    # modality toggles
    ap.add_argument("--no-url", action="store_true")
    ap.add_argument("--no-js", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-dom", action="store_true")
    ap.add_argument("--no-cheap", action="store_true")
    # adv/canon
    ap.add_argument("--js-raw-field", default=None)
    ap.add_argument("--no-js-canonicalize", action="store_true")
    ap.add_argument("--html-canonicalize", action="store_true")
    ap.add_argument("--html-field", default="html")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = MultiModalDataset(
        args.train_jsonl,
        use_url=not args.no_url,
        use_js=not args.no_js,
        use_text=not args.no_text,
        use_dom=not args.no_dom,
        use_cheap=not args.no_cheap,
        js_raw_field=args.js_raw_field,
        js_canonicalize=not args.no_js_canonicalize,
    html_canonicalize=bool(args.html_canonicalize),
    html_field=args.html_field,
    )
    va_ds = MultiModalDataset(
        args.val_jsonl,
        use_url=not args.no_url,
        use_js=not args.no_js,
        use_text=not args.no_text,
        use_dom=not args.no_dom,
        use_cheap=not args.no_cheap,
        js_raw_field=args.js_raw_field,
        js_canonicalize=not args.no_js_canonicalize,
    html_canonicalize=bool(args.html_canonicalize),
    html_field=args.html_field,
    )
    te_ds = MultiModalDataset(
        args.test_jsonl,
        use_url=not args.no_url,
        use_js=not args.no_js,
        use_text=not args.no_text,
        use_dom=not args.no_dom,
        use_cheap=not args.no_cheap,
        js_raw_field=args.js_raw_field,
        js_canonicalize=not args.no_js_canonicalize,
    html_canonicalize=bool(args.html_canonicalize),
    html_field=args.html_field,
    )
    coll = MultiModalCollator()
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)  # type: ignore[arg-type]
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]

    model = CrossModalTransformerFusion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tr_dl) * args.epochs))

    bcel = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    bad = 0
    patience = 3
    for ep in range(args.epochs):
        _ = train_one_epoch(model, tr_dl, opt, sched, device)
        ids_v, logits_v, labels_v = eval_logits(model, va_dl, device)
        val_loss = float(bcel(logits_v, labels_v.float()).item()) if logits_v.numel() else float("inf")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), map_location=device))
    ids_v, logits_v, labels_v = eval_logits(model, va_dl, device)
    ids_t, logits_t, labels_t = eval_logits(model, te_dl, device)

    # Fit temp scaling
    if logits_v.numel():
        ts, meta = fit_temperature(logits_v, labels_v)
        torch.save({"log_T": ts.log_T.detach().cpu()}, os.path.join(args.out_dir, "temp_scale.pt"))
        with open(os.path.join(args.out_dir, "calibration_val.json"), "w", encoding="utf-8") as f:
            json.dump({"T": meta.T, "val_loss_before": meta.val_loss_before, "val_loss_after": meta.val_loss_after}, f, indent=2)
    else:
        ts = TemperatureScaler()

    with torch.no_grad():
        p_val = torch.sigmoid(logits_v / torch.exp(ts.log_T)).numpy() if logits_v.numel() else np.zeros((0,), dtype=float)
        p_test = torch.sigmoid(logits_t / torch.exp(ts.log_T)).numpy() if logits_t.numel() else np.zeros((0,), dtype=float)

    # Metrics + thresholds on test
    pr = pr_auc(labels_t.numpy().astype(int).tolist(), p_test.tolist()) if p_test.size else 0.0
    roc = roc_auc(labels_t.numpy().astype(int).tolist(), p_test.tolist()) if p_test.size else 0.0
    thresholds = {}
    for tpr in (0.95, 0.90):
        fpr, thr = fpr_at_tpr(labels_t.numpy().astype(int).tolist(), p_test.tolist(), tpr) if p_test.size else (1.0, 1.0)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    cal = {"metrics": {"pr_auc": pr, "roc_auc": roc}, "thresholds": thresholds}
    with open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    # Dump preds in the same format as other heads for alignment downstream
    def dump(path: str, ids: List[str], labels: torch.Tensor, probs: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for id_, y, p in zip(ids, labels.numpy().astype(int).tolist(), probs.tolist()):
                f.write(json.dumps({"id": id_, "label": int(y), "prob": float(p)}))
                f.write("\n")
    dump(os.path.join(args.out_dir, "preds_val.jsonl"), ids_v, labels_v, p_val)
    dump(os.path.join(args.out_dir, "preds_test.jsonl"), ids_t, labels_t, p_test)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
