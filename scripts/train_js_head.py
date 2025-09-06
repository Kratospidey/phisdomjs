#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phisdom.data.new_heads import JsSeqDataset, PaddedSeqCollator
from phisdom.models.heads import JsCharCNN
from phisdom.models.calibration import fit_temperature


def train_one_epoch(model, loader, opt, sched, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].float().to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_logits(model, loader, device):
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


def main():
    ap = argparse.ArgumentParser(description="Train JS Char-CNN head and calibrate")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--output-dir", default="artifacts/js_charcnn")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.15)
    # Phase 6: allow training from raw augmented text when js_charseq missing
    ap.add_argument("--raw-field", type=str, default=None, help="Optional raw JS text field to encode on-the-fly (e.g., js_augmented)")
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = JsSeqDataset(args.train_jsonl, raw_field=args.raw_field)
    va_ds = JsSeqDataset(args.val_jsonl, raw_field=args.raw_field)
    coll = PaddedSeqCollator(pad_idx=0, max_len=args.max_len)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)  # type: ignore[arg-type]
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]

    model = JsCharCNN(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tr_dl) * args.epochs))

    best_val = float("inf")
    patience = 3
    bad = 0
    bcel = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, tr_dl, opt, sched, device)
        logits, labels = eval_logits(model, va_dl, device)
        val_loss = float(bcel(logits, labels.float()).item()) if logits.numel() else float("inf")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break

    # Reload best and compute final val logits
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt"), map_location=device))
    val_logits, val_labels = eval_logits(model, va_dl, device)
    # Fit temperature scaler
    if val_logits.numel():
        ts, meta = fit_temperature(val_logits, val_labels)
        torch.save({"log_T": ts.log_T.detach().cpu()}, os.path.join(args.output_dir, "temp_scale.pt"))
        with open(os.path.join(args.output_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"T": meta.T, "val_loss_before": meta.val_loss_before, "val_loss_after": meta.val_loss_after}, f, indent=2)
        torch.save({"logits": val_logits, "labels": val_labels}, os.path.join(args.output_dir, "val_logits.pt"))


if __name__ == "__main__":
    main()
