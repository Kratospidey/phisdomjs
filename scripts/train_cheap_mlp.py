#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
from typing import List, Dict, Any
from contextlib import suppress

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phisdom.data.new_heads import CheapFeaturesDataset, CheapFeaturesCollator
from phisdom.data.cheap_features import CHEAP_FEATURES
from phisdom.models.heads import CheapMLP
from phisdom.models.calibration import fit_temperature


def train_one_epoch(model, loader, opt, sched, device, crit, log_grad_norm: bool = False, feat_var_tracker: Dict[str, Any] | None = None, diag_interval: int = 0, grad_snapshots: List[float] | None = None):
    model.train()
    total = 0.0
    n = 0
    grad_norms: List[float] = [] if log_grad_norm else []
    for b_idx, batch in enumerate(loader):
        x = batch["features"].to(device)
        y = batch["labels"].float().to(device)

        # Update feature running variance (each sample an observation)
        if feat_var_tracker is not None:
            with torch.no_grad():
                flat = x.detach()
                if feat_var_tracker['count'] == 0:
                    feat_var_tracker['mean'] = flat.mean(0)
                    feat_var_tracker['M2'] = torch.zeros_like(feat_var_tracker['mean'])
                    feat_var_tracker['count'] = flat.size(0)
                else:
                    batch_mean = flat.mean(0)
                    batch_count = flat.size(0)
                    delta = batch_mean - feat_var_tracker['mean']
                    total_count = feat_var_tracker['count'] + batch_count
                    new_mean = feat_var_tracker['mean'] + delta * (batch_count / total_count)
                    batch_var = flat.var(0, unbiased=False)
                    feat_var_tracker['M2'] += batch_var * batch_count + (delta ** 2) * feat_var_tracker['count'] * batch_count / total_count
                    feat_var_tracker['mean'] = new_mean
                    feat_var_tracker['count'] = total_count

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        if log_grad_norm:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    pn = p.grad.detach().data.norm(2).item()
                    total_norm += pn ** 2
            gnorm = total_norm ** 0.5
            grad_norms.append(gnorm)
            if diag_interval > 0 and (b_idx % diag_interval == 0) and grad_snapshots is not None:
                grad_snapshots.append(float(gnorm))
        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n), (sum(grad_norms) / max(1, len(grad_norms)) if log_grad_norm else None)


@torch.no_grad()
def eval_logits(model, loader, device):
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


def main():
    ap = argparse.ArgumentParser(description="Train Cheap-Feature MLP and calibrate")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--output-dir", default="artifacts/cheap_mlp")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    # Diagnostics & weighting
    ap.add_argument("--pos-weight", type=float, default=None, help="Manual positive class weight; if unset and --auto-pos-weight, compute from train labels")
    ap.add_argument("--auto-pos-weight", action="store_true")
    ap.add_argument("--diagnostics-dir", default=None, help="Directory for diagnostics (default output-dir/diagnostics)")
    ap.add_argument("--log-hist-every", type=int, default=1, help="Epoch frequency for logits histogram")
    ap.add_argument("--no-grad-norm", action="store_true")
    ap.add_argument("--no-feature-var", action="store_true")
    ap.add_argument("--loss", choices=["bce","focal"], default="bce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=None)
    ap.add_argument("--diag-interval", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = CheapFeaturesDataset(args.train_jsonl)
    va_ds = CheapFeaturesDataset(args.val_jsonl)
    coll = CheapFeaturesCollator()
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)  # type: ignore[arg-type]
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]

    model = CheapMLP(in_dim=len(CHEAP_FEATURES), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tr_dl) * args.epochs))

    # Class weighting
    pos_weight_tensor = None
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor(float(args.pos_weight), device=device)
    elif args.auto_pos_weight:
        pos = 0
        neg = 0
        for item in tr_ds:
            y = int(item['label']) if 'label' in item else int(item.get('labels', 0))
            if y == 1:
                pos += 1
            else:
                neg += 1
        if pos > 0:
            pw = neg / pos
            print(f"Auto-computed pos_weight = {pw:.3f} (neg={neg}, pos={pos})")
            pos_weight_tensor = torch.tensor(pw, device=device)
        else:
            print("Warning: No positives found for pos_weight computation")

    if args.loss == "bce":
        if pos_weight_tensor is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            crit = nn.BCEWithLogitsLoss()
    else:
        gamma = float(args.focal_gamma)
        if args.focal_alpha is not None:
            alpha = float(args.focal_alpha)
        else:
            pos = 0; neg = 0
            for item in tr_ds:
                yv = int(item['label']) if 'label' in item else int(item.get('labels', 0))
                if yv == 1: pos += 1
                else: neg += 1
            tot = pos + neg
            alpha = (pos / tot) if tot > 0 else 0.5
        print(f"Using Focal Loss (gamma={gamma}, alpha={alpha:.4f})")
        def focal_loss(logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p = torch.sigmoid(logits)
            pt = p * targets + (1 - p) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            return (alpha_t * (1 - pt).pow(gamma) * bce).mean()
        crit = focal_loss

    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    feat_var_tracker = None if args.no_feature_var else {"count": 0, "mean": None, "M2": None}
    history: Dict[str, List[float]] = {"val_loss": [], "train_loss": [], "grad_norm": []}

    def dump_histogram(epoch: int, logits: torch.Tensor, split: str):
        if logits.numel() == 0 or epoch % max(1, args.log_hist_every) != 0:
            return
        with suppress(ImportError):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            arr = logits.detach().cpu().float()
            plt.figure(figsize=(4, 3))
            plt.hist(arr.numpy(), bins=50, color='purple', alpha=0.8)
            plt.title(f"Logits Histogram ({split}) epoch {epoch}")
            plt.xlabel("logit")
            plt.ylabel("count")
            out_path = os.path.join(diagnostics_dir, f"logits_{split}_ep{epoch}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

    best_val = float("inf")
    patience = 3
    bad = 0
    bcel = crit
    grad_snapshots: List[float] = []
    for ep in range(args.epochs):
        tr_loss, grad_norm = train_one_epoch(
            model,
            tr_dl,
            opt,
            sched,
            device,
            crit,
            log_grad_norm=not args.no_grad_norm,
            feat_var_tracker=feat_var_tracker,
            diag_interval=args.diag_interval,
            grad_snapshots=grad_snapshots,
        )
        logits, labels = eval_logits(model, va_dl, device)
        val_loss = float(bcel(logits, labels.float()).item()) if logits.numel() else float("inf")
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        if grad_norm is not None:
            history["grad_norm"].append(float(grad_norm))
        dump_histogram(ep, logits, "val")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break

    # Persist diagnostics
    try:
        diag: Dict[str, Any] = {"history": history, "grad_snapshots": grad_snapshots[:200]}
        if feat_var_tracker is not None and feat_var_tracker['count'] > 1:
            var = feat_var_tracker['M2'] / (feat_var_tracker['count'] - 1)
            diag['feature_mean'] = feat_var_tracker['mean'].detach().cpu().tolist()
            diag['feature_var'] = var.detach().cpu().tolist()
        with open(os.path.join(diagnostics_dir, "training_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write diagnostics: {e}")

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt"), map_location=device))
    # Save a tiny model config for robust evaluation
    try:
        cfg = {"in_dim": len(CHEAP_FEATURES), "hidden": int(args.hidden), "dropout": 0.1}
        with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass
    val_logits, val_labels = eval_logits(model, va_dl, device)
    if val_logits.numel():
        ts, meta = fit_temperature(val_logits, val_labels)
        torch.save({"log_T": ts.log_T.detach().cpu()}, os.path.join(args.output_dir, "temp_scale.pt"))
        with open(os.path.join(args.output_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"T": meta.T, "val_loss_before": meta.val_loss_before, "val_loss_after": meta.val_loss_after}, f, indent=2)
        torch.save({"logits": val_logits, "labels": val_labels}, os.path.join(args.output_dir, "val_logits.pt"))


if __name__ == "__main__":
    main()
