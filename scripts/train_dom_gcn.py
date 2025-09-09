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

from phisdom.data.new_heads import DomGraphDataset, DomGraphCollator
from phisdom.models.heads import DomGCN
from phisdom.models.calibration import fit_temperature


def train_one_epoch(model, loader, opt, sched, device, crit, log_grad_norm: bool = False, feat_var_tracker: Dict[str, Any] | None = None):
    """Train one epoch; optionally accumulate feature variance statistics & grad norms.

    feat_var_tracker structure (mutated in place):
        {
          'count': int total_samples,
          'mean': running mean tensor (feature_dim,),
          'M2': running squared diff accumulator tensor (feature_dim,)
        }
    Uses Welford's algorithm for numerical stability.
    """
    model.train()
    total = 0.0
    n = 0
    grad_norms: List[float] = [] if log_grad_norm else []

    for batch in loader:
        x_nodes = batch["node_feats_raw"].to(device)
        x_edges = batch["edge_index"].to(device)
        x_bidx = batch["batch_index"].to(device)
        y = batch["labels"].float().to(device)

        # Feature variance tracking (node features pre-aggregation)
        if feat_var_tracker is not None:
            with torch.no_grad():
                # Ensure floating dtype for statistics (datasets may provide long/int features)
                flat = x_nodes.detach().float()  # shape (total_nodes_in_batch, feat_dim)
                # Collapse node dimension, treat each node as observation.
                if flat.numel():
                    if feat_var_tracker['count'] == 0:
                        feat_var_tracker['mean'] = flat.mean(0)
                        feat_var_tracker['M2'] = torch.zeros_like(feat_var_tracker['mean'])
                        feat_var_tracker['count'] = flat.size(0)
                        # For first chunk we set variance M2=0; subsequent chunks update incrementally.
                    else:
                        # Welford update per-node then aggregate; to avoid large loops process in one batch
                        batch_mean = flat.mean(0)
                        batch_count = flat.size(0)
                        delta = batch_mean - feat_var_tracker['mean']
                        total_count = feat_var_tracker['count'] + batch_count
                        # Update mean
                        new_mean = feat_var_tracker['mean'] + delta * (batch_count / total_count)
                        # Approximate M2 combining (treat within-batch variance ~ var(flat))
                        batch_var = flat.var(0, unbiased=False)
                        feat_var_tracker['M2'] += batch_var * batch_count + (delta ** 2) * feat_var_tracker['count'] * batch_count / total_count
                        feat_var_tracker['mean'] = new_mean
                        feat_var_tracker['count'] = total_count

        opt.zero_grad(set_to_none=True)
        logits = model(x_nodes, x_edges, x_bidx)
        loss = crit(logits, y)
        loss.backward()

        if log_grad_norm:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2).item()
                    total_norm += param_norm ** 2
            grad_norms.append(total_norm ** 0.5)

        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    avg_loss = total / max(1, n)
    return avg_loss, (sum(grad_norms) / max(1, len(grad_norms)) if log_grad_norm else None)


@torch.no_grad()
def eval_logits(model, loader, device):
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


def main():
    ap = argparse.ArgumentParser(description="Train DOM GCN head and calibrate")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--output-dir", default="artifacts/dom_gcn")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    # Diagnostics & weighting
    ap.add_argument("--pos-weight", type=float, default=None, help="Manual positive class weight for BCE; if not set and --auto-pos-weight, compute from train labels")
    ap.add_argument("--auto-pos-weight", action="store_true", help="Auto compute pos_weight = N_neg / N_pos from training set")
    ap.add_argument("--diagnostics-dir", default=None, help="Directory to write diagnostics (defaults to output-dir/diagnostics)")
    ap.add_argument("--log-hist-every", type=int, default=1, help="Epoch frequency for logits histogram logging (requires matplotlib)")
    ap.add_argument("--no-grad-norm", action="store_true", help="Disable gradient norm tracking")
    ap.add_argument("--no-feature-var", action="store_true", help="Disable node feature variance tracking")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = DomGraphDataset(args.train_jsonl)
    va_ds = DomGraphDataset(args.val_jsonl)
    coll = DomGraphCollator()
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)  # type: ignore[arg-type]
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)  # type: ignore[arg-type]

    model = DomGCN(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tr_dl) * args.epochs))

    # Class weighting
    pos_weight_tensor = None
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor(float(args.pos_weight), device=device)
    elif args.auto_pos_weight:
        # one pass over dataset labels
        pos = 0
        neg = 0
        for item in tr_ds:  # DomGraphDataset returns dict with 'label'
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
            print("Warning: No positive samples found for auto pos_weight computation")

    if pos_weight_tensor is not None:
        print(f"Using pos_weight={pos_weight_tensor.item():.4f}")
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        crit = nn.BCEWithLogitsLoss()

    # Diagnostics setup
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    feat_var_tracker = None if args.no_feature_var else {"count": 0, "mean": None, "M2": None}
    history: Dict[str, List[float]] = {"val_loss": [], "train_loss": [], "grad_norm": []}

    def dump_histogram(epoch: int, logits: torch.Tensor, split: str):
        if logits.numel() == 0:
            return
        if epoch % max(1, args.log_hist_every) != 0:
            return
        with suppress(ImportError):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            arr = logits.detach().cpu().float()
            plt.figure(figsize=(4, 3))
            plt.hist(arr.numpy(), bins=50, color='steelblue', alpha=0.85)
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
    bcel = crit  # For consistency later
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

    # Persist diagnostics summary
    try:
        diag: Dict[str, Any] = {"history": history}
        if feat_var_tracker is not None and feat_var_tracker["count"] > 1:
            var = feat_var_tracker["M2"] / (feat_var_tracker["count"] - 1)
            diag["feature_mean"] = feat_var_tracker["mean"].detach().cpu().tolist()
            diag["feature_var"] = var.detach().cpu().tolist()
        with open(os.path.join(diagnostics_dir, "training_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write diagnostics: {e}")

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
