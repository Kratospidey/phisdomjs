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



def normalize_cheap_features(batch):
    """Normalize cheap features to prevent gradient explosion."""
    if 'cheap_features' in batch:
        cf = batch['cheap_features']
        
        # Clip extreme outliers (>1e10) to reasonable range
        cf = torch.clamp(cf, min=-1e6, max=1e6)
        
        # Log transform for large positive values
        cf_positive = cf > 0
        cf_log = torch.where(cf_positive, torch.log1p(cf), cf)
        
        # Standard normalization per feature
        # Use robust statistics (median, IQR) to handle remaining outliers
        median = torch.median(cf_log, dim=0, keepdim=True)[0]
        q75 = torch.quantile(cf_log, 0.75, dim=0, keepdim=True)
        q25 = torch.quantile(cf_log, 0.25, dim=0, keepdim=True)
        iqr = q75 - q25
        iqr = torch.where(iqr < 1e-6, torch.ones_like(iqr), iqr)  # Avoid division by zero
        
        cf_normalized = (cf_log - median) / iqr
        
        # Final clipping to [-5, 5] for numerical stability
        cf_normalized = torch.clamp(cf_normalized, min=-5.0, max=5.0)
        
        batch['cheap_features'] = cf_normalized
    
    return batch


def train_one_epoch(model, loader, opt, sched, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    nan_batches = 0
    for batch_idx, batch in enumerate(loader):
        labels = batch.pop("labels").float().to(device)
        batch = normalize_cheap_features(batch)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch).squeeze(-1)
        
        # Check for NaN/inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"🚨 NaN/Inf logits detected in batch {batch_idx}, skipping...")
            nan_batches += 1
            continue
            
        loss = crit(logits, labels)
        
        # Check for NaN/inf in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"🚨 NaN/Inf loss detected in batch {batch_idx}, skipping...")
            nan_batches += 1
            continue
            
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"🚨 NaN/Inf gradient in {name}, zeroing...")
                    param.grad.zero_()
                    has_nan_grad = True
        
        if has_nan_grad:
            nan_batches += 1
            continue
            
        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    
    if nan_batches > 0:
        print(f"⚠️  Skipped {nan_batches} batches due to NaN/Inf values")
    
    return total / max(1, n)


@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    ids: List[str] = []
    nan_samples = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        # accept either 'ids' or 'id' from the collator
        _ids = batch.pop("ids", None) or batch.pop("id", None)
        if _ids is not None:
            ids.extend(list(_ids))
        batch = normalize_cheap_features(batch)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        logits = model(batch).squeeze(-1)
        
        # Check for NaN/inf in evaluation logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"🚨 NaN/Inf logits in evaluation, replacing with zeros...")
            nan_count = torch.isnan(logits).sum().item() + torch.isinf(logits).sum().item()
            nan_samples += nan_count
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    
    if nan_samples > 0:
        print(f"⚠️  Replaced {nan_samples} NaN/Inf predictions with zeros during evaluation")
        
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
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--disable-tqdm", action="store_true")
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
    dl_kwargs = dict(num_workers=int(args.num_workers), pin_memory=bool(args.pin_memory), persistent_workers=bool(args.persistent_workers) and int(args.num_workers) > 0)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll, **dl_kwargs)  # type: ignore[arg-type]
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll, **dl_kwargs)  # type: ignore[arg-type]
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll, **dl_kwargs)  # type: ignore[arg-type]

    # Infer cheap feature dimension safely without consuming a training batch
    cheap_dim = None
    if not args.no_cheap:
        try:
            # Sample a single row to detect cheap feature dimension
            sample_row = tr_ds[0]
            sample_batch = coll([sample_row])  # collator expects a list
            
            # Look for cheap features in various field names
            cf = sample_batch.get("cheap_features") or sample_batch.get("cheap")
            if cf is not None and hasattr(cf, 'shape'):
                cheap_dim = int(cf.shape[-1])
                print(f"Detected cheap feature dimension: {cheap_dim}")
            else:
                # Fallback to constant dimension from feature list
                from phisdom.data.cheap_features import CHEAP_FEATURES
                cheap_dim = len(CHEAP_FEATURES)
                print(f"Using cheap feature dimension from constant: {cheap_dim}")
        except Exception as e:
            print(f"Warning: Could not detect cheap feature dimension, using lazy adaptation: {e}")
            cheap_dim = None  # Let LazyLinear handle it
    
    # Create model with detected or adaptive cheap dimension
    model = CrossModalTransformerFusion(
        d_model=128,
        n_heads=4, 
        n_layers=2,
        dropout=0.1,
        use_url=not args.no_url,
        use_js=not args.no_js,
        use_text=not args.no_text,
        use_dom=not args.no_dom,
        use_cheap=not args.no_cheap,
        cheap_dim=cheap_dim  # None for lazy adaptation, int for fixed size
    ).to(device)
    
    # Enhanced optimizer with numerical stability
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        eps=1e-8,  # Larger epsilon for stability
        amsgrad=True  # More stable variant
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tr_dl) * args.epochs))

    # Add parameter initialization for stability
    def init_weights_stable(m):
        if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)  # Smaller std

    model.apply(init_weights_stable)
    print("✓ Applied stable weight initialization")

    bcel = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    bad = 0
    patience = 3
    model_saved = False  # Track if we ever saved a model
    for ep in range(args.epochs):
        _ = train_one_epoch(model, tr_dl, opt, sched, device)
        ids_v, logits_v, labels_v = eval_logits(model, va_dl, device)
        val_loss = float(bcel(logits_v, labels_v.float()).item()) if logits_v.numel() else float("inf")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            model_saved = True
        else:
            bad += 1
            if bad >= patience:
                break
    
    # Ensure we have a saved model even if validation was problematic
    if not model_saved:
        print("Warning: No model was saved during training, saving current state")
        torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

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
