#!/usr/bin/env python
"""Refit fusion/meta logistic model using calibrated per-head probabilities.

Loads calibrated per-head val/test predictions (preds_val_calibrated.jsonl / preds_test_calibrated.jsonl)
for each specified head, fits a logistic regression (Newton IRLS) on validation set, then produces
fusion calibrated predictions for val/test and evaluates metrics (Brier, ECE, ROC AUC, PR AUC).

Outputs:
  artifacts/fusion_calibrated/weights.json           (weights + bias)
  artifacts/fusion_calibrated/preds_val.jsonl
  artifacts/fusion_calibrated/preds_test.jsonl
  artifacts/fusion_calibrated/metrics.json           (val/test metrics)
  artifacts/diagnostics/fusion_calibrated.md         (markdown summary)
"""
from __future__ import annotations
import os, json, argparse, math
import numpy as np
from typing import List, Tuple

HEAD_DIRS = ["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]  # dom_gcn dropped

def load_calibrated(head: str, split: str):
    path=f"artifacts/{head}/preds_{split}_calibrated.jsonl"
    if not os.path.isfile(path):
        path_fallback=f"artifacts/{head}/preds_{split}.jsonl"  # fallback if head chosen=raw
        path=path_fallback
    probs=[]; labels=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'prob' not in js or 'label' not in js: continue
            probs.append(float(js['prob'])); labels.append(int(js['label']))
    return np.array(labels), np.array(probs)

def brier(y,p):
    return float(np.mean((p-y)**2)) if len(y) else float('nan')

def ece(y,p,bins=10):
    if len(y)==0: return float('nan')
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges)-1
    tot=0.0
    for b in range(bins):
        m=idx==b
        if not np.any(m): continue
        conf=np.mean(p[m]); acc=np.mean(y[m]); w=np.mean(m)
        tot += w*abs(conf-acc)
    return float(tot)

def roc_auc(y,p):
    if len(y)==0: return float('nan')
    pairs=sorted(zip(p,y), key=lambda x:x[0])
    labs=[yy for _,yy in pairs]; pos=sum(labs); neg=len(labs)-pos
    if pos==0 or neg==0: return float('nan')
    scores=[s for s,_ in pairs]
    ranks=[0.0]*len(pairs); i=0; r=1
    while i < len(scores):
        j=i
        while j+1 < len(scores) and scores[j+1]==scores[i]:
            j+=1
        avg=(r + (r + (j-i)))/2
        for k in range(i,j+1): ranks[k]=avg
        r=j+2; i=j+1
    sum_ranks_pos=sum(rank for rank,l in zip(ranks,labs) if l==1)
    u=sum_ranks_pos - pos*(pos+1)/2
    return float(u/(pos*neg))

def pr_auc(y,p):
    if len(y)==0: return float('nan')
    pairs=sorted(zip(p,y), key=lambda x:-x[0])
    P=sum(l for _,l in pairs)
    if P==0: return float('nan')
    tp=0; fp=0; fn=P
    prev_r=0.0; prev_prec=1.0; auc=0.0; last=None
    for s,l in pairs:
        if last is not None and s!=last:
            r=tp/(tp+fn)
            prec=tp/max(1,tp+fp)
            auc += (r-prev_r)*prev_prec
            prev_r=r; prev_prec=prec
        if l==1: tp+=1; fn-=1
        else: fp+=1
        last=s
    r=tp/(tp+fn if (tp+fn)>0 else 1)
    auc += (r-prev_r)*prev_prec
    return float(max(0.0,min(1.0,auc)))

def fit_logistic(X,y, max_iter=200, tol=1e-6):
    # Add bias
    Xb=np.column_stack([X, np.ones(len(X))])
    w=np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z=Xb @ w
        p=1/(1+np.exp(-z))
        g=Xb.T @ (p - y)
        W=p*(1-p)
        H=Xb.T @ (Xb*W[:,None]) + 1e-6*np.eye(Xb.shape[1])
        step=np.linalg.solve(H,g)
        w-=step
        if np.max(np.abs(step)) < tol:
            break
    return w

def evaluate(y, p):
    return {
        'brier': brier(y,p),
        'ece': ece(y,p),
        'roc_auc': roc_auc(y,p),
        'pr_auc': pr_auc(y,p)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEAD_DIRS)
    ap.add_argument('--out-dir', default='artifacts/fusion_calibrated')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Load matrices
    matrices={}
    labels_ref=None
    for split in ('val','test'):
        head_arrays=[]; lengths=[]; labels_list=[]
        for h in args.heads:
            y, p = load_calibrated(h, split)
            head_arrays.append((h,y,p))
            lengths.append(len(p))
        if not lengths:
            continue
        min_len=min(lengths)
        if len(set(lengths))>1:
            lengths_str=" ".join(f"{h}:{len(p)}" for h,_,p in head_arrays)
            print(f"[warn] length mismatch {split}: {lengths_str} -> truncating to {min_len}")
        feats=[]
        for h,y,p in head_arrays:
            feats.append(p[:min_len])
            labels_list.append(y[:min_len])
        # Assume labels consistent; take first
        labels_ref=labels_list[0]
        X=np.vstack(feats).T
        matrices[split]={'X':X,'y':labels_ref}
    # Fit on val
    X_val=matrices['val']['X']; y_val=matrices['val']['y']
    w=fit_logistic(X_val, y_val)
    # Predict
    def fuse(X):
        z=X @ w[:-1] + w[-1]
        return 1/(1+np.exp(-z))
    preds={}
    for split in ('val','test'):
        X=matrices[split]['X']; y=matrices[split]['y']
        p=fuse(X)
        preds[split]={'p':p,'y':y, 'metrics': evaluate(y,p)}
        out_jsonl=os.path.join(args.out_dir, f"preds_{split}.jsonl")
        with open(out_jsonl,'w',encoding='utf-8') as f:
            for yy,pp in zip(y,p):
                f.write(json.dumps({'label': int(yy), 'prob': float(pp), 'split': split, 'model': 'fusion_calibrated'})+'\n')
    weights={'weights': w[:-1].tolist(), 'bias': float(w[-1]), 'heads': args.heads}
    with open(os.path.join(args.out_dir,'weights.json'),'w',encoding='utf-8') as f:
        json.dump(weights,f,indent=2)
    metrics_out={split: preds[split]['metrics'] for split in preds}
    with open(os.path.join(args.out_dir,'metrics.json'),'w',encoding='utf-8') as f:
        json.dump(metrics_out,f,indent=2)
    # Markdown summary
    md=["# Fusion Calibrated Metrics","","Split | Brier | ECE | ROC AUC | PR AUC","----- | ----- | --- | ------- | ------"]
    for split in ('val','test'):
        m=preds[split]['metrics']
        md.append(" | ".join([
            split,
            f"{m['brier']:.6f}", f"{m['ece']:.4f}", f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}"
        ]))
    diag_dir='artifacts/diagnostics'
    os.makedirs(diag_dir, exist_ok=True)
    with open(os.path.join(diag_dir,'fusion_calibrated.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(md)+"\n")
    print("Wrote fusion calibrated artifacts to", args.out_dir)

if __name__=='__main__':
    main()
