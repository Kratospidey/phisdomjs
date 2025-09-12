#!/usr/bin/env python
"""Cross-validation (OOF) logistic fusion over calibrated head probabilities.

Because only validation split calibrated predictions are available (no explicit
train preds), we approximate CV by partitioning the validation set into k folds.
Process:
 1. Build ID inner join matrix of head calibrated probs (val & test splits)
 2. Split validation rows into k folds (stratified by label if possible)
 3. For each fold: fit logistic on k-1 folds, predict held-out (OOF)
 4. Aggregate OOF predictions for unbiased calibration metrics
 5. Refit final logistic on all validation data and evaluate on test set
 6. Select operating threshold on OOF predictions (recall >= target)
 7. Write artifacts similar to single-fit fusion plus OOF metrics.

Outputs: artifacts/fusion_calibrated_ids_cv/
  - preds_val_oof.jsonl (id,label,prob)
  - preds_test.jsonl
  - weights.json (final model weights)
  - metrics.json (oof + test + thresholds)
  - summary.md
"""
from __future__ import annotations
import os, json, argparse, math, random
import numpy as np

HEADS_DEFAULT=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_split(head, split):
    path=f"artifacts/{head}/preds_{split}_calibrated.jsonl"
    if not os.path.isfile(path):
        path=f"artifacts/{head}/preds_{split}.jsonl"
    data={}
    if not os.path.isfile(path):
        return data
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            _id=js.get('id');
            if _id is None: continue
            data[_id]={'label': int(js['label']), 'prob': float(js['prob'])}
    return data

def inner_join(heads, split):
    per=[(h, load_split(h, split)) for h in heads]
    id_sets=[set(d.keys()) for _,d in per if d]
    if not id_sets: return []
    common=set.intersection(*id_sets)
    rows=[]
    for _id in common:
        feats=[]; label=None
        for h,d in per:
            feats.append(d[_id]['prob'])
            if label is None: label=d[_id]['label']
        rows.append({'id': _id,'label': label,'features': feats})
    return rows

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def fit_logistic(X,y,max_iter=300,tol=1e-6):
    Xb=np.column_stack([X, np.ones(len(X))])
    w=np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z=Xb @ w; p=_sigmoid(z); p=np.clip(p,1e-8,1-1e-8); g=Xb.T @ (p-y); W=p*(1-p)
        H=Xb.T @ (Xb*W[:,None]) + 1e-6*np.eye(Xb.shape[1])
        step=np.linalg.solve(H,g); w-=step
        if np.max(np.abs(step))<tol: break
    return w

def pick_threshold(y,p,target_recall=0.95):
    order=np.argsort(-p); y=y[order]; p=p[order]; P=y.sum(); tp=0; fp=0; best=None; neg=len(y)-P
    for score,label in zip(p,y):
        if label==1: tp+=1
        else: fp+=1
        recall=tp/P if P>0 else 0; prec=tp/max(1,tp+fp); fpr=fp/max(1,neg)
        if recall>=target_recall and (best is None or prec>best['precision']+1e-12):
            best={'threshold': score,'precision': prec,'recall': recall,'fpr': fpr}
    if best is None and len(p):
        prec=tp/max(1,tp+fp); recall=tp/max(1,P); fpr=fp/max(1,neg)
        best={'threshold': p[-1],'precision':prec,'recall': recall,'fpr': fpr}
    return best or {'threshold':1.0,'precision':float('nan'),'recall':float('nan'),'fpr':float('nan')}

def brier(y,p): return float(np.mean((p-y)**2))

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges)-1; tot=0.0
    for b in range(bins):
        m=idx==b
        if not np.any(m): continue
        tot += np.mean(m)*abs(np.mean(p[m])-np.mean(y[m]))
    return float(tot)

def roc_auc(y,p):
    pairs=sorted(zip(p,y), key=lambda x:x[0]); labs=[l for _,l in pairs]; pos=sum(labs); neg=len(labs)-pos
    if pos==0 or neg==0: return float('nan')
    scores=[s for s,_ in pairs]; ranks=[0.0]*len(scores); i=0; r=1
    while i < len(scores):
        j=i
        while j+1 < len(scores) and scores[j+1]==scores[i]: j+=1
        avg=(r+(r+(j-i)))/2
        for k in range(i,j+1): ranks[k]=avg
        r=j+2; i=j+1
    sum_r_pos=sum(rank for rank,l in zip(ranks,labs) if l==1)
    u=sum_r_pos - pos*(pos+1)/2
    return float(u/(pos*neg))

def pr_auc(y,p):
    arr=sorted(zip(p,y), key=lambda t:-t[0]); P=sum(y)
    if P==0: return float('nan')
    tp=0; fp=0; fn=P; prev_r=0.0; prev_prec=1.0; auc=0.0; last=None
    for s,yy in arr:
        if last is not None and s!=last:
            r=tp/(tp+fn); prec=tp/max(1,tp+fp); auc += (r-prev_r)*prev_prec; prev_r=r; prev_prec=prec
        if yy==1: tp+=1; fn-=1
        else: fp+=1
        last=s
    r=tp/(tp+fn if (tp+fn)>0 else 1); auc += (r-prev_r)*prev_prec
    return float(max(0.0,min(1.0,auc)))

def eval_metrics(y,p):
    return {'brier': brier(y,p), 'ece': ece(y,p), 'roc_auc': roc_auc(y,p), 'pr_auc': pr_auc(y,p)}

def stratified_folds(y, k, seed=42):
    rng=random.Random(seed)
    pos=[i for i,v in enumerate(y) if v==1]
    neg=[i for i,v in enumerate(y) if v==0]
    rng.shuffle(pos); rng.shuffle(neg)
    folds=[[] for _ in range(k)]
    for i,idx in enumerate(pos): folds[i%k].append(idx)
    for i,idx in enumerate(neg): folds[i%k].append(idx)
    return folds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEADS_DEFAULT)
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--out-dir', default='artifacts/fusion_calibrated_ids_cv')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows_val=inner_join(args.heads,'val')
    rows_test=inner_join(args.heads,'test')
    if not rows_val:
        raise SystemExit('No joined validation rows; cannot run CV fusion.')
    X_val=np.array([r['features'] for r in rows_val]); y_val=np.array([r['label'] for r in rows_val])
    X_test=np.array([r['features'] for r in rows_test]); y_test=np.array([r['label'] for r in rows_test]) if rows_test else (np.zeros((0,len(args.heads))), np.zeros(0))
    # Determine viable fold count given class counts
    if len(set(map(int, y_val.tolist()))) < 2:
        print('[WARN] Single-class validation; skipping CV and using class prior for OOF.')
        folds=[]
        oof=np.full(len(y_val), float(y_val.mean()), dtype=float)
    else:
        pos=int(y_val.sum()); neg=len(y_val)-pos
        k=max(2, min(args.k_folds, pos, neg))
        folds=stratified_folds(y_val, k)
        oof=np.zeros(len(y_val))
        for fi, fold in enumerate(folds):
            train_idx=[i for i in range(len(y_val)) if i not in fold]
            if not train_idx or not fold:
                continue
            # Guard: ensure both classes present in train fold
            if len(set(map(int, y_val[train_idx].tolist()))) < 2:
                # fallback to prior for this fold
                oof[fold]=float(y_val[train_idx].mean())
                continue
            w=fit_logistic(X_val[train_idx], y_val[train_idx])
            z=X_val[fold] @ w[:-1] + w[-1]; oof[fold]=_sigmoid(z)
    oof=np.zeros(len(y_val))
    for fi, fold in enumerate(folds):
        train_idx=[i for i in range(len(y_val)) if i not in fold]
        if not train_idx or not fold:
            continue
        w=fit_logistic(X_val[train_idx], y_val[train_idx])
        z=X_val[fold] @ w[:-1] + w[-1]; oof[fold]=1/(1+np.exp(-z))
    # Fallback: any untouched (rare when tiny data) -> fit full model value
    if 'oof' in locals() and np.any(oof==0):
        w_full=fit_logistic(X_val,y_val)
        z=X_val @ w_full[:-1] + w_full[-1]
        mask=(oof==0)
        oof[mask]=_sigmoid(z[mask])
    oof_metrics=eval_metrics(y_val,oof)
    op=pick_threshold(y_val,oof,target_recall=args.target_recall)
    # Final model
    w_final=fit_logistic(X_val,y_val)
    def fuse(X):
        return _sigmoid(X @ w_final[:-1] + w_final[-1])
    p_test=fuse(X_test) if len(X_test) else np.zeros(0)
    test_metrics=eval_metrics(y_test,p_test) if len(X_test) else {}
    # Operating point on test
    if len(p_test):
        test_pred=(p_test>=op['threshold']).astype(int)
        tp=int(((test_pred==1)&(y_test==1)).sum()); fp=int(((test_pred==1)&(y_test==0)).sum())
        fn=int(((test_pred==0)&(y_test==1)).sum()); tn=int(((test_pred==0)&(y_test==0)).sum())
        test_op={'threshold': op['threshold'],'precision': tp/max(1,tp+fp),'recall': tp/max(1,tp+fn),'fpr': fp/max(1,fp+tn),'tp': tp,'fp': fp,'fn': fn,'tn': tn}
    else:
        test_op={}
    # Write outputs
    with open(os.path.join(args.out_dir,'preds_val_oof.jsonl'),'w',encoding='utf-8') as f:
        for r,p in zip(rows_val,oof):
            f.write(json.dumps({'id': r['id'],'label': r['label'],'prob': float(p),'split':'val','model':'fusion_cv_oof'})+'\n')
    if len(p_test):
        with open(os.path.join(args.out_dir,'preds_test.jsonl'),'w',encoding='utf-8') as f:
            for r,p in zip(rows_test,p_test):
                f.write(json.dumps({'id': r['id'],'label': r['label'],'prob': float(p),'split':'test','model':'fusion_cv_final'})+'\n')
    with open(os.path.join(args.out_dir,'weights.json'),'w',encoding='utf-8') as f:
        json.dump({'weights': w_final[:-1].tolist(),'bias': float(w_final[-1]),'heads': args.heads}, f, indent=2)
    with open(os.path.join(args.out_dir,'metrics.json'),'w',encoding='utf-8') as f:
        json.dump({'oof': oof_metrics,'oof_operating_point': op,'test': test_metrics,'test_operating_point': test_op}, f, indent=2)
    # Markdown
    md=[
        '# CV Fusion (OOF) Summary','',
        f'Heads: {",".join(args.heads)}',
        f'Folds: {len(folds)}',
        '',
        'Metric | OOF',
        '------ | ---',
        f'Brier | {oof_metrics["brier"]:.6f}',
        f'ECE | {oof_metrics["ece"]:.6f}',
        f'ROC AUC | {oof_metrics["roc_auc"]:.4f}',
        f'PR AUC | {oof_metrics["pr_auc"]:.4f}',
        '',
        'OOF Operating Point (Recall-constrained) | Value',
        '------------------------------------------ | -----',
        f'Threshold | {op["threshold"]:.6f}',
        f'Precision | {op["precision"]:.4f}',
        f'Recall | {op["recall"]:.4f}',
        f'FPR | {op["fpr"]:.4f}',
    ]
    if test_metrics:
        md += [
            '',
            'Metric | Test',
            '------ | ----',
            f'Brier | {test_metrics["brier"]:.6f}',
            f'ECE | {test_metrics["ece"]:.6f}',
            f'ROC AUC | {test_metrics["roc_auc"]:.4f}',
            f'PR AUC | {test_metrics["pr_auc"]:.4f}',
        ]
    with open(os.path.join(args.out_dir,'summary.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(md)+'\n')
    print('Wrote CV fusion outputs to', args.out_dir)

if __name__=='__main__':
    main()
