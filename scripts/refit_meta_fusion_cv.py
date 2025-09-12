#!/usr/bin/env python
"""Phase 5: True stacking with engineered features + CV OOF meta fusion.

Features per sample (for each ID intersection across heads):
  - Raw calibrated head probabilities (H)
  - Aggregate stats: min, max, mean, std of head probs (4)
  - Pairwise absolute differences for head pairs (H*(H-1)/2)
  - (Optional) existing fusion_calibrated_ids probability (1) if available

Cross-validation (k folds) over validation set produces OOF predictions used
to select threshold. Final model fit on all validation data applied to test.

Outputs: artifacts/meta_fusion_cv/
  - preds_val_oof.jsonl
  - preds_test.jsonl
  - metrics.json (oof + test + operating points)
  - weights.json (list per feature + bias)
  - feature_map.json (names)
  - summary.md
"""
from __future__ import annotations
import os, json, argparse, itertools, random
import numpy as np

HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_split(head, split):
    path=f"artifacts/{head}/preds_{split}_calibrated.jsonl"
    if not os.path.isfile(path):
        path=f"artifacts/{head}/preds_{split}.jsonl"
    data={}
    if not os.path.isfile(path): return data
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            _id=js.get('id');
            if _id is None: continue
            data[_id]={'label': int(js['label']), 'prob': float(js['prob'])}
    return data

def load_fusion_prob(split):
    path=f"artifacts/fusion_calibrated_ids/preds_{split}.jsonl"
    if not os.path.isfile(path): return {}
    out={}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if js.get('id'):
                out[js['id']]=js['prob']
    return out

def build_matrix(heads, split, include_fusion):
    per=[(h, load_split(h, split)) for h in heads]
    id_sets=[set(d.keys()) for _,d in per if d]
    if not id_sets: return [], []
    common=set.intersection(*id_sets)
    fusion_map=load_fusion_prob(split) if include_fusion else {}
    rows=[]
    feature_names=[]
    # Base head names
    feature_names.extend([f"prob_{h}" for h in heads])
    # aggregate stats names (will append after building each row)
    agg_names=['prob_min','prob_max','prob_mean','prob_std']
    # pairwise diff names
    pair_names=[f"absdiff_{a}_{b}" for a,b in itertools.combinations(heads,2)]
    if include_fusion: feature_names.append('prob_fusion_calibrated_ids')
    feature_names.extend(agg_names)
    feature_names.extend(pair_names)
    for _id in common:
        probs=[d[_id]['prob'] for _,d in per]
        label=per[0][1][_id]['label']
        feats=list(probs)
        if include_fusion:
            feats.append(float(fusion_map.get(_id,0.0)))
        # aggregates
        arr=np.array(probs)
        feats.extend([arr.min(), arr.max(), arr.mean(), arr.std(ddof=0)])
        # pairwise differences
        for i in range(len(probs)):
            for j in range(i+1, len(probs)):
                feats.append(abs(probs[i]-probs[j]))
        rows.append({'id': _id,'label': label,'features': feats})
    return rows, feature_names

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def fit_logistic(X, y, max_iter=400, tol=1e-6):
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z = Xb @ w
        p = _sigmoid(z)
        # clip to avoid p*(1-p) under/overflow
        p = np.clip(p, 1e-8, 1 - 1e-8)
        g = Xb.T @ (p - y)
        W = p * (1 - p)
        H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(Xb.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w

def stratified_folds(y,k,seed=42):
    rng=random.Random(seed)
    pos=[i for i,v in enumerate(y) if v==1]; neg=[i for i,v in enumerate(y) if v==0]
    rng.shuffle(pos); rng.shuffle(neg)
    folds=[[] for _ in range(k)]
    for i,idx in enumerate(pos): folds[i%k].append(idx)
    for i,idx in enumerate(neg): folds[i%k].append(idx)
    return folds

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
    p = np.clip(p, 0.0, 1.0)
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges, right=True)-1; tot=0.0
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
def metrics(y,p): return {'brier': brier(y,p),'ece': ece(y,p),'roc_auc': roc_auc(y,p),'pr_auc': pr_auc(y,p)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEADS)
    ap.add_argument('--include-fusion-prob', action='store_true')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--out-dir', default='artifacts/meta_fusion_cv')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows_val, feat_names = build_matrix(args.heads,'val', args.include_fusion_prob)
    rows_test, _ = build_matrix(args.heads,'test', args.include_fusion_prob)
    if not rows_val:
        raise SystemExit('No validation rows for meta fusion CV.')
    X_val=np.array([r['features'] for r in rows_val]); y_val=np.array([r['label'] for r in rows_val])
    X_test=np.array([r['features'] for r in rows_test]) if rows_test else np.zeros((0,len(feat_names)))
    # If single-class, skip CV and use constant-prob baseline
    if len(set(map(int, y_val.tolist()))) < 2:
        print('[WARN] Single-class validation labels; skipping CV. Using class prior as OOF.')
        prior=float(y_val.mean())
        oof=np.full(len(y_val), prior, dtype=float)
    else:
        pos=int(y_val.sum()); neg=len(y_val)-pos
        k=max(2, min(args.k_folds, pos, neg))
        folds=stratified_folds(y_val, k)
        oof=np.zeros(len(y_val))
        for fold in folds:
            train_idx=[i for i in range(len(y_val)) if i not in fold]
            if not train_idx or not fold: continue
            w=fit_logistic(X_val[train_idx], y_val[train_idx])
            z=X_val[fold] @ w[:-1] + w[-1]
            oof[fold]=_sigmoid(z)
        # Fill any zeros (rare edge) with full model preds
        if np.any(oof==0):
            w_full=fit_logistic(X_val,y_val); z=X_val @ w_full[:-1] + w_full[-1]
            mask=(oof==0); oof[mask]=_sigmoid(z[mask])
    oof_m=metrics(y_val,oof)
    op=pick_threshold(y_val,oof,target_recall=args.target_recall)
    # Final model
    w_final=fit_logistic(X_val,y_val)
    def predict(X): return _sigmoid(X @ w_final[:-1] + w_final[-1])
    p_test=predict(X_test) if len(X_test) else np.zeros(0)
    test_m=metrics(np.array([r['label'] for r in rows_test]), p_test) if len(X_test) else {}
    if len(p_test):
        test_pred=(p_test>=op['threshold']).astype(int)
        y_test=np.array([r['label'] for r in rows_test])
        tp=int(((test_pred==1)&(y_test==1)).sum()); fp=int(((test_pred==1)&(y_test==0)).sum())
        fn=int(((test_pred==0)&(y_test==1)).sum()); tn=int(((test_pred==0)&(y_test==0)).sum())
        test_op={'threshold': op['threshold'],'precision': tp/max(1,tp+fp),'recall': tp/max(1,tp+fn),'fpr': fp/max(1,fp+tn),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
    else:
        test_op={}
    # Outputs
    with open(os.path.join(args.out_dir,'preds_val_oof.jsonl'),'w',encoding='utf-8') as f:
        for r,p in zip(rows_val,oof):
            f.write(json.dumps({'id': r['id'],'label': r['label'],'prob': float(p),'split':'val','model':'meta_fusion_cv_oof'})+'\n')
    if len(p_test):
        with open(os.path.join(args.out_dir,'preds_test.jsonl'),'w',encoding='utf-8') as f:
            for r,p in zip(rows_test,p_test):
                f.write(json.dumps({'id': r['id'],'label': r['label'],'prob': float(p),'split':'test','model':'meta_fusion_cv_final'})+'\n')
    with open(os.path.join(args.out_dir,'weights.json'),'w',encoding='utf-8') as f:
        json.dump({'weights': w_final[:-1].tolist(),'bias': float(w_final[-1])}, f, indent=2)
    with open(os.path.join(args.out_dir,'feature_map.json'),'w',encoding='utf-8') as f:
        json.dump({'features': feat_names}, f, indent=2)
    with open(os.path.join(args.out_dir,'metrics.json'),'w',encoding='utf-8') as f:
        json.dump({'oof': oof_m,'oof_operating_point': op,'test': test_m,'test_operating_point': test_op}, f, indent=2)
    # Markdown
    md=['# Meta Fusion CV Summary','', f'Heads: {" ,".join(args.heads)}', f'Feature count: {len(feat_names)}','', 'Metric | OOF','------ | ---', f"Brier | {oof_m['brier']:.6f}", f"ECE | {oof_m['ece']:.6f}", f"ROC AUC | {oof_m['roc_auc']:.4f}", f"PR AUC | {oof_m['pr_auc']:.4f}", '', 'OOF Operating Point | Value','------------------- | -----', f"Threshold | {op['threshold']:.6f}", f"Precision | {op['precision']:.4f}", f"Recall | {op['recall']:.4f}", f"FPR | {op['fpr']:.4f}"]
    if test_m:
        md += ['', 'Metric | Test','------ | ----', f"Brier | {test_m['brier']:.6f}", f"ECE | {test_m['ece']:.6f}", f"ROC AUC | {test_m['roc_auc']:.4f}", f"PR AUC | {test_m['pr_auc']:.4f}"]
    with open(os.path.join(args.out_dir,'summary.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(md)+'\n')
    print('Wrote meta fusion CV outputs to', args.out_dir)

if __name__=='__main__':
    main()
