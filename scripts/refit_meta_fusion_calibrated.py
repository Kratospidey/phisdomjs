#!/usr/bin/env python
"""Meta fusion stacking: features = calibrated head probs + id-joined fusion calibrated prob.
Performs ID inner join across all heads and fusion_calibrated_ids predictions, trains logistic meta model, selects threshold.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

HEADS_DEFAULT=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]  # dom_gcn dropped
FUSION_JOIN_DIR='artifacts/fusion_calibrated_ids'

def load_head(head, split):
    path=f"artifacts/{head}/preds_{split}_calibrated.jsonl"
    if not os.path.isfile(path):
        path=f"artifacts/{head}/preds_{split}.jsonl"
    data={}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            js=json.loads(ln)
            _id=js.get('id');
            if _id is None: continue
            data[_id]=float(js['prob'])
    return data

def load_fusion(split):
    path=os.path.join(FUSION_JOIN_DIR, f"preds_{split}.jsonl")
    data={}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            js=json.loads(ln)
            data[js['id']]= {'prob': float(js['prob']), 'label': int(js['label'])}
    return data

def join(heads, split):
    fusion=load_fusion(split)
    head_maps=[(h, load_head(h, split)) for h in heads]
    ids=set(fusion.keys())
    for _,m in head_maps: ids &= set(m.keys())
    rows=[]
    for _id in ids:
        feats=[m[_id] for _,m in head_maps]
        feats.append(fusion[_id]['prob'])  # append fusion calibrated prob
        rows.append({'id': _id, 'label': fusion[_id]['label'], 'features': feats})
    return rows

def fit_logistic(X,y, max_iter=200, tol=1e-6):
    Xb=np.column_stack([X, np.ones(len(X))]); w=np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z=Xb @ w; p=1/(1+np.exp(-z)); g=Xb.T @ (p-y); W=p*(1-p)
        H=Xb.T @ (Xb*W[:,None]) + 1e-6*np.eye(Xb.shape[1])
        step=np.linalg.solve(H,g); w-=step
        if np.max(np.abs(step))<tol: break
    return w

def pick_threshold(y,p,target_recall):
    order=np.argsort(-p); y=y[order]; p=p[order]; P=y.sum(); tp=0; fp=0; best=None; neg=len(y)-P
    for score,label in zip(p,y):
        if label==1: tp+=1
        else: fp+=1
        recall=tp/P if P>0 else 0; prec=tp/max(1,tp+fp); fpr=fp/max(1,neg)
        if recall>=target_recall and (best is None or prec>best['precision']+1e-12):
            best={'threshold':score,'precision':prec,'recall':recall,'fpr':fpr}
    if best is None and len(p):
        prec=tp/max(1,tp+fp); recall=tp/max(1,P); fpr=fp/max(1,neg)
        best={'threshold':p[-1],'precision':prec,'recall':recall,'fpr':fpr}
    return best or {'threshold':1.0,'precision':float('nan'),'recall':float('nan'),'fpr':float('nan')}

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

def brier(y,p): return float(np.mean((p-y)**2))

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges)-1; tot=0.0
    for b in range(bins):
        m=idx==b
        if not np.any(m): continue
        conf=np.mean(p[m]); acc=np.mean(y[m]); w=np.mean(m)
        tot += w*abs(conf-acc)
    return float(tot)

def roc_auc(y,p):
    pairs=sorted(zip(p,y), key=lambda x:x[0]); labs=[l for _,l in pairs]; pos=sum(labs); neg=len(labs)-pos
    if pos==0 or neg==0: return float('nan')
    scores=[s for s,_ in pairs]; ranks=[0.0]*len(scores); i=0; r=1
    while i < len(scores):
        j=i
        while j+1 < len(scores) and scores[j+1]==scores[i]: j+=1
        avg=(r + (r + (j-i)))/2
        for k in range(i,j+1): ranks[k]=avg
        r=j+2; i=j+1
    sum_r_pos=sum(rank for rank,l in zip(ranks,labs) if l==1)
    u=sum_r_pos - pos*(pos+1)/2
    return float(u/(pos*neg))

def evaluate(y,p):
    return {'brier': brier(y,p),'ece': ece(y,p),'roc_auc': roc_auc(y,p),'pr_auc': pr_auc(y,p)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEADS_DEFAULT)
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--out-dir', default='artifacts/meta_fusion_calibrated')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows_val=join(args.heads,'val'); rows_test=join(args.heads,'test')
    if not rows_val: raise SystemExit('No rows for meta fusion (val).')
    X_val=np.array([r['features'] for r in rows_val]); y_val=np.array([r['label'] for r in rows_val])
    X_test=np.array([r['features'] for r in rows_test]); y_test=np.array([r['label'] for r in rows_test])
    w=fit_logistic(X_val,y_val)
    def infer(X): z=X @ w[:-1] + w[-1]; return 1/(1+np.exp(-z))
    p_val=infer(X_val); p_test=infer(X_test)
    metrics_val=evaluate(y_val,p_val); metrics_test=evaluate(y_test,p_test)
    op=pick_threshold(y_val,p_val,args.target_recall)
    test_pred=(p_test>=op['threshold']).astype(int)
    tp=int(((test_pred==1)&(y_test==1)).sum()); fp=int(((test_pred==1)&(y_test==0)).sum())
    fn=int(((test_pred==0)&(y_test==1)).sum()); tn=int(((test_pred==0)&(y_test==0)).sum())
    op_test={'precision': tp/max(1,tp+fp), 'recall': tp/max(1,tp+fn), 'fpr': fp/max(1,fp+tn), 'threshold': op['threshold'], 'tp': tp,'fp': fp,'fn': fn,'tn': tn}
    for split, rows, probs in [('val', rows_val, p_val),('test', rows_test, p_test)]:
        outp=os.path.join(args.out_dir, f'preds_{split}.jsonl')
        with open(outp,'w',encoding='utf-8') as f:
            for r,p in zip(rows,probs):
                f.write(json.dumps({'id': r['id'], 'label': r['label'], 'prob': float(p), 'split': split, 'model': 'meta_fusion_calibrated'})+'\n')
    with open(os.path.join(args.out_dir,'weights.json'),'w',encoding='utf-8') as f:
        json.dump({'weights': w[:-1].tolist(), 'bias': float(w[-1]), 'feature_heads': args.heads + ['fusion_calibrated_ids']}, f, indent=2)
    with open(os.path.join(args.out_dir,'metrics.json'),'w',encoding='utf-8') as f:
        json.dump({'val': metrics_val, 'test': metrics_test, 'operating_point_val': op, 'operating_point_test': op_test}, f, indent=2)
    print('Wrote meta fusion calibrated to', args.out_dir)

if __name__=='__main__':
    main()
