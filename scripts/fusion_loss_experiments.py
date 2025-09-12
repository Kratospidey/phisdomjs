#!/usr/bin/env python
"""Phase 7: Loss function experiments for fusion (validation only).

Compares logistic (baseline), focal (gamma=2), weighted logistic (balanced), and squared loss.
Uses calibrated head probabilities joined with IDs (validation split) fitting separate logistic models optimizing each surrogate.
"""
from __future__ import annotations
import os, json, argparse, numpy as np

HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_split(head, split):
    path=f"artifacts/{head}/preds_{split}_calibrated.jsonl"
    if not os.path.isfile(path): path=f"artifacts/{head}/preds_{split}.jsonl"
    data={}
    if not os.path.isfile(path): return data
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            _id=js.get('id');
            if _id: data[_id]={'label': int(js['label']),'prob': float(js['prob'])}
    return data

def inner_join(heads, split):
    per=[(h,load_split(h,split)) for h in heads]
    ids= set.intersection(*[set(d.keys()) for _,d in per]) if per else set()
    rows=[]
    for _id in ids:
        rows.append({'id': _id,'label': per[0][1][_id]['label'],'features': [d[_id]['prob'] for _,d in per]})
    return rows

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def fit_logistic(X,y, loss='logistic', gamma=2.0, class_weight=None, iters=400):
    Xb=np.column_stack([X, np.ones(len(X))]); w=np.zeros(Xb.shape[1])
    for _ in range(iters):
        z=Xb @ w; p=_sigmoid(z); p=np.clip(p,1e-8,1-1e-8)
        if loss=='logistic':
            grad=Xb.T @ (p - y)
        elif loss=='focal':
            # Derivative of focal approximate: (p - y)* ( (1-p)^gamma * (y==1) + p^gamma * (y==0) )
            mod=((1-p)**gamma)*y + (p**gamma)*(1-y)
            grad=Xb.T @ ((p - y)*mod)
        elif loss=='squared':
            grad=Xb.T @ ((p - y)*p*(1-p)*2)  # chain rule through logistic
        elif loss=='weighted':
            cw = class_weight or {1:1.0,0:1.0}
            wp=cw.get(1,1.0); wn=cw.get(0,1.0)
            weights = wp*y + wn*(1-y)
            grad=Xb.T @ ((p - y)*weights)
        else:
            grad=Xb.T @ (p - y)
        H=Xb.T @ (Xb*(p*(1-p))[:,None]) + 1e-6*np.eye(Xb.shape[1])
        step=np.linalg.solve(H,grad); w-=step
        if np.max(np.abs(step))<1e-6:
            break
    return w

def eval_metrics(y,p):
    brier=float(np.mean((p-y)**2))
    # PR AUC simple
    arr=sorted(zip(p,y), key=lambda t:-t[0]); P=sum(y); tp=0; fp=0; fn=P; prev_r=0.0; prev_prec=1.0; auc=0.0; last=None
    if P>0:
        for s,l in arr:
            if last is not None and s!=last:
                r=tp/(tp+fn); prec=tp/max(1,tp+fp); auc+=(r-prev_r)*prev_prec; prev_r=r; prev_prec=prec
            if l==1: tp+=1; fn-=1
            else: fp+=1
            last=s
        r=tp/(tp+fn if (tp+fn)>0 else 1); auc+=(r-prev_r)*prev_prec
    return {'brier': brier,'pr_auc': auc if P>0 else float('nan')}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEADS)
    ap.add_argument('--out-json', default='artifacts/diagnostics/fusion_loss_experiments.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/fusion_loss_experiments.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    rows=inner_join(args.heads,'val')
    if not rows: raise SystemExit('No rows for loss experiments.')
    X=np.array([r['features'] for r in rows]); y=np.array([r['label'] for r in rows])
    # Guard: if labels are single-class, use class prior and skip fitting to avoid degenerate Hessians
    if len(set(map(int, y.tolist()))) < 2:
        prior=float(y.mean())
        results=[{'loss': 'constant','metrics': {'brier': float(np.mean((prior - y)**2)), 'pr_auc': float('nan')}, 'weights': [0.0]*X.shape[1], 'bias': float(np.log(prior/(1-prior)) if 0 < prior < 1 else 0.0)}]
        best=results[0]
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json,'w',encoding='utf-8') as f:
            json.dump({'results': results,'best': best,'note': 'single-class labels; used prior'}, f, indent=2)
        with open(args.out_md,'w',encoding='utf-8') as f:
            f.write('# Fusion Loss Experiments\n\nSingle-class labels; skipped fitting.\n')
        print('Wrote loss experiment results.')
        return
    pos=y.sum(); neg=len(y)-pos
    cw={1: 0.5/(pos/len(y)), 0:0.5/(neg/len(y))} if pos and neg else {1:1,0:1}
    configs=[('logistic',{}),('focal',{'loss':'focal'}),('weighted',{'loss':'weighted','class_weight':cw}),('squared',{'loss':'squared'})]
    results=[]
    for name,kw in configs:
        w=fit_logistic(X,y, **kw)
        p=_sigmoid(X @ w[:-1] + w[-1])
        p=np.clip(p,1e-8,1-1e-8)
        m=eval_metrics(y,p)
        results.append({'loss': name,'metrics': m,'weights': w[:-1].tolist(),'bias': float(w[-1])})
    # pick best by PR AUC then Brier
    best=sorted(results, key=lambda r:(-r['metrics']['pr_auc'], r['metrics']['brier']))[0]
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump({'results': results,'best': best,'class_weights': cw}, f, indent=2)
    lines=['# Fusion Loss Experiments','', 'Loss | PR AUC | Brier','---- | ------ | -----']
    for r in results:
        lines.append(f"{r['loss']} | {r['metrics']['pr_auc']:.4f} | {r['metrics']['brier']:.6f}")
    lines += ['', 'Best:', '', json.dumps(best,indent=2)]
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Wrote loss experiment results.')

if __name__=='__main__':
    main()
