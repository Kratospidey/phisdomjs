#!/usr/bin/env python
"""Per-head calibration and operating point selection (Phase 4).

For each atomic head (url, text, dom, js variants, etc.):
 1. Load val predictions (probabilities + labels)
 2. Fit two calibrators:
      - Platt (logistic on logit transform of prob clipped)
      - Isotonic regression (monotonic mapping)
 3. Evaluate on val: Brier score, ECE (10-bin), PR AUC, ROC AUC
 4. Choose best calibrator (lowest Brier, tie-break by higher PR AUC)
 5. Apply chosen calibrator to test predictions
 6. Determine operating threshold achieving recall >= target_recall (default 0.95) that maximizes precision
 7. Emit:
      artifacts/diagnostics/head_calibration/{head}.json  (metrics + chosen calibrator)
      artifacts/diagnostics/head_calibration/{head}_operating_point.json
 8. Produce consolidated markdown + CSV summary.

Assumptions: prediction JSONLs at artifacts/{head}/preds_val.jsonl and preds_test.jsonl with fields: id,label,prob.
"""
from __future__ import annotations
import os, json, math, argparse
from typing import List, Dict, Tuple

import numpy as np

HEAD_DIRS = ["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]  # dom_gcn dropped

def load_preds(path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    labels=[]; probs=[]; ids=[]
    if not os.path.isfile(path):
        return np.zeros(0), np.zeros(0), []
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip();
            if not ln: continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'label' not in js or 'prob' not in js: continue
            labels.append(int(js['label']))
            probs.append(float(js['prob']))
            ids.append(js.get('id'))
    return np.array(labels), np.array(probs), ids

def brier(labels, probs):
    return float(np.mean((probs - labels)**2)) if len(labels) else float('nan')

def ece(labels, probs, bins=10):
    if len(labels)==0: return float('nan')
    edges=np.linspace(0,1,bins+1)
    idx=np.digitize(probs, edges) - 1
    total=0.0
    for b in range(bins):
        mask=idx==b
        if not np.any(mask):
            continue
        conf=np.mean(probs[mask])
        acc=np.mean(labels[mask])
        w = np.mean(mask)
        total += w * abs(conf-acc)
    return float(total)

def roc_auc(labels, probs):
    if len(labels)==0: return float('nan')
    pairs=sorted(zip(probs, labels), key=lambda x:x[0])
    labs=[y for _,y in pairs]; pos=sum(labs); neg=len(labs)-pos
    if pos==0 or neg==0: return float('nan')
    ranks=[0.0]*len(pairs); i=0; r=1
    scores=[s for s,_ in pairs]
    while i < len(scores):
        j=i
        while j+1 < len(scores) and scores[j+1]==scores[i]:
            j+=1
        avg=(r + (r + (j-i)))/2
        for k in range(i,j+1): ranks[k]=avg
        r=j+2; i=j+1
    sum_ranks_pos=sum(rank for rank,y in zip(ranks,labs) if y==1)
    u=sum_ranks_pos - pos*(pos+1)/2
    return float(u/(pos*neg))

def pr_auc(labels, probs):
    if len(labels)==0: return float('nan')
    pairs=sorted(zip(probs, labels), key=lambda x:-x[0])
    P=sum(y for _,y in pairs)
    if P==0: return float('nan')
    tp=0; fp=0; fn=P
    prev_recall=0.0; prev_prec=1.0; auc=0.0; last=None
    for s,y in pairs:
        if last is not None and s!=last:
            recall=tp/(tp+fn)
            prec=tp/max(1,tp+fp)
            auc += (recall-prev_recall)*prev_prec
            prev_recall=recall; prev_prec=prec
        if y==1: tp+=1; fn-=1
        else: fp+=1
        last=s
    recall=tp/(tp+fn if (tp+fn)>0 else 1)
    auc += (recall-prev_recall)*prev_prec
    return float(max(0.0,min(1.0,auc)))

def platt_fit(labels, probs):
    # Avoid 0/1 extremes
    eps=1e-6
    x=np.clip(probs, eps, 1-eps)
    # Use logit transform as feature
    logit=np.log(x/(1-x))
    # Add bias
    X=np.column_stack([logit, np.ones(len(logit))])
    y=labels.astype(float)
    # Simple iterative reweighted least squares (Newton) for logistic regression with one feature
    w=np.zeros(2)
    for _ in range(100):
        z=X @ w
        p=1/(1+np.exp(-z))
        g=X.T @ (p - y)
        W=p*(1-p)
        H=X.T @ (X*W[:,None])
        # regularization to stabilize
        H+=1e-6*np.eye(2)
        step=np.linalg.solve(H, g)
        w-=step
        if np.max(np.abs(step))<1e-6:
            break
    def transform(p_new):
        p_new=np.clip(p_new, eps, 1-eps)
        z=np.log(p_new/(1-p_new))
        z2=w[0]*z + w[1]
        return 1/(1+np.exp(-z2))
    return {'type':'platt','w':w.tolist()}, transform

def isotonic_fit(labels, probs):
    # Simple pooled adjacent violators since sklearn not assumed
    order=np.argsort(probs)
    x=probs[order]
    y=labels[order]
    # Start with each point as a block
    blocks=[{'sum':y[i], 'n':1, 'val':y[i]} for i in range(len(y))]
    changed=True
    while changed:
        changed=False
        i=0
        new=[]
        while i < len(blocks):
            j=i
            while j+1 < len(blocks) and blocks[j]['val'] > blocks[j+1]['val']:
                # merge blocks j and j+1
                merged={'sum': blocks[j]['sum']+blocks[j+1]['sum'], 'n': blocks[j]['n']+blocks[j+1]['n']}
                merged['val']=merged['sum']/merged['n']
                blocks[j+1]=merged
                del blocks[j]
                changed=True
                j=max(j-1,0)
            i+=1
        # no explicit action; loop until stable
    # Expand to piecewise constant function
    preds=[]; idx=0
    for b in blocks:
        for _ in range(b['n']):
            preds.append(b['val'])
    iso_vals=np.array(preds)
    calibrated=np.empty_like(iso_vals)
    calibrated[order]=iso_vals
    def transform(p_new):
        # nearest neighbor step lookup
        p_new=np.asarray(p_new)
        inds=np.searchsorted(x, p_new, side='right')-1
        inds=np.clip(inds,0,len(calibrated)-1)
        return calibrated[inds]
    return {'type':'isotonic'}, transform

def pick_operating_point(labels, probs, target_recall=0.95):
    # descending threshold
    order=np.argsort(-probs)
    probs_sorted=probs[order]; labels_sorted=labels[order]
    P=labels.sum()
    if P==0:
        return {'threshold':1.0,'precision':float('nan'),'recall':float('nan'),'fpr':float('nan')}
    tp=0; fp=0; best=None
    neg_total=len(labels)-P
    last_score=None
    precision=0.0; recall=0.0; fpr=0.0
    for score, y in zip(probs_sorted, labels_sorted):
        if y==1: tp+=1
        else: fp+=1
        recall=tp/P
        precision=tp/max(1,tp+fp)
        fpr=fp/max(1,neg_total)
        if recall >= target_recall:
            if (best is None) or (precision > best['precision']+1e-12):
                best={'threshold':score,'precision':precision,'recall':recall,'fpr':fpr}
    if best is None:
        best={'threshold':probs_sorted[-1] if len(probs_sorted) else 1.0,'precision':precision,'recall':recall,'fpr':fpr}
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*', default=HEAD_DIRS)
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--out-dir', default='artifacts/diagnostics/head_calibration')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary=[]
    for head in args.heads:
        base=f"artifacts/{head}"
        val_path=os.path.join(base,'preds_val.jsonl')
        test_path=os.path.join(base,'preds_test.jsonl')
        y_val,p_val,ids_val=load_preds(val_path)
        y_test,p_test,ids_test=load_preds(test_path)
        if len(y_val)==0:
            continue
        cal_platt, platt_fn = platt_fit(y_val, p_val)
        p_val_platt=platt_fn(p_val)
        cal_iso, iso_fn = isotonic_fit(y_val, p_val)
        p_val_iso=iso_fn(p_val)
        def metrics(y,p):
            return {'brier': brier(y,p), 'ece': ece(y,p), 'roc_auc': roc_auc(y,p), 'pr_auc': pr_auc(y,p)}
        raw_m=metrics(y_val,p_val); platt_m=metrics(y_val,p_val_platt); iso_m=metrics(y_val,p_val_iso)
        choices=[('raw',raw_m,None),('platt',platt_m,platt_fn),('isotonic',iso_m,iso_fn)]
        chosen_name, chosen_metrics, chosen_fn = sorted(choices, key=lambda c:(c[1]['brier'], -c[1]['pr_auc']))[0]
        p_val_cal = chosen_fn(p_val) if chosen_fn else p_val
        p_test_cal = chosen_fn(p_test) if chosen_fn else p_test
        op=pick_operating_point(y_val, p_val_cal, target_recall=args.target_recall)
        test_pred=(p_test_cal >= op['threshold']).astype(int)
        tp=int(((test_pred==1) & (y_test==1)).sum()); fp=int(((test_pred==1) & (y_test==0)).sum())
        fn=int(((test_pred==0) & (y_test==1)).sum()); tn=int(((test_pred==0) & (y_test==0)).sum())
        test_precision= tp / max(1,tp+fp); test_recall = tp / max(1,tp+fn); test_fpr = fp / max(1,fp+tn)
        head_record={'head': head,'val': {'raw': raw_m, 'platt': platt_m, 'isotonic': iso_m,'chosen': chosen_name, 'chosen_metrics': chosen_metrics,'operating_point': op,'calibrators': {'platt': cal_platt,'isotonic': cal_iso}},'test': {'threshold': op['threshold'],'precision': test_precision,'recall': test_recall,'fpr': test_fpr,'tp': tp,'fp': fp,'fn': fn,'tn': tn}}
        with open(os.path.join(args.out_dir, f"{head}.json"),'w',encoding='utf-8') as f:
            json.dump(head_record,f,indent=2)
        for split, y_arr, p_arr, ids_arr in [("val", y_val, p_val_cal, ids_val),("test", y_test, p_test_cal, ids_test)]:
            outp=os.path.join(base, f"preds_{split}_calibrated.jsonl")
            with open(outp,'w',encoding='utf-8') as wf:
                for idx,(yy, pp) in enumerate(zip(y_arr, p_arr)):
                    rec={'label': int(yy), 'prob': float(pp)}
                    if ids_arr and idx < len(ids_arr) and ids_arr[idx] is not None:
                        rec['id']=ids_arr[idx]
                    wf.write(json.dumps(rec)+"\n")
        summary.append({'head': head,'chosen': chosen_name,'val_pr_auc': chosen_metrics['pr_auc'],'val_brier': chosen_metrics['brier'],'op_thr': op['threshold'],'op_precision_val': op['precision'],'op_recall_val': op['recall'],'test_precision': test_precision,'test_recall': test_recall,'test_fpr': test_fpr})
    # Output consolidated markdown + CSV
    md_lines=["# Per-Head Calibration Summary","", "Head | Chosen | Val PR AUC | Val Brier | Thr (val) | Val Prec@R>=target | Val Recall | Test Precision | Test Recall | Test FPR", "---- | ------ | ---------- | --------- | --------- | ------------------ | ---------- | -------------- | ----------- | --------"]
    for r in sorted(summary, key=lambda x:x['head']):
        md_lines.append(" | ".join([
            r['head'], r['chosen'], f"{r['val_pr_auc']:.4f}", f"{r['val_brier']:.4f}", f"{r['op_thr']:.4f}", f"{r['op_precision_val']:.4f}", f"{r['op_recall_val']:.4f}", f"{r['test_precision']:.4f}", f"{r['test_recall']:.4f}", f"{r['test_fpr']:.4f}"
        ]))
    with open(os.path.join(args.out_dir,'summary.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(md_lines)+"\n")
    import csv
    with open(os.path.join(args.out_dir,'summary.csv'),'w',newline='',encoding='utf-8') as cf:
        writer=csv.writer(cf)
        writer.writerow([c.strip() for c in md_lines[2].split('|')])
        for r in sorted(summary, key=lambda x:x['head']):
            writer.writerow([r['head'], r['chosen'], f"{r['val_pr_auc']:.4f}", f"{r['val_brier']:.4f}", f"{r['op_thr']:.4f}", f"{r['op_precision_val']:.4f}", f"{r['op_recall_val']:.4f}", f"{r['test_precision']:.4f}", f"{r['test_recall']:.4f}", f"{r['test_fpr']:.4f}"])
    print(f"Wrote calibration summaries to {args.out_dir}")

if __name__=='__main__':
    main()
