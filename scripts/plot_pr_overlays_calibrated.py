#!/usr/bin/env python
"""Plot PR curve overlays: raw heads vs calibrated heads vs fusion vs fusion_calibrated."""
from __future__ import annotations
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

def read_preds(path):
    y=[]; p=[]
    if not os.path.isfile(path): return y,p
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            js=json.loads(ln)
            y.append(int(js.get('label',0)))
            p.append(float(js.get('prob',0.0)))
    return y,p

def pr_auc(y,p):
    if not y: return float('nan')
    arr=sorted(zip(p,y), key=lambda t:-t[0])
    P=sum(y)
    if P==0: return float('nan')
    tp=0; fp=0; fn=P
    prev_r=0.0; prev_prec=1.0; auc=0.0; last=None
    for s,yy in arr:
        if last is not None and s!=last:
            r=tp/(tp+fn)
            prec=tp/max(1,tp+fp)
            auc += (r-prev_r)*prev_prec
            prev_r=r; prev_prec=prec
        if yy==1: tp+=1; fn-=1
        else: fp+=1
        last=s
    r=tp/(tp+fn if (tp+fn)>0 else 1)
    auc += (r-prev_r)*prev_prec
    return float(max(0.0,min(1.0,auc)))

def pr_curve(y,p):
    arr=sorted(zip(p,y), key=lambda t:-t[0])
    tp=0; fp=0; P=sum(y)
    R=[]; P_list=[]
    for s,yy in arr:
        if yy==1: tp+=1
        else: fp+=1
        rec=tp/max(1,P); prec=tp/max(1,tp+fp)
        R.append(rec); P_list.append(prec)
    return np.array(R), np.array(P_list)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val','test'], default='test')
    ap.add_argument('--out', default='artifacts/diagnostics/pr_overlays.png')
    ap.add_argument('--heads', nargs='*', default=['url_head','text_head','js_codet5p','js_charcnn','js_charcnn_aug'])  # dom_gcn dropped
    args=ap.parse_args()
    plt.figure(figsize=(8,6))
    colors=['C0','C1','C2','C3','C4','C5']
    for i,h in enumerate(args.heads):
        raw_path=f"artifacts/{h}/preds_{args.split}.jsonl"
        cal_path=f"artifacts/{h}/preds_{args.split}_calibrated.jsonl"
        y_raw,p_raw=read_preds(raw_path)
        y_cal,p_cal=read_preds(cal_path)
        if p_raw:
            R,Pc=pr_curve(y_raw,p_raw); auc=pr_auc(y_raw,p_raw)
            plt.plot(R,Pc, linestyle='--', color=colors[i%len(colors)], label=f"{h} raw (AP={auc:.3f})")
        if p_cal:
            R,Pc=pr_curve(y_cal,p_cal); auc=pr_auc(y_cal,p_cal)
            plt.plot(R,Pc, linestyle='-', color=colors[i%len(colors)], label=f"{h} cal (AP={auc:.3f})")
    # Fusion baselines
    y_f,p_f=read_preds(f"artifacts/fusion/preds_{args.split}.jsonl")
    if p_f:
        R,Pc=pr_curve(y_f,p_f); auc=pr_auc(y_f,p_f)
        plt.plot(R,Pc, color='black', linewidth=2, label=f"fusion (AP={auc:.3f})")
    y_fc,p_fc=read_preds(f"artifacts/fusion_calibrated/preds_{args.split}.jsonl")
    if p_fc:
        R,Pc=pr_curve(y_fc,p_fc); auc=pr_auc(y_fc,p_fc)
        plt.plot(R,Pc, color='magenta', linewidth=2, label=f"fusion_calibrated (AP={auc:.3f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Overlays ({args.split})')
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=160)
    print('Saved', args.out)

if __name__=='__main__':
    main()
