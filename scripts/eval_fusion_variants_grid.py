#!/usr/bin/env python
"""Evaluate fusion variants at fixed recall grid (val & test).
Variants: original fusion (id-joined), fusion_calibrated_ids, meta_fusion_calibrated.
Outputs markdown + CSV with precision, FPR at recall targets.
"""
from __future__ import annotations
import os, json, argparse, csv
import numpy as np

VARIANTS={
  'fusion_ids': 'artifacts/fusion_ids/preds_{split}.jsonl',
  'fusion_calibrated_ids': 'artifacts/fusion_calibrated_ids/preds_{split}.jsonl',
  'meta_fusion_calibrated': 'artifacts/meta_fusion_calibrated/preds_{split}.jsonl'
}

RECALL_TARGETS=[0.90,0.92,0.94,0.95,0.96,0.97,0.98]

def load_preds(path):
    if not os.path.isfile(path): return [],[]
    ys=[]; ps=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            js=json.loads(ln)
            ys.append(int(js.get('label',0)))
            ps.append(float(js.get('prob',0.0)))
    return np.array(ys), np.array(ps)

def find_metrics(y,p, recall_targets):
    order=np.argsort(-p); y=y[order]; p=p[order]
    P=y.sum(); tp=0; fp=0; neg=len(y)-P
    out={rt:{'precision':float('nan'),'fpr':float('nan'),'threshold':float('nan')} for rt in recall_targets}
    remaining=set(recall_targets)
    for score,label in zip(p,y):
        if label==1: tp+=1
        else: fp+=1
        recall=tp/P if P>0 else 0; prec=tp/max(1,tp+fp); fpr=fp/max(1,neg)
        hit=[rt for rt in list(remaining) if recall>=rt]
        for rt in sorted(hit):
            if np.isnan(out[rt]['precision']) or prec>out[rt]['precision']+1e-12:
                out[rt]={'precision':prec,'fpr':fpr,'threshold':score}
            remaining.remove(rt)
        if not remaining: break
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-md', default='artifacts/diagnostics/fusion_variants_grid.md')
    ap.add_argument('--out-csv', default='artifacts/diagnostics/fusion_variants_grid.csv')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    rows=[]
    for variant, template in VARIANTS.items():
        for split in ('val','test'):
            path=template.format(split=split)
            y,p=load_preds(path)
            if len(y)==0: continue
            metrics=find_metrics(y,p, RECALL_TARGETS)
            for rt in RECALL_TARGETS:
                m=metrics[rt]
                rows.append({'variant': variant,'split': split,'recall_target': rt,'precision': m['precision'],'fpr': m['fpr'],'threshold': m['threshold']})
    # Markdown
    lines=["# Fusion Variants Recall Grid","","Variant | Split | Recall Target | Precision | FPR | Threshold","------- | ----- | ------------- | --------- | --- | ---------"]
    for r in rows:
        lines.append(" | ".join([r['variant'], r['split'], f"{r['recall_target']:.2f}", f"{r['precision']:.4f}", f"{r['fpr']:.4f}", f"{r['threshold']:.6f}"]))
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    with open(args.out_csv,'w',newline='',encoding='utf-8') as cf:
        w=csv.writer(cf); w.writerow(['variant','split','recall_target','precision','fpr','threshold'])
        for r in rows: w.writerow([r['variant'],r['split'],f"{r['recall_target']:.2f}",f"{r['precision']:.6f}",f"{r['fpr']:.6f}",f"{r['threshold']:.8f}"])
    print('Wrote', args.out_md, 'and CSV')

if __name__=='__main__':
    main()
