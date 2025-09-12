#!/usr/bin/env python
"""Generate a small markdown report comparing single-fit fusion vs CV fusion and meta CV.

Reads metrics from artifacts and produces artifacts/diagnostics/compare_stacking.md
"""
from __future__ import annotations
import os, json, argparse

FILES={
    'Fusion (ID)': 'artifacts/fusion_calibrated_ids/metrics.json',
    'Fusion CV (OOF)': 'artifacts/fusion_calibrated_ids_cv/metrics.json',
    'Meta CV': 'artifacts/meta_fusion_cv/metrics.json',
}

def load(path):
    if not os.path.isfile(path): return None
    try: return json.load(open(path,'r'))
    except Exception: return None

def pick_op(m):
    if not m: return None
    return m.get('operating_point_test') or m.get('test',{}).get('operating_point_test')

def pick_test_metrics(m):
    return m.get('test') if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-md', default='artifacts/diagnostics/compare_stacking.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    rows=[]
    for name,path in FILES.items():
        js=load(path)
        if js is None: continue
        op = pick_op(js)
        tm = pick_test_metrics(js) or {}
        rows.append((name, op.get('precision') if op else None, op.get('recall') if op else None, op.get('fpr') if op else None, tm.get('brier'), tm.get('ece')))
    lines=['# Stacking Comparison','', 'Model | Precision | Recall | FPR | Brier | ECE','----- | --------- | ------ | --- | ----- | ---']
    for name,prec,rec,fpr,brier,ece in rows:
        def fmt(x):
            return f"{x:.4f}" if isinstance(x,(int,float)) and x is not None else '-'
        lines.append(' | '.join([name, fmt(prec), fmt(rec), fmt(fpr), fmt(brier), fmt(ece)]))
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Wrote stacking comparison to', args.out_md)

if __name__=='__main__':
    main()
