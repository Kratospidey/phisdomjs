#!/usr/bin/env python
"""Detect anomalous heads based on probability distribution statistics.
Flags heads whose mean prob > high_mean_threshold or variance < low_var_threshold.
Outputs JSON + markdown.
"""
from __future__ import annotations
import os, json, argparse, math
import numpy as np

HEAD_DIRS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]  # dom_gcn removed


def load_probs(head, split='val'):
    path=f"artifacts/{head}/preds_{split}.jsonl"
    ps=[]
    if os.path.isfile(path):
        with open(path,'r',encoding='utf-8') as f:
            for ln in f:
                if not ln.strip(): continue
                try: js=json.loads(ln)
                except Exception: continue
                if 'prob' in js: ps.append(float(js['prob']))
    return np.array(ps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--heads', nargs='*', default=HEAD_DIRS)
    ap.add_argument('--high-mean-threshold', type=float, default=0.95)
    ap.add_argument('--low-var-threshold', type=float, default=1e-3)
    ap.add_argument('--out-json', default='artifacts/diagnostics/head_anomalies.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/head_anomalies.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    records=[]; flagged=[]
    for h in args.heads:
        probs=load_probs(h, args.split)
        if probs.size==0: continue
        mean=float(probs.mean()); var=float(probs.var()); mn=float(probs.min()); mx=float(probs.max())
        rec={'head': h,'mean': mean,'var': var,'min': mn,'max': mx}
        if mean>args.high_mean_threshold or var<args.low_var_threshold:
            rec['flag']=True; flagged.append(h)
        records.append(rec)
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump({'split': args.split,'high_mean_threshold': args.high_mean_threshold,'low_var_threshold': args.low_var_threshold,'records': records}, f, indent=2)
    lines=["# Head Anomalies","",f"Split: {args.split}",f"High Mean Thresh: {args.high_mean_threshold}",f"Low Var Thresh: {args.low_var_threshold}","","Head | Mean | Var | Min | Max | Flag","---- | ---- | --- | --- | --- | ----"]
    for r in records:
        lines.append(" | ".join([r['head'], f"{r['mean']:.4f}", f"{r['var']:.6f}", f"{r['min']:.4f}", f"{r['max']:.4f}", 'YES' if r.get('flag') else '']))
    lines.append("")
    if flagged:
        lines.append("Flagged heads: "+", ".join(flagged))
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    print('Wrote', args.out_md)

if __name__=='__main__':
    main()
