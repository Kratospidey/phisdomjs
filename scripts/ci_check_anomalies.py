#!/usr/bin/env python
"""CI anomaly gate for head probability distributions.

Runs the same logic as detect_head_anomalies but can fail the build if any
heads are flagged. Intended use in automation before accepting new model
artifacts.

Exit codes:
  0 - success, no anomalies (or anomalies allowed with --allow-flags)
  3 - anomalies detected and not allowed
"""
from __future__ import annotations
import os, json, argparse, sys
import numpy as np

HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_probs(path):
    probs=[]
    if not os.path.isfile(path):
        return np.array(probs)
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'prob' in js: probs.append(float(js['prob']))
    return np.array(probs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--heads', nargs='*', default=HEADS)
    ap.add_argument('--high-mean-threshold', type=float, default=0.95)
    ap.add_argument('--low-var-threshold', type=float, default=1e-3)
    ap.add_argument('--allow-flags', action='store_true')
    ap.add_argument('--out-md', default='artifacts/diagnostics/ci_anomaly_check.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    lines=['# CI Anomaly Check','',f'Split: {args.split}',f'Mean Threshold: {args.high_mean_threshold}',f'Var Threshold: {args.low_var_threshold}','', 'Head | Mean | Var | Min | Max | Flag', '---- | ---- | --- | --- | --- | ----']
    flagged=[]
    for h in args.heads:
        path=f'artifacts/{h}/preds_{args.split}.jsonl'
        probs=load_probs(path)
        if probs.size==0: continue
        mean=float(probs.mean()); var=float(probs.var()); mn=float(probs.min()); mx=float(probs.max())
        flag=mean>args.high_mean_threshold or var<args.low_var_threshold
        if flag: flagged.append(h)
        lines.append(' | '.join([h, f"{mean:.4f}", f"{var:.6f}", f"{mn:.4f}", f"{mx:.4f}", 'YES' if flag else '']))
    if flagged:
        lines += ['', 'Flagged heads: '+', '.join(flagged)]
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Anomaly check complete. Flagged:', flagged)
    if flagged and not args.allow_flags:
        sys.exit(3)

if __name__=='__main__':
    main()
