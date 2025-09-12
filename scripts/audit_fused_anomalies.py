#!/usr/bin/env python
"""Find IDs where dom + js_code predictions exist but fused/meta_fused missing.

Requires strict baseline combined predictions for a split (default test).
Outputs list + counts for investigation.
"""
from __future__ import annotations
import json, os, argparse
from typing import List, Set


def load_preds(path: str):
    out=[]
    if not os.path.isfile(path): return out
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', default='artifacts/baseline_strict')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='artifacts/diagnostics/fused_anomalies.json')
    args=ap.parse_args()
    preds = load_preds(os.path.join(args.baseline_dir, f'combined_preds_{args.split}.jsonl'))
    dom_ids={p['id'] for p in preds if p.get('model')=='dom'}
    js_ids={p['id'] for p in preds if p.get('model')=='js_code'}
    fused_ids={p['id'] for p in preds if p.get('model')=='fused'}
    meta_ids={p['id'] for p in preds if p.get('model')=='meta_fused'}
    candidates = (dom_ids & js_ids) - fused_ids
    meta_candidates = (dom_ids & js_ids) - meta_ids
    out_data = {
        'split': args.split,
        'dom_count': len(dom_ids),
        'js_code_count': len(js_ids),
        'fused_count': len(fused_ids),
        'meta_fused_count': len(meta_ids),
        'dom_js_overlap': len(dom_ids & js_ids),
        'missing_fused_despite_dom_js': len(candidates),
        'missing_meta_fused_despite_dom_js': len(meta_candidates),
        'example_ids': sorted(list(candidates))[:50],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(out_data,f,indent=2)
    print(f"Fused anomaly audit complete. Missing fused count: {len(candidates)}")

if __name__=='__main__':
    main()
