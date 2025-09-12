#!/usr/bin/env python
"""Soft imputation fusion: for IDs missing one head, impute missing head prob with per-head global prevalence.

Process:
 1. Load standardized head predictions for dom/js/url/text.
 2. Build union of IDs.
 3. For each head, map id->prob; missing gets head_mean_prob.
 4. Average (or logistic fallback) to produce soft fused probability.
 5. Output preds_val_soft.jsonl / preds_test_soft.jsonl.
"""
from __future__ import annotations
import os, json, argparse, math
from typing import Dict, List

HEADS = {
    'dom': 'artifacts/markup_run',
    'js_code': 'artifacts/js_codet5p',
    'url': 'artifacts/url_head',
    'text': 'artifacts/text_head'
}

def load_preds(dir_path: str, split: str):
    path = os.path.join(dir_path, f'preds_{split}.jsonl')
    out=[]
    if not os.path.isfile(path): return out
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='artifacts/fusion_soft')
    ap.add_argument('--full', action='store_true', help='Also build soft fusion for *_full splits when available')
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base_splits=['val','test']
    all_splits=list(base_splits)
    if args.full and os.path.isfile('data/pages_val_full.jsonl') and os.path.isfile('data/pages_test_full.jsonl'):
        all_splits += ['val_full','test_full']
    for split in all_splits:
        head_preds = {h: load_preds(d, split) for h,d in HEADS.items()}
        id_union=set()
        per_head_map={}
        per_head_mean={}
        labels_map={}
        for h,preds in head_preds.items():
            m={}
            probs=[]
            for r in preds:
                rid=str(r.get('id'))
                if not rid: continue
                id_union.add(rid)
                prob=float(r.get('prob',0.5))
                probs.append(prob)
                m[rid]=prob
                labels_map[rid]=int(r.get('label',0))  # last wins but labels should match
            per_head_map[h]=m
            per_head_mean[h]= sum(probs)/len(probs) if probs else 0.5
        out_path = os.path.join(args.out_dir, f'preds_{split}.jsonl')
        with open(out_path,'w',encoding='utf-8') as f:
            for rid in sorted(id_union):
                label=labels_map.get(rid,0)
                vals=[]
                for h in HEADS.keys():
                    vals.append(per_head_map[h].get(rid, per_head_mean[h]))
                fused_prob = sum(vals)/len(vals) if vals else 0.5
                obj={'id': rid,'label': label,'prob': fused_prob,'model':'fusion_soft','split':split}
                f.write(json.dumps(obj)); f.write('\n')
        print(f"[soft_fusion] Wrote {split} soft fused preds covering {len(id_union)} IDs")
    print('Soft imputation fusion complete.')

if __name__=='__main__':
    main()
