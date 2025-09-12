#!/usr/bin/env python
"""Audit modality coverage gaps on extended (*_full) splits.

Reads *_full prediction files for dom/js_code/url/text heads plus optional js_charcnn variants
and reports per-head coverage counts and combinations, along with IDs missing specific heads.

Usage:
  python scripts/audit_extended_modality_gaps.py \
      --val data/pages_val_full.jsonl --test data/pages_test_full.jsonl \
      --out artifacts/diagnostics/extended_modality_gaps.json

Outputs JSON with structure:
{
  "splits": {
     "val_full": { "total_gold": int, "per_head": {head: n_ids}, "combos": {"dom+url": n, ...}, "missing": {"js_code": [... up to limit ...], ...} },
     "test_full": { ... }
  }
}
"""
from __future__ import annotations
import argparse, os, json, itertools
from typing import Dict, Set, List

HEAD_DIRS = {
    'dom': 'artifacts/markup_run',
    'js_code': 'artifacts/js_codet5p',
    'url': 'artifacts/url_head',
    'text': 'artifacts/text_head',
}

EXTRA_JS = {
    'js_charcnn_base': 'artifacts/js_charcnn',
    'js_charcnn_aug': 'artifacts/js_charcnn_aug'
}

def load_ids(pred_dir: str, split_full: str) -> Set[str]:
    fname = f'preds_{split_full}.jsonl'
    path = os.path.join(pred_dir, fname)
    cleaned = path + '.cleaned'
    if os.path.isfile(cleaned):
        path = cleaned
    ids = set()
    if not os.path.isfile(path):
        return ids
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                js=json.loads(ln)
            except Exception:
                continue
            rid = js.get('id')
            if rid is not None:
                ids.add(str(rid))
    return ids

def load_gold(path: str) -> List[str]:
    ids=[]
    if not os.path.isfile(path):
        return ids
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                js=json.loads(ln)
            except Exception:
                continue
            rid=js.get('id')
            if rid is not None:
                ids.append(str(rid))
    return ids

def audit(split_name: str, gold_path: str, include_extra_js: bool, missing_limit: int) -> Dict:
    gold_ids = load_gold(gold_path)
    head_sets: Dict[str, Set[str]] = {}
    for h, d in HEAD_DIRS.items():
        head_sets[h]=load_ids(d, split_name)
    if include_extra_js:
        for h,d in EXTRA_JS.items():
            head_sets[h]=load_ids(d, split_name)
    union_ids = set().union(*head_sets.values()) if head_sets else set()
    combos: Dict[str,int] = {}
    # compute combinations of presence among primary four heads
    primary = ['dom','js_code','url','text']
    for rid in union_ids:
        present = [h for h in primary if rid in head_sets.get(h,set())]
        key = '+'.join(present) if present else 'none'
        combos[key]=combos.get(key,0)+1
    per_head_counts = {h: len(s) for h,s in head_sets.items()}
    missing: Dict[str,List[str]] = {}
    for h in primary:
        miss = [rid for rid in gold_ids if rid not in head_sets.get(h,set())]
        if miss:
            missing[h]=miss[:missing_limit]
    return {
        'total_gold': len(gold_ids),
        'union_ids': len(union_ids),
        'per_head': per_head_counts,
        'combos': combos,
        'missing_examples': missing,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--val', default='data/pages_val_full.jsonl')
    ap.add_argument('--test', default='data/pages_test_full.jsonl')
    ap.add_argument('--out', default='artifacts/diagnostics/extended_modality_gaps.json')
    ap.add_argument('--extra-js', action='store_true', help='Include js_charcnn variants')
    ap.add_argument('--missing-limit', type=int, default=50)
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result={'splits':{}}
    if os.path.isfile(args.val):
        result['splits']['val_full']=audit('val_full', args.val, args.extra_js, args.missing_limit)
    if os.path.isfile(args.test):
        result['splits']['test_full']=audit('test_full', args.test, args.extra_js, args.missing_limit)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(f"Extended modality gap audit written: {args.out}")

if __name__=='__main__':
    main()
