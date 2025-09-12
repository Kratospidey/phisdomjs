#!/usr/bin/env python
"""Analyze js_code coverage gaps for *_full splits.
Uses extended modality gap audit plus raw dataset to categorize missing js_code IDs by js_code content presence.
"""
from __future__ import annotations
import json, os, argparse
from typing import Dict


def load_gap_ids(gap_path: str, split: str) -> list[str]:
    if not os.path.isfile(gap_path):
        return []
    with open(gap_path,'r',encoding='utf-8') as f:
        js=json.load(f)
    return js.get('splits',{}).get(split,{}).get('missing_examples',{}).get('js_code',[])


def index_dataset(path: str) -> Dict[str, dict]:
    out={}
    if not os.path.isfile(path):
        return out
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: js=json.loads(ln)
            except Exception:
                continue
            rid=str(js.get('id')) if js.get('id') is not None else None
            if not rid: continue
            out[rid]=js
    return out


def summarize(ids: list[str], data: Dict[str, dict]):
    stats={'total_missing': len(ids), 'no_record':0, 'no_js_field':0, 'empty_js':0, 'short_js':0, 'has_js':0}
    examples={'no_js_field':[], 'empty_js':[], 'short_js':[], 'has_js':[]}
    for rid in ids:
        row=data.get(rid)
        if not row:
            stats['no_record']+=1
            continue
        js_code=row.get('js_code')
        if js_code is None:
            stats['no_js_field']+=1
            if len(examples['no_js_field'])<10: examples['no_js_field'].append(rid)
            continue
        if isinstance(js_code, str):
            length=len(js_code)
        else:
            try:
                length=len(json.dumps(js_code))
            except Exception:
                length=0
        if length==0:
            stats['empty_js']+=1
            if len(examples['empty_js'])<10: examples['empty_js'].append(rid)
        elif length<200:
            stats['short_js']+=1
            if len(examples['short_js'])<10: examples['short_js'].append(rid)
        else:
            stats['has_js']+=1
            if len(examples['has_js'])<10: examples['has_js'].append(rid)
    return stats, examples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gaps', default='artifacts/diagnostics/extended_modality_gaps.json')
    ap.add_argument('--val', default='data/pages_val_full.jsonl')
    ap.add_argument('--test', default='data/pages_test_full.jsonl')
    ap.add_argument('--out', default='artifacts/diagnostics/js_code_gap_analysis.json')
    args=ap.parse_args()
    val_missing=load_gap_ids(args.gaps,'val_full')
    test_missing=load_gap_ids(args.gaps,'test_full')
    val_data=index_dataset(args.val)
    test_data=index_dataset(args.test)
    val_stats, val_examples=summarize(val_missing, val_data)
    test_stats, test_examples=summarize(test_missing, test_data)
    result={'val_full': {'stats': val_stats, 'examples': val_examples}, 'test_full': {'stats': test_stats, 'examples': test_examples}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(f"JS code gap analysis written: {args.out}")

if __name__=='__main__':
    main()
