#!/usr/bin/env python
"""Diagnose JS model coverage gaps.

For canonical splits (train/val/test) this script reports:
  - Total canonical IDs
  - IDs with js_code predictions (from baseline_strict combined)
  - IDs without js_code predictions
  - Among missing, presence of DOM predictions (dom) to show JS-only failure
  - Simple heuristic: whether raw JS fields exist in data record (looking for keys: js_charseq, js_raw, js_augmented)

Outputs JSON + MD under artifacts/diagnostics.
"""
from __future__ import annotations
import json, os, argparse
from typing import Dict, List, Set

JS_KEYS = ["js_charseq","js_raw","js_augmented","js"]


def load_jsonl(path: str) -> List[dict]:
    rows=[]
    if not os.path.isfile(path):
        return rows
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except Exception: pass
    return rows


def canonical_ids(split: str) -> List[str]:
    return [str(r.get('id')) for r in load_jsonl(f'data/pages_{split}.jsonl') if r.get('id') is not None]


def load_baseline_preds(baseline_dir: str, split: str) -> List[dict]:
    path=os.path.join(baseline_dir,f'combined_preds_{split}.jsonl')
    if not os.path.isfile(path): return []
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except Exception: pass
    return out


def analyze_split(split: str, baseline_dir: str):
    canon = canonical_ids(split)
    canon_set = set(canon)
    preds = load_baseline_preds(baseline_dir, split)
    js_ids = {p['id'] for p in preds if p.get('model')=='js_code'}
    dom_ids = {p['id'] for p in preds if p.get('model')=='dom'}
    missing_js = canonical_ids(split)
    missing_js = [cid for cid in canon if cid not in js_ids]
    # load raw data to inspect JS key presence
    records_by_id = {}
    for r in load_jsonl(f'data/pages_{split}.jsonl'):
        rid=str(r.get('id'))
        if rid:
            records_by_id[rid]=r
    js_key_presence_counts = {k:0 for k in JS_KEYS}
    missing_detail=[]
    for mid in missing_js:
        rec = records_by_id.get(mid, {})
        has_js_field = False
        keys_present=[]
        for k in JS_KEYS:
            if k in rec and rec[k]:
                js_key_presence_counts[k]+=1
                has_js_field=True
                keys_present.append(k)
        missing_detail.append({
            'id': mid,
            'has_dom_pred': mid in dom_ids,
            'js_keys_present': keys_present,
        })
    # summarize root causes
    dom_only = sum(1 for d in missing_detail if d['has_dom_pred'] and not d['js_keys_present'])
    no_modalities = sum(1 for d in missing_detail if not d['has_dom_pred'] and not d['js_keys_present'])
    with_js_but_no_pred = sum(1 for d in missing_detail if d['js_keys_present'])
    return {
        'split': split,
        'canonical_size': len(canon_set),
        'js_covered': len(js_ids),
        'js_missing': len(missing_js),
        'js_coverage': len(js_ids)/len(canon_set) if canon_set else 0.0,
        'dom_present_for_missing': sum(1 for d in missing_detail if d['has_dom_pred']),
        'missing_breakdown': {
            'dom_pred_but_no_js_field': dom_only,
            'no_dom_pred_no_js_field': no_modalities,
            'js_field_present_but_no_pred': with_js_but_no_pred,
        },
        'js_key_presence_counts': js_key_presence_counts,
        'sample_missing': missing_detail[:50],
    }


def write_md(path: str, reports: List[dict]):
    lines=["# JS Coverage Audit","",]
    for r in reports:
        lines.append(f"## Split: {r['split']}")
        lines.append(f"Canonical: {r['canonical_size']}  JS covered: {r['js_covered']}  Missing: {r['js_missing']} (cov {r['js_coverage']:.3f})")
        lines.append("Breakdown:")
        for k,v in r['missing_breakdown'].items():
            lines.append(f"- {k}: {v}")
        lines.append("JS key presence in missing (counts):")
        for k,v in r['js_key_presence_counts'].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    with open(path,'w',encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', default='artifacts/baseline_strict')
    ap.add_argument('--splits', nargs='*', default=['val','test'])
    args=ap.parse_args()
    os.makedirs('artifacts/diagnostics', exist_ok=True)
    reps=[analyze_split(s, args.baseline_dir) for s in args.splits]
    with open('artifacts/diagnostics/js_coverage_audit.json','w',encoding='utf-8') as f:
        json.dump({'reports': reps}, f, indent=2)
    write_md('artifacts/diagnostics/js_coverage_audit.md', reps)
    print('JS coverage audit complete.')

if __name__=='__main__':
    main()
