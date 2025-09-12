#!/usr/bin/env python
"""Summarize js_code gap analysis JSON into markdown.
Reads artifacts/diagnostics/js_code_gap_full.json (or provided path) and emits a markdown summary with counts and percentages.
"""
from __future__ import annotations
import json, os, argparse

def pct(part, whole):
    if not whole:
        return 0.0
    return 100.0 * part / whole

def render_split(name, stats):
    total=stats['total_missing']
    lines=[f"### {name}", f"Total missing js_code predictions: {total}"]
    for key in ['no_record','no_js_field','empty_js','short_js','has_js']:
        val=stats.get(key,0)
        lines.append(f"- {key}: {val} ({pct(val,total):.1f}%)")
    lines.append("")
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gap-json', default='artifacts/diagnostics/js_code_gap_full.json')
    ap.add_argument('--out', default='artifacts/diagnostics/js_code_gap_summary.md')
    args=ap.parse_args()
    if not os.path.isfile(args.gap_json):
        raise SystemExit(f"Gap JSON not found: {args.gap_json}")
    with open(args.gap_json,'r',encoding='utf-8') as f:
        data=json.load(f)
    lines=["# js_code Coverage Gap Summary","", "This summarizes why pages lack js_code head predictions in *_full splits."]
    for split in ['val_full','test_full']:
        if split in data:
            lines.append(render_split(split, data[split]['stats']))
    # high level consolidation
    combined_keys=['no_record','no_js_field','empty_js','short_js','has_js']
    agg={k:0 for k in combined_keys}
    tot=0
    for split in ['val_full','test_full']:
        if split in data:
            s=data[split]['stats']
            tot+=s['total_missing']
            for k in combined_keys:
                agg[k]+=s.get(k,0)
    if tot:
        lines.append("### Combined")
        lines.append(f"Total missing across splits: {tot}")
        for k in combined_keys:
            lines.append(f"- {k}: {agg[k]} ({pct(agg[k],tot):.1f}%)")
    content='\n'.join(lines).rstrip()+"\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote summary markdown: {args.out}")

if __name__=='__main__':
    main()
