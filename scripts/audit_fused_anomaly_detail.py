#!/usr/bin/env python
"""Detailed inspection of a fused anomaly ID where dom+js_code exist but fused missing.

Reports availability of each head, raw probabilities, and suggests causes.
"""
from __future__ import annotations
import json, os, argparse
from typing import Dict

HEAD_DIRS = {
    'dom': 'artifacts/markup_run',
    'js_code': 'artifacts/js_codet5p',
    'fused': 'artifacts/fusion',
    'meta_fused': 'artifacts/fusion_meta'
}


def load_pred_by_id(dir_path: str, split: str, target_id: str):
    path = os.path.join(dir_path, f'preds_{split}.jsonl')
    if not os.path.isfile(path):
        return None
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            try:
                js=json.loads(ln)
            except Exception:
                continue
            if str(js.get('id')) == target_id:
                return js
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--id', required=True)
    ap.add_argument('--split', default='test')
    args=ap.parse_args()
    report={}
    for head, d in HEAD_DIRS.items():
        report[head] = load_pred_by_id(d, args.split, args.id)
    # Cause heuristics
    cause = 'unknown'
    if report['dom'] and report['js_code'] and not report['fused']:
        cause = 'fusion_alignment_excluded_or_bug'
    print(json.dumps({'id': args.id, 'split': args.split, 'preds': report, 'cause_guess': cause}, indent=2))

if __name__=='__main__':
    main()
