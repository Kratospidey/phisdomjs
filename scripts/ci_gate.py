#!/usr/bin/env python
"""Phase 10: CI gating on key metrics.

Loads previous baseline metrics and current enhanced metrics; enforces relative degradation thresholds.
If baseline absent and --create-baseline, saves current as baseline.
Exit codes: 0 success, 5 gate failure.
"""
from __future__ import annotations
import os, json, argparse, sys

BASELINE_FILE='artifacts/diagnostics/ci_baseline.json'
CURRENT_FILE='artifacts/diagnostics/enhanced_evaluation.json'

def load(path):
    if not os.path.isfile(path): return None
    try: return json.load(open(path,'r'))
    except Exception: return None

def extract_precision(entry):
    # Try fusion calibrated test precision
    metrics=entry.get('fusion_calibrated_ids') or {}
    op_test=metrics.get('operating_point_test') or metrics.get('test',{}).get('operating_point_test') or {}
    return op_test.get('precision')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-precision-drop', type=float, default=0.02)
    ap.add_argument('--max-fpr-increase', type=float, default=0.02)
    ap.add_argument('--create-baseline', action='store_true')
    ap.add_argument('--force', action='store_true', help='Ignore gates (always pass)')
    args=ap.parse_args()
    cur=load(CURRENT_FILE)
    if cur is None: sys.exit('Missing current enhanced evaluation. Run evaluate_enhanced.py first.')
    base=load(BASELINE_FILE)
    if base is None and args.create_baseline:
        os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
        json.dump(cur, open(BASELINE_FILE,'w'), indent=2)
        print('Created baseline; passing (first run).')
        return
    if base is None:
        print('No baseline present and --create-baseline not set; passing by default.')
        return
    cur_prec=extract_precision(cur) or 0
    base_prec=extract_precision(base) or cur_prec
    precision_drop = (base_prec - cur_prec)/base_prec if base_prec>0 else 0
    # FPR attempt
    def extract_fpr(entry):
        metrics=entry.get('fusion_calibrated_ids') or {}
        op=metrics.get('operating_point_test') or metrics.get('test',{}).get('operating_point_test') or {}
        return op.get('fpr')
    cur_fpr=extract_fpr(cur) or 0
    base_fpr=extract_fpr(base) or cur_fpr
    fpr_increase = (cur_fpr - base_fpr)/base_fpr if base_fpr>0 else 0
    pass_gate = (precision_drop <= args.max_precision_drop) and (fpr_increase <= args.max_fpr_increase)
    print(f"Precision drop: {precision_drop:.4f} (limit {args.max_precision_drop})  FPR increase: {fpr_increase:.4f} (limit {args.max_fpr_increase})")
    if not pass_gate and not args.force:
        print('CI Gate FAILED')
        sys.exit(5)
    print('CI Gate PASSED')

if __name__=='__main__':
    main()
