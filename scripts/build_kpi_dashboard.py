#!/usr/bin/env python
"""Phase 11: KPI dashboard builder.

Appends current key metrics to a log file and regenerates a markdown dashboard.
Key metrics: fusion test precision, recall, FPR, Brier, ECE, cascade best cost, refiner uplift.
"""
from __future__ import annotations
import os, json, argparse, datetime

EVAL_FILE='artifacts/diagnostics/enhanced_evaluation.json'
LOG_FILE='artifacts/diagnostics/kpi_timeseries.jsonl'
DASH_FILE='artifacts/diagnostics/kpi_dashboard.md'

def load_json(path):
    if not os.path.isfile(path): return None
    try: return json.load(open(path,'r'))
    except Exception: return None

def extract_metrics(data):
    fusion=data.get('fusion_calibrated_ids',{})
    op=fusion.get('operating_point_test') or fusion.get('test',{}).get('operating_point_test') or {}
    test=fusion.get('test',{})
    cascade=data.get('cascade_v2',{})
    best_cascade=(cascade.get('best') or {})
    refiner=data.get('confusion_refiner',{})
    return {
        'fusion_precision': op.get('precision'),
        'fusion_recall': op.get('recall'),
        'fusion_fpr': op.get('fpr'),
        'fusion_brier': test.get('brier'),
        'fusion_ece': test.get('ece'),
        'cascade_cost': best_cascade.get('expected_cost'),
        'cascade_recall': best_cascade.get('recall'),
        'refiner_uplift': refiner.get('precision_uplift')
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-file', default=EVAL_FILE)
    ap.add_argument('--log-file', default=LOG_FILE)
    ap.add_argument('--out-md', default=DASH_FILE)
    ap.add_argument('--out-csv', default='artifacts/diagnostics/kpi_timeseries.csv')
    args=ap.parse_args()
    data=load_json(args.eval_file)
    if data is None: raise SystemExit('Missing enhanced evaluation file.')
    metrics=extract_metrics(data)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    entry={'ts': datetime.datetime.utcnow().isoformat()+'Z', **metrics}
    with open(args.log_file,'a',encoding='utf-8') as f: f.write(json.dumps(entry)+'\n')
    # Rebuild dashboard
    rows=[]
    with open(args.log_file,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: rows.append(json.loads(ln))
            except Exception: pass
    lines=['# KPI Dashboard','', 'TS | Fusion Precision | Fusion Recall | FPR | Brier | ECE | Cascade Cost | Cascade Recall | Refiner Uplift','-- | ---------------- | ------------- | --- | ----- | --- | ------------ | --------------- | --------------']
    for r in rows[-200:]:
        lines.append(' | '.join([r.get('ts',''), *(f"{r.get(k,0):.4f}" if isinstance(r.get(k), (int,float)) and r.get(k) is not None else '-' for k in ['fusion_precision','fusion_recall','fusion_fpr','fusion_brier','fusion_ece','cascade_cost','cascade_recall','refiner_uplift'])]))
    with open(args.out_md,'w',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    # Optional CSV export
    try:
        import csv
        with open(args.out_csv,'w', newline='', encoding='utf-8') as cf:
            w=csv.writer(cf)
            w.writerow(['ts','fusion_precision','fusion_recall','fusion_fpr','fusion_brier','fusion_ece','cascade_cost','cascade_recall','refiner_uplift'])
            for r in rows:
                w.writerow([r.get('ts',''), r.get('fusion_precision'), r.get('fusion_recall'), r.get('fusion_fpr'), r.get('fusion_brier'), r.get('fusion_ece'), r.get('cascade_cost'), r.get('cascade_recall'), r.get('refiner_uplift')])
    except Exception:
        pass
    print('Updated KPI dashboard.')

if __name__=='__main__':
    main()
