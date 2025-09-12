#!/usr/bin/env python
"""Phase 9: Consolidated enhanced evaluation aggregator.

Collects metrics from multiple artifacts if present; skips missing gracefully.
Produces JSON + markdown summary for downstream CI gating and dashboard.
"""
from __future__ import annotations
import os, json, argparse, glob

SOURCE_FILES=[
    'artifacts/fusion_calibrated_ids/metrics.json',
    'artifacts/fusion_calibrated_ids_cv/metrics.json',
    'artifacts/meta_fusion_cv/metrics.json',
    'artifacts/diagnostics/cascade_v2_eval.json',
    'artifacts/diagnostics/cascade_band_search.json',
    'artifacts/diagnostics/confusion_refiner.json'
]

def load_json(path):
    if not os.path.isfile(path): return None
    try: return json.load(open(path,'r'))
    except Exception: return None

def extract():
    out={}
    fusion=load_json('artifacts/fusion_calibrated_ids/metrics.json')
    if fusion: out['fusion_calibrated_ids']=fusion
    fusion_cv=load_json('artifacts/fusion_calibrated_ids_cv/metrics.json')
    if fusion_cv: out['fusion_calibrated_ids_cv']=fusion_cv
    meta_cv=load_json('artifacts/meta_fusion_cv/metrics.json')
    if meta_cv: out['meta_fusion_cv']=meta_cv
    cascade=load_json('artifacts/diagnostics/cascade_v2_eval.json')
    if cascade: out['cascade_v2']=cascade
    cascade_band=load_json('artifacts/diagnostics/cascade_band_search.json')
    if cascade_band: out['cascade_band']=cascade_band
    refiner=load_json('artifacts/diagnostics/confusion_refiner.json')
    if refiner: out['confusion_refiner']=refiner
    loss=load_json('artifacts/diagnostics/fusion_loss_experiments.json')
    if loss: out['loss_experiments']=loss
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-json', default='artifacts/diagnostics/enhanced_evaluation.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/enhanced_evaluation.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    data=extract()
    with open(args.out_json,'w',encoding='utf-8') as f: json.dump(data,f,indent=2)
    # Build KPI table
    def op_fields(entry, key):
        m=entry.get(key, {})
        op=m.get('operating_point_test') or m.get('test',{}).get('operating_point_test') or {}
        return op.get('precision'), op.get('recall'), op.get('fpr'), (m.get('test') or {}).get('brier'), (m.get('test') or {}).get('ece')
    kpi=[('fusion_calibrated_ids','Fusion (ID)'), ('fusion_calibrated_ids_cv','Fusion CV (OOF)'), ('meta_fusion_cv','Meta CV')]
    rows=[]
    for key,title in kpi:
        if key in data:
            prec,rec,fpr,brier,ece=op_fields(data, key)
            rows.append((title, prec,rec,fpr,brier,ece))
    # Cascade & refiner
    cascade=data.get('cascade_v2') or {}
    best=cascade.get('best') or {}
    refiner=data.get('confusion_refiner') or {}
    lines=['# Enhanced Evaluation','', 'Model | Precision | Recall | FPR | Brier | ECE','----- | --------- | ------ | --- | ----- | ---']
    for title,prec,rec,fpr,brier,ece in rows:
        lines.append(' | '.join([title, *(f"{x:.4f}" if isinstance(x,(int,float)) and x is not None else '-' for x in [prec,rec,fpr,brier,ece])]))
    lines += ['','## Cascade (V2)','', json.dumps(best, indent=2) if best else 'No cascade best result found.', '', '## Refiner', '', json.dumps(refiner, indent=2) if refiner else 'No refiner result found.']
    with open(args.out_md,'w',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print('Wrote enhanced evaluation summary.')

if __name__=='__main__':
    main()
