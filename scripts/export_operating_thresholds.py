#!/usr/bin/env python
"""Export consolidated operating thresholds for heads and fusion variants to a single JSON for deployment."""
from __future__ import annotations
import os, json, argparse

HEAD_CAL_DIR='artifacts/diagnostics/head_calibration'
FUSION_IDS='artifacts/fusion_calibrated_ids/metrics.json'
META_CAL='artifacts/meta_fusion_calibrated/metrics.json'
GRID='artifacts/diagnostics/fusion_variants_grid.md'

def load_head_thresholds():
    out=[]
    if not os.path.isdir(HEAD_CAL_DIR): return out
    for fn in os.listdir(HEAD_CAL_DIR):
        if not fn.endswith('.json'): continue
        js=json.load(open(os.path.join(HEAD_CAL_DIR,fn),'r',encoding='utf-8'))
        head=js['head']
        if head == 'dom_gcn':
            continue  # dropped head
        op=js['val']['operating_point']
        out.append({'model': head, 'threshold': op['threshold'], 'precision_val': op['precision'], 'recall_val': op['recall'], 'fpr_val': op['fpr']})
    return out

def load_variant_ops():
    out=[]
    def try_file(path, name):
        if os.path.isfile(path):
            js=json.load(open(path,'r',encoding='utf-8'))
            opv=js.get('operating_point_val'); opt=js.get('operating_point_test')
            if opv and opt:
                out.append({'model': name, 'threshold': opv['threshold'], 'precision_val': opv['precision'], 'recall_val': opv['recall'], 'fpr_val': opv['fpr'], 'precision_test': opt['precision'], 'recall_test': opt['recall'], 'fpr_test': opt['fpr']})
    try_file(FUSION_IDS,'fusion_calibrated_ids')
    try_file(META_CAL,'meta_fusion_calibrated')
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/diagnostics/operating_thresholds.json')
    args=ap.parse_args()
    data={'heads': load_head_thresholds(),'fusion_variants': load_variant_ops()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2)
    print('Wrote', args.out)

if __name__=='__main__':
    main()
