#!/usr/bin/env python
"""Phase 3: Prepare retraining plan incorporating hard negatives.

This script inspects the augmented training dataset produced by
augment_with_hard_negatives.py, summarizes class balance changes, and emits a
proposed retraining command plan for each head. It doesn't actually retrain
models (since head training logic is project-specific) but gives reproducible
commands / JSON manifest to plug into automation.

Outputs:
  artifacts/diagnostics/hard_negative_retrain_plan.json
  artifacts/diagnostics/hard_negative_retrain_plan.md

The plan includes suggested class weighting if negatives dwarf positives and a
recommended sampling cap for new hard negatives to avoid overwhelming the
signal (configurable via --max-hard-negative-fraction).
"""
from __future__ import annotations
import os, json, argparse, math

HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_jsonl(path):
    if not os.path.isfile(path): return []
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def summarize(records):
    pos=sum(1 for r in records if r.get('label')==1)
    neg=sum(1 for r in records if r.get('label')==0)
    hard=sum(1 for r in records if r.get('hard_negative'))
    return {'total': len(records),'positives': pos,'negatives': neg,'hard_negatives': hard,'positive_ratio': pos/max(1,len(records)),'hard_negative_ratio': hard/max(1,neg)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--aug-train', default='data/pages_train_augmented.jsonl')
    ap.add_argument('--base-train', default='data/pages_train.jsonl')
    ap.add_argument('--max-hard-negative-fraction', type=float, default=0.30, help='Max fraction of negatives to come from hard negatives; may downsample if exceeded.')
    ap.add_argument('--out-json', default='artifacts/diagnostics/hard_negative_retrain_plan.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/hard_negative_retrain_plan.md')
    ap.add_argument('--target-recall', type=float, default=0.95)
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    aug=load_jsonl(args.aug_train)
    base=load_jsonl(args.base_train)
    if not aug:
        raise SystemExit('Augmented training file missing or empty; run augmentation first.')
    s_aug=summarize(aug); s_base=summarize(base) if base else None
    hard_negatives=[r for r in aug if r.get('hard_negative')]
    # Compute recommended sample of hard negatives
    neg_total=s_aug['negatives']
    hard_allowed=int(args.max_hard_negative_fraction * neg_total)
    hard_current=s_aug['hard_negatives']
    downsample_needed = hard_current > hard_allowed
    plan={'base_summary': s_base,'aug_summary': s_aug,'constraints': {'max_hard_negative_fraction': args.max_hard_negative_fraction},'hard_negative_sampling': {'current': hard_current,'allowed': hard_allowed,'downsample': downsample_needed},'heads': []}
    # Suggested class weight: inverse prevalence
    pos_ratio=s_aug['positives']/max(1,s_aug['total'])
    neg_ratio=1-pos_ratio
    if pos_ratio>0 and neg_ratio>0:
        weight_pos=0.5/pos_ratio
        weight_neg=0.5/neg_ratio
    else:
        weight_pos=weight_neg=1.0
    for h in HEADS:
        plan['heads'].append({
            'head': h,
            'train_data': args.aug_train,
            'class_weights': {'positive': weight_pos,'negative': weight_neg},
            'notes': 'Apply same preprocessing; ensure random seed fixed for reproducibility.'
        })
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump(plan,f,indent=2)
    # Markdown
    md=["# Hard Negative Retraining Plan","",f"Augmented file: {args.aug_train}"]
    if s_base:
        md += ["","## Baseline vs Augmented","Metric | Base | Augmented","------ | ---- | ---------"]
        for k in ['total','positives','negatives','hard_negatives','positive_ratio','hard_negative_ratio']:
            md.append(f"{k} | {s_base.get(k,'-')} | {s_aug.get(k)}")
    md += ["","## Hard Negative Sampling","Current | Allowed | Downsample?","------- | ------- | ----------",f"{hard_current} | {hard_allowed} | {downsample_needed}"]
    md += ["","## Class Weights","positive | negative","-------- | --------",f"{weight_pos:.4f} | {weight_neg:.4f}"]
    md += ["","## Head Training Commands (illustrative)","Head | Command","---- | -------"]
    for h in HEADS:
        cmd=f"python train_head.py --head {h} --train {args.aug_train} --class-weight-pos {weight_pos:.4f} --class-weight-neg {weight_neg:.4f}"
        if downsample_needed:
            cmd += f" --max-hard-neg-fraction {args.max_hard_negative_fraction}"
        md.append(f"{h} | `{cmd}`")
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(md)+'\n')
    print('Wrote retraining plan to', args.out_json)

if __name__=='__main__':
    main()
