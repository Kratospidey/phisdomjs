#!/usr/bin/env python
"""Phase 8: Confusion band refiner training.

Identifies samples whose fusion prob is within [thr - margin, thr + margin] on validation.
Trains a small sklearn GradientBoostingClassifier using head probabilities as features
to re-evaluate those cases only. Reports potential precision uplift if applied.
"""
from __future__ import annotations
import os, json, argparse, numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    GradientBoostingClassifier=None

HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]

def load_preds(model_dir, split='val'):
    path=os.path.join(model_dir,f'preds_{split}.jsonl')
    out=[]
    if not os.path.isfile(path): return out
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'id' in js and 'prob' in js and 'label' in js: out.append(js)
    return out

def load_head(head, split):
    path=f'artifacts/{head}/preds_{split}_calibrated.jsonl'
    if not os.path.isfile(path): path=f'artifacts/{head}/preds_{split}.jsonl'
    m={}
    if not os.path.isfile(path): return m
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if js.get('id'): m[js['id']]=js
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fusion-dir', default='artifacts/fusion_calibrated_ids')
    ap.add_argument('--split', default='val')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--out-json', default='artifacts/diagnostics/confusion_refiner.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/confusion_refiner.md')
    ap.add_argument('--holdout-frac', type=float, default=0.3)
    ap.add_argument('--eval-on-test', action='store_true', help='If set, train on val-band and evaluate band on test for true uplift')
    args=ap.parse_args()
    if GradientBoostingClassifier is None:
        raise SystemExit('sklearn not available for refiner.')
    oph='artifacts/diagnostics/operating_thresholds.json'; thr=0.5
    if os.path.isfile(oph):
        js=json.load(open(oph,'r'))
        for fv in js.get('fusion_variants',[]):
            if fv['model']=='fusion_calibrated_ids': thr=fv['threshold']
    fusion=load_preds(args.fusion_dir,args.split)
    if not fusion: raise SystemExit('Missing fusion predictions.')
    head_maps={h: load_head(h,args.split) for h in HEADS}
    common=set.intersection(*[set(m.keys()) for m in head_maps.values()]) & set(r['id'] for r in fusion)
    rows=[r for r in fusion if r['id'] in common]
    # confusion band selection
    band=[r for r in rows if abs(r['prob']-thr)<=args.margin]
    X=np.array([[head_maps[h][r['id']]['prob'] for h in HEADS] for r in band]) if band else np.zeros((0,len(HEADS)))
    y=np.array([r['label'] for r in band]) if band else np.zeros(0)
    model=None
    probs=np.array([])
    refined_band_indices=None
    if len(y)>=20 and y.sum()>0 and (len(y)-y.sum())>0:
        # Split into train/holdout indices
        n=len(y)
        idx=np.arange(n)
        # simple stratified split
        pos_idx=idx[y==1]; neg_idx=idx[y==0]
        n_hold_pos=max(1,int(len(pos_idx)*args.holdout_frac)); n_hold_neg=max(1,int(len(neg_idx)*args.holdout_frac))
        hold_pos=pos_idx[:n_hold_pos]; hold_neg=neg_idx[:n_hold_neg]
        hold=np.concatenate([hold_pos, hold_neg]); train_idx=np.array([i for i in idx if i not in set(hold)])
        model=GradientBoostingClassifier(random_state=42, max_depth=2)
        model.fit(X[train_idx], y[train_idx])
        probs=model.predict_proba(X[hold])[:,1]
        # Replace only holdout with refined predictions; others keep base
        refined_band_indices=hold
    else:
        probs=np.array([])
    # Simulate uplift: replace fusion prob with refiner prob (band only) keeping fusion threshold
    improved_tp=0; improved_fp=0; base_tp=0; base_fp=0
    # Map band row index to refined prob when available
    refined_map={}
    if refined_band_indices is not None:
        for k,bi in enumerate(refined_band_indices):
            refined_map[band[bi]['id']]=probs[k]
    for r in rows:
        base_pred = 1 if r['prob']>=thr else 0
        if probs.size and r['id'] in refined_map:
            new_pred=1 if refined_map[r['id']]>=thr else 0
        else:
            new_pred=base_pred
        if base_pred==1 and r['label']==1: base_tp+=1
        if base_pred==1 and r['label']==0: base_fp+=1
        if new_pred==1 and r['label']==1: improved_tp+=1
        if new_pred==1 and r['label']==0: improved_fp+=1
    base_precision=base_tp/max(1, base_tp+base_fp)
    new_precision=improved_tp/max(1, improved_tp+improved_fp)
    uplift=new_precision-base_precision
    out={'threshold': thr,'margin': args.margin,'band_size': len(band),'base_precision': base_precision,'refined_precision': new_precision,'precision_uplift': uplift,'model_trained': model is not None}

    # Optional: evaluate on test band if requested
    if args.eval_on_test and model is not None:
        fusion_te=load_preds(args.fusion_dir,'test')
        if fusion_te:
            head_maps_te={h: load_head(h,'test') for h in HEADS}
            common_te=set.intersection(*[set(m.keys()) for m in head_maps_te.values()]) & set(r['id'] for r in fusion_te)
            rows_te=[r for r in fusion_te if r['id'] in common_te]
            band_te=[r for r in rows_te if abs(r['prob']-thr)<=args.margin]
            if band_te:
                X_te=np.array([[head_maps_te[h][r['id']].get('prob',0.0) for h in HEADS] for r in band_te])
                y_te=np.array([r['label'] for r in band_te])
                try:
                    probs_te=model.predict_proba(X_te)[:,1]
                except Exception:
                    probs_te=np.zeros(len(band_te))
                # Apply same threshold to compute precision uplift within band
                base_tp=base_fp=improved_tp=improved_fp=0
                for r,p_new in zip(band_te, probs_te):
                    base_pred = 1 if r['prob']>=thr else 0
                    new_pred = 1 if p_new>=thr else 0
                    if base_pred==1 and r['label']==1: base_tp+=1
                    if base_pred==1 and r['label']==0: base_fp+=1
                    if new_pred==1 and r['label']==1: improved_tp+=1
                    if new_pred==1 and r['label']==0: improved_fp+=1
                base_prec_te=base_tp/max(1, base_tp+base_fp)
                new_prec_te=improved_tp/max(1, improved_tp+improved_fp)
                out['test_band']={'size': len(band_te),'base_precision': base_prec_te,'refined_precision': new_prec_te,'precision_uplift': new_prec_te-base_prec_te}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
    lines=['# Confusion Refiner','',f'Threshold: {thr:.6f}  Margin: {args.margin}','', f'Band size: {len(band)}', f'Model trained: {model is not None}', '', f'Base Precision: {base_precision:.4f}', f'Refined Precision: {new_precision:.4f}', f'Uplift: {uplift:.4f}']
    with open(args.out_md,'w',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print('Wrote confusion refiner results.')

if __name__=='__main__':
    main()
