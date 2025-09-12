#!/usr/bin/env python
"""Multi-band cascade optimization with cost model.

Stages:
  Stage 1: url_head threshold t1 -> if below t1 -> predict benign immediately
  Stage 2: (optional) cheap heads subset aggregated by simple mean; threshold t2 -> if below t2 -> benign
  Stage 3: full fusion_calibrated_ids probability threshold (from operating thresholds) for final decision.

We search grids of t1 and t2 (quantiles of corresponding score distributions) and
compute:
  - overall recall / precision / FPR
  - pass rates between stages
  - expected cost per sample given per-head evaluation costs

Output: artifacts/diagnostics/cascade_v2_eval.md + json.
"""
from __future__ import annotations
import os, json, argparse, math
import numpy as np

HEAD_COSTS={
    'url_head': 1.0,
    'text_head': 5.0,
    'js_charcnn': 8.0,
    'js_charcnn_aug': 8.0,
    'js_codet5p': 60.0,
    'fusion_calibrated_ids': 2.0  # logistic combination overhead
}

CHEAP_HEADS=['text_head','js_charcnn','js_charcnn_aug']  # exclude heavy codet5p

def load_preds(path):
    rows=[]
    if not os.path.isfile(path):
        return rows
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'id' in js and 'prob' in js and 'label' in js:
                rows.append(js)
    return rows

def index(rows):
    return {r['id']: r for r in rows}

def quantile_grid(probs, n=25):
    if len(probs)==0: return [0.0]
    qs=np.linspace(0,1,n)
    vals=np.quantile(probs, qs)
    return sorted(set([float(f"{v:.6f}") for v in vals]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--out-json', default='artifacts/diagnostics/cascade_v2_eval.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/cascade_v2_eval.md')
    ap.add_argument('--cheap-heads', nargs='*', default=CHEAP_HEADS)
    ap.add_argument('--fusion-dir', default='artifacts/fusion_calibrated_ids')
    ap.add_argument('--url-head-dir', default='artifacts/url_head')
    ap.add_argument('--head-dir-base', default='artifacts')
    ap.add_argument('--head-costs-json', default=None, help='Optional JSON file with head cost overrides {head: cost}')
    ap.add_argument('--cost-scale', type=float, default=1.0, help='Scale all costs by this factor for sensitivity analysis')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # Override head costs if provided
    head_costs=dict(HEAD_COSTS)
    if args.head_costs_json and os.path.isfile(args.head_costs_json):
        try:
            overrides=json.load(open(args.head_costs_json,'r'))
            for k,v in overrides.items():
                try:
                    head_costs[k]=float(v)
                except Exception:
                    pass
        except Exception:
            pass
    if args.cost_scale and args.cost_scale != 1.0:
        head_costs={k: float(v)*float(args.cost_scale) for k,v in head_costs.items()}
    # Load thresholds
    fusion_thr=0.5
    oph='artifacts/diagnostics/operating_thresholds.json'
    if os.path.isfile(oph):
        js=json.load(open(oph,'r'))
        for fv in js.get('fusion_variants',[]):
            if fv['model']=='fusion_calibrated_ids':
                fusion_thr=fv['threshold']
                break
    # Load preds
    fusion_preds=load_preds(os.path.join(args.fusion_dir, f'preds_{args.split}.jsonl'))
    url_preds=load_preds(os.path.join(args.url_head_dir, f'preds_{args.split}.jsonl'))
    cheap_maps={h: index(load_preds(os.path.join(args.head_dir_base, h, f'preds_{args.split}.jsonl'))) for h in args.cheap_heads}
    url_map=index(url_preds)
    fusion_map=index(fusion_preds)
    # Common IDs
    common=set(url_map.keys()) & set(fusion_map.keys())
    for h,m in cheap_maps.items():
        common &= set(m.keys())
    common=list(common)
    if not common:
        raise SystemExit('No common IDs across required heads.')
    labels=np.array([url_map[i]['label'] for i in common])
    url_scores=np.array([url_map[i]['prob'] for i in common])
    cheap_mean=np.array([np.mean([cheap_maps[h][i]['prob'] for h in args.cheap_heads]) for i in common])
    fusion_scores=np.array([fusion_map[i]['prob'] for i in common])
    pos=labels.sum(); neg=len(labels)-pos
    q1=quantile_grid(url_scores)
    q2=quantile_grid(cheap_mean)
    results=[]
    best=None
    for t1 in q1:
        stage1_mask=url_scores>=t1
        pass1=stage1_mask.mean()
        # cost so far: all pay url_head cost + those passing pay cheap head costs
        for t2 in q2:
            stage2_mask=cheap_mean[stage1_mask]>=t2
            # final evaluation set
            final_mask=np.zeros(len(labels),dtype=bool)
            # Only those both stage masks pass are evaluated by fusion
            idx_stage2=np.where(stage1_mask)[0][stage2_mask]
            final_mask[idx_stage2]=True
            # Predictions:
            # Default benign unless passes final fusion threshold
            preds=np.zeros(len(labels),dtype=int)
            preds[final_mask & (fusion_scores>=fusion_thr)]=1
            tp=int(((preds==1)&(labels==1)).sum()); fp=int(((preds==1)&(labels==0)).sum())
            fn=int(((preds==0)&(labels==1)).sum()); tn=int(((preds==0)&(labels==0)).sum())
            recall=tp/max(1,pos); precision=tp/max(1,tp+fp); fpr=fp/max(1,neg)
            pass2=len(idx_stage2)/len(labels)
            final_eval=final_mask.mean()
            # Cost model
            cost = head_costs.get('url_head',1.0)
            cost += pass1 * sum(head_costs.get(h,5.0) for h in args.cheap_heads)
            cost += final_eval * (head_costs.get('js_codet5p',60.0) + head_costs.get('fusion_calibrated_ids',2.0))
            rec={'t1': t1,'t2': t2,'recall': recall,'precision': precision,'fpr': fpr,'pass_rate_stage1': pass1,'pass_rate_stage2': pass2,'final_eval_rate': final_eval,'expected_cost': cost}
            results.append(rec)
            if recall >= args.target_recall:
                if best is None or cost < best['expected_cost'] - 1e-9:
                    best=rec
    results=sorted(results, key=lambda r:(-r['recall'], r['expected_cost']))
    out={'target_recall': args.target_recall,'fusion_threshold': fusion_thr,'results': results[:500],'best': best,'head_costs': head_costs}
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2)
    # Markdown summary (top 40 rows)
    lines=['# Cascade V2 Evaluation','',f'Split: {args.split}  Target Recall: {args.target_recall}  Fusion Thr: {fusion_thr:.6f}','', 't1 | t2 | Recall | Precision | FPR | Pass1 | Pass2 | FinalEval | Cost','-- | -- | ------ | --------- | --- | ----- | ----- | -------- | ----']
    for r in results[:40]:
        row_parts=[
            f"{r['t1']:.6f}",
            f"{r['t2']:.6f}",
            f"{r['recall']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['fpr']:.4f}",
            f"{r['pass_rate_stage1']:.3f}",
            f"{r['pass_rate_stage2']:.3f}",
            f"{r['final_eval_rate']:.3f}",
            f"{r['expected_cost']:.2f}"
        ]
        lines.append(' | '.join(row_parts))
    if best:
        lines += ['', '## Best (Min Cost meeting Recall constraint)', '', json.dumps(best, indent=2)]
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Wrote', args.out_md)

if __name__=='__main__':
    main()
