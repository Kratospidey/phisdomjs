#!/usr/bin/env python
"""Phase 6: Multi-band cascade exhaustive/limited search with Pareto frontier.

Bands (example):
 1. url_head threshold t1
 2. cheap aggregate (mean of text_head + js_charcnn + js_charcnn_aug) threshold t2
 3. heavy head (js_codet5p) threshold t3 (optional: if below t3 => benign)
 4. fusion or meta probability final threshold (pre-determined) -> label

We search quantile-based grids for t1,t2,t3 and compute cost & metrics.
Return top K Pareto (cost vs recall) plus best by cost under recall constraint.
"""
from __future__ import annotations
import os, json, argparse, numpy as np

COSTS={'url_head':1.0,'text_head':5.0,'js_charcnn':8.0,'js_charcnn_aug':8.0,'js_codet5p':60.0,'fusion':2.0}
CHEAP=['text_head','js_charcnn','js_charcnn_aug']

def load_preds(path):
    out=[]
    if not os.path.isfile(path): return out
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: js=json.loads(ln)
            except Exception: continue
            if 'id' in js and 'prob' in js and 'label' in js: out.append(js)
    return out

def quantiles(arr, n):
    if len(arr)==0: return [0.0]
    qs=np.linspace(0,1,n)
    return sorted(set(float(f"{v:.6f}") for v in np.quantile(arr,qs)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--grid', type=int, default=15, help='Quantile grid points per band')
    ap.add_argument('--out-json', default='artifacts/diagnostics/cascade_band_search.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/cascade_band_search.md')
    ap.add_argument('--fusion-dir', default='artifacts/fusion_calibrated_ids')
    ap.add_argument('--meta-dir', default='artifacts/meta_fusion_cv')
    ap.add_argument('--use-meta', action='store_true')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # fusion probability source
    final_dir=args.meta_dir if args.use_meta else args.fusion_dir
    final_file=os.path.join(final_dir, f'preds_{args.split}.jsonl')
    fusion=load_preds(final_file)
    url=load_preds(f'artifacts/url_head/preds_{args.split}.jsonl')
    js_c=load_preds(f'artifacts/js_charcnn/preds_{args.split}.jsonl')
    js_ca=load_preds(f'artifacts/js_charcnn_aug/preds_{args.split}.jsonl')
    text=load_preds(f'artifacts/text_head/preds_{args.split}.jsonl')
    heavy=load_preds(f'artifacts/js_codet5p/preds_{args.split}.jsonl')
    maps={k:{r['id']:r for r in v} for k,v in [('fusion',fusion),('url_head',url),('js_charcnn',js_c),('js_charcnn_aug',js_ca),('text_head',text),('js_codet5p',heavy)]}
    common=set(maps['fusion'].keys()) & set(maps['url_head'].keys()) & set(maps['js_charcnn'].keys()) & set(maps['js_charcnn_aug'].keys()) & set(maps['text_head'].keys()) & set(maps['js_codet5p'].keys())
    if not common: raise SystemExit('No common IDs for cascade search.')
    ids=list(common)
    labels=np.array([maps['fusion'][i]['label'] for i in ids])
    url_p=np.array([maps['url_head'][i]['prob'] for i in ids])
    cheap_mean=np.array([np.mean([maps[h][i]['prob'] for h in CHEAP]) for i in ids])
    heavy_p=np.array([maps['js_codet5p'][i]['prob'] for i in ids])
    final_p=np.array([maps['fusion'][i]['prob'] for i in ids])
    # operating threshold for final
    final_thr=0.5
    oph='artifacts/diagnostics/operating_thresholds.json'
    if os.path.isfile(oph):
        js=json.load(open(oph,'r'))
        key='fusion_calibrated_ids' if not args.use_meta else 'meta_fusion_calibrated'
        for fv in js.get('fusion_variants',[]):
            if fv['model']==key: final_thr=fv['threshold']
    q1=quantiles(url_p,args.grid)
    q2=quantiles(cheap_mean,args.grid)
    q3=quantiles(heavy_p,args.grid)
    pos=labels.sum(); neg=len(labels)-pos
    results=[]; best=None
    for t1 in q1:
        s1=url_p>=t1
        pass1=s1.mean()
        for t2 in q2:
            s2=(cheap_mean>=t2) & s1
            pass2=s2.mean()
            for t3 in q3:
                s3=(heavy_p>=t3) & s2
                pass3=s3.mean()
                preds=np.zeros(len(labels),dtype=int)
                preds[s3 & (final_p>=final_thr)]=1
                tp=int(((preds==1)&(labels==1)).sum()); fp=int(((preds==1)&(labels==0)).sum())
                fn=int(((preds==0)&(labels==1)).sum()); tn=int(((preds==0)&(labels==0)).sum())
                recall=tp/max(1,pos); precision=tp/max(1,tp+fp); fpr=fp/max(1,neg)
                # cost expectation
                cost=COSTS['url_head']
                cost += pass1*sum(COSTS[h] for h in CHEAP)
                cost += pass2*COSTS['js_codet5p']
                cost += pass3*COSTS['fusion']
                rec={'t1':t1,'t2':t2,'t3':t3,'recall':recall,'precision':precision,'fpr':fpr,'pass1':pass1,'pass2':pass2,'pass3':pass3,'expected_cost':cost}
                results.append(rec)
                if recall>=args.target_recall and (best is None or cost<best['expected_cost']-1e-9):
                    best=rec
    # Pareto frontier (min cost for recall bins)
    results_sorted=sorted(results, key=lambda r:(-r['recall'], r['expected_cost']))
    frontier=[]; seen=set()
    for r in results_sorted:
        bucket=round(r['recall'],3)
        if bucket in seen: continue
        frontier.append(r); seen.add(bucket)
        if len(frontier)>=50: break
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump({'target_recall': args.target_recall,'final_threshold': final_thr,'best': best,'frontier': frontier,'searched': len(results)}, f, indent=2)
    lines=['# Cascade Band Search','',f'Split: {args.split}  Target Recall: {args.target_recall}  Final Thr: {final_thr:.6f}','', 't1 | t2 | t3 | Recall | Precision | FPR | Pass1 | Pass2 | Pass3 | Cost','-- | -- | -- | ------ | --------- | --- | ----- | ----- | ----- | ----']
    for r in frontier[:40]:
        lines.append(' | '.join([f"{r['t1']:.5f}",f"{r['t2']:.5f}",f"{r['t3']:.5f}",f"{r['recall']:.4f}",f"{r['precision']:.4f}",f"{r['fpr']:.4f}",f"{r['pass1']:.3f}",f"{r['pass2']:.3f}",f"{r['pass3']:.3f}",f"{r['expected_cost']:.2f}"]))
    if best:
        lines += ['','## Best (Min Cost meeting Recall constraint)','',json.dumps(best,indent=2)]
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Wrote cascade band search results.')

if __name__=='__main__':
    main()
