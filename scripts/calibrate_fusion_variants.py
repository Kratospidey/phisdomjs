#!/usr/bin/env python
"""Compare calibration / threshold metrics for fusion variants.
Reads metrics_* files or recomputes from combined baseline records.
Generates markdown table with ROC/PR/FPR@TPR90/95 and coverage.
"""
from __future__ import annotations
import json, os, argparse, math
from typing import Dict, List

VARIANTS = ["fused","fused_covmax","fused_soft","meta_fused"]
# Mapping from reporting variant alias to underlying model field stored in prediction JSONLs
MODEL_FOR_VARIANT = {
    'fused': ['fused','fusion'],
    'fused_covmax': ['fused_covmax','fusion_covmax'],
    'fused_soft': ['fused_soft','fusion_soft'],
    'meta_fused': ['meta_fused','fusion_meta']
}


def load_combined(baseline_dir: str, split: str):
    path=os.path.join(baseline_dir, f"combined_preds_{split}.jsonl")
    out=[]
    if not os.path.isfile(path): return out
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip();
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out


def metrics_for_variant(records, variant: str):
    model_names = MODEL_FOR_VARIANT.get(variant, [variant])
    recs=[r for r in records if r.get('model') in model_names]
    if not recs:
        return None
    labels=[r['label'] for r in recs]
    probs=[r['prob'] for r in recs]
    import numpy as np
    labels_arr=np.array(labels); probs_arr=np.array(probs)
    P=labels_arr.sum(); N=len(labels_arr)-P
    def roc_auc():
        # probability-of-rank (same method as exporter)
        pairs=sorted(zip(probs_arr, labels_arr), key=lambda x: x[0])
        scores=[p for p,_ in pairs]; labs=[y for _,y in pairs]
        pos=sum(labs); neg=len(labs)-pos
        if pos==0 or neg==0: return float('nan')
        ranks=[0.0]*len(scores); i=0; r=1
        while i < len(scores):
            j=i
            while j+1 < len(scores) and scores[j+1]==scores[i]:
                j+=1
            avg=(r + (r + (j-i)))/2
            for k in range(i,j+1): ranks[k]=avg
            r=j+2; i=j+1
        sum_ranks_pos=sum(rank for rank,y in zip(ranks,labs) if y==1)
        u=sum_ranks_pos - pos*(pos+1)/2
        return u/(pos*neg)
    def pr_auc():
        pairs=sorted(zip(probs_arr, labels_arr), key=lambda x: -x[0])
        tp=0; fp=0; fn=int(P)
        if fn==0: return float('nan')
        prev_recall=0.0; prev_prec=1.0; auc=0.0; last=None
        for s,y in pairs:
            if last is not None and s!=last:
                recall=tp/(tp+fn)
                prec=tp/max(1,tp+fp)
                auc += (recall-prev_recall)*prev_prec
                prev_recall=recall; prev_prec=prec
            if y==1: tp+=1; fn-=1
            else: fp+=1
            last=s
        recall=tp/(tp+fn if (tp+fn)>0 else 1)
        auc += (recall-prev_recall)*prev_prec
        return max(0.0,min(1.0,auc))
    def fpr_at_tpr(target):
        pairs=sorted(zip(probs_arr, labels_arr), key=lambda x: -x[0])
        P=sum(1 for _,y in pairs if y==1); N=len(pairs)-P
        tp=0; fp=0; last=None
        best_fpr=float('nan'); best_thr=float('nan')
        for s,y in pairs + [(-float('inf'), None)]:
            if last is not None and s!=last:
                tpr=tp/max(1,P); fpr=fp/max(1,N)
                if tpr>=target:
                    best_fpr=fpr; best_thr=last; break
            if y==1: tp+=1
            else: fp+=1
            last=s
        return best_fpr, best_thr
    roc=roc_auc(); pr=pr_auc(); fpr90, thr90=fpr_at_tpr(0.90); fpr95, thr95=fpr_at_tpr(0.95)
    return {
        'n': len(labels_arr), 'pos': int(P), 'prevalence': (P/len(labels_arr) if len(labels_arr) else float('nan')),
        'roc_auc': roc, 'pr_auc': pr, 'fpr90': fpr90, 'thr90': thr90, 'fpr95': fpr95, 'thr95': thr95
    }


def _load_full_variant_preds(variant: str) -> dict:
    """Load full variant prediction JSONLs directly from artifact dirs.
    Returns {'val': [recs], 'test': [recs]} or empty if not present.
    """
    dir_map = {
        'fused': 'artifacts/fusion',
        'fused_covmax': 'artifacts/fusion_covmax',
        'fused_soft': 'artifacts/fusion_soft',
        'meta_fused': 'artifacts/fusion_meta'
    }
    base = dir_map.get(variant)
    out={'val':[], 'test':[]}
    if not base: return out
    for split in ['val','test']:
        path=os.path.join(base, f"preds_{split}_full.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path,'r',encoding='utf-8') as f:
            for ln in f:
                ln=ln.strip();
                if not ln: continue
                try: js=json.loads(ln)
                except Exception: continue
                out[split].append(js)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', default='artifacts/baseline_strict')
    ap.add_argument('--out', default='artifacts/diagnostics/fusion_calibration_report.md')
    ap.add_argument('--include-full', action='store_true', help='Also compute calibration for *_full variant prediction files (direct directory read)')
    ap.add_argument('--csv-out', default=None, help='Optional path to also write a CSV of the table')
    ap.add_argument('--fenced-out', default=None, help='Optional path to write a fenced markdown table (```) to reduce wrapping artifacts')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows=[]
    for split in ['val','test']:
        all_records=load_combined(args.baseline_dir, split)
        for v in VARIANTS:
            m=metrics_for_variant(all_records, v)
            if not m: continue
            rows.append({'split': split, 'variant': v, **m})
    # Optional full variants
    if args.include_full:
        for v in VARIANTS:
            full_preds=_load_full_variant_preds(v)
            for split in ['val','test']:
                recs=full_preds.get(split, [])
                if not recs:
                    continue
                # Override model field so that different fusion strategies sharing raw 'fusion' tag in files are separated
                canon_name = MODEL_FOR_VARIANT.get(v, [v])[0]
                recs_tagged=[{**r, 'model': canon_name} for r in recs]
                m=metrics_for_variant(recs_tagged, v)
                if not m:
                    continue
                rows.append({'split': f"{split}_full", 'variant': v, **m})
    # Build markdown
    # Sort rows for stable output: variant then split
    split_order={s:i for i,s in enumerate(["val","test","val_full","test_full"]) }
    rows.sort(key=lambda r: (r['variant'], split_order.get(r['split'], 999)))
    # Compute baseline (fused) PR per split for PR gain calculation
    baseline_pr={}
    for r in rows:
        if r['variant']=='fused' and not math.isnan(r['pr_auc']):
            baseline_pr[r['split']]=r['pr_auc']
    # Compute fpr delta and pr delta, track min fpr delta per split
    min_fpr_delta={}
    enriched=[]
    for r in rows:
        fpr_delta=(r['fpr95']-r['fpr90']) if (not math.isnan(r['fpr90']) and not math.isnan(r['fpr95'])) else float('nan')
        pr_base=baseline_pr.get(r['split'])
        pr_delta=(r['pr_auc']-pr_base) if (pr_base is not None and not math.isnan(r['pr_auc'])) else float('nan')
        rr={**r,'fpr_delta':fpr_delta,'pr_delta':pr_delta}
        enriched.append(rr)
        if not math.isnan(fpr_delta):
            cur=min_fpr_delta.get(r['split'])
            if cur is None or fpr_delta < cur:
                min_fpr_delta[r['split']]=fpr_delta
    rows=enriched
    # Identify threshold recommendations per split (lowest FPR at targets)
    recommendations=[]
    by_split={}
    for r in rows:
        by_split.setdefault(r['split'], []).append(r)
    for split, items in by_split.items():
        # best at 90
        best90=sorted([i for i in items if not math.isnan(i['fpr90'])], key=lambda x:(x['fpr90'], -x['pr_auc']))[:1]
        best95=sorted([i for i in items if not math.isnan(i['fpr95'])], key=lambda x:(x['fpr95'], -x['pr_auc']))[:1]
        if best90:
            recommendations.append({'split': split, 'target':'TPR90', 'variant': best90[0]['variant'], 'thr': best90[0]['thr90'], 'fpr': best90[0]['fpr90']})
        if best95:
            recommendations.append({'split': split, 'target':'TPR95', 'variant': best95[0]['variant'], 'thr': best95[0]['thr95'], 'fpr': best95[0]['fpr95']})
    header = "Variant | Split | N | Pos | Prev | ROC | PR | PRΔ | FPR@TPR90 | Thr90 | FPR@TPR95 | Thr95 | FPRΔ(95-90)"
    sep = "------ | ----- | --- | --- | ---- | --- | -- | --- | ---------- | ----- | ---------- | ----- | -----------"
    lines=["# Fusion Calibration Comparison","", header, sep]
    for r in rows:
        name=r['variant']
        fpr_delta=r['fpr_delta']
        if r['split'] in min_fpr_delta and not math.isnan(fpr_delta) and abs(fpr_delta - min_fpr_delta[r['split']]) < 1e-12:
            name = name + '*'
        lines.append(" | ".join([
            name, r['split'], str(r['n']), str(r['pos']), f"{r['prevalence']:.4f}", f"{r['roc_auc']:.4f}",
            f"{r['pr_auc']:.4f}", f"{r['pr_delta']:.4f}" if not math.isnan(r['pr_delta']) else 'nan',
            f"{r['fpr90']:.4f}", f"{r['thr90']:.4f}", f"{r['fpr95']:.4f}", f"{r['thr95']:.4f}", f"{fpr_delta:.4f}" if not math.isnan(fpr_delta) else 'nan'
        ]))
    # Append recommendations section
    lines.append("")
    lines.append("## Threshold Recommendations")
    lines.append("")
    lines.append("Split | Target | Variant | Thr | FPR")
    lines.append("----- | ------ | ------- | --- | ---")
    for rec in sorted(recommendations, key=lambda x:(x['split'], x['target'])):
        lines.append(f"{rec['split']} | {rec['target']} | {rec['variant']} | {rec['thr']:.4f} | {rec['fpr']:.4f}")
    content='\n'.join(lines).rstrip()+"\n"
    with open(args.out,'w',encoding='utf-8') as f:
        f.write(content)
    if args.csv_out:
        import csv
        with open(args.csv_out,'w',newline='',encoding='utf-8') as cf:
            writer=csv.writer(cf)
            writer.writerow([c.strip() for c in header.split('|')])
            table_only=[l for l in lines[4:] if l and not l.startswith('## Threshold Recommendations')]
            for r_line in table_only:
                if '|' not in r_line: continue
                parts=[p.strip() for p in r_line.split('|')]
                if len(parts) >= 13:  # ensure it's a data row
                    writer.writerow(parts[:13])
    if args.fenced_out:
        # Separate original markdown body (lines) already contains recommendations
        table_block=[header, sep]
        data_rows=[l for l in lines[4:] if not l.startswith('## Threshold Recommendations') and '|' in l and not l.startswith('Split ')]
        fenced_lines=["# Fusion Calibration Comparison (Fenced)","", "```", *table_block, *data_rows, "```", "", *lines[-7:]]
        with open(args.fenced_out,'w',encoding='utf-8') as ff:
            ff.write('\n'.join(fenced_lines))
    print(f"Calibration comparison written to {args.out}" + (f", CSV -> {args.csv_out}" if args.csv_out else "") + (f", fenced -> {args.fenced_out}" if args.fenced_out else ""))

if __name__=='__main__':
    main()
