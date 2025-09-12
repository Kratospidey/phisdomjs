#!/usr/bin/env python
"""Cascade band optimization scaffold (Phase 6).

Goal: Choose stage-1 head (e.g., URL) threshold band to short-circuit obvious negatives while preserving >= target recall at stage 2 (fusion).
This scaffold:
  - Loads stage1 head preds and fusion calibrated preds (ID-joined).
  - Sweeps candidate stage1 thresholds, measuring: retained recall, precision impact, % traffic passed to stage2.
  - Writes a table to markdown for manual selection (future: automated optimization with cost weights).

Future extensions:
  - Add latency / cost modeling per head.
  - Multi-band cascade (low threshold accept, mid route to second model, high threshold auto-block).
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

URL_HEAD='artifacts/url_head/preds_val.jsonl'
FUSION='artifacts/fusion_calibrated_ids/preds_val.jsonl'
OUT_MD='artifacts/diagnostics/cascade_band_eval.md'


def load(path):
    data=[]
    if not os.path.isfile(path): return data
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: data.append(json.loads(ln))
            except Exception: pass
    return data


def index_by_id(recs):
    return {r.get('id'): r for r in recs if r.get('id')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url-head', default=URL_HEAD)
    ap.add_argument('--fusion', default=FUSION)
    ap.add_argument('--out-md', default=OUT_MD)
    ap.add_argument('--grid', type=int, default=50, help='Number of threshold points (uniform in observed prob range)')
    ap.add_argument('--target-recall', type=float, default=0.95)
    args=ap.parse_args()
    url=load(args.url_head); fusion=load(args.fusion)
    url_map=index_by_id(url); fusion_map=index_by_id(fusion)
    common=list(set(url_map.keys()) & set(fusion_map.keys()))
    if not common:
        raise SystemExit('No overlapping IDs for cascade evaluation.')
    url_probs=np.array([url_map[i]['prob'] for i in common])
    labels=np.array([fusion_map[i]['label'] for i in common])
    fusion_probs=np.array([fusion_map[i]['prob'] for i in common])
    # Sort thresholds ascending for negative filtering: below threshold => auto-negative
    lo, hi = float(url_probs.min()), float(url_probs.max())
    thr_grid=np.linspace(lo, hi, args.grid)
    P=labels.sum(); neg=len(labels)-P
    rows=[]
    for thr in thr_grid:
        mask_pass = url_probs >= thr  # send to fusion
        # Stage1 auto-negatives are url_probs < thr
        # Compute recall retained after Stage1 filtering using fusion later (approx assume fusion recall same on passed subset)
        passed_labels=labels[mask_pass]
        pass_rate = mask_pass.mean()
        # Approx retained recall = positives passed / total positives
        retained_recall = passed_labels.sum()/P if P>0 else 0
        # Approx fusion precision on passed subset (use raw fusion probs threshold at 0.5 fallback) — placeholder
        fusion_sub=fusion_probs[mask_pass]; fusion_lab=labels[mask_pass]
        # crude op: choose threshold that yields recall >= target on subset
        order=np.argsort(-fusion_sub); tps=0; fps=0; pos_sub=fusion_lab.sum(); best_prec=0
        for score,lab in zip(fusion_sub[order], fusion_lab[order]):
            if lab==1: tps+=1
            else: fps+=1
            rec_sub=tps/max(1,pos_sub)
            if rec_sub >= args.target_recall:
                prec=tps/max(1,tps+fps)
                best_prec=prec
                break
        rows.append({'thr': thr,'pass_rate': pass_rate,'retained_recall': retained_recall,'approx_precision_at_target': best_prec})
    # Pick Pareto (retained_recall>=target_recall) minimizing pass_rate
    feasible=[r for r in rows if r['retained_recall']>=args.target_recall]
    best=min(feasible, key=lambda r:r['pass_rate']) if feasible else None
    lines=["# Cascade Band Evaluation","",f"URL head prob range: [{lo:.6f},{hi:.6f}]",f"Target Recall Stage1+2: {args.target_recall}","","Thr | Pass Rate | Retained Recall | Approx Fusion Precision@R>=target","---- | --------- | --------------- | -----------------------------"]
    for r in rows:
        lines.append(" | ".join([f"{r['thr']:.6f}", f"{r['pass_rate']:.4f}", f"{r['retained_recall']:.4f}", f"{r['approx_precision_at_target']:.4f}"]))
    if best:
        lines.append("\nBest (min pass_rate with retained_recall>=target):" )
        lines.append(f"Threshold {best['thr']:.6f} pass_rate={best['pass_rate']:.4f} retained_recall={best['retained_recall']:.4f} approx_precision={best['approx_precision_at_target']:.4f}")
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    print('Wrote', args.out_md)

if __name__=='__main__':
    main()
