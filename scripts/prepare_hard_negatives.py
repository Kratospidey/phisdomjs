#!/usr/bin/env python
"""Prepare hard negative seed list from FP clusters.

Reads fp_clusters.json and selects specified clusters (default: 1,3) as benign themes.
Samples up to N examples per cluster and writes:
  artifacts/diagnostics/hard_negative_seeds.jsonl (id + cluster + reason)
Also produces markdown summary.
"""
from __future__ import annotations
import os, json, argparse, random

DEFAULT_CLUSTERS=[1,3]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clusters', type=int, nargs='*', default=DEFAULT_CLUSTERS)
    ap.add_argument('--per-cluster', type=int, default=30)
    ap.add_argument('--fp-json', default='artifacts/diagnostics/fp_clusters.json')
    ap.add_argument('--out-jsonl', default='artifacts/diagnostics/hard_negative_seeds.jsonl')
    ap.add_argument('--out-md', default='artifacts/diagnostics/hard_negative_seeds.md')
    args=ap.parse_args()
    if not os.path.isfile(args.fp_json):
        raise SystemExit('Missing fp_clusters.json. Run cluster_false_positives.py first.')
    data=json.load(open(args.fp_json,'r',encoding='utf-8'))
    clusters={c['cluster']: c for c in data['clusters']}
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    lines=["# Hard Negative Seeds","",f"Source: {args.fp_json}",""]
    total=0
    with open(args.out_jsonl,'w',encoding='utf-8') as outj:
        for cid in args.clusters:
            if cid not in clusters: continue
            c=clusters[cid]
            ids=c['sample_ids'] if len(c['sample_ids'])>=args.per_cluster else c['sample_ids']
            # If cluster has more IDs than provided sample list, we cannot expand; keep existing
            chosen=ids[:args.per_cluster]
            total+=len(chosen)
            lines.append(f"## Cluster {cid}")
            lines.append(f"Size (reported subset): {len(chosen)}  Top Terms: {', '.join(c['top_terms'])}")
            for _id in chosen:
                outj.write(json.dumps({'id': _id, 'cluster': cid, 'label': 0, 'reason': 'benign_cluster_pattern'})+'\n')
            lines.append('')
    lines.append(f"Total seeds: {total}")
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    print('Wrote', args.out_jsonl, 'and markdown summary')

if __name__=='__main__':
    main()
