#!/usr/bin/env python
"""Produce ID-joined original fusion predictions restricted to intersection of all heads (for fair comparison)."""
from __future__ import annotations
import os, json, argparse
HEADS=["url_head","text_head","js_codet5p","js_charcnn","js_charcnn_aug"]  # dom_gcn dropped

def load_set(head, split):
    path=f"artifacts/{head}/preds_{split}.jsonl"
    ids=set()
    if os.path.isfile(path):
        for ln in open(path,'r',encoding='utf-8'):
            if not ln.strip(): continue
            js=json.loads(ln); _id=js.get('id');
            if _id: ids.add(_id)
    return ids

def load_fusion(split):
    path=f"artifacts/fusion/preds_{split}.jsonl"
    data={}
    if os.path.isfile(path):
        for ln in open(path,'r',encoding='utf-8'):
            if not ln.strip(): continue
            js=json.loads(ln); _id=js.get('id')
            if _id: data[_id]=js
    return data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='artifacts/fusion_ids')
    args=ap.parse_args()
    for split in ('val','test'):
        head_sets=[load_set(h, split) for h in HEADS]
        common=set.intersection(*head_sets) if head_sets else set()
        fusion=load_fusion(split)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(f"{args.out_dir}/preds_{split}.jsonl",'w',encoding='utf-8') as f:
            for _id in common:
                if _id in fusion:
                    rec=fusion[_id]
                    rec['model']='fusion_ids'
                    f.write(json.dumps(rec)+'\n')
    print('Wrote ID-joined fusion preds to', args.out_dir)

if __name__=='__main__':
    main()
