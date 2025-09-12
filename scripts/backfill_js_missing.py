#!/usr/bin/env python
"""Backfill js_code predictions for IDs with js_charseq but missing js_code preds.

Steps:
 1. Load strict baseline combined predictions (val/test) to find missing js_code IDs.
 2. From canonical pages_{split}.jsonl load records; filter those with js_charseq and missing js_code pred.
 3. Write temporary subset JSONL(s).
 4. Run eval_js_codet5p.py in a lightweight mode (reusing existing model & calibration) on subset to produce probs.
 5. Merge new preds into artifacts/js_codet5p/preds_{split}.jsonl (append then dedupe by id keeping existing first or replacing? We replace only if id absent).
 6. Re-run export script optionally (user can do after).

Note: We bypass temperature recalibration; we reuse stored temperature if available, else raw probabilities.
"""
from __future__ import annotations
import json, os, argparse, tempfile, sys
from typing import List, Dict, Set

MODEL_DIR = "artifacts/js_codet5p"

def load_combined(baseline_dir: str, split: str) -> List[dict]:
    path = os.path.join(baseline_dir, f"combined_preds_{split}.jsonl")
    if not os.path.isfile(path):
        return []
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip();
            if not ln: continue
            try: rows.append(json.loads(ln))
            except Exception: pass
    return rows

def load_js_code_ids(preds_dir: str, split: str) -> Set[str]:
    path = os.path.join(preds_dir, f"preds_{split}.jsonl")
    ids=set()
    if not os.path.isfile(path):
        return ids
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            try:
                js=json.loads(ln)
            except Exception:
                continue
            rid=js.get('id')
            if rid is not None:
                ids.add(str(rid))
    return ids

def load_canonical(split: str) -> List[dict]:
    path=f"data/pages_{split}.jsonl"
    if not os.path.isfile(path): return []
    out=[]
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for ln in f:
            ln=ln.strip();
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def find_missing_ids(baseline_dir: str, split: str) -> Set[str]:
    combined = load_combined(baseline_dir, split)
    js_ids = {r['id'] for r in combined if r.get('model')=='js_code'}
    return js_ids

def build_subset(split: str, baseline_dir: str):
    existing_js_ids = load_js_code_ids(MODEL_DIR, split)
    combined = load_combined(baseline_dir, split)
    combined_js_ids = {r['id'] for r in combined if r.get('model')=='js_code'}
    # Use combined_js_ids in case model dir outdated
    js_present = existing_js_ids | combined_js_ids
    records = load_canonical(split)
    subset=[]
    for rec in records:
        rid=str(rec.get('id'))
        if not rid: continue
        if rid in js_present:
            continue
        # require js_charseq presence
        if rec.get('js_charseq'):
            subset.append(rec)
    return subset

def write_jsonl(path: str, rows: List[dict]):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)); f.write('\n')

def merge_preds(split: str, new_path: str):
    orig_path = os.path.join(MODEL_DIR, f"preds_{split}.jsonl")
    existing = {}
    if os.path.isfile(orig_path):
        with open(orig_path,'r',encoding='utf-8') as f:
            for ln in f:
                try:
                    js=json.loads(ln)
                except Exception:
                    continue
                if 'id' in js:
                    existing[str(js['id'])]=js
    added=0
    with open(new_path,'r',encoding='utf-8') as f:
        for ln in f:
            try: js=json.loads(ln)
            except Exception: continue
            rid=str(js.get('id'))
            if rid and rid not in existing:
                existing[rid]=js; added+=1
    with open(orig_path,'w',encoding='utf-8') as f:
        for v in existing.values():
            f.write(json.dumps(v)); f.write('\n')
    print(f"[backfill] Added {added} new js_code preds to {orig_path}")

def run_subset_eval(model_dir: str, subset_path: str, split: str):
    # Minimal inline evaluation (avoid recalibration) replicating core of eval_js_codet5p.predict
    import numpy as np, torch
    from transformers import AutoTokenizer, T5EncoderModel
    from phisdom.data.js import JsonlJsDataset
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5EncoderModel.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # type: ignore[misc]
    ds = JsonlJsDataset(subset_path)
    clf_path = os.path.join(model_dir, 'classifier.pt')
    clf_w=None
    if os.path.exists(clf_path):
        state=torch.load(clf_path, map_location=device)
        w=state.get('weight'); b=state.get('bias')
        if w is not None:
            clf_w=(w.to(device), b.to(device) if b is not None else None)
    probs=[]; bs=8
    with torch.no_grad():
        for i in range(0,len(ds),bs):
            batch=[ds[j] for j in range(i, min(len(ds), i+bs))]
            texts=[r.get('text','') for r in batch]
            enc=tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
            enc={k:v.to(device) for k,v in enc.items()}
            out=model(**enc)
            last_hidden=out.last_hidden_state
            mask=enc['attention_mask'].unsqueeze(-1).type_as(last_hidden)
            pooled=(last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            if clf_w is not None:
                W,B=clf_w; logits=pooled @ W.T; logits = logits + (B if B is not None else 0)
            else:
                logits=pooled @ torch.zeros((pooled.size(1),2), device=device)
            logits=logits.detach().cpu().numpy(); m=logits.max(axis=1, keepdims=True); e=(np.exp(logits-m)); p1=e[:,1]/(e[:,0]+e[:,1]); probs.extend(p1.tolist())
    # Temperature scaling reuse
    temp=1.0
    calib_path=os.path.join(model_dir,'calibration.json')
    if os.path.exists(calib_path):
        try:
            with open(calib_path,'r',encoding='utf-8') as f: cal=json.load(f); temp=float(cal.get('temperature',1.0))
        except Exception: pass
    import math
    def _apply_temp(p):
        eps=1e-6; p=min(1-eps, max(eps,p)); logit=math.log(p/(1-p)); logit/=temp; e=math.exp(logit); return e/(1+e)
    probs=[_apply_temp(p) for p in probs]
    out_path=os.path.join(model_dir, f"preds_{split}_backfill.jsonl")
    with open(out_path,'w',encoding='utf-8') as f:
        for i,p in enumerate(probs):
            r=ds[i]
            o={'id': r.get('id', str(i)), 'label': int(r.get('label',0)), 'prob': float(p)}
            f.write(json.dumps(o)); f.write('\n')
    return out_path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', default='artifacts/baseline_strict')
    ap.add_argument('--splits', nargs='*', default=['val','test'])
    args=ap.parse_args()
    for split in args.splits:
        subset = build_subset(split, args.baseline_dir)
        if not subset:
            print(f"[backfill] No subset needed for {split}")
            continue
        with tempfile.NamedTemporaryFile('w', delete=False, suffix=f'_{split}.jsonl') as tmp:
            path=tmp.name
            for r in subset:
                tmp.write(json.dumps(r)); tmp.write('\n')
        print(f"[backfill] Split {split}: running inference on {len(subset)} records")
        new_preds = run_subset_eval(MODEL_DIR, path, split)
        merge_preds(split, new_preds)
    print("Backfill complete. Re-run export_predictions.py to refresh baseline.")

if __name__=='__main__':
    main()
