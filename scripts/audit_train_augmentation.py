#!/usr/bin/env python
"""Audit whether pages_train_aug.jsonl provides novel examples vs pages_train.jsonl.

Outputs a JSON + Markdown report with:
  - line counts
  - count of shared IDs (if 'id' field present) else hash-based identical lines
  - proportion identical
  - sample differing records (up to 20)
  - recommendation (drop / regenerate / keep)
"""
from __future__ import annotations
import json, os, hashlib, argparse, random
from typing import List, Dict, Any

random.seed(0)

def read_jsonl(path: str) -> List[Any]:
    out=[]
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

def sha(line: str) -> str:
    return hashlib.sha256(line.encode('utf-8','ignore')).hexdigest()

def read_raw(path: str) -> List[str]:
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', default='data/pages_train.jsonl')
    ap.add_argument('--aug', default='data/pages_train_aug.jsonl')
    ap.add_argument('--out-dir', default='artifacts/diagnostics')
    args=ap.parse_args()
    if not (os.path.isfile(args.base) and os.path.isfile(args.aug)):
        raise SystemExit('Base or augmented file missing')
    os.makedirs(args.out_dir, exist_ok=True)
    raw_base = read_raw(args.base)
    raw_aug = read_raw(args.aug)
    hashes_base = {sha(x): x for x in raw_base}
    hashes_aug = {sha(x): x for x in raw_aug}
    shared = set(hashes_base.keys()) & set(hashes_aug.keys())
    # Try ID-based diff
    js_base = read_jsonl(args.base)
    js_aug = read_jsonl(args.aug)
    ids_base = {str(o.get('id')) for o in js_base if 'id' in o}
    ids_aug = {str(o.get('id')) for o in js_aug if 'id' in o}
    id_overlap = ids_base & ids_aug if ids_base and ids_aug else set()
    identical_ratio = len(shared)/max(1,len(hashes_aug))
    novelty_ratio = 1 - identical_ratio
    recommendation = 'drop_as_redundant' if identical_ratio > 0.98 else ('regenerate_more_diverse' if identical_ratio>0.9 else 'keep')
    differing_hashes = list(set(hashes_aug.keys()) - set(hashes_base.keys()))
    random.shuffle(differing_hashes)
    sample_new = [hashes_aug[h] for h in differing_hashes[:20]]
    report = {
        'base_lines': len(raw_base),
        'aug_lines': len(raw_aug),
        'identical_lines': len(shared),
        'identical_ratio': identical_ratio,
        'novelty_ratio': novelty_ratio,
        'id_overlap': len(id_overlap),
        'id_base': len(ids_base),
        'id_aug': len(ids_aug),
        'sample_new_records': sample_new,
        'recommendation': recommendation,
    }
    with open(os.path.join(args.out_dir,'train_augmentation_audit.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    # Markdown
    md_lines=["# Train Augmentation Audit","",f"Base lines: {len(raw_base)}",f"Aug lines: {len(raw_aug)}",f"Identical lines: {len(shared)} ({identical_ratio:.3%})",f"Novelty ratio: {novelty_ratio:.3%}",f"ID overlap: {len(id_overlap)} / base {len(ids_base)} / aug {len(ids_aug)}","",f"Recommendation: {recommendation}"]
    if sample_new:
        md_lines.append("\n## Sample novel augmented lines (truncated)")
        for s in sample_new:
            md_lines.append(f"- {s[:120]}")
    with open(os.path.join(args.out_dir,'train_augmentation_audit.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print('Augmentation audit complete.')

if __name__=='__main__':
    main()
