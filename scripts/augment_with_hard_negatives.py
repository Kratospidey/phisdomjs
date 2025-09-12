#!/usr/bin/env python
"""Augment training set with harvested hard negative seeds.

Reads hard negative seed IDs (benign false positives) and merges their full
records into a copy of the base training JSONL set, tagging them so downstream
training can weight or track them.

Design:
 1. Load seeds from artifacts/diagnostics/hard_negative_seeds.jsonl
    (each line: {id, cluster, label=0, reason})
 2. Search source data files (train, val, test, full variants) for matching IDs
    keeping FIRST occurrence (assumed identical across splits).
 3. Write augmented training file:
        data/pages_train_augmented.jsonl  (original training + new negatives)
    and manifest summary JSON + markdown in artifacts/diagnostics.

Safeguards:
 - Skip seeds already present in training (avoid duplication)
 - Enforce label=0 for inserted records
 - If record not found, record as missing (reported in manifest)

Assumptions:
 - Base training file: data/pages_train.jsonl (override with --train-file)
 - Records are JSON objects with at minimum: id, label, url (others optional)
 - IDs are unique per file.

Exit code 0 even if some seeds missing (reported) so pipeline can proceed; use
--strict to fail if any missing.
"""
from __future__ import annotations
import os, json, argparse, sys
from typing import Dict

SEEDS_FILE_DEFAULT = 'artifacts/diagnostics/hard_negative_seeds.jsonl'
SEARCH_FILES_DEFAULT = [
    'data/pages_train_full.jsonl',
    'data/pages_train.jsonl',
    'data/pages_val_full.jsonl',
    'data/pages_val.jsonl',
    'data/pages_test_full.jsonl',
    'data/pages_test.jsonl'
]

def load_jsonl(path: str):
    if not os.path.isfile(path):
        return []
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

def index_records(paths):
    idx: Dict[str, dict] = {}
    provenance: Dict[str, str] = {}
    for p in paths:
        if not os.path.isfile(p):
            continue
        for rec in load_jsonl(p):
            _id = rec.get('id')
            if _id and _id not in idx:
                idx[_id] = rec
                provenance[_id] = p
    return idx, provenance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds-file', default=SEEDS_FILE_DEFAULT)
    ap.add_argument('--train-file', default='data/pages_train.jsonl')
    ap.add_argument('--search-files', nargs='*', default=SEARCH_FILES_DEFAULT)
    ap.add_argument('--out-train', default='data/pages_train_augmented.jsonl')
    ap.add_argument('--manifest-json', default='artifacts/diagnostics/hard_negative_augmentation.json')
    ap.add_argument('--manifest-md', default='artifacts/diagnostics/hard_negative_augmentation.md')
    ap.add_argument('--strict', action='store_true', help='Fail if any seed record missing in source files')
    args = ap.parse_args()

    if not os.path.isfile(args.seeds_file):
        print('Missing seeds file:', args.seeds_file, file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.manifest_json), exist_ok=True)

    # Load seeds
    seeds = []
    with open(args.seeds_file,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                js = json.loads(ln)
            except Exception:
                continue
            if js.get('id'):
                seeds.append(js)

    # Index available records across search space
    idx, provenance = index_records(args.search_files)

    # Load base training
    base_train = load_jsonl(args.train_file)
    train_ids = {r.get('id') for r in base_train if r.get('id')}

    augmented = list(base_train)
    added = []
    missing = []
    skipped = []
    for seed in seeds:
        _id = seed['id']
        if _id in train_ids:
            skipped.append(_id)
            continue
        src = idx.get(_id)
        if not src:
            missing.append(_id)
            continue
        rec = dict(src)  # shallow copy
        rec['label'] = 0  # enforce benign
        rec['hard_negative'] = True
        rec['hard_negative_cluster'] = seed.get('cluster')
        rec['hard_negative_reason'] = seed.get('reason', 'benign_cluster_pattern')
        rec['hard_negative_source_split'] = provenance.get(_id)
        augmented.append(rec)
        added.append(_id)

    # Write augmented training file
    with open(args.out_train,'w',encoding='utf-8') as f:
        for rec in augmented:
            f.write(json.dumps(rec)+'\n')

    manifest = {
        'base_train_file': args.train_file,
        'augmented_train_file': args.out_train,
        'seeds_file': args.seeds_file,
        'searched_files': args.search_files,
        'counts': {
            'base': len(base_train),
            'seeds_total': len(seeds),
            'added': len(added),
            'missing': len(missing),
            'skipped_existing': len(skipped),
            'final_total': len(augmented)
        },
        'added_ids_sample': added[:20],
        'missing_ids_sample': missing[:20]
    }
    with open(args.manifest_json,'w',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)

    # Markdown summary
    md_lines = [
        '# Hard Negative Augmentation',
        '',
        f'Seeds file: {args.seeds_file}',
        f'Base train file: {args.train_file}',
        f'Augmented output: {args.out_train}',
        '',
        'Metric | Count',
        '------ | -----',
        f'Base examples | {len(base_train)}',
        f'Seeds (total) | {len(seeds)}',
        f'Added | {len(added)}',
        f'Skipped (already present) | {len(skipped)}',
        f'Missing (not found) | {len(missing)}',
        f'Final total | {len(augmented)}',
        '',
    ]
    if added:
        md_lines.append('## Added Sample IDs')
        md_lines.append(', '.join(added[:50]))
        md_lines.append('')
    if missing:
        md_lines.append('## Missing Seed IDs (sample)')
        md_lines.append(', '.join(missing[:50]))
        md_lines.append('')
    with open(args.manifest_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(md_lines)+'\n')

    print(f"Augmented training: +{len(added)} seeds (missing {len(missing)}). Output -> {args.out_train}")
    if args.strict and missing:
        print('Strict mode: missing seeds present, failing.', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
