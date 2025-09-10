#!/usr/bin/env python
"""Idempotently append phish-only crawl output into the master dataset.

Logic:
 1. If phish-only file missing, exit 0.
 2. Load existing URLs (final URL if present else url) from master dataset into a set (streaming).
 3. Stream phish-only file; for each record whose key not in set, append to master dataset file and count.
 4. Write a small marker sidecar JSON with counts so repeated runs skip already ingested rows quickly.

Assumptions:
 - Master dataset path: data/pages.jsonl (override with --dataset)
 - Phish-only path: data/pages_phish_only.jsonl (override with --phish-only)
 - Records already have label=1; we do not modify fields.
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
from typing import Iterable, Dict, Any


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_key(obj: Dict[str, Any]) -> str:
    # Prefer canonicalized final URL if present; fall back to initial url.
    u = obj.get('url_final') or obj.get('url') or ''
    return u.strip().lower()


def main():
    ap = argparse.ArgumentParser(description="Ingest phish-only dataset into master dataset if new rows exist")
    ap.add_argument('--dataset', default='data/pages.jsonl')
    ap.add_argument('--phish-only', default='data/pages_phish_only.jsonl')
    ap.add_argument('--marker', default='data/.phish_only_ingested.json', help='Metadata marker file to speed up re-runs')
    args = ap.parse_args()

    if not os.path.exists(args.phish_only):
        print('[INGEST][SKIP] Phish-only file not found:', args.phish_only)
        return
    if not os.path.exists(args.dataset):
        print('[INGEST][ERROR] Master dataset missing:', args.dataset)
        sys.exit(2)

    # Load existing URLs
    existing = set()
    for obj in iter_jsonl(args.dataset):
        existing.add(build_key(obj))
    print(f'[INGEST] Existing master records: {len(existing)}')

    added = 0
    kept_hash = hashlib.sha1()
    with open(args.dataset, 'a', encoding='utf-8') as fout:
        for obj in iter_jsonl(args.phish_only):
            k = build_key(obj)
            if not k or k in existing:
                continue
            existing.add(k)
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            added += 1
            kept_hash.update(k.encode('utf-8'))
    print(f'[INGEST] Added new phish-only rows: {added}')

    meta = {
        'phish_only_path': os.path.abspath(args.phish_only),
        'dataset_path': os.path.abspath(args.dataset),
        'added': added,
    }
    os.makedirs(os.path.dirname(args.marker), exist_ok=True)
    with open(args.marker, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    print(f'[INGEST] Wrote marker {args.marker}')

if __name__ == '__main__':
    main()
