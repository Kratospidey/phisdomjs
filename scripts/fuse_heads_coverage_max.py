#!/usr/bin/env python
"""Run fusion with coverage_max strategy and export predictions (tag _covmax)."""
from __future__ import annotations
import argparse, os, json
from subprocess import check_call

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag', default='_covmax')
    ap.add_argument('--out-dir', default='artifacts/fusion_covmax')
    ap.add_argument('--min-heads', type=int, default=2)
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Reuse fuse_heads with coverage_max
    cmd = [
        'python','scripts/fuse_heads.py',
        '--alignment-strategy','coverage_max',
        '--min-heads', str(args.min_heads),
        '--dom-dir','artifacts/markup_run',
        '--js-dir','artifacts/js_codet5p',
        '--url-dir','artifacts/url_head',
        '--text-dir','artifacts/text_head',
        '--val-jsonl','data/pages_val.jsonl',
        '--test-jsonl','data/pages_test.jsonl',
        '--out-dir', args.out_dir,
        '--normalize-dom-model'
    ]
    check_call(cmd)
    # Copy outputs with tag
    for split in ['val','test']:
        src = os.path.join(args.out_dir, f'preds_{split}.jsonl')
        if os.path.isfile(src):
            dst = os.path.join(args.out_dir, f'preds_{split}{args.tag}.jsonl')
            try:
                import shutil; shutil.copyfile(src,dst)
            except Exception: pass
    print('coverage_max fusion complete.')

if __name__=='__main__':
    main()
