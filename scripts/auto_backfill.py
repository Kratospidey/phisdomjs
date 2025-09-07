#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def needs_backfill(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                keys = obj.keys()
                markers = [
                    # legacy compact fields
                    "host_hyphens", "has_punycode",
                    "form_fp_hash64", "num_pw", "form_css_sig_hash64",
                    "js_eval_like", "key_listeners_total",
                    "favicon_dhash64", "logo_phash64",
                    "title_host_jaccard_q8",
                    # Phase 1 additions
                    "url_charseq", "js_charseq", "dom_graph", "text_title", "text_visible",
                ]
                return not any(k in keys for k in markers)
    except Exception:
        return True
    return True


def main():
    ap = argparse.ArgumentParser(description="Auto-detect and backfill compact fields for JSONL files if needed")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--tls-timeout", type=float, default=3.0)
    ap.add_argument("--dns-timeout", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-lines", type=int, default=2000)
    ap.add_argument("--disable-visuals", action="store_true", help="Skip favicon/logo to reduce memory")
    ap.add_argument("--max-image-bytes", type=int, default=262144)
    args = ap.parse_args()

    any_run = False
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            continue
        if needs_backfill(path):
            print(f"[AUTO-BACKFILL] Updating {path} ...", flush=True)
            cmd = [sys.executable, "scripts/backfill_fields.py", "--inputs", str(path)]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.network:
                cmd.append("--network")
            cmd.extend(["--tls-timeout", str(args.tls_timeout), "--dns-timeout", str(args.dns_timeout)])
            if args.workers and args.workers > 1:
                cmd.extend(["--workers", str(args.workers), "--batch-lines", str(args.batch_lines)])
            # Adaptive: for very large base files, disable visuals by default unless explicitly allowed
            try:
                sz = path.stat().st_size
            except Exception:
                sz = 0
            disable_visuals_flag = args.disable_visuals or (sz > 1_500_000_000)
            if disable_visuals_flag:
                cmd.append("--disable-visuals")
            if args.max_image_bytes:
                cmd.extend(["--max-image-bytes", str(args.max_image_bytes)])
            r = subprocess.run(cmd)
            if r.returncode != 0:
                sys.exit(r.returncode)
            any_run = True
        else:
            print(f"[AUTO-BACKFILL] {path} already has compact fields; skipping.")

    if not any_run:
        print("[AUTO-BACKFILL] No files required backfilling.")


if __name__ == "__main__":
    main()
