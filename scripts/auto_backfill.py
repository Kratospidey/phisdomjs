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
                    "host_hyphens", "has_punycode",
                    "form_fp_hash64", "num_pw", "form_css_sig_hash64",
                    "js_eval_like", "key_listeners_total",
                    "favicon_dhash64", "logo_phash64",
                    "title_host_jaccard_q8",
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
