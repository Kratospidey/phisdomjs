#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import deque


def needs_backfill(path: Path, scan_lines: int = 200) -> bool:
    """Return True if file appears to need backfilling.
    Heuristic scans up to `scan_lines` JSON rows from the head and tail.
    - If both compact and non-compact rows are seen (mixed), return True.
    - If only compact rows are seen, return False.
    - If only non-compact rows are seen, return True.
    - If nothing can be parsed, return True (be conservative).
    """
    if not path.exists() or path.stat().st_size == 0:
        # Nothing to do (or empty file)
        return False
    # Keys that indicate compact/enriched fields already present
    markers = {
        # legacy compact fields
        "host_hyphens", "has_punycode",
        "form_fp_hash64", "num_pw", "form_css_sig_hash64",
        "js_eval_like", "key_listeners_total",
        "favicon_dhash64", "logo_phash64",
        "title_host_jaccard_q8",
        # Phase 1 additions
        "url_charseq", "js_charseq", "dom_graph", "text_title", "text_visible",
    }
    head_has = head_no = 0
    tail_has = tail_no = 0
    parsed_any = False
    try:
        tail_buf = deque(maxlen=max(1, scan_lines * 4))  # keep more raw lines to ensure enough JSON rows in tail
        with path.open("r", encoding="utf-8") as f:
            # Scan head while building tail buffer
            head_seen = 0
            for raw in f:
                s = raw.strip()
                if not s:
                    tail_buf.append(raw)
                    continue
                # Always feed into tail buffer
                tail_buf.append(raw)
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                parsed_any = True
                if head_seen < scan_lines:
                    head_seen += 1
                    if any(k in obj for k in markers):
                        head_has += 1
                    else:
                        head_no += 1
        # Now scan tail buffer for up to scan_lines JSON rows
        tail_seen = 0
        for raw in reversed(tail_buf):
            if tail_seen >= scan_lines:
                break
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            tail_seen += 1
            parsed_any = True
            if any(k in obj for k in markers):
                tail_has += 1
            else:
                tail_no += 1
    except Exception:
        # If any IO/parsing error at file level, err on processing
        return True

    any_has = (head_has + tail_has) > 0
    any_no = (head_no + tail_no) > 0
    if not parsed_any:
        return True
    if any_has and any_no:
        # Mixed: likely appended new rows without compact fields
        return True
    if any_has and not any_no:
        # All scanned rows look compact -> skip
        return False
    # Only non-compact seen (or insufficient head indicators) -> process
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
    ap.add_argument("--drop-raw", action="store_true", help="Drop large raw fields after enrichment")
    ap.add_argument("--force", action="store_true", help="Force backfill regardless of existing compact fields")
    ap.add_argument("--scan-lines", type=int, default=200, help="How many JSON rows to sample from head/tail when deciding if backfill is needed")
    args = ap.parse_args()

    any_run = False
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            continue
        if args.force or needs_backfill(path, scan_lines=max(1, int(args.scan_lines))):
            reason = "(forced)" if args.force else ""
            print(f"[AUTO-BACKFILL] Updating {path} {reason}...", flush=True)
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
            if args.drop_raw:
                cmd.append("--drop-raw")
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
