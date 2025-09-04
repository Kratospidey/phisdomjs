#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import os
import random
from typing import Iterable, List, Tuple
import io
import zipfile


def _read_urls_any(path: str) -> List[str]:
    urls: List[str] = []
    # If a zip archive is provided (e.g., Tranco export), try reading the first CSV/TXT within
    if path.lower().endswith(".zip") and zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                # Prefer CSV, else any text-like file
                names = sorted(zf.namelist())
                inner_name = None
                for n in names:
                    ln = n.lower()
                    if ln.endswith(".csv"):
                        inner_name = n
                        break
                if inner_name is None:
                    for n in names:
                        ln = n.lower()
                        if ln.endswith(".txt"):
                            inner_name = n
                            break
                if inner_name is not None:
                    with zf.open(inner_name, 'r') as fh:
                        text = io.TextIOWrapper(fh, encoding='utf-8', errors='ignore')
                        sample = text.read(4096)
                        text.seek(0)
                        # Try CSV first
                        try:
                            sniffer = csv.Sniffer()
                            dialect = sniffer.sniff(sample)
                        except Exception:
                            dialect = None
                        if dialect:
                            reader = csv.DictReader(text, dialect=dialect)
                        else:
                            reader = csv.DictReader(text)
                        if reader.fieldnames:
                            lower_fields = [h.lower() for h in reader.fieldnames]
                            has_url = "url" in lower_fields
                            has_domain = "domain" in lower_fields or "host" in lower_fields
                            if has_url or has_domain:
                                for row in reader:
                                    if has_url:
                                        u = (row.get("url") or row.get("URL") or "").strip()
                                        if u:
                                            urls.append(u)
                                    else:
                                        d = (row.get("domain") or row.get("host") or "").strip()
                                        if d:
                                            urls.append(d)
                                if urls:
                                    return urls
                        # Fall back to reading one URL/domain per line
                        text.seek(0)
                        for line in text:
                            u = line.strip()
                            if not u:
                                continue
                            lower = u.lower()
                            if lower == "url" or lower.startswith("url,"):
                                continue
                            urls.append(u)
                        if urls:
                            return urls
        except Exception:
            # On zip read failure, fall back to regular file handling below
            pass
    # Try CSV with a 'url' column first
    try:
        with open(path, newline="", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            dialect = None
            try:
                dialect = sniffer.sniff(sample)
            except Exception:
                dialect = None
            reader = csv.DictReader(f, dialect=dialect) if dialect else csv.DictReader(f)
            if reader.fieldnames:
                lower_fields = [h.lower() for h in reader.fieldnames]
                has_url = "url" in lower_fields
                has_domain = "domain" in lower_fields or "host" in lower_fields
                if has_url or has_domain:
                    for row in reader:
                        if has_url:
                            u = (row.get("url", "") or row.get("URL", "")).strip()
                            if u:
                                urls.append(u)
                        else:
                            d = (row.get("domain") or row.get("host") or "").strip()
                            if d:
                                urls.append(d)
                    if urls:
                        return urls
    except Exception:
        pass
    # Fall back to one URL per line (OpenPhish text feed style)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if not u:
                    continue
                lower = u.lower()
                # Skip common header lines in txt exports
                if lower == "url" or lower.startswith("url,"):
                    continue
                urls.append(u)
    except Exception:
        pass
    return urls


def _tag(urls: Iterable[str], label: int, source: str) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        # Ensure scheme present for Playwright; if missing, prefix http://
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        out.append((u, label, source))
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge OpenPhish/PhishTank/Tranco into one seed CSV (url,label,source)")
    parser.add_argument("--openphish", help="Path to OpenPhish URLs (txt or csv)")
    parser.add_argument("--phishtank", help="Path to PhishTank (CSV or txt)")
    parser.add_argument("--tranco", help="Path to Tranco URLs (csv/txt)")
    parser.add_argument("--out", required=True, help="Output CSV (url,label,source)")
    # Legacy per-source limits
    parser.add_argument("--limit-phish", type=int, default=None, help="Limit number of phish URLs from each phish source")
    parser.add_argument("--limit-benign", type=int, default=None, help="Limit number of benign URLs")
    # New: single target-size with phishing ratio control
    parser.add_argument("--target-size", type=int, default=None, help="Approximate total rows (excluding header). Overrides per-source limits if set.")
    parser.add_argument("--phish-ratio", type=float, default=0.5, help="Fraction of phishing rows when using --target-size (0..1). Default 0.5")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    parser.add_argument("--seed", type=int, default=None, help="Seed for shuffling; default is random")
    args = parser.parse_args()

    # Load sources (keep as lists for flexible allocation)
    op: List[str] = []
    pt: List[str] = []
    tr: List[str] = []
    if args.openphish and os.path.exists(args.openphish):
        op = _read_urls_any(args.openphish)
    if args.phishtank and os.path.exists(args.phishtank):
        pt = _read_urls_any(args.phishtank)
    if args.tranco and os.path.exists(args.tranco):
        tr = _read_urls_any(args.tranco)

    rows: List[Tuple[str, int, str]] = []

    if args.target_size is not None and args.target_size > 0:
        # Compute desired phishing and benign counts
        pr = min(max(args.phish_ratio, 0.0), 1.0)
        phish_target = int(round(args.target_size * pr))
        benign_target = max(args.target_size - phish_target, 0)

        # Allocate phishing between OpenPhish and PhishTank proportionally to availability
        op_n = len(op)
        pt_n = len(pt)
        total_avail = max(op_n + pt_n, 1)
        op_take = min(int(round(phish_target * (op_n / total_avail))), op_n)
        pt_take = min(phish_target - op_take, pt_n)
        # If rounding left some slack and one source still has items, fill
        if op_take + pt_take < phish_target:
            slack = phish_target - (op_take + pt_take)
            # Prefer the larger remaining source
            op_rem = max(op_n - op_take, 0)
            take_more_op = min(slack, op_rem)
            op_take += take_more_op
            slack -= take_more_op
            if slack > 0:
                pt_take += min(slack, max(pt_n - pt_take, 0))

        rows.extend(_tag(op[:op_take], 1, "openphish"))
        rows.extend(_tag(pt[:pt_take], 1, "phishtank"))
        rows.extend(_tag(tr[:benign_target], 0, "tranco"))
    else:
        # Legacy per-source limiting
        if op:
            if args.limit_phish is not None:
                op = op[: args.limit_phish]
            rows.extend(_tag(op, 1, "openphish"))
        if pt:
            if args.limit_phish is not None:
                pt = pt[: args.limit_phish]
            rows.extend(_tag(pt, 1, "phishtank"))
        if tr:
            if args.limit_benign is not None:
                tr = tr[: args.limit_benign]
            rows.extend(_tag(tr, 0, "tranco"))

    # Deduplicate by URL, preferring first occurrence
    seen = set()
    deduped: List[Tuple[str, int, str]] = []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        deduped.append(r)

    if args.shuffle:
        random.Random(args.seed).shuffle(deduped)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label", "source"])
        for u, y, s in deduped:
            w.writerow([u, y, s])


if __name__ == "__main__":
    main()
