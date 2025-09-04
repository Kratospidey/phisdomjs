#!/usr/bin/env python
"""
Generate data/feeds/tranco.csv using the official tranco package.
Supports optional args to pick date/list-id, include subdomains, and desired count.
Falls back to an empty CSV with header on failure or missing library.
"""
import csv
import os
import sys
import argparse


def write_empty(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Tranco CSV via tranco library")
    parser.add_argument("--out", default=os.path.join("data", "feeds", "tranco.csv"))
    parser.add_argument("--date", default=None, help="YYYY-MM-DD daily list to retrieve (default: latest)")
    parser.add_argument("--list-id", default=None, help="Specific Tranco list ID (mutually exclusive with --date)")
    parser.add_argument("--subdomains", action="store_true", help="Include subdomains (only for daily list)")
    parser.add_argument("--count", type=int, default=100000, help="How many domains to write (default: 100000)")
    args = parser.parse_args()

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        from tranco import Tranco  # type: ignore
    except Exception as e:
        print(f"WARN: tranco not available: {e}")
        write_empty(out)
        print(f"Wrote empty fallback: {out}")
        return 0

    try:
        t = Tranco(cache=True, cache_dir=".tranco")
        if args.list_id and args.date:
            raise ValueError("Provide either --list-id or --date, not both")
        if args.list_id:
            lst = t.list(list_id=args.list_id)
        else:
            lst = t.list(date=args.date, subdomains=args.subdomains)
        n = max(1, args.count)
        top = lst.top(n)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url"])
            for d in top:
                w.writerow([f"http://{d}/"])  # keep simple HTTP to avoid SSL issues when crawling
        print(f"Wrote {len(top)} rows to {out}")
        return 0
    except Exception as e:
        print(f"WARN: Tranco generation failed: {e}")
        write_empty(out)
        print(f"Wrote empty fallback: {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
