#!/usr/bin/env python
"""
Generate data/feeds/tranco.csv using the tranco package.
If unavailable or an error occurs, write an empty CSV with just a header.
"""
import csv
import os
import sys


def write_empty(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url"])


def main() -> int:
    out = os.path.join("data", "feeds", "tranco.csv")
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
        lst = t.list()
        top = lst.top(100000)  # adjust as needed
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
