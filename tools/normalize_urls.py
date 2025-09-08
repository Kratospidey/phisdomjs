#!/usr/bin/env python
"""
Ensure every row has a usable URL in `url_final` by falling back to `url_raw` or `url`.
Usage:
  PYTHONPATH=src python tools/normalize_urls.py --splits val test train
"""
from __future__ import annotations
import argparse, json, os


def normalize(paths: list[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            continue
        out_rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                u = o.get("url_final") or o.get("url_raw") or o.get("url")
                if u:
                    o["url_final"] = u
                out_rows.append(o)
        with open(path, "w", encoding="utf-8") as f:
            for o in out_rows:
                f.write(json.dumps(o))
                f.write("\n")
        print(f"[normalize_urls] updated {path} rows={len(out_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    args = ap.parse_args()
    paths = [os.path.join(args.data_dir, f"pages_{s}.jsonl") for s in args.splits]
    normalize(paths)


if __name__ == "__main__":
    main()
