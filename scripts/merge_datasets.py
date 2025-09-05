#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List, Set

from phisdom.data.schema import load_jsonl


def merge(inputs: List[str]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	seen_ids: Set[str] = set()
	seen_urls: Set[str] = set()
	for p in inputs:
		rows = load_jsonl(p)
		for r in rows:
			rid = str(r.get("id", ""))
			url = str(r.get("url", ""))
			# Deduplicate favoring first occurrence (internal pages first)
			if rid and rid in seen_ids:
				continue
			if url and url in seen_urls:
				continue
			out.append(r)
			if rid:
				seen_ids.add(rid)
			if url:
				seen_urls.add(url)
	return out


def main():
	ap = argparse.ArgumentParser(description="Merge multiple JSONL datasets, de-duplicating by id/url")
	ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files (order matters; earlier has priority)")
	ap.add_argument("--out", required=True, help="Output JSONL path")
	args = ap.parse_args()

	rows = merge(args.inputs)
	with open(args.out, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False))
			f.write("\n")
	print(f"[INFO] Merged {len(args.inputs)} datasets -> {len(rows)} rows: {args.out}")


if __name__ == "__main__":
	main()
