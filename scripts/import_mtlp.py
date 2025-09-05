#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List


def sniff_has_header(path: str) -> bool:
	try:
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			sample = f.read(4096)
		dialect = csv.Sniffer().sniff(sample)
		return csv.Sniffer().has_header(sample)
	except Exception:
		return True


def read_mtlp_csv(path: str) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	has_header = sniff_has_header(path)
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		if has_header:
			reader = csv.DictReader(f)
			for r in reader:
				rows.append({k.strip(): v for k, v in r.items()})
		else:
			# Fallback: assume canonical order [url, html, whois, screenshot_id, label?]
			reader = csv.reader(f)
			for cols in reader:
				if not cols:
					continue
				d: Dict[str, Any] = {}
				if len(cols) >= 1:
					d["URL"] = cols[0]
				if len(cols) >= 2:
					d["HTML"] = cols[1]
				if len(cols) >= 3:
					d["WHOIS"] = cols[2]
				if len(cols) >= 4:
					d["Screenshot ID"] = cols[3]
				if len(cols) >= 5:
					d["Label"] = cols[4]
				rows.append(d)
	return rows


def normalize_etld1(url: str) -> str:
	try:
		import tldextract
		tx = tldextract.extract(url)
		return ".".join([p for p in [tx.domain, tx.suffix] if p]) or ""
	except Exception:
		return ""


def coerce_int(v: Any, default: int = 0) -> int:
	try:
		return int(v)
	except Exception:
		return default


def coerce_str(v: Any) -> str:
	if v is None:
		return ""
	s = str(v)
	# Some CSVs embed literal "\n"; unescape once
	s = s.replace("\\n", "\n")
	return s


def to_jsonl(rows: List[Dict[str, Any]], out_path: str, source: str = "mtlp") -> None:
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	now = time.time()
	with open(out_path, "w", encoding="utf-8") as f:
		for i, r in enumerate(rows):
			url = coerce_str(r.get("URL") or r.get("url"))
			if not url:
				continue
			html = coerce_str(r.get("HTML") or r.get("html"))
			whois = coerce_str(r.get("WHOIS") or r.get("whois"))
			sid = coerce_str(r.get("Screenshot ID") or r.get("screenshot_id") or r.get("screenshot"))
			# Label: try to infer (dataset may not include); default to 0/1 via heuristics or skip if unknown
			label = r.get("Label") or r.get("label")
			lab = None
			if label is not None and str(label).strip() != "":
				try:
					lab = int(label)
				except Exception:
					s = str(label).lower()
					if s in ("phish", "phishing", "1", "true", "p"):
						lab = 1
					elif s in ("benign", "legit", "0", "false", "a"):
						lab = 0
			# If unknown, try to infer from screenshot prefix p_/a_
			if lab is None:
				sname = sid.strip().lower()
				if sname.startswith("p_"):
					lab = 1
				elif sname.startswith("a_"):
					lab = 0
			if lab is None:
				# As a last resort, skip unlabeled
				continue
			rec: Dict[str, Any] = {
				"id": f"mtlp_{i}",
				"url": url,
				"etld1": normalize_etld1(url),
				"timestamp": now,  # dataset may lack exact crawl time
				"source": source,
				"label": int(lab),
				"html": html,
				"scripts": [],  # Not available; could be extracted from HTML later if needed
				"headers": {"whois": whois, "screenshot_id": sid},
			}
			f.write(json.dumps(rec, ensure_ascii=False))
			f.write("\n")


def main():
	ap = argparse.ArgumentParser(description="Import MTLP CSV into JSONL compatible with PhisDOM schema")
	ap.add_argument("--csv", required=True, help="Path to MTLP_Dataset.csv")
	ap.add_argument("--out", required=True, help="Output JSONL path, e.g., data/external/mtlp.jsonl")
	args = ap.parse_args()

	rows = read_mtlp_csv(args.csv)
	to_jsonl(rows, args.out)
	print(f"[INFO] Wrote MTLP JSONL: {args.out} ({len(rows)} rows read; unlabeled rows may be dropped)")


if __name__ == "__main__":
	main()
