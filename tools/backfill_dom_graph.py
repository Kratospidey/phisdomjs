#!/usr/bin/env python
"""
Backfill a minimal DOM graph from available HTML if missing, to improve DOM-GCN coverage.
Uses BeautifulSoup to parse the HTML and builds a simple parent-child tree.

Usage:
  PYTHONPATH=src python tools/backfill_dom_graph.py --splits val test train
"""
from __future__ import annotations
import argparse, json, os
from bs4 import BeautifulSoup


def make_graph_from_html(html: str):
    soup = BeautifulSoup(html or "", "html.parser")
    nodes = []
    edges = []

    def add_node(tag: str, depth: int) -> int:
        nid = len(nodes)
        # map to minimal features expected by DomGraphCollator (t_hash, c_hash, depth, xbin)
        nodes.append({"t_hash": hash(tag) % (1 << 16), "c_hash": 0, "depth": depth, "xbin": 0})
        return nid

    def walk(el, depth: int, parent: int | None):
        if getattr(el, "name", None) is None:
            return
        nid = add_node(str(el.name), depth)
        if parent is not None:
            edges.append([parent, nid])
        for c in getattr(el, "children", []):
            try:
                walk(c, depth + 1, nid)
            except Exception:
                continue

    root = soup.body or soup
    walk(root, 0, None)
    return {"nodes": nodes, "edges": edges}


essential_html_keys = ("html_norm", "html", "html_raw", "markup", "page_html")


def backfill(paths: list[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            continue
        out_rows = []
        updated = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                g = o.get("dom_graph") if isinstance(o, dict) else None
                ok = bool(isinstance(g, dict) and (g.get("nodes") or []))
                if not ok:
                    html = ""
                    for k in essential_html_keys:
                        v = o.get(k)
                        if isinstance(v, str) and v:
                            html = v; break
                    if html:
                        try:
                            o["dom_graph"] = make_graph_from_html(html)
                            updated += 1
                        except Exception:
                            pass
                out_rows.append(o)
        with open(path, "w", encoding="utf-8") as f:
            for o in out_rows:
                f.write(json.dumps(o))
                f.write("\n")
        print(f"[backfill_dom_graph] updated {updated} rows in {path} (total {len(out_rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    args = ap.parse_args()
    paths = [os.path.join(args.data_dir, f"pages_{s}.jsonl") for s in args.splits]
    backfill(paths)


if __name__ == "__main__":
    main()
