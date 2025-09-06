#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from typing import Dict, List

from phisdom.data.schema import iter_jsonl


def subset_jsonl(in_path: str, out_path: str, indices: List[int]) -> None:
    # Stream over input and write out selected indices without loading all rows
    wanted = set(int(i) for i in indices)
    if not wanted:
        # Create empty file
        open(out_path, "w").close()
        return
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(iter_jsonl(in_path)):
            if i in wanted:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Slice a JSONL dataset into splits using indices JSON")
    parser.add_argument("--dataset", required=True, help="Input JSONL from crawler")
    parser.add_argument("--splits", required=True, help="JSON with keys train/val/test mapping to index arrays")
    parser.add_argument("--out-dir", required=True, help="Output directory for pages_train/val/test.jsonl")
    args = parser.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits: Dict[str, List[int]] = json.load(f)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    subset_jsonl(args.dataset, os.path.join(args.out_dir, "pages_train.jsonl"), splits.get("train", []))
    subset_jsonl(args.dataset, os.path.join(args.out_dir, "pages_val.jsonl"), splits.get("val", []))
    subset_jsonl(args.dataset, os.path.join(args.out_dir, "pages_test.jsonl"), splits.get("test", []))


if __name__ == "__main__":
    main()
