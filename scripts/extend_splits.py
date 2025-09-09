#!/usr/bin/env python
"""Extend existing splits with any newly appended rows in the master dataset.

Goal: You previously generated `data/splits.json` when the dataset had ~10k rows.
After crawling more, `data/pages.jsonl` now has extra rows (indices beyond the
maximum referenced in the stored splits). Re‑running a full re-splitting would
invalidate existing trained model artifacts whose predictions align with the
old split files. Instead we append the new indices as TEST examples only,
leaving TRAIN/VAL untouched for comparability.

This script:
1. Loads existing splits JSON (expects keys train/val/test OR with `cutoff`).
2. Counts lines in the dataset JSONL (streaming; does not load all into RAM).
3. Determines which indices (0..N_total-1) are missing from the union of
   existing splits. If indices are missing only at the tail (expected case),
   they are treated as newly appended rows.
4. Appends those new indices to the TEST split (deduplicated + sorted).
5. Writes a new splits file (default: splits_full.json) preserving the original
   `cutoff` if present and adding metadata with `extended_from` path and counts.
6. Optionally slices out updated pages_{train,val,test}_full.jsonl alongside the
   originals (controlled by --slice-out-dir).

Safe: Original splits + slice files remain unchanged; consumers can opt into
the extended versions gradually.

Usage:
  python scripts/extend_splits.py \
    --dataset data/pages.jsonl \
    --splits data/splits.json \
    --out data/splits_full.json \
    --slice-out-dir data

After running you will have (example):
  data/splits_full.json
  data/pages_train_full.jsonl
  data/pages_val_full.jsonl
  data/pages_test_full.jsonl

You then need to generate predictions for ONLY the new test indices (those
reported under `new_test_indices`) to have metrics over the full dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any

from phisdom.data.schema import iter_jsonl


def iter_line_count(path: str) -> int:
    cnt = 0
    for _ in iter_jsonl(path):
        cnt += 1
    return cnt


def load_splits(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def slice_extended_data(extended_splits: Dict[str, Any], dataset_path: str, slice_out_dir: str) -> None:
    """Helper to generate slice files from existing extended splits."""
    os.makedirs(slice_out_dir, exist_ok=True)
    
    train = extended_splits.get("train", [])
    val = extended_splits.get("val", [])
    test = extended_splits.get("test", [])
    
    train_out = os.path.join(slice_out_dir, "pages_train_full.jsonl")
    val_out = os.path.join(slice_out_dir, "pages_val_full.jsonl")
    test_out = os.path.join(slice_out_dir, "pages_test_full.jsonl")
    
    subset_jsonl(dataset_path, train_out, train)
    subset_jsonl(dataset_path, val_out, val)
    subset_jsonl(dataset_path, test_out, test)
    
    print(f"[EXTEND] Refreshed slice files: {os.path.basename(train_out)}, {os.path.basename(val_out)}, {os.path.basename(test_out)}")


def subset_jsonl(in_path: str, out_path: str, indices: List[int]) -> None:
    wanted = set(int(i) for i in indices)
    with open(out_path, "w", encoding="utf-8") as f_out:
        if not wanted:
            return
        for i, row in enumerate(iter_jsonl(in_path)):
            if i in wanted:
                f_out.write(json.dumps(row, ensure_ascii=False))
                f_out.write("\n")


def main() -> None:  # pragma: no cover - CLI utility
    ap = argparse.ArgumentParser(description="Append newly crawled rows to TEST split without disturbing existing train/val")
    ap.add_argument("--dataset", required=True, help="Path to master pages.jsonl dataset")
    ap.add_argument("--splits", required=True, help="Existing splits JSON (original)")
    ap.add_argument("--out", required=True, help="Path to write extended splits JSON (e.g., data/splits_full.json)")
    ap.add_argument("--slice-out-dir", default=None, help="If set, also write pages_{train,val,test}_full.jsonl here")
    args = ap.parse_args()

    total = iter_line_count(args.dataset)
    print(f"[EXTEND] Dataset lines: {total}")

    # Check if extended splits already exist and are up-to-date
    if os.path.exists(args.out):
        try:
            existing_extended = load_splits(args.out)
            if existing_extended.get("total_examples") == total and existing_extended.get("extended"):
                print(f"[EXTEND] Extended splits already up-to-date for {total} examples. Skipping.")
                # Still do slicing if requested to ensure slice files are current
                if args.slice_out_dir:
                    print(f"[EXTEND] Ensuring slice files are current...")
                    slice_extended_data(existing_extended, args.dataset, args.slice_out_dir)
                return
        except Exception as e:
            print(f"[EXTEND] Could not read existing extended splits ({e}). Regenerating...")

    splits_raw = load_splits(args.splits)
    # Support format with or without cutoff metadata
    train = list(map(int, splits_raw.get("train", [])))
    val = list(map(int, splits_raw.get("val", [])))
    test = list(map(int, splits_raw.get("test", [])))

    covered = set(train) | set(val) | set(test)
    max_old = max(covered) if covered else -1
    print(f"[EXTEND] Max index in existing splits: {max_old}")

    # Identify missing indices (should normally be a contiguous tail)
    missing = [i for i in range(total) if i not in covered]
    if not missing:
        print("[EXTEND] No new indices detected; splits already cover full dataset.")
        out_obj = {
            **splits_raw,
            "extended": False,
            "total_examples": total,
        }
        write_json(args.out, out_obj)
        return

    tail_contiguous = missing == list(range(min(missing), total))
    if not tail_contiguous:
        # Rare: holes within existing index range; still append all to test but warn.
        print(
            f"[EXTEND][WARN] Missing indices are not a simple tail. Count={len(missing)}. They will all be appended to TEST regardless."
        )
    else:
        print(f"[EXTEND] New tail indices detected: {missing[0]}..{total-1} (n={len(missing)})")

    # Append to test
    new_test = sorted(set(test).union(missing))
    extended_obj = {
        k: v for k, v in splits_raw.items() if k not in {"train", "val", "test"}
    }
    extended_obj.update(
        {
            "train": sorted(train),
            "val": sorted(val),
            "test": new_test,
            "extended": True,
            "new_test_indices": missing,
            "original_test_size": len(test),
            "new_test_size": len(new_test),
            "total_examples": total,
            "extended_from": os.path.abspath(args.splits),
        }
    )
    write_json(args.out, extended_obj)
    print(
        f"[EXTEND] Wrote extended splits: train={len(train)} val={len(val)} test(old={len(test)} -> new={len(new_test)}) path={args.out}"
    )

    if args.slice_out_dir:
        os.makedirs(args.slice_out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.out))[0].replace("splits", "pages")
        # We'll produce *_full.jsonl names explicitly for clarity
        train_out = os.path.join(args.slice_out_dir, "pages_train_full.jsonl")
        val_out = os.path.join(args.slice_out_dir, "pages_val_full.jsonl")
        test_out = os.path.join(args.slice_out_dir, "pages_test_full.jsonl")
        print(f"[EXTEND] Writing sliced full split files to {args.slice_out_dir}")
        subset_jsonl(args.dataset, train_out, train)
        subset_jsonl(args.dataset, val_out, val)
        subset_jsonl(args.dataset, test_out, new_test)
        print(
            f"[EXTEND] Done slicing: {os.path.basename(train_out)}, {os.path.basename(val_out)}, {os.path.basename(test_out)}"
        )
        if missing:
            print(
                f"[EXTEND] NOTE: You now need predictions for {len(missing)} new test rows (last index now {total-1})."
            )


if __name__ == "__main__":  # pragma: no cover
    main()
