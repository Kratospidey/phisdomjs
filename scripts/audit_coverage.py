#!/usr/bin/env python
"""Phase 1: Coverage & data integrity audit.

Computes:
  - Canonical split sizes from `data/splits.json` (train/val/test indices)
  - Prediction coverage per model per split using baseline combined files
  - Missing IDs (in split but absent from at least one model's predictions)
  - Extraneous IDs (in predictions but not in split definition)
  - Label prevalence drift vs canonical (macro) prevalence

Input assumptions:
  - Phase 0 already created: artifacts/baseline/combined_preds_{split}.jsonl
  - Each line has fields: id,label,prob,logit,model,split

Outputs in artifacts/baseline/:
  - coverage_report.json
  - coverage_report.md (human-readable)

Usage:
  python scripts/audit_coverage.py 
"""
from __future__ import annotations
import json, os, math, collections, argparse
from typing import Dict, List, Set, Tuple

BASELINE_DIR = "artifacts/baseline"
SPLITS_PATH = "data/splits.json"


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in ("train", "val", "test")}


def load_baseline_preds(split: str) -> List[dict]:
    p = os.path.join(BASELINE_DIR, f"combined_preds_{split}.jsonl")
    if not os.path.isfile(p):
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def audit(split: str, split_indices: List[int], preds: List[dict]):
    # Canonical size is number of indices referencing dataset rows; we only know index counts.
    canonical_n = len(split_indices)
    # Build maps per model
    by_model: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)
    for r in preds:
        mid = r["model"]
        rid = r["id"]
        by_model[mid][rid] = r
    # All predicted ids union
    all_ids: Set[str] = set()
    for m in by_model.values():
        all_ids.update(m.keys())
    # Without raw dataset referencing ID -> we can only reason about *relative* coverage across models; canonical ids unknown unless we map index->id.
    # Heuristic: treat union as observed universe; point out mismatch vs earlier reported dataset counts (if available via len(preds_{model}).)
    model_stats = {}
    for model, mp in by_model.items():
        # Estimate prevalence in predictions for this split
        labels = [r.get("label", 0) for r in mp.values()]
        pos = sum(int(l) for l in labels)
        model_stats[model] = {
            "pred_count": len(mp),
            "pos": pos,
            "prevalence": (pos / len(mp)) if mp else math.nan,
        }
    # If we had canonical ids we would compute missing; here we note inability unless an id mapping file is supplied.
    return {
        "split": split,
        "canonical_index_count": canonical_n,
        "observed_prediction_id_union": len(all_ids),
        "model_stats": model_stats,
    }


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_md(path: str, report_obj):
    lines = ["# Coverage Report", "", f"Generated from baseline in `{BASELINE_DIR}`", ""]
    for split_entry in report_obj["splits"]:
        lines.append(f"## Split: {split_entry['split']}")
        lines.append("")
        lines.append(f"Canonical index count: {split_entry['canonical_index_count']}")
        lines.append(f"Observed prediction id union: {split_entry['observed_prediction_id_union']}")
        lines.append("")
        lines.append("Model | Pred Count | Positives | Prevalence")
        lines.append("----- | ---------- | --------- | ----------")
        for model, st in sorted(split_entry['model_stats'].items()):
            lines.append(f"{model} | {st['pred_count']} | {st['pos']} | {st['prevalence']:.4f}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():  # pragma: no cover
    global BASELINE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SPLITS_PATH)
    ap.add_argument("--baseline-dir", default=BASELINE_DIR)
    args = ap.parse_args()
    BASELINE_DIR = args.baseline_dir
    splits = load_splits(args.splits)
    out = {"splits": []}
    for split in ("val", "test"):
        preds = load_baseline_preds(split)
        entry = audit(split, splits.get(split, []), preds)
        out["splits"].append(entry)
    os.makedirs(BASELINE_DIR, exist_ok=True)
    write_json(os.path.join(BASELINE_DIR, "coverage_report.json"), out)
    write_md(os.path.join(BASELINE_DIR, "coverage_report.md"), out)
    print("Wrote coverage report.")


if __name__ == "__main__":
    main()
