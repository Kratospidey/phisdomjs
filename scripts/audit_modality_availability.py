#!/usr/bin/env python
"""Modality availability matrix for fused/meta_fused.

Reads baseline combined predictions (strict or otherwise) plus per-model files
(if needed) and reports, for canonical test & val IDs:
  - Which active models produced predictions.
  - Missing IDs for fused/meta_fused and which underlying single-head models
    are present for each missing ID (to diagnose gating causes: e.g., missing DOM).

Active models inferred from allowlist argument or from snapshot.json.

Output:
  artifacts/baseline/modality_availability.json
  artifacts/baseline/modality_availability.md

Usage:
  python scripts/audit_modality_availability.py \
    --baseline-dir artifacts/baseline \
    --splits val test \
    --models dom js_code fused meta_fused url text js_charcnn_base js_charcnn_aug
"""
from __future__ import annotations
import json, os, argparse, collections
from typing import Dict, List, Set

DEF_MODELS = ["dom","js_code","fused","meta_fused","url","text","js_charcnn_base","js_charcnn_aug"]

def load_combined(baseline_dir: str, split: str) -> List[dict]:
    path = os.path.join(baseline_dir, f"combined_preds_{split}.jsonl")
    if not os.path.isfile(path):
        return []
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip();
            if not ln: continue
            try: rows.append(json.loads(ln))
            except Exception: continue
    return rows

def load_canonical_ids(split: str) -> List[str]:
    path = f"data/pages_{split}.jsonl"
    if not os.path.isfile(path): return []
    ids=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            try:
                js=json.loads(ln)
            except Exception:
                continue
            rid=js.get("id")
            if rid: ids.append(str(rid))
    return ids

def build_matrix(rows: List[dict], models: List[str]):
    by_model: Dict[str, Set[str]] = {m:set() for m in models}
    for r in rows:
        m=r.get("model"); rid=r.get("id")
        if m in by_model and rid:
            by_model[m].add(rid)
    return by_model

def analyze_split(split: str, baseline_dir: str, models: List[str]):
    combined = load_combined(baseline_dir, split)
    canonical_ids = load_canonical_ids(split)
    canon_set = set(canonical_ids)
    by_model = build_matrix(combined, models)
    fused_missing = sorted(list(canon_set - by_model.get("fused", set())))
    meta_missing = sorted(list(canon_set - by_model.get("meta_fused", set())))
    def presence_map(missing_ids: List[str]):
        out=[]
        for rid in missing_ids:
            present=[m for m in models if rid in by_model[m]]
            out.append({"id": rid, "present_models": present})
        return out
    fused_diag = presence_map(fused_missing)
    meta_diag = presence_map(meta_missing)
    # Aggregate cause counts for fused (expect requiring DOM + js_code at least)
    def cause_counts(diag):
        counts=collections.Counter()
        for entry in diag:
            present=set(entry["present_models"])
            have_dom = "dom" in present
            have_js = "js_code" in present
            key = (
                ("dom" if have_dom else "_") + "+" + ("js" if have_js else "_")
            )
            counts[key]+=1
        return counts
    fused_cause = cause_counts(fused_diag)
    meta_cause = cause_counts(meta_diag)
    return {
        "split": split,
        "canonical_size": len(canon_set),
        "coverage": {m: len(by_model[m]) for m in models},
        "fused_missing": len(fused_missing),
        "meta_missing": len(meta_missing),
        "fused_missing_diag": fused_diag[:50],  # truncate
        "meta_missing_diag": meta_diag[:50],
        "fused_cause_counts": fused_cause,
        "meta_cause_counts": meta_cause,
    }

def write_md(path: str, reports: List[dict]):
    lines=["# Modality Availability","",]
    for rep in reports:
        lines.append(f"## Split: {rep['split']}")
        lines.append("")
        lines.append(f"Canonical size: {rep['canonical_size']}")
        lines.append("Model Coverage (count)")
        for m, cnt in rep['coverage'].items():
            lines.append(f"- {m}: {cnt}")
        lines.append(f"Fused missing: {rep['fused_missing']}")
        lines.append(f"Meta fused missing: {rep['meta_missing']}")
        lines.append("Fused cause counts (dom+js matrix):")
        for k,v in rep['fused_cause_counts'].items():
            lines.append(f"  - {k}: {v}")
        lines.append("Meta fused cause counts (dom+js matrix):")
        for k,v in rep['meta_cause_counts'].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():  # pragma: no cover
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-dir",default="artifacts/baseline")
    ap.add_argument("--splits", nargs="*", default=["val","test"])
    ap.add_argument("--models", nargs="*", default=DEF_MODELS)
    args=ap.parse_args()
    reports=[]
    for split in args.splits:
        reports.append(analyze_split(split, args.baseline_dir, args.models))
    os.makedirs(args.baseline_dir, exist_ok=True)
    with open(os.path.join(args.baseline_dir,"modality_availability.json"),"w",encoding="utf-8") as f:
        json.dump({"reports":reports}, f, indent=2)
    write_md(os.path.join(args.baseline_dir,"modality_availability.md"), reports)
    print("Modality availability audit complete.")

if __name__ == "__main__":
    main()
