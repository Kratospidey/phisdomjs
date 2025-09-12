#!/usr/bin/env python
"""Extended Phase 1 Coverage & Integrity Audit.

Implements detailed requirements:
  1. Map canonical IDs from dataset split files (pages_val.jsonl, pages_test.jsonl).
  2. Detect per-model duplicate prediction rows (frequency > 1) and aggregate.
  3. Identify extra (augmented) IDs not in canonical set; simple heuristics.
  4. Regenerate strict prediction files with exactly one row per canonical ID per model (mean aggregation for duplicates).
  5. Compute prevalence drift deltas vs canonical prevalence.
  6. Produce cleaned metrics on strict test set (and val if differences).
  7. Optional enhancements: per-ID frequency histogram, intersection prevalence, warning thresholds (union inflation > 1.2x).

Outputs:
  artifacts/baseline/extended_coverage_report.json
  artifacts/baseline/extended_coverage_report.md
  artifacts/baseline/strict/combined_preds_{split}_strict.jsonl
  artifacts/baseline/strict/metrics_{split}_strict.json

NOTE: Assumes Phase 0 baseline combined prediction files exist.
"""
from __future__ import annotations
import os, json, math, argparse, collections, statistics
from typing import Dict, List, Tuple, Iterable

BASELINE_DIR = "artifacts/baseline"
DATA_DIR = "data"

SPLIT_FILES = {
    "val": "pages_val.jsonl",
    "test": "pages_test.jsonl",
}

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Many dataset lines might not be JSON: attempt parse else skip
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def load_canonical(split: str) -> Tuple[List[str], Dict[str, int]]:
    fname = SPLIT_FILES.get(split)
    if not fname:
        return [], {}
    path = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(path):
        return [], {}
    rows = read_jsonl(path)
    ids = []
    labels = {}
    # Expect each row has at least 'id' and 'label'
    missing_id_ct = 0
    for r in rows:
        rid = r.get("id")
        if rid is None:
            missing_id_ct += 1
            continue
        ids.append(rid)
        if "label" in r:
            try:
                labels[rid] = int(r["label"])
            except Exception:
                pass
    if missing_id_ct:
        print(f"[WARN] {missing_id_ct} rows missing id in canonical {split} file")
    return ids, labels

def load_baseline_preds(split: str) -> List[dict]:
    path = os.path.join(BASELINE_DIR, f"combined_preds_{split}.jsonl")
    if not os.path.isfile(path):
        return []
    return read_jsonl(path)

def aggregate_predictions(preds: List[dict]) -> Dict[str, Dict[str, dict]]:
    by_model: Dict[str, Dict[str, List[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in preds:
        mid = r.get("model")
        rid = r.get("id")
        if mid is None or rid is None:
            continue
        by_model[mid][rid].append(r)
    aggregated: Dict[str, Dict[str, dict]] = {}
    for model, id_map in by_model.items():
        out = {}
        for rid, lst in id_map.items():
            if len(lst) == 1:
                out[rid] = {**lst[0], "duplicate_count": 1}
            else:
                probs = [x.get("prob", 0.0) for x in lst]
                logits = [x.get("logit", math.nan) for x in lst if isinstance(x.get("logit"), (int, float))]
                mean_prob = float(statistics.fmean(probs)) if probs else 0.0
                mean_logit = float(statistics.fmean(logits)) if logits else float("nan")
                base = lst[0]
                merged = {k: base.get(k) for k in ("id","model","split","label")}
                merged.update({"prob": mean_prob, "logit": mean_logit, "duplicate_count": len(lst)})
                out[rid] = merged
        aggregated[model] = out
    return aggregated

def compute_metrics(y_true: List[int], y_prob: List[float]):
    # Minimal dependency metric implementations (ROC AUC, PR AUC, FPR@TPR90/95)
    from math import sqrt
    # Sort by prob desc for PR/AUC calcs
    paired = list(zip(y_prob, y_true))
    if not paired:
        return {"roc_auc": None, "pr_auc": None, "fpr_at_tpr90": None, "thr_tpr90": None, "fpr_at_tpr95": None, "thr_tpr95": None}
    # ROC AUC
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        roc = None
    else:
        ranked = sorted(paired, key=lambda x: x[0])
        rank_sum = 0.0
        for i,(p,l) in enumerate(ranked, start=1):
            if l == 1:
                rank_sum += i
        roc = (rank_sum - pos*(pos+1)/2) / (pos*neg)
    # PR AUC (step-wise)
    paired_desc = sorted(paired, key=lambda x: -x[0])
    tp=0; fp=0
    precisions=[]; recalls=[]
    for p,l in paired_desc:
        if l==1: tp+=1
        else: fp+=1
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / pos if pos>0 else 0.0
        precisions.append(prec); recalls.append(rec)
    pr_auc = 0.0
    prev_rec=0.0
    for pr,rc in zip(precisions,recalls):
        pr_auc += pr*(rc-prev_rec)
        prev_rec = rc
    # FPR@TPR function
    def fpr_at_target(target: float):
        thresh_metrics = sorted({p for p,_ in paired}, reverse=True)
        best_thr=None; best_fpr=None
        for thr in thresh_metrics:
            tp_l=fp_l=tn_l=fn_l=0
            for p,l in paired:
                if p>=thr:
                    if l==1: tp_l+=1
                    else: fp_l+=1
                else:
                    if l==1: fn_l+=1
                    else: tn_l+=1
            tpr = tp_l/(tp_l+fn_l) if (tp_l+fn_l)>0 else 0.0
            if tpr>=target:
                fpr = fp_l/(fp_l+tn_l) if (fp_l+tn_l)>0 else 0.0
                best_thr=thr; best_fpr=fpr
                break
        return best_fpr,best_thr
    fpr90,thr90 = fpr_at_target(0.90)
    fpr95,thr95 = fpr_at_target(0.95)
    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "fpr_at_tpr90": fpr90,
        "thr_tpr90": thr90,
        "fpr_at_tpr95": fpr95,
        "thr_tpr95": thr95,
    }

def build_strict(aggregated: Dict[str, Dict[str, dict]], canonical_ids: List[str], canonical_labels: Dict[str,int]):
    strict_rows = []
    for model, id_map in aggregated.items():
        for rid in canonical_ids:
            if rid not in id_map:
                continue  # missing id; keep gap for coverage calc
            r = id_map[rid]
            # enforce canonical label
            if rid in canonical_labels:
                r = {**r, "label": canonical_labels[rid]}
            strict_rows.append(r)
    return strict_rows

def write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def prevalence(labels: List[int]):
    return sum(labels)/len(labels) if labels else 0.0

def generate_reports(split: str, canonical_ids: List[str], canonical_labels: Dict[str,int], aggregated: Dict[str, Dict[str,dict]]):
    canonical_set = set(canonical_ids)
    canonical_prevalence = prevalence([canonical_labels.get(rid,0) for rid in canonical_ids])
    report_models = {}
    near_complete_models = []
    for model, id_map in aggregated.items():
        ids_pred = set(id_map.keys())
        missing = canonical_set - ids_pred
        extra = ids_pred - canonical_set
        coverage = len(ids_pred & canonical_set) / len(canonical_set) if canonical_set else 0.0
        if coverage >= 0.97:
            near_complete_models.append(model)
        # prevalence using canonical denominator
        pos = sum(1 for rid in canonical_ids if rid in id_map and id_map[rid].get("label",0)==1)
        model_prev = pos / len(canonical_ids) if canonical_ids else 0.0
        drift = (model_prev - canonical_prevalence) / canonical_prevalence if canonical_prevalence>0 else None
        dup_counts = [r.get("duplicate_count",1) for r in id_map.values()]
        freq_hist = collections.Counter(dup_counts)
        report_models[model] = {
            "coverage": coverage,
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_sample": list(sorted(missing))[:5],
            "extra_sample": list(sorted(extra))[:5],
            "prevalence": model_prev,
            "prevalence_drift": drift,
            "duplicate_freq_hist": dict(freq_hist),
        }
    intersection_ids = set(canonical_ids)
    for m in near_complete_models:
        intersection_ids &= set(aggregated[m].keys())
    intersection_prev = prevalence([canonical_labels.get(rid,0) for rid in intersection_ids]) if intersection_ids else None
    warning_union_inflation = None
    union_ids = set().union(*[set(m.keys()) for m in aggregated.values()]) if aggregated else set()
    if len(canonical_set)>0 and len(union_ids) > 1.2 * len(canonical_set):
        warning_union_inflation = {
            "canonical": len(canonical_set),
            "union": len(union_ids),
            "inflation_factor": len(union_ids)/len(canonical_set)
        }
    # Metrics per model on strict (where present)
    metrics_per_model = {}
    for model, id_map in aggregated.items():
        y_true=[]; y_prob=[]
        for rid in canonical_ids:
            r = id_map.get(rid)
            if not r: continue
            y_true.append(int(r.get("label",0)))
            y_prob.append(float(r.get("prob",0.0)))
        metrics_per_model[model] = compute_metrics(y_true,y_prob)
        metrics_per_model[model]["n_strict"] = len(y_true)
    return {
        "split": split,
        "canonical_size": len(canonical_ids),
        "canonical_prevalence": canonical_prevalence,
        "intersection_prevalence_near_complete_models": intersection_prev,
        "near_complete_models": near_complete_models,
        "models": report_models,
        "warning_union_inflation": warning_union_inflation,
        "metrics_strict": metrics_per_model,
    }

def write_markdown(path: str, reports: List[dict]):
    lines=["# Extended Coverage Report","", "Generated extended audit."]
    for rep in reports:
        lines.append(f"## Split: {rep['split']}")
        lines.append("")
        lines.append(f"Canonical size: {rep['canonical_size']}")
        lines.append(f"Canonical prevalence: {rep['canonical_prevalence']:.4f}")
        if rep['intersection_prevalence_near_complete_models'] is not None:
            lines.append(f"Intersection prevalence (near-complete models): {rep['intersection_prevalence_near_complete_models']:.4f}")
        if rep['warning_union_inflation']:
            w = rep['warning_union_inflation']
            lines.append(f"WARNING: Union inflation factor {w['inflation_factor']:.2f} (union={w['union']} vs canonical={w['canonical']})")
        lines.append("")
        lines.append("Model | Coverage | Missing | Extra | Prev | Drift | Dups(max) | Strict N | ROC | PR | FPR@TPR90 | Thr90")
        lines.append("----- | -------- | ------- | ----- | ---- | ----- | --------- | ------- | --- | -- | ---------- | -----")
        for model, stats in sorted(rep['models'].items()):
            m = rep['metrics_strict'][model]
            dups_hist = stats['duplicate_freq_hist']
            max_dup = max(dups_hist) if dups_hist else 1
            lines.append(
                f"{model} | {stats['coverage']:.3f} | {stats['missing_count']} | {stats['extra_count']} | "
                f"{stats['prevalence']:.4f} | {stats['prevalence_drift'] if stats['prevalence_drift'] is not None else 'NA'} | {max_dup} | "
                f"{m['n_strict']} | {m['roc_auc'] if m['roc_auc'] is not None else 'NA'} | {m['pr_auc'] if m['pr_auc'] is not None else 'NA'} | "
                f"{m['fpr_at_tpr90'] if m['fpr_at_tpr90'] is not None else 'NA'} | {m['thr_tpr90'] if m['thr_tpr90'] is not None else 'NA'}"
            )
        lines.append("")
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():  # pragma: no cover
    global BASELINE_DIR, DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=BASELINE_DIR)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model-allowlist", nargs="*", default=None, help="If set, restrict report to these model names (others ignored)")
    args = ap.parse_args()
    BASELINE_DIR = args.baseline_dir
    DATA_DIR = args.data_dir
    os.makedirs(BASELINE_DIR, exist_ok=True)
    strict_dir = os.path.join(BASELINE_DIR, "strict")
    os.makedirs(strict_dir, exist_ok=True)
    reports = []
    for split in ("val","test"):
        canonical_ids, canonical_labels = load_canonical(split)
        preds = load_baseline_preds(split)
        aggregated = aggregate_predictions(preds)
        if args.model_allowlist:
            aggregated = {m:v for m,v in aggregated.items() if m in set(args.model_allowlist)}
        strict_rows = build_strict(aggregated, canonical_ids, canonical_labels)
        strict_path = os.path.join(strict_dir, f"combined_preds_{split}_strict.jsonl")
        write_jsonl(strict_path, strict_rows)
        report = generate_reports(split, canonical_ids, canonical_labels, aggregated)
        reports.append(report)
        # Per-split metrics file (strict)
        metrics_out = {m: report['metrics_strict'][m] for m in report['metrics_strict']}
        with open(os.path.join(strict_dir, f"metrics_{split}_strict.json"),"w",encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
    with open(os.path.join(BASELINE_DIR,"extended_coverage_report.json"),"w",encoding="utf-8") as f:
        json.dump({"splits": reports}, f, indent=2)
    write_markdown(os.path.join(BASELINE_DIR,"extended_coverage_report.md"), reports)
    print("Extended coverage audit complete.")

if __name__ == "__main__":
    main()
