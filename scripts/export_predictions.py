#!/usr/bin/env python
"""Phase 0 baseline snapshot exporter.

Collects existing per-head prediction JSONL files under `artifacts/` and
produces a frozen baseline bundle in `artifacts/baseline/`:

Outputs:
  - combined_preds_{split}.jsonl  (schema: id,label,prob,logit,model,split)
  - metrics_{split}.json          (per-model + macro metrics)
  - snapshot.json                 (run metadata)

We do NOT recompute model outputs here—only aggregate what already exists.
Safe to re-run; files are overwritten.

Usage:
    python scripts/export_predictions.py \
        [--artifacts-dir artifacts] \
        [--out-dir artifacts/baseline] \
        [--mode strict|extended|mixed] \
        [--model-allowlist dom js_code ...] \
        [--separate-full] [--guard-inflation 1.05] [--allow-inflation]

Modes:
    strict   -> prefer base preds_{split}.jsonl even if *_full exists (drop extras)
    extended -> prefer *_full variants when present (current default behavior)
    mixed    -> export both (base to combined, full to combined_full) if --separate-full

Guard:
    Computes union of IDs per split vs canonical pages_{split}.jsonl if available.
    Abort if union_factor > guard-inflation unless --allow-inflation.

Note: Some directories (e.g. url_head) omit `model` / `split` fields; we infer
them from directory and filename. Full vs non-full variants: if both
`preds_test_full.jsonl` and `preds_test.jsonl` exist, we prefer the longer
(_full) variant (more coverage) if it has >= the number of lines of the
non-full file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional

try:  # optional speedups
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore

RELEVANT_DIR_HINTS = {
    # directory name -> canonical model label in snapshot
    "markup_run": "dom",
    "js_codet5p": "js_code",
    "fusion": "fused",
    "fusion_covmax": "fused_covmax",
    "fusion_soft": "fused_soft",
    "fusion_meta": "meta_fused",
    "url_head": "url",
    "text_head": "text",
    "js_charcnn": "js_charcnn_base",
    "js_charcnn_aug": "js_charcnn_aug",
    "cheap_mlp": "cheap_mlp",
}

PRED_FILENAMES = [  # order influences selection logic per mode
    ("preds_val_full.jsonl", "val"),
    ("preds_val.jsonl", "val"),
    ("preds_test_full.jsonl", "test"),
    ("preds_test.jsonl", "test"),
]

# NOTE: After introduction of auto head-tag inference and corrected split name
# parsing, coverage_max (_full) and soft imputation full variants emit standard
# preds_{split}_full.jsonl filenames inside their respective directories. No
# special-case mapping required here; discovery logic above will pick them up
# and treat them consistently with other heads.


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().splitlines()


def _maybe_json(line: str) -> Optional[dict]:
    if not line.strip():
        return None
    try:
        if orjson is not None:
            return orjson.loads(line)
        return json.loads(line)
    except Exception:
        return None


def _logit(p: float, eps: float = 1e-12) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


@dataclass
class Record:
    id: str
    label: int
    prob: float
    model: str
    split: str
    logit: float

    def to_json(self) -> str:
        obj = {
            "id": self.id,
            "label": self.label,
            "prob": self.prob,
            "logit": self.logit,
            "model": self.model,
            "split": self.split,
        }
        if orjson is not None:
            return orjson.dumps(obj).decode()
        return json.dumps(obj, separators=(",", ":"))


def discover_prediction_files(artifacts_dir: str, mode: str = "extended", separate_full: bool = False) -> Tuple[List[Tuple[str,str,str]], List[Tuple[str,str,str]]]:
    """Return (primary_files, full_only_files).

    primary_files -> (model, split, path) respecting mode selection.
    full_only_files -> only populated when separate_full and mode in {mixed,strict} (to capture *_full outside primary export).
    """
    primary: List[Tuple[str,str,str]] = []
    full_only: List[Tuple[str,str,str]] = []
    for entry in sorted(os.listdir(artifacts_dir)):
        dpath = os.path.join(artifacts_dir, entry)
        if not os.path.isdir(dpath) or entry not in RELEVANT_DIR_HINTS:
            continue
        model_name = RELEVANT_DIR_HINTS[entry]
        # collect all candidates
        candidates: Dict[str, Dict[str, Tuple[str,int]]] = {"base":{}, "full":{}}
        for fname, split in PRED_FILENAMES:
            fpath = os.path.join(dpath, fname)
            if not os.path.isfile(fpath):
                continue
            is_full = fname.endswith("_full.jsonl")
            try:
                n_lines = sum(1 for _ in open(fpath, "r", encoding="utf-8"))
            except Exception:
                n_lines = 0
            bucket = "full" if is_full else "base"
            prev = candidates[bucket].get(split)
            if prev is None or n_lines >= prev[1]:
                candidates[bucket][split] = (fpath, n_lines)
        # decide selection
        splits = set(candidates["base"].keys()) | set(candidates["full"].keys())
        for split in splits:
            base = candidates["base"].get(split)
            full = candidates["full"].get(split)
            selected: Optional[Tuple[str,int]] = None
            if mode == "strict":
                selected = base or full  # fallback if only full exists
                if separate_full and full and base and full!=base:
                    full_only.append((model_name, split, full[0]))
            elif mode == "extended":
                selected = full or base
            elif mode == "mixed":
                selected = base or full
                if separate_full and full and (not base or full!=base):
                    full_only.append((model_name, split, full[0]))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if selected:
                primary.append((model_name, split, selected[0]))
    return primary, full_only


def load_predictions(model: str, split: str, path: str) -> List[Record]:
    lines = _read_lines(path)
    out: List[Record] = []
    for ln in lines:
        js = _maybe_json(ln)
        if not js:
            continue
        pid = str(js.get("id") or js.get("page_id") or js.get("hash") or "")
        if not pid:
            continue
        label = int(js.get("label", 0))
        # Some heads might store probabilities with different key naming; attempt fallbacks
        prob = js.get("prob")
        if prob is None:
            prob = js.get("p") or js.get("phish_prob")
        if prob is None:
            continue
        try:
            prob = float(prob)  # type: ignore[arg-type]
        except Exception:
            continue
        rec = Record(
            id=pid,
            label=label,
            prob=prob,
            model=model,
            split=split,
            logit=_logit(prob),
        )
        out.append(rec)
    return out


def _fpr_at_tpr(labels: List[int], probs: List[float], target_tpr: float) -> Tuple[float, float]:
    # Sort descending by score
    pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tp = 0
    fp = 0
    last_s = None
    best_thr = float("nan")
    best_fpr = float("nan")
    for s, y in pairs + [(-float("inf"), None)]:  # sentinel
        if last_s is not None and s != last_s:
            tpr = tp / P
            fpr = fp / N
            if tpr >= target_tpr:
                best_thr = last_s
                best_fpr = fpr
                break
        if y == 1:
            tp += 1
        elif y == 0:
            fp += 1
        last_s = s
    return best_fpr, best_thr


def _roc_auc(labels: List[int], probs: List[float]) -> float:
    # Probability-of-rank implementation (ties averaged)
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    scores = [p for p, _ in pairs]
    labs = [y for _, y in pairs]
    pos = sum(labs)
    neg = len(labs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    # Ranks with tie average
    ranks = [0.0] * len(scores)
    i = 0
    r = 1
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[j + 1] == scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        r = j + 2
        i = j + 1
    sum_ranks_pos = sum(rank for rank, y in zip(ranks, labs) if y == 1)
    u = sum_ranks_pos - pos * (pos + 1) / 2
    return u / (pos * neg)


def _pr_auc(labels: List[int], probs: List[float]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    fn = sum(labels)
    if fn == 0:
        return float("nan")
    prev_recall = 0.0
    prev_precision = 1.0
    auc = 0.0
    last_s = None
    for s, y in pairs:
        if last_s is not None and s != last_s:
            recall = tp / (tp + fn)
            precision = tp / max(1, tp + fp)
            auc += (recall - prev_recall) * prev_precision
            prev_recall = recall
            prev_precision = precision
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        last_s = s
    recall = tp / (tp + fn if (tp + fn) > 0 else 1)
    auc += (recall - prev_recall) * prev_precision
    return max(0.0, min(1.0, auc))


def compute_metrics(records: List[Record]) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[Record]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)
    metrics: Dict[str, Dict[str, float]] = {}
    for model, recs in by_model.items():
        labels = [r.label for r in recs]
        probs = [r.prob for r in recs]
        pos = sum(labels)
        total = len(labels)
        roc = _roc_auc(labels, probs)
        pr = _pr_auc(labels, probs)
        fpr90, thr90 = _fpr_at_tpr(labels, probs, 0.90)
        fpr95, thr95 = _fpr_at_tpr(labels, probs, 0.95)
        metrics[model] = {
            "n": float(total),
            "pos": float(pos),
            "prevalence": (pos / total) if total else float("nan"),
            "roc_auc": roc,
            "pr_auc": pr,
            "fpr_at_tpr90": fpr90,
            "thr_tpr90": thr90,
            "fpr_at_tpr95": fpr95,
            "thr_tpr95": thr95,
        }
    # Macro averages
    if metrics:
        keys = [k for k in next(iter(metrics.values())).keys() if k not in {"n", "pos"}]
        macro: Dict[str, float] = {}
        for k in keys:
            vals = [m[k] for m in metrics.values() if not (isinstance(m[k], float) and math.isnan(m[k]))]
            if vals:
                macro[k] = sum(vals) / len(vals)
        metrics["_macro_avg"] = macro
    return metrics


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if orjson is not None:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode())
        else:
            json.dump(obj, f, indent=2)


def _load_canonical_ids(split: str) -> List[str]:
    path = f"data/pages_{split}.jsonl"
    if not os.path.isfile(path):
        return []
    ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                js=json.loads(line)
            except Exception:
                continue
            rid = js.get("id")
            if rid:
                ids.append(str(rid))
    return ids

def main():  # pragma: no cover - CLI utility
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts", help="Root artifacts directory")
    ap.add_argument("--out-dir", default="artifacts/baseline", help="Baseline output directory")
    ap.add_argument("--mode", choices=["strict","extended","mixed"], default="extended")
    ap.add_argument("--model-allowlist", nargs="*", default=None, help="Restrict to these model names (post-mapping labels)")
    ap.add_argument("--separate-full", action="store_true", help="When strict/mixed, also export *_full predictions to combined_full file set")
    ap.add_argument("--guard-inflation", type=float, default=1.05, help="Max allowed union inflation factor vs canonical before abort")
    ap.add_argument("--allow-inflation", action="store_true", help="Bypass union inflation guard")
    ap.add_argument("--splits", nargs="*", default=["val","test"], help="Which splits to export (supports 'train' if per-model preds exist)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files, full_only = discover_prediction_files(args.artifacts_dir, mode=args.mode, separate_full=args.separate_full)
    if args.model_allowlist:
        allow = set(args.model_allowlist)
        files = [t for t in files if t[0] in allow]
        full_only = [t for t in full_only if t[0] in allow]
    if not files:
        print("No prediction files discovered; aborting.", file=sys.stderr)
        sys.exit(1)

    # dynamic splits container
    by_split: Dict[str, List[Record]] = {s: [] for s in args.splits}
    for model, split, path in files:
        recs = load_predictions(model, split, path)
        by_split.setdefault(split, []).extend(recs)
        print(f"[BASE] Loaded {len(recs):5d} records from {model}/{split} ({os.path.relpath(path)})")
    # Guard: union inflation vs canonical if canonical IDs available
    for split, recs in list(by_split.items()):
        canonical_ids = set(_load_canonical_ids(split))
        if not canonical_ids:
            continue
        union_ids = {r.id for r in recs}
        inflation = (len(union_ids)/len(canonical_ids)) if canonical_ids else 1.0
        if inflation > args.guard_inflation and not args.allow_inflation:
            print(f"[GUARD][FAIL] Split {split} union inflation {inflation:.2f} exceeds guard {args.guard_inflation}. Aborting.", file=sys.stderr)
            sys.exit(2)
        if inflation > 1.0:
            print(f"[GUARD][WARN] Split {split} union inflation {inflation:.2f} (union={len(union_ids)} vs canonical={len(canonical_ids)})")

    # Optional separate full export
    full_by_split: Dict[str, List[Record]] = {"val": [], "test": []}
    if full_only:
        for model, split, path in full_only:
            recs = load_predictions(model, split, path)
            full_by_split.setdefault(split, []).extend(recs)
            print(f"[FULL] Loaded {len(recs):5d} records from {model}/{split} ({os.path.relpath(path)})")

    snapshot_meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": os.path.basename(__file__),
        "artifacts_dir": args.artifacts_dir,
        "models": sorted({m for m, _, _ in files}),
        "mode": args.mode,
        "separate_full": bool(args.separate_full),
        "model_allowlist": args.model_allowlist,
    }

    for split, recs in by_split.items():
        if not recs:
            continue
        out_path = os.path.join(args.out_dir, f"combined_preds_{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(r.to_json())
                f.write("\n")
        metrics = compute_metrics(recs)
        write_json(os.path.join(args.out_dir, f"metrics_{split}.json"), metrics)
        print(f"Wrote {len(recs)} combined records + metrics for split={split}")

    if full_only:
        full_dir = os.path.join(args.out_dir, "full_variants")
        os.makedirs(full_dir, exist_ok=True)
        for split, recs in full_by_split.items():
            if not recs:
                continue
            out_path = os.path.join(full_dir, f"combined_preds_{split}_full.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(r.to_json()); f.write("\n")
            metrics = compute_metrics(recs)
            write_json(os.path.join(full_dir, f"metrics_{split}_full.json"), metrics)
            print(f"[FULL] Wrote {len(recs)} combined full records + metrics for split={split}")

    write_json(os.path.join(args.out_dir, "snapshot.json"), snapshot_meta)
    # Train augmentation redundancy check
    train_base = os.path.join("data","pages_train.jsonl")
    train_aug = os.path.join("data","pages_train_aug.jsonl")
    if os.path.isfile(train_base) and os.path.isfile(train_aug):
        try:
            import hashlib
            def _hash(fp):
                h=hashlib.sha256()
                with open(fp,'rb') as f: h.update(f.read())
                return h.hexdigest()
            hb=_hash(train_base); ha=_hash(train_aug)
            if hb==ha:
                print("[AUG][WARN] pages_train_aug.jsonl identical to pages_train.jsonl (no augmentation effect).")
                snapshot_meta["train_aug_identical"] = True
                write_json(os.path.join(args.out_dir, "snapshot.json"), snapshot_meta)
        except Exception:
            pass
    print("Baseline snapshot complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
