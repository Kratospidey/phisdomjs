# PhisDOM: End-to-end Pipeline, Models, and Results

This document explains the full project: data pipeline, crawlers, models, training/eval, outputs, and how to interpret the results. It also maps important files and configuration switches so you can reproduce and modify runs quickly.

## Overview

PhisDOM detects phishing pages using a two-head approach:
- DOM head: a transformer on page markup (MarkupLM)
- JS head: a transformer on JavaScript (CodeT5+)
These heads are calibrated and optionally fused (logistic fusion) for improved accuracy. The dataset is built entirely from feeds you control (OpenPhish, PhishTank, Tranco) via an automated pipeline.

## Data pipeline

Inputs and steps:
- Feeds (configs/feeds.yaml)
  - OpenPhish: text of phishing URLs
  - PhishTank: CSV (requires env: PHISHTANK_APP_KEY and PHISHTANK_UA)
  - Tranco: top domains (benign candidates), generated via the official tranco Python package
- Unify (scripts/unify_feeds.py)
  - Dedup and merge feeds to a single CSV `data/seed.csv` with columns: url,label,source
  - Control total size and label ratio with SEED_SIZE and PHISH_RATIO
- Crawl (scripts/crawl_playwright.py)
  - Async Playwright crawler fetches DOM and linked JS, writing JSONL to `data/pages.jsonl`
  - Robust defaults: realistic UA, ignore HTTPS errors, retries, backoff, block heavy assets (images/media/fonts)
  - Resume-safe: on reruns, it skips URLs already present in `pages.jsonl`
- Splits (scripts/make_splits.py, scripts/slice_dataset.py)
  - Builds train/val/test partitions and writes `data/pages_{train,val,test}.jsonl`

Key Make targets:
- `make feeds` → fetch feeds and generate Tranco
- `make unify` → produce `data/seed.csv`
- `make crawl` → produce/append `data/pages.jsonl` (resumable)
- `make splits slice` → produce split JSONLs
- `make train eval report` → train/evaluate DOM head and produce reports
- Optional JS head + fusion: `make train-js eval-js fuse`

Important Make variables:
- SEED_SIZE: desired total size of seed (e.g., 20000)
- PHISH_RATIO: fraction of phishing in seed (0.0-1.0)
- CRAWL_CONCURRENCY, CRAWL_TIMEOUT, CRAWL_RETRIES: crawler knobs
- CRAWL: set to false/0/no to skip crawling and use existing `data/pages.jsonl`
- AUTO_CUTOFF, VAL_FRAC: split generation knobs
- XAI_DEVICE: device for explanation generation (cuda/cpu)

## Crawler details

- Script: `scripts/crawl_playwright.py`
- Inputs: `--input-csv data/seed.csv` → CSV with url,label,source
- Output: `--out-jsonl data/pages.jsonl` with records:
  - id (sha1(url)), url, etld1, timestamp, source, label, html, scripts[], headers
- Behavior:
  - Resumable: loads existing URLs from out JSONL and skips them
  - Concurrency: per-worker Playwright contexts
  - Robustness: retries with exponential backoff; graceful shutdown

Resuming crawling:
- Stop anytime (Ctrl+C). Re-run `make crawl` later; it will continue where it left off by skipping existing URLs.
- To skip crawling in an end-to-end run and proceed with ML, use `make CRAWL=false e2e`.

## Data format

- `data/seed.csv`: url,label,source
  - label: 1=phish, 0=benign
- `data/pages.jsonl`: JSONL of PageRecord
  - html: normalized DOM
  - scripts: list of {src, inline, text, attrs}
- `data/pages_{train,val,test}.jsonl`: subsets according to `data/splits.json`

## Models

- DOM head: MarkupLM fine-tuned on normalized page HTML
  - Config: `configs/markup_base.yaml` (see `markup_*` variants)
  - Output dir: `artifacts/markup_run`
- JS head: CodeT5+ fine-tuned on JavaScript snippets
  - Train/Eval scripts: `scripts/train_js_codet5p.py`, `scripts/eval_js_codet5p.py`
  - Output dir: `artifacts/js_codet5p`
- Fusion: Logistic regression on calibrated DOM and JS scores
  - Script: `scripts/fuse_heads.py`
  - Output dir: `artifacts/fusion`

## Training and evaluation

- DOM training: `make train`
- DOM evaluation: `make eval` → produces metrics, calibration, and plots in `artifacts/markup_run`
- JS head (optional): `make train-js eval-js`
- Fusion (optional): `make fuse`
- Report generation: `make report` and `make report-xai` (adds LIME/SHAP explanations)

## Results snapshot (example)

Dataset summary:
- Train: 10823 docs (benign 5618, phish 5205)
- Val: 922 docs (benign 635, phish 287)
- Test: 2919 docs (benign 1556, phish 1363)

Calibration & Metrics (Test):
- DOM (MarkupLM)
  - PR-AUC: 0.9726 | ROC-AUC: 0.9715
  - TPR@0.95 → Threshold=0.1053, FPR=0.1427
  - TPR@0.90 → Threshold=0.3342, FPR=0.0675
- JS (CodeT5+)
  - PR-AUC: 0.9544 | ROC-AUC: 0.9594
  - TPR@0.95 → Threshold=0.0806, FPR=0.2791
  - TPR@0.90 → Threshold=0.2029, FPR=0.0730
- Fused (DOM+JS)
  - PR-AUC: 0.9733 | ROC-AUC: 0.9784
  - TPR@0.95 → Threshold=0.0757, FPR=0.1114
  - TPR@0.90 → Threshold=0.3861, FPR=0.0398

Interpretation:
- Fusion improves overall ranking quality (higher ROC/PR AUC) and reduces FPR at matched TPR, indicating complementary signals between DOM and JS.
- Thresholds are post-calibration operating points; use them to hit desired TPR/FPR trade-offs.

## Key plots (artifacts/markup_run)

- PR and ROC curves: per split (train/val/test) and combined
- Accuracy by split and fused vs. individual heads
- Reliability (calibration) plots: DOM, JS, and fused
- Confusion matrices at TPR≈0.90 and 0.95 for val/test

Filenames (examples):
- pr_curve_test.png, pr_curve_test_js.png, pr_curve_test_fused.png
- roc_curve_test.png, roc_curve_test_js.png, roc_curve_test_fused.png
- reliability_test.png, reliability_test_js.png, reliability_test_fused.png
- confusion_test_fused_tpr90.png, confusion_test_fused_tpr95.png

## Explanations (optional XAI)

- Generated via `make report-xai` (LIME and SHAP for DOM and JS)
- HTML files under `artifacts/markup_run/report/` (lime_*.html, shap_*.html)
- Use these to inspect which tokens/fragments influenced predictions

## Configuration quick reference

- Makefile knobs (env vars):
  - SEED_SIZE, PHISH_RATIO
  - CRAWL_CONCURRENCY, CRAWL_TIMEOUT, CRAWL_RETRIES
  - CRAWL (true/false)
  - AUTO_CUTOFF, VAL_FRAC, XAI_DEVICE, MAXLEN, BATCH, EPOCHS, LR
- Feeds config: `configs/feeds.yaml`
- Model configs: `configs/markup_*.yaml`

## File map (selected)

- configs/
  - feeds.yaml — feed locations
  - markup_*.yaml — model/training configs for DOM head
- scripts/
  - fetch_feeds.py — download/process feeds
  - gen_tranco.py — generate Tranco CSV via tranco package
  - unify_feeds.py — merge feeds to seed.csv
  - crawl_playwright.py — Playwright crawler (resumable)
  - make_splits.py — generate splits.json
  - slice_dataset.py — create train/val/test JSONLs
  - train_markup.py, eval_markup.py — DOM head training/eval
  - train_js_codet5p.py, eval_js_codet5p.py — JS head training/eval
  - fuse_heads.py — logistic fusion of calibrated scores
  - report_eval.py — plots, calibration, and explanations
  - run_e2e.py — Python orchestration for the full pipeline
- data/
  - feeds/ — raw feeds (openphish.txt, phishtank.csv, tranco.csv)
  - seed.csv — unified URLs with labels
  - pages.jsonl — crawled pages (resumable)
  - pages_{train,val,test}.jsonl — split datasets
- artifacts/
  - markup_run/ — DOM head model, metrics, plots, explanations
  - js_codet5p/ — JS head artifacts
  - fusion/ — fusion artifacts

## Reproducible runs

End-to-end with a target seed and longer crawl timeouts:
```bash
make SEED_SIZE=20000 PHISH_RATIO=0.5 CRAWL_TIMEOUT=8.0 CRAWL_RETRIES=4 e2e
```
Skip crawling and use existing JSONL:
```bash
make CRAWL=false e2e
```
Resume crawling later (continues from where you left off):
```bash
make CRAWL=true CRAWL_TIMEOUT=8.0 CRAWL_RETRIES=4 crawl
```

## Troubleshooting

- PhishTank feed is empty: ensure PHISHTANK_APP_KEY and PHISHTANK_UA are set
- Too many crawl SKIPs: increase `CRAWL_TIMEOUT`, reduce `CRAWL_CONCURRENCY`, ensure network connectivity
- Resuming doesn’t skip: verify `data/pages.jsonl` exists and isn’t corrupt; the crawler will log how many existing URLs were loaded

---
Last updated: 2025-09-04