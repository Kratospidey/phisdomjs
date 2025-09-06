# Phase 1: Schema and Extractors (URL/JS char-CNN, DOM graph, Visible Text)

This phase adds storage-light fields to `PageRecord` and deterministic extractors:

- url_raw: original URL string
- url_charseq: list[int] (<=256) URL character indices
- js_charseq: list[int] (<=2048) JS character indices (concatenated)
- text_title: capped `<title>`
- text_visible: capped visible text (scripts/styles/hidden removed)
- dom_graph: compact DOM tree graph `{n, nodes:[{t,c,d,x}], edges:[[i,j]]}`

These are populated during crawl and via `scripts/backfill_fields.py` for existing JSONL.
# End-to-end pipeline (feeds → seed → crawl → splits → train → eval)

This repo provides an automated, leak-safe pipeline to build a phishing detection dataset and train a DOM transformer (MarkupLM) end-to-end.

## Overview

Stages:
- Fetch feeds: OpenPhish (phish), PhishTank (phish), Tranco (benign)
- Unify feeds → `data/seed.csv` (url,label,source)
- Crawl pages to DOM JSONL via Playwright → `data/pages.jsonl`
- Time-aware, group-disjoint splits → `data/splits.json`
- Slice datasets → `data/pages_{train,val,test}.jsonl`
- Train MarkupLM → `artifacts/markup_run`
- Evaluate + temperature calibration → metrics and thresholds

## Prereqs

- Python env with: `torch`, `transformers`, `playwright`, `tldextract`, `beautifulsoup4`, `lxml`, `requests`, `PyYAML`, `tqdm`, `tranco` (for Tranco fetch)
- Install Playwright browser once:

```bash
python -m playwright install chromium
```

- PhishTank authenticated feed (recommended):
  - Export env vars before running fetch:

```bash
export PHISHTANK_APP_KEY=your_key_here
export PHISHTANK_UA="phishtank/your-ua"
```

## Quick start (Make targets)

Fetch, unify, crawl, split, train, eval in one go:

```bash
make e2e
```

Individual steps:

```bash
make feeds     # fetch feeds (generates Tranco CSV via Python, fetches OpenPhish/PhishTank)
make unify     # merge into data/seed.csv
make crawl     # crawl with Playwright
make splits    # create time-aware, group-disjoint splits
make slice     # create train/val/test JSONLs
make train     # fine-tune MarkupLM
make eval      # evaluate + calibrate, dumps metrics and thresholds
```

## One-shot script

Alternatively, run the orchestrator:

```bash
PYTHONPATH=src python scripts/run_e2e.py
```

## Configuration

- `configs/feeds.yaml` controls sources:
  - OpenPhish: public text
  - PhishTank: authenticated `.bz2` CSV (env-driven URL, decompressed automatically)
  - Tranco: local `data/feeds/tranco.csv` generated via `make tranco`
- `configs/markup_base.yaml` controls model/training.

## Repro tips

- Keep `data/seed.csv` under version control if you want reproducible crawls.
- For larger experiments, increase limits in the Makefile (`LIMIT_PHISH`, `LIMIT_BENIGN`) and `num_epochs`/`batch_size` in `configs/markup_base.yaml`.
- For CUDA, PyTorch will automatically use GPU; mixed precision enabled when available.

## Troubleshooting

- Playwright missing: install and run `python -m playwright install chromium`.
- PhishTank 403/404: ensure `PHISHTANK_APP_KEY` and a descriptive `PHISHTANK_UA` are exported.
- Tranco rate/HTTP errors: the package caches locally; try again later.

## Outputs

- `artifacts/markup_run/calibration.json` contains PR/ROC AUC and threshold(s).
- `artifacts/markup_run/preds_{val,test}.jsonl` contain per-sample probabilities.

