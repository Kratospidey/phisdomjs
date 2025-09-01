# Leak-Safe, Interpretable Phishing-Page Detection via DOM+JS+URL Ensembles

## Problem
Phishing sites evolve quickly and evade single-view detectors. We need a fast, interpretable model that:
- Prevents train/test leakage
- Generalizes across datasets
- Gives calibrated probabilities for thresholded alerts
- Runs at extension-speed on saved pages

## Threat model
Web attackers craft HTML/JS that mimics brands, obfuscates strings/identifiers, injects logos, or delays behavior. We assume no need to break CAPTCHAs or execute privileged APIs; we collect only page code/headers in a sandboxed headless browser (≤3 s), with navigation and outbound requests disabled after initial load.

## Scope & datasets
- Phish: live feeds (OpenPhish, PhishTank) crawled to HTML/JS with provenance logs; add one academic phishing HTML corpus if license permits.
- Benign: Tranco top sites subset (stratified by TLD/category).
- Homelab crawl: ~10,000 pages total; obey robots.txt; polite throttle; store raw HTML, normalized DOM, extracted scripts, headers/metadata.

## Leakage prevention
- eTLD+1 disjoint between splits
- Time-aware test strictly after train
- Cross-dataset generalization (train on source A, test on B)
- Export split indices and seeds

## Features
- DOM tokens: path tokens tag>child[attr], with selected attributes (type, rel, name, href, src, alt); cap vocab; unknown bucket.
- JS tokens: static AST → API n-grams (e.g., document.forms, eval, atob, fetch, crypto.subtle); tiny dynamic capture records counts (no keystrokes/credentials).
- URL tokens: character n-grams; length; entropy-like features (optional).

## Models (advanced heads)
- DOM head: MarkupLM-base (HTML/XPath-aware).
- JS head: CodeT5+ 220M (code encoder; encoder-only mode).
- URL head: char-CNN (fast); optional 4th head: MobileBERT generalist (per PhishLang) if extra robustness is desired.
- Fusion: calibrated head probabilities → mean fuse (start) or meta-LR on validation predictions.

## Calibration
Post-hoc temperature scaling (optionally adaptive) / Platt; report ECE, NLL, reliability diagrams; ship calibrated thresholds.

## Metrics
- Primary: PR-AUC, ROC-AUC
- Operating points: FPR at TPR=95% (and at 90%)
- Efficiency: CPU latency on saved HTML (model head <10 ms; parse time reported separately)
- Throughput: pages/sec single CPU thread

## Baselines
URL-only char n-gram LR, HTML BoW LR/SVM; ablations: DOM-only, JS-only, URL-only.

## Robustness suite
HTML minify/whitespace; JS minify + identifier renaming; brand/logo injection; trivial template changes; report metric deltas with 95% CIs.

## Interpretability
Per-page “Why flagged?”: top DOM paths and top JS APIs contributing to score (token attribution).
Global importances across test; example gallery (true/false positives).

## Reproducibility
Deterministic seeds; pinned deps; hashed vocab files; exported split indices; scripts and YAML configs; CI bootstrap CIs.
Release: checkpoints, calibration scalers, fusion weights; exact command logs.

## Ethics & safety
No collection of credentials or user inputs; sandboxed headless browser; disable navigation/external requests post-load; honor robots.txt; respect feed licenses (PhishTank CC BY-SA 2.5; OpenPhish academic terms).
Dual-use: detection can aid defenders but also inform attackers—publish responsibly; omit sensitive indicators that enable trivial evasion.

## Week-by-week (6 weeks)
1) Data plumbing + crawl harness + data cards + split code
2) Feature extractors + vocab freeze
3) Train heads + baselines; prelim results
4) Calibration + fusion + cross-dataset tests
5) Robustness suite + interpretability + CPU bench
6) Final tables/figures; write IEEE paper; repo polish

## Success thresholds (suggested)
- PR-AUC ≥ 0.90; ROC-AUC ≥ 0.95 on main test
- FPR ≤ 2–3% @ TPR=95% for the fused, calibrated model
- CPU model-head latency < 10 ms; total replay (parse+infer) reported

## Data cards (templates)

### OpenPhish (Phishing Feeds)
- Provenance: curated live phishing URLs; near-real-time
- License/terms: see “Phishing Feeds” and “Academic Use”; some feeds paid—log exact feed and date
- Biases: skews to reported/visible phish; brand/region bias
- Caveats: URLs expire quickly; crawling failures due to takedowns

### PhishTank
- Provenance: community-submitted, community-verified phish
- License: CC BY-SA 2.5 (cite; keep attribution)
- Biases: relies on user reports; verification lag
- Caveats: duplicates with other feeds; snapshot aging

### Tranco (Benign)
- Provenance: multi-source, 30-day aggregated ranking; manipulation-hardened
- License: see site; download logs with date/version
- Biases: popularity ≠ safety; includes trackers/ad-heavy sites
- Caveats: site variants by geo; dynamic content

## Notes on the advanced heads
- MarkupLM (DOM head): HTML/XML encoder with XPath embeddings; ideal for structural DOM cues. Use MarkupLMProcessor to prepare inputs (tokens + XPath sequences).
- CodeT5+ (JS head): modern (2023) code LLM; set to encoder-only for classification; feed linearized AST/API token stream; use sentencepiece tokenizer.
- MobileBERT (optional generalist head): keeps the ensemble light and “client-side” viable; PhishLang (2024) shows MobileBERT can catch contextual phishing patterns quickly.

### Variant A: Transformer (single/dual-encoder)
- HTML-centric: fine-tune MarkupLM-base on each page’s raw HTML (processor extracts nodes/XPaths).
- With JS: dual-encoder (MarkupLM + CodeT5+) → concat [CLS] → small MLP head.
- If inputs overflow 512: slide windows + max-pool logits, or switch to Longformer.

### Variant 2: Transformer for code context
- Concatenate HTML DOM and JS token sequence; small transformer to attend across entire source.
- Pre-train on unlabeled HTML pages if possible; then fine-tune on phishing task.

### Variant 3: DOM GNN + Sequence hybrid
- Graph the DOM (nodes: elements; edges: tree structure) + a JS sequence encoder; fuse and classify.

---
This file is a living one-pager to guide implementation. Keep it updated with decisions, split hashes, and release artifacts.
