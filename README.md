# phisdom

Leak-safe, interpretable phishing-page detection via DOM+JS+URL ensembles.

## What’s here now
- Project charter one-pager (`PROJECT_CHARTER.md`)
- Minimal Python package skeleton (`src/phisdom`) with:
  - URL feature extractor (char n-grams, entropy, length)
  - Group/time-aware split helper (no pandas required)
  - Basic metrics (ROC-AUC, PR-AUC, FPR@TPR)
  - Simple temperature scaler (post-hoc calibration)
  - Tiny CLI demo
- Lightweight unit tests (`tests/`) runnable with the standard library

## Quickstart

Run tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

CLI demo:

```bash
python -m phisdom.cli --url "http://login.example.co-security.com/reset?acct=123"
```

## Documentation

- Full project report: see `REPORT.md` for pipeline, configs, models, and results.

## Make knobs (crawler)

- CRAWL_CONCURRENCY: parallel browser contexts
- CRAWL_TIMEOUT: per-navigation timeout seconds
- CRAWL_RETRIES: retries per URL
- TLS_TIMEOUT: timeout for TLS metadata fetch (per host)
- DNS_TIMEOUT: timeout for DNS TTL/MX lookup (per domain)
- MOBILE_PROFILE: when true, perform a second quick mobile-profile load to compute cloaking deltas
- GPU: when true, enable GPU-friendly Chromium flags during crawling

Quick 5-URL smoke crawl to verify lightweight fields:

```bash
make MOBILE_PROFILE=true GPU=true smoke
```

## Roadmap (abridged)
- Data plumbing/crawl harness and data cards
- DOM/JS feature extractors and vocab freeze
- Train heads + baselines; prelim results
- Calibration + fusion + cross-dataset tests
- Robustness suite + interpretability + CPU bench

Contributions welcome. Keep ethics and licenses in mind (see charter).
