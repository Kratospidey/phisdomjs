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

## Roadmap (abridged)
- Data plumbing/crawl harness and data cards
- DOM/JS feature extractors and vocab freeze
- Train heads + baselines; prelim results
- Calibration + fusion + cross-dataset tests
- Robustness suite + interpretability + CPU bench

Contributions welcome. Keep ethics and licenses in mind (see charter).
