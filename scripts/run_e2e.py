#!/usr/bin/env python
from __future__ import annotations
import os
import sys
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable]
ENV = os.environ.copy()
# Ensure phisdom package is importable by called scripts
ENV["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + ENV.get("PYTHONPATH", "") if ENV.get("PYTHONPATH") else "")


def run(cmd: list[str], cwd: Path = ROOT):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd), env=ENV)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    # 1) Fetch feeds (requires PHISHTANK_APP_KEY and PHISHTANK_UA for PhishTank)
    run(PY + [str(ROOT / "scripts/fetch_feeds.py"), "--config", str(ROOT / "configs/feeds.yaml")])

    # 2) Unify feeds
    Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
    run(PY + [str(ROOT / "scripts/unify_feeds.py"),
              "--openphish", str(ROOT / "data/feeds/openphish.txt"),
              "--phishtank", str(ROOT / "data/feeds/phishtank.csv"),
              "--tranco", str(ROOT / "data/feeds/tranco.csv"),
              "--out", str(ROOT / "data/seed.csv"),
              "--limit-phish", "1000", "--limit-benign", "1000", "--shuffle", "--seed", "1337"])

    # 3) Crawl
    run(PY + [str(ROOT / "scripts/crawl_playwright.py"),
              "--input-csv", str(ROOT / "data/seed.csv"),
              "--out-jsonl", str(ROOT / "data/pages.jsonl"),
              "--concurrency", "4", "--timeout-s", "3.0"])

    # 4) Splits
    run(PY + [str(ROOT / "scripts/make_splits.py"),
              "--jsonl", str(ROOT / "data/pages.jsonl"),
              "--out", str(ROOT / "data/splits.json"),
              "--auto-cutoff-percentile", "80", "--val-fraction", "0.1"])

    # 5) Slice
    run(PY + [str(ROOT / "scripts/slice_dataset.py"),
              "--jsonl", str(ROOT / "data/pages.jsonl"),
              "--splits", str(ROOT / "data/splits.json"),
              "--out-prefix", str(ROOT / "data/pages_")])

    # 6) Train (fine-tunes MarkupLM head so classifier weights are initialized)
    run(PY + [str(ROOT / "scripts/train_markup.py"), "--config", str(ROOT / "configs/markup_base.yaml")])

    # 7) Eval
    cfg_path = ROOT / "configs/markup_base.yaml"
    # Default output dir from config
    out_dir = ROOT / "artifacts/markup_run"
    run(PY + [str(ROOT / "scripts/eval_markup.py"),
              "--model-dir", str(out_dir),
              "--val-jsonl", str(ROOT / "data/pages_val.jsonl"),
              "--test-jsonl", str(ROOT / "data/pages_test.jsonl"),
              "--max-length", "512"])

    print("\nEnd-to-end run complete. Artifacts in:", out_dir)
    cal = out_dir / "calibration.json"
    if cal.exists():
        print("Calibration summary:\n", cal.read_text())


if __name__ == "__main__":
    main()
