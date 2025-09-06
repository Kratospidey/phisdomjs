#!/usr/bin/env python
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
import json

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


def _needs_backfill(jsonl_path: Path) -> bool:
    try:
        if not jsonl_path.exists():
            return False
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Representative new fields
                keys = obj.keys() if isinstance(obj, dict) else []
                required_any = [
                    "host_hyphens", "has_punycode",
                    "form_fp_hash64", "num_pw", "form_css_sig_hash64",
                    "js_eval_like", "key_listeners_total",
                    "favicon_dhash64", "logo_phash64",
                    "title_host_jaccard_q8",
                ]
                return not any(k in keys for k in required_any)
    except Exception:
        return True
    return True


def main():
    # 1) Fetch feeds (requires PHISHTANK_APP_KEY and PHISHTANK_UA for PhishTank)
    run(PY + [str(ROOT / "scripts/fetch_feeds.py"), "--config", str(ROOT / "configs/feeds.yaml")])

    # 2) Unify feeds
    Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
    # Build unify args with either target-size or legacy per-source limits
    unify_cmd = [str(ROOT / "scripts/unify_feeds.py"),
                 "--openphish", str(ROOT / "data/feeds/openphish.txt"),
                 "--phishtank", str(ROOT / "data/feeds/phishtank.csv"),
                 "--tranco", str(ROOT / "data/feeds/tranco.csv"),
                 "--out", str(ROOT / "data/seed.csv")]
    seed_size = os.environ.get("SEED_SIZE")
    if seed_size:
        phish_ratio = os.environ.get("PHISH_RATIO", "0.5")
        unify_cmd += ["--target-size", seed_size, "--phish-ratio", phish_ratio]
    else:
        unify_cmd += ["--limit-phish", os.environ.get("LIMIT_PHISH", "5000"),
                      "--limit-benign", os.environ.get("LIMIT_BENIGN", "5000")]
    unify_cmd += ["--shuffle"]
    run(PY + unify_cmd)

    # 3) Crawl
    run(PY + [str(ROOT / "scripts/crawl_playwright.py"),
              "--input-csv", str(ROOT / "data/seed.csv"),
              "--out-jsonl", str(ROOT / "data/pages.jsonl"),
              "--concurrency", "4", "--timeout-s", "3.0"])

    # 3b) Backfill compact fields if needed (dynamic)
    pages_path = ROOT / "data/pages.jsonl"
    if _needs_backfill(pages_path):
        print("[E2E] Detected missing compact fields, running backfill on pages.jsonl...")
        bf_cmd = [str(ROOT / "scripts/backfill_fields.py"), "--inputs", str(pages_path), "--overwrite"]
        if os.environ.get("E2E_BACKFILL_NETWORK", "1") not in ("0", "false", "False"):
            bf_cmd.append("--network")
        run(PY + bf_cmd)

    # 4) Splits
    run(PY + [str(ROOT / "scripts/make_splits.py"),
              "--dataset", str(ROOT / "data/pages.jsonl"),
              "--out", str(ROOT / "data/splits.json"),
              "--auto-cutoff-percentile", os.environ.get("AUTO_CUTOFF", "80"),
              "--val-frac", os.environ.get("VAL_FRAC", "0.1")])

    # 5) Slice
    run(PY + [str(ROOT / "scripts/slice_dataset.py"),
              "--jsonl", str(ROOT / "data/pages.jsonl"),
              "--splits", str(ROOT / "data/splits.json"),
              "--out-prefix", str(ROOT / "data/pages_")])

    # 5b) Backfill sliced splits if needed (dynamic)
    for split in ("train", "val", "test"):
        p = ROOT / f"data/pages_{split}.jsonl"
        if _needs_backfill(p):
            print(f"[E2E] Backfilling pages_{split}.jsonl...")
            bf_cmd = [str(ROOT / "scripts/backfill_fields.py"), "--inputs", str(p), "--overwrite"]
            if os.environ.get("E2E_BACKFILL_NETWORK", "1") not in ("0", "false", "False"):
                bf_cmd.append("--network")
            run(PY + bf_cmd)

    # 6) Train (fine-tunes MarkupLM head so classifier weights are initialized)
    run(PY + [str(ROOT / "scripts/train_markup.py"), "--config", str(ROOT / "configs/markup_base.yaml")])

    # 7) Eval
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
