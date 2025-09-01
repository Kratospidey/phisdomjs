#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
import time
import bz2
from typing import Dict, Any

import requests
import yaml


DEFAULT_HEADERS = {"User-Agent": "phisdom/0.1 (+research; contact: local)"}


def expand_env(s: str) -> str:
    """Expand ${VARS} using environment variables."""
    return os.path.expandvars(s)


def unresolved_vars_present(s: str) -> bool:
    return "${" in s and "}" in s


def fetch_to(path: str, url: str, headers: Dict[str, str] | None = None, timeout: float = 20.0) -> bytes:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    content = r.content
    # Write raw response; caller may choose to post-process
    with open(path, "wb") as f:
        f.write(content)
    return content


def main():
    parser = argparse.ArgumentParser(description="Fetch live feed files (OpenPhish/PhishTank/Tranco) to local paths")
    parser.add_argument("--config", required=True, help="YAML with endpoints and output paths")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    feeds = cfg.get("feeds", {})
    delay = float(cfg.get("delay_s", 0.5))
    any_ok = False
    for name, spec in feeds.items():
        raw_url = spec.get("url")
        out = spec.get("out")
        decompress = (spec.get("decompress") or "").strip().lower()  # e.g., "bz2" | ""
        headers_map = spec.get("headers") or {}
        # Expand envs in url and header values
        url = expand_env(raw_url or "")
        headers = {k: expand_env(str(v)) for k, v in headers_map.items()}

        if not raw_url or not out:
            print(f"SKIP {name}: missing url/out in config", file=sys.stderr)
            continue
        if unresolved_vars_present(url):
            print(f"SKIP {name}: unresolved env vars in url '{raw_url}'. Export required env and retry.", file=sys.stderr)
            continue

        print(f"Fetching {name}: {url} -> {out}")
        try:
            content = fetch_to(out, url, headers=headers)
            any_ok = True
            if decompress == "bz2":
                try:
                    decompressed = bz2.decompress(content)
                except OSError:
                    # If server already returns decompressed content
                    decompressed = content
                # Write decompressed to target out (overwriting the original)
                with open(out, "wb") as f:
                    f.write(decompressed)
                print(f"Decompressed {name} (bz2) -> {out}")
        except Exception as e:
            print(f"WARN: failed to fetch {name}: {e}", file=sys.stderr)
        time.sleep(delay)

    if not any_ok:
        print("ERROR: no feeds fetched successfully. Check your configuration and network.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
