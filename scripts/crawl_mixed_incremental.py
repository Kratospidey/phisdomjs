#!/usr/bin/env python3
"""
Incremental crawler that:
1. Uses existing crawled data if available
2. Fetches URLs from Tranco (benign) and multiple phishing feeds
3. Supports custom phish/benign ratios and phish-only mode
4. Refill loop to backfill phishing shortfalls across sources
5. Optional overshoot to exceed the incremental target for more phish
6. Updates existing pages.jsonl file and prevents duplicates
7. Optional source column for observability in temp CSV
"""

import argparse
import asyncio
import csv
import os
import sys
from typing import Set, List, Tuple, Dict
from pathlib import Path
import requests
import tempfile
import re
from urllib.parse import urlsplit, urlunsplit

# Import existing crawler functionality
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from tranco import Tranco
    from crawl_playwright import crawl, _load_existing_urls
except ImportError as e:
    print(f"ERROR: Could not import crawler functions: {e}")
    print("Make sure you're in the phisdom directory and have installed: pip install tranco dnspython requests")
    sys.exit(1)

# -----------------------
# Phishing feed endpoints
# -----------------------

# OpenPhish (community)
OPENPHISH_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
OPENPHISH_LEGACY = "https://openphish.com/feed.txt"  # fallback

# Phishing.Database (GitHub mirror)
PHISHINGDB_FEEDS = [
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-NOW.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-today.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-last-hour.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt",
]

# CERT-PL (domains)
CERTPL_DOMAINS = "https://hole.cert.pl/domains/v2/domains.txt"

# BlocklistProject (domains)
BLOCKLISTPROJECT_PHISHING = "https://raw.githubusercontent.com/blocklistproject/Lists/master/phishing.txt"

# Phishing Army (domains)
PHISHING_ARMY = "https://phishing.army/download/phishing_army_blocklist.txt"
PHISHING_ARMY_EXT = "https://phishing.army/download/phishing_army_blocklist_extended.txt"

# JPCERT phishurl-list (CSV per year)
JPCERT_INDEX = "https://raw.githubusercontent.com/JPCERTCC/phishurl-list/main/{year}/phishurl-list-{year}.csv"

# URLhaus recent CSV
URLHAUS_RECENT_CSV = "https://urlhaus.abuse.ch/downloads/csv_recent/"

# HTTP UA
UA = {"User-Agent": "phisdom-crawler/1.1 (+research)"}

# -----------------------
# Utilities
# -----------------------

def _canonicalize_url(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    parts = urlsplit(u)
    scheme = parts.scheme.lower() if parts.scheme else "http"
    netloc = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, netloc, path, parts.query, ""))

def _canonicalize_set(urls: Set[str]) -> Set[str]:
    return {_canonicalize_url(u) for u in urls if u}

def _fetch_text(url: str, timeout=60) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        if r.status_code == 200:
            return r.text
        print(f"Warning: {url} returned HTTP {r.status_code}")
    except Exception as e:
        print(f"Warning: failed to fetch {url}: {e}")
    return ""

def _fetch_lines(url: str, timeout=60) -> List[str]:
    text = _fetch_text(url, timeout)
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]

# -----------------------
# Source helpers
# -----------------------

def _fetch_domainlist_as_urls(url: str, source: str, limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    urls: List[Tuple[str, int, str]] = []
    lines = _fetch_lines(url)
    print(f"{source} domains: {len(lines)} lines")
    for dom in lines:
        dom = dom.replace("0.0.0.0 ", "").lstrip("||").rstrip("^").strip()
        if not dom or "." not in dom or " " in dom:
            continue
        u = f"http://{dom}/"
        c = _canonicalize_url(u)
        if c not in existing_canon:
            urls.append((c, 1, source))
            if len(urls) >= limit:
                break
    return urls

def _fetch_openphish(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    urls: List[Tuple[str, int, str]] = []
    lines = _fetch_lines(OPENPHISH_URL)
    if not lines:
        lines = _fetch_lines(OPENPHISH_LEGACY)
    print(f"OpenPhish lines: {len(lines)}")
    for url in lines:
        c = _canonicalize_url(url)
        if c and c not in existing_canon:
            urls.append((c, 1, "openphish"))
            if len(urls) >= limit:
                break
    return urls

def _fetch_phishingdb(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    urls: List[Tuple[str, int, str]] = []
    for feed_url in PHISHINGDB_FEEDS:
        print(f"Fetching from {feed_url}...")
        lines = _fetch_lines(feed_url)
        print(f"Got {len(lines)} lines")
        for url in lines:
            c = _canonicalize_url(url)
            if c and c not in existing_canon:
                urls.append((c, 1, "phishingdb"))
                if len(urls) >= limit:
                    return urls
    return urls

def _fetch_phishtank_if_configured(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    api_key = os.environ.get("PHISHTANK_API_KEY")
    if not api_key:
        print("PhishTank: No API key, skipping")
        return []
    try:
        url = f"http://data.phishtank.com/data/{api_key}/online-valid.csv"
        text = _fetch_text(url, timeout=60)
        if not text:
            print("PhishTank: empty response, skipping")
            return []
        urls: List[Tuple[str, int, str]] = []
        reader = csv.DictReader([ln for ln in text.splitlines() if ln and not ln.startswith("#")])
        for row in reader:
            raw = (row.get("url") or "").strip()
            if not raw:
                continue
            c = _canonicalize_url(raw)
            if c and c not in existing_canon:
                urls.append((c, 1, "phishtank"))
                if len(urls) >= limit:
                    break
        print(f"PhishTank lines (unique): {len(urls)}")
        return urls
    except Exception as e:
        print(f"PhishTank error: {e}")
        return []

def _fetch_certpl_domains(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    return _fetch_domainlist_as_urls(CERTPL_DOMAINS, "cert-pl", limit, existing_canon)

def _fetch_blocklistproject_domains(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    return _fetch_domainlist_as_urls(BLOCKLISTPROJECT_PHISHING, "blocklistproject", limit, existing_canon)

def _fetch_phishingarmy(limit: int, existing_canon: Set[str], extended: bool = False) -> List[Tuple[str, int, str]]:
    src = "phishing-army-extended" if extended else "phishing-army"
    url = PHISHING_ARMY_EXT if extended else PHISHING_ARMY
    return _fetch_domainlist_as_urls(url, src, limit, existing_canon)

def _fetch_jpcert(limit: int, existing_canon: Set[str], years=range(2025, 2018, -1)) -> List[Tuple[str, int, str]]:
    urls: List[Tuple[str, int, str]] = []
    for y in years:
        text = _fetch_text(JPCERT_INDEX.format(year=y))
        if not text:
            continue
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        # Skip header if present
        start_idx = 1 if lines and ("URL" in lines[0] or "url" in lines[0]) else 0
        reader = csv.reader(lines[start_idx:])
        for row in reader:
            if len(row) < 2:
                continue
            raw = row[1].strip().strip('"')
            if not raw:
                continue
            c = _canonicalize_url(raw)
            if c and c not in existing_canon:
                urls.append((c, 1, f"jpcert-{y}"))
                if len(urls) >= limit:
                    return urls
    return urls

def _fetch_urlhaus_phishing_tag(limit: int, existing_canon: Set[str], filter_phish: bool = True) -> List[Tuple[str, int, str]]:
    text = _fetch_text(URLHAUS_RECENT_CSV)
    if not text:
        return []
    urls: List[Tuple[str, int, str]] = []
    rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    reader = csv.reader(rows)
    # Rough columns: dateadded,url,url_status,threat,tags,...
    for row in reader:
        if len(row) < 4:
            continue
        url = row[1].strip()
        threat = (row[3] or "").strip().lower() if len(row) > 3 else ""
        if filter_phish and "phish" not in threat:
            continue
        c = _canonicalize_url(url)
        if c and c not in existing_canon:
            urls.append((c, 1, "urlhaus"))
            if len(urls) >= limit:
                break
    return urls

# -----------------------
# Fetch orchestration
# -----------------------

def _gather_from_sources(limit: int, existing_canon: Set[str], urlhaus_filter_phish: bool) -> Tuple[List[Tuple[str, int, str]], Dict[str, int]]:
    if limit <= 0:
        return [], {}

    all_urls: List[Tuple[str, int, str]] = []
    per_source: Dict[str, int] = {}

    source_funcs = [
        _fetch_openphish,
        _fetch_phishingdb,
        _fetch_phishtank_if_configured,
        _fetch_certpl_domains,
        _fetch_blocklistproject_domains,
        lambda n, s: _fetch_phishingarmy(n, s, extended=False),
        lambda n, s: _fetch_phishingarmy(n, s, extended=True),
        _fetch_jpcert,
        lambda n, s: _fetch_urlhaus_phishing_tag(n, s, filter_phish=urlhaus_filter_phish),
    ]

    remaining = limit
    # Initial fair split
    initial_chunk = max(1, remaining // len(source_funcs))
    for fn in source_funcs:
        got = fn(initial_chunk, existing_canon)
        all_urls.extend(got)
        for u, _, src in got:
            existing_canon.add(u)
            per_source[src] = per_source.get(src, 0) + 1
    remaining = limit - len(all_urls)

    # Refill loop (cycle until limit or exhaustion)
    safety_rounds = 6
    while remaining > 0 and safety_rounds > 0:
        progress = 0
        for fn in source_funcs:
            if remaining <= 0:
                break
            want = min(500, remaining)
            got = fn(want, existing_canon)
            if not got:
                continue
            all_urls.extend(got)
            for u, _, src in got:
                existing_canon.add(u)
                per_source[src] = per_source.get(src, 0) + 1
            grabbed = len(got)
            remaining -= grabbed
            progress += grabbed
        if progress == 0:
            break
        safety_rounds -= 1

    return all_urls[:limit], per_source

def fetch_phishing_urls(limit: int, existing_canon: Set[str], urlhaus_filter_phish: bool = True) -> Tuple[List[Tuple[str, int, str]], Dict[str, int]]:
    urls, counts = _gather_from_sources(limit, existing_canon, urlhaus_filter_phish=urlhaus_filter_phish)
    if len(urls) < limit:
        print(f"[WARN] Phish shortfall: {len(urls)}/{limit} (only {len(urls)} found)")
    return urls, counts

def fetch_tranco_urls(limit: int, existing_canon: Set[str]) -> Tuple[List[Tuple[str, int, str]], Dict[str, int]]:
    t = Tranco(cache=True)
    top_sites = t.list().top(1_000_000)
    urls: List[Tuple[str, int, str]] = []
    for domain in top_sites:
        if len(urls) >= limit:
            break
        for scheme in ["https", "http"]:
            url = f"{scheme}://{domain}/"
            c = _canonicalize_url(url)
            if c not in existing_canon:
                urls.append((c, 0, "tranco"))
                existing_canon.add(c)
                break
    print(f"Tranco URLs: {len(urls)}")
    return urls, {"tranco": len(urls)}

def fetch_mixed_urls(needed: int, phish_ratio: float, existing_urls: Set[str],
                     phish_only: bool, allow_overshoot: bool, overshoot_limit: int,
                     urlhaus_filter_phish: bool) -> Tuple[List[Tuple[str, int, str]], Dict[str, int]]:
    existing_canon = _canonicalize_set(existing_urls)

    if phish_only:
        phish_count = needed
        benign_count = 0
    else:
        phish_count = int(needed * phish_ratio)
        benign_count = max(0, needed - phish_count)

    print(f"Incremental target: {phish_count} phishing + {benign_count} benign = {needed} total")

    per_source_counts: Dict[str, int] = {}

    # Phishing
    phish_urls, phish_src = fetch_phishing_urls(phish_count, existing_canon, urlhaus_filter_phish=urlhaus_filter_phish)
    print(f"Got {len(phish_urls)} phishing URLs")
    for k, v in phish_src.items():
        per_source_counts[k] = per_source_counts.get(k, 0) + v

    # Benign
    benign_urls: List[Tuple[str, int, str]] = []
    if not phish_only and benign_count > 0:
        b_urls, b_src = fetch_tranco_urls(benign_count, existing_canon)
        print(f"Got {len(b_urls)} benign URLs")
        benign_urls = b_urls
        for k, v in b_src.items():
            per_source_counts[k] = per_source_counts.get(k, 0) + v

    all_urls = phish_urls + benign_urls

    # Overshoot for phish shortfall
    requested_phish = phish_count
    got_phish = len(phish_urls)
    shortfall = max(0, requested_phish - got_phish)
    if shortfall > 0 and allow_overshoot and overshoot_limit > 0:
        extra_want = min(shortfall, overshoot_limit)
        print(f"[OVERSHOOT] Trying to fetch +{extra_want} additional phish to meet requested mix...")
        extra_phish, extra_src = fetch_phishing_urls(extra_want, existing_canon, urlhaus_filter_phish=urlhaus_filter_phish)
        print(f"[OVERSHOOT] Got +{len(extra_phish)} additional phish")
        all_urls.extend(extra_phish)
        for k, v in extra_src.items():
            per_source_counts[k] = per_source_counts.get(k, 0) + v

    # Batch stats
    phish_added = sum(1 for _, lbl, _ in all_urls if lbl == 1)
    benign_added = sum(1 for _, lbl, _ in all_urls if lbl == 0)
    total_added = len(all_urls)
    if total_added:
        print(f"Requested phish={requested_phish}, got={phish_added}; requested benign={benign_count}, got={benign_added}")
        print(f"Achieved phish ratio in this batch: {phish_added/total_added:.3f} ({phish_added}/{total_added})")

    return all_urls, per_source_counts

# -----------------------
# Main CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Incremental crawler with richer phishing sources and refill loop")
    parser.add_argument("--target-count", type=int, default=10000, help="Overall target total URLs (existing + new)")
    parser.add_argument("--phish-ratio", type=float, default=0.5, help="Desired phishing ratio for the *incremental* additions (0.0..1.0)")
    parser.add_argument("--phish-only", action="store_true", help="Fetch only phishing URLs for this run (ignores phish-ratio)")
    parser.add_argument("--allow-overshoot", action="store_true", help="Allow exceeding the incremental target to fetch extra phish if sources under-deliver")
    parser.add_argument("--overshoot-limit", type=int, default=1000, help="Max extra phish to add beyond the incremental target when overshoot is enabled")
    parser.add_argument("--include-source-in-csv", action="store_true", help="Write a 'source' column into the temp CSV (3rd column)")
    # BooleanOptionalAction for easy --urlhaus-filter-phish / --no-urlhaus-filter-phish
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:
        bool_action = None
    if bool_action is not None:
        parser.add_argument("--urlhaus-filter-phish", action=bool_action, default=True,
                            help="If True, keep only URLhaus rows with threat~='phish*' (default: True)")
    else:
        parser.add_argument("--urlhaus-filter-phish", action="store_true",
                            help="If set, keep only URLhaus rows with threat~='phish*' (no negative form available on this Python)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent crawlers")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Timeout per page in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries per URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be crawled without actually crawling")
    parser.add_argument("--output", default="data/pages.jsonl", help="Output file path")
    parser.add_argument("--per-url-cap-s", type=float, default=45.0, help="Hard wall per-URL total time budget across retries (default 45s)")
    parser.add_argument("--progress-interval-s", type=float, default=10.0, help="How often to print progress lines (seconds, default 10)")
    # Pass-through crawler filtering / circuit-breaker flags
    parser.add_argument("--skip-numeric-sld-min-digits", type=int, default=8,
                        help="Skip SLDs composed only of digits with length >= this (0 disables)")
    parser.add_argument("--skip-numeric-allow", type=str, default="360,163,12306",
                        help="Comma list of numeric SLDs to always allow")
    parser.add_argument("--skip-suffix", type=str, default="",
                        help="Comma list of SLD suffixes to skip (e.g., -cdn,-cache)")
    parser.add_argument("--adaptive-block-after", type=int, default=3,
                        help="Auto-block eTLD+1 after this many timeout/cap events (0 disables)")
    args = parser.parse_args()

    if not 0 <= args.phish_ratio <= 1:
        print("ERROR: phish-ratio must be between 0.0 and 1.0")
        sys.exit(1)

    # Load existing
    output_path = Path(args.output)
    existing_urls = set()
    if output_path.exists():
        existing_urls = _load_existing_urls(str(output_path))
        print(f"Loaded {len(existing_urls)} existing URLs from {output_path}")

    needed = args.target_count - len(existing_urls)
    if needed <= 0:
        print(f"Already have {len(existing_urls)} URLs (target: {args.target_count}). Nothing to do.")
        return

    print(f"Need {needed} more URLs to reach target of {args.target_count}")

    # Fetch new URLs
    new_urls, per_source = fetch_mixed_urls(
        needed=needed,
        phish_ratio=args.phish_ratio,
        existing_urls=existing_urls,
        phish_only=args.phish_only,
        allow_overshoot=args.allow_overshoot,
        overshoot_limit=args.overshoot_limit,
        urlhaus_filter_phish=getattr(args, "urlhaus_filter_phish", True),
    )

    if not new_urls:
        print("No new URLs found!")
        return

    print(f"Found {len(new_urls)} new URLs to crawl")
    if per_source:
        print("Per-source additions (this batch):")
        for src, cnt in sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {src:24s} {cnt:6d}")

    if args.dry_run:
        print("DRY RUN - sample of what would be crawled:")
        for url, label, source in new_urls[:20]:
            print(f"  {url} (label={label}, source={source})")
        if len(new_urls) > 20:
            print(f"  ... and {len(new_urls) - 20} more")
        return

    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        if args.include_source_in_csv:
            writer.writerow(['url', 'label', 'source'])
            for url, label, source in new_urls:
                writer.writerow([url, label, source])
        else:
            writer.writerow(['url', 'label'])
            for url, label, source in new_urls:
                writer.writerow([url, label])
        temp_file = f.name

    try:
        print(f"Starting crawler with {args.concurrency} workers...")
        asyncio.run(crawl(
            input_csv=temp_file,
            out_jsonl=str(output_path),
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            block_assets=True,
            capture_external_js=True,
            retries=args.retries,
            resume=True,
            tls_timeout=10.0,
            dns_timeout=5.0,
            mobile_profile=False,
            use_gpu=False,
            per_url_cap_s=args.per_url_cap_s,
            progress_interval_s=args.progress_interval_s,
            skip_numeric_sld_min_digits=args.skip_numeric_sld_min_digits,
            skip_numeric_allow=args.skip_numeric_allow,
            skip_suffix=args.skip_suffix,
            adaptive_block_after=args.adaptive_block_after,
        ))

        if output_path.exists():
            final_urls = _load_existing_urls(str(output_path))
            print(f"Crawling complete! Total URLs: {len(final_urls)}")
        else:
            print("Warning: Output file not created")

    finally:
        try:
            os.unlink(temp_file)
        except Exception:
            pass

if __name__ == "__main__":
    main()
