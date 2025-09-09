#!/usr/bin/env python3
"""
Incremental crawler that:
1. Uses existing crawled data if available
2. Fetches URLs from Tranco (benign) and OpenPhish/PhishTank (phishing)
3. Supports custom phish/benign ratios
4. Crawls until target number of URLs is reached
5. Updates existing pages.jsonl file
6. Prevents duplicates and cleans up temp files
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import Set, List, Tuple
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


# Phishing feed sources
OPENPHISH_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
OPENPHISH_LEGACY = "https://openphish.com/feed.txt"  # fallback

# Phishing.Database: use GitHub raw (the phish.co.za mirror often returns HTML)
PHISHINGDB_FEEDS = [
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-NOW.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-today.txt", 
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-last-hour.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt",
]

# Domain-only feeds (we'll convert to URLs)
CERTPL_DOMAINS = "https://hole.cert.pl/domains/v2/domains.txt"
BLOCKLISTPROJECT_PHISHING = "https://raw.githubusercontent.com/blocklistproject/Lists/master/phishing.txt"

UA = {"User-Agent": "phisdom-crawler/1.0 (+research)"}


def _canonicalize_url(u: str) -> str:
    """Make deduplication robust across http/https, trailing slashes, and fragments."""
    u = u.strip()
    if not u:
        return u
    parts = urlsplit(u)
    # Normalize scheme/host
    scheme = parts.scheme.lower() if parts.scheme else "http"
    netloc = parts.netloc.lower()
    # Drop fragment, normalize path (strip duplicate slashes & trailing slash except root)
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    # Keep query as-is (many phish use query tokens)
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _canonicalize_set(urls: Set[str]) -> Set[str]:
    """Canonicalize an entire set once for fast membership checks."""
    return {_canonicalize_url(u) for u in urls}


def _fetch_text(url, timeout=60):
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        if r.status_code == 200:
            return r.text
        print(f"Warning: {url} returned HTTP {r.status_code}")
    except Exception as e:
        print(f"Warning: failed to fetch {url}: {e}")
    return ""


def _fetch_lines(url, timeout=60):
    text = _fetch_text(url, timeout)
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def _fetch_domainlist_as_urls(url: str, source: str, limit: int, existing_canon: Set[str]):
    """Turn domain lists into crawlable URLs (root path)."""
    urls = []
    lines = _fetch_lines(url)
    print(f"{source} domains: {len(lines)} lines")
    for dom in lines:
        # domains may include leading "0.0.0.0 " or AdGuard syntax; keep only host
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
    """Fetch URLs from OpenPhish."""
    urls = []
    lines = _fetch_lines(OPENPHISH_URL)
    print(f"OpenPhish lines: {len(lines)}")
    for url in lines:
        c = _canonicalize_url(url)
        if c not in existing_canon:
            urls.append((c, 1, "openphish"))
            if len(urls) >= limit:
                break
    return urls


def _fetch_phishingdb(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch URLs from Phishing.Database (multiple feeds)."""
    urls = []
    for feed_url in PHISHINGDB_FEEDS:
        print(f"Fetching from {feed_url}...")
        lines = _fetch_lines(feed_url)
        print(f"Got {len(lines)} lines")
        for url in lines:
            c = _canonicalize_url(url)
            if c not in existing_canon:
                urls.append((c, 1, "phishingdb"))
                if len(urls) >= limit:
                    break
        if len(urls) >= limit:
            break
    return urls


def _fetch_phishtank_if_configured(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch URLs from PhishTank if API key is available."""
    api_key = os.environ.get("PHISHTANK_API_KEY")
    if not api_key:
        print("PhishTank: No API key, skipping")
        return []
    
    try:
        # Download PhishTank CSV if API key is provided
        url = f"http://data.phishtank.com/data/{api_key}/online-valid.csv"
        r = requests.get(url, timeout=60, headers=UA)
        if r.status_code != 200:
            print(f"PhishTank: HTTP {r.status_code}, skipping")
            return []
        
        urls = []
        reader = csv.DictReader(r.text.splitlines())
        for row in reader:
            raw = row.get("url", "").strip()
            if not raw:
                continue
            c = _canonicalize_url(raw)
            if c not in existing_canon:
                urls.append((c, 1, "phishtank"))
                if len(urls) >= limit:
                    break
        print(f"PhishTank lines: {len(urls)}")
        return urls
    except Exception as e:
        print(f"PhishTank error: {e}")
        return []


def _fetch_certpl_domains(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch domains from CERT-PL and convert to URLs."""
    return _fetch_domainlist_as_urls(CERTPL_DOMAINS, "cert-pl", limit, existing_canon)


def _fetch_blocklistproject_domains(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch domains from BlocklistProject phishing list and convert to URLs."""
    return _fetch_domainlist_as_urls(BLOCKLISTPROJECT_PHISHING, "blocklistproject", limit, existing_canon)


def fetch_phishing_urls(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch phishing URLs from all available sources."""
    all_urls = []
    sources = [
        _fetch_openphish,
        _fetch_phishingdb,
        _fetch_phishtank_if_configured,
        _fetch_certpl_domains,
        _fetch_blocklistproject_domains,
    ]
    
    per_source = max(1, limit // len(sources))
    for source_func in sources:
        try:
            source_urls = source_func(per_source, existing_canon)
            all_urls.extend(source_urls)
            # Update the canonicalized set to avoid duplicates across sources
            for url, _, _ in source_urls:
                existing_canon.add(url)
            if len(all_urls) >= limit:
                break
        except Exception as e:
            print(f"Error fetching from {source_func.__name__}: {e}")
    
    return all_urls[:limit]


def fetch_tranco_urls(limit: int, existing_canon: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch benign URLs from Tranco."""
    t = Tranco(cache=True)
    top_sites = t.list().top(1_000_000)  # Get up to 1M domains
    
    urls = []
    for rank, domain in enumerate(top_sites, 1):
        if len(urls) >= limit:
            break
        
        # Try both http and https versions
        for scheme in ["https", "http"]:
            url = f"{scheme}://{domain}/"
            c = _canonicalize_url(url)
            if c not in existing_canon:
                urls.append((c, 0, "tranco"))
                existing_canon.add(c)
                break
    
    print(f"Tranco URLs: {len(urls)}")
    return urls


def fetch_mixed_urls(target_count: int, phish_ratio: float, existing_urls: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch mixed phishing and benign URLs according to specified ratio."""
    existing_canon = _canonicalize_set(existing_urls)
    
    phish_count = int(target_count * phish_ratio)
    benign_count = target_count - phish_count
    
    print(f"Target: {phish_count} phishing + {benign_count} benign = {target_count} total")
    
    # Fetch phishing URLs
    phish_urls = fetch_phishing_urls(phish_count, existing_canon)
    print(f"Got {len(phish_urls)} phishing URLs")
    
    # Fetch benign URLs
    benign_urls = fetch_tranco_urls(benign_count, existing_canon)
    print(f"Got {len(benign_urls)} benign URLs")
    
    all_urls = phish_urls + benign_urls
    print(f"Total new URLs: {len(all_urls)}")
    return all_urls


def main():
    parser = argparse.ArgumentParser(description="Incremental crawler with mixed URL sources")
    parser.add_argument("--target-count", type=int, default=10000, 
                       help="Target number of total URLs to crawl")
    parser.add_argument("--phish-ratio", type=float, default=0.5,
                       help="Ratio of phishing URLs (0.0 to 1.0)")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="Number of concurrent crawlers")
    parser.add_argument("--timeout-s", type=float, default=10.0,
                       help="Timeout per page in seconds")
    parser.add_argument("--retries", type=int, default=3,
                       help="Number of retries per URL")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be crawled without actually crawling")
    parser.add_argument("--output", default="data/pages.jsonl",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Validation
    if not 0 <= args.phish_ratio <= 1:
        print("ERROR: phish-ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Load existing URLs
    output_path = Path(args.output)
    existing_urls = set()
    if output_path.exists():
        existing_urls = _load_existing_urls(str(output_path))
        print(f"Loaded {len(existing_urls)} existing URLs from {output_path}")
    
    # Calculate how many new URLs we need
    needed = args.target_count - len(existing_urls)
    if needed <= 0:
        print(f"Already have {len(existing_urls)} URLs (target: {args.target_count}). Nothing to do.")
        return
    
    print(f"Need {needed} more URLs to reach target of {args.target_count}")
    
    # Fetch new URLs
    new_urls = fetch_mixed_urls(needed, args.phish_ratio, existing_urls)
    
    if not new_urls:
        print("No new URLs found!")
        return
    
    print(f"Found {len(new_urls)} new URLs to crawl")
    
    if args.dry_run:
        print("DRY RUN - would crawl these URLs:")
        for url, label, source in new_urls[:10]:  # Show first 10
            print(f"  {url} (label={label}, source={source})")
        if len(new_urls) > 10:
            print(f"  ... and {len(new_urls) - 10} more")
        return
    
    # Create temporary file with new URLs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label'])
        for url, label, source in new_urls:
            writer.writerow([url, label])
        temp_file = f.name
    
    try:
        # Run crawler
        print(f"Starting crawler with {args.concurrency} workers...")
        asyncio.run(crawl(
            input_csv=temp_file,
            out_jsonl=str(output_path),
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            block_assets=True,
            capture_external_js=True,
            retries=args.retries,
            resume=True,  # Always resume to append to existing file
            tls_timeout=10.0,
            dns_timeout=5.0,
            mobile_profile=False,
            use_gpu=False
        ))
        
        # Verify results
        if output_path.exists():
            final_urls = _load_existing_urls(str(output_path))
            print(f"Crawling complete! Total URLs: {len(final_urls)}")
        else:
            print("Warning: Output file not created")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass


if __name__ == "__main__":
    main()
