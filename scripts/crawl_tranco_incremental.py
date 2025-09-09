#!/usr/bin/env python3
"""
Incremental crawler that:
1. Uses existing crawled data if available
2. Fetches URLs directly from Tranco (benign) and OpenPhish/PhishTank (phishing)
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

# Import existing crawler functionality
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from tranco import Tranco
    from crawl_playwright import crawl, _load_existing_urls
except ImportError as e:
    print(f"ERROR: Could not import crawler functions: {e}")
    print("Make sure you're in the phisdom directory and have installed: pip install tranco dnspython")
    sys.exit(1)


def load_existing_crawled_urls(pages_jsonl: str) -> Set[str]:
    """Load URLs that have already been crawled"""
    if not os.path.exists(pages_jsonl):
        return set()
    
    try:
        return _load_existing_urls(pages_jsonl)
    except Exception as e:
        print(f"Warning: Could not load existing URLs from {pages_jsonl}: {e}")
        return set()


def fetch_phishing_urls(needed_count: int, existing_urls: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch phishing URLs from OpenPhish and PhishTank feeds."""
    print(f"Fetching phishing URLs (need {needed_count})...")
    
    phish_urls = []
    
    # Try OpenPhish first
    try:
        print("Fetching from OpenPhish...")
        response = requests.get("https://openphish.com/feed.txt", timeout=30)
        if response.status_code == 200:
            urls = response.text.strip().split('\n')
            for url in urls:
                url = url.strip()
                if url and url not in existing_urls:
                    phish_urls.append((url, 1, "openphish"))
                    if len(phish_urls) >= needed_count:
                        break
            print(f"Got {len(phish_urls)} URLs from OpenPhish")
    except Exception as e:
        print(f"Failed to fetch OpenPhish: {e}")
    
    # If we still need more, try PhishTank (requires API key)
    if len(phish_urls) < needed_count:
        try:
            api_key = os.getenv('PHISHTANK_APP_KEY')
            if api_key:
                print("Fetching from PhishTank...")
                response = requests.post(
                    "http://data.phishtank.com/data/{}/online-valid.json".format(api_key),
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    for entry in data:
                        url = entry.get('url', '').strip()
                        if url and url not in existing_urls and len(phish_urls) < needed_count:
                            phish_urls.append((url, 1, "phishtank"))
                    print(f"Got {len(phish_urls)} total URLs including PhishTank")
            else:
                print("PHISHTANK_APP_KEY not set, skipping PhishTank")
        except Exception as e:
            print(f"Failed to fetch PhishTank: {e}")
    
    return phish_urls[:needed_count]

def fetch_tranco_urls(needed_count: int, existing_urls: Set[str]) -> List[Tuple[str, int, str]]:
    """Fetch benign URLs from Tranco that are not in existing_urls."""
    print(f"Fetching benign URLs from Tranco (need {needed_count})...")
    
    t = Tranco(cache=True, cache_dir='.tranco')
    latest_list = t.list()
    print(f"Retrieved Tranco list ID: {latest_list.list_id}")
    
    # Calculate how many to fetch from Tranco (up to 1M available)
    buffer_factor = 1.5
    fetch_limit = min(int(needed_count * buffer_factor), 1000000)
    print(f"Fetching Tranco ranks 1-{fetch_limit}...")
    
    top_domains = latest_list.top(fetch_limit)
    
    new_urls = []
    for domain in top_domains:
        if len(new_urls) >= needed_count:
            break
        
        url = f"http://{domain}"
        if url not in existing_urls:
            new_urls.append((url, 0, "tranco"))  # label=0 for benign
    
    print(f"Found {len(new_urls)} new benign URLs from Tranco")
    return new_urls

def fetch_mixed_urls(target_count: int, existing_urls: Set[str], phish_ratio: float = 0.5) -> List[Tuple[str, int, str]]:
    """Fetch mixed URLs according to specified ratio."""
    current_count = len(existing_urls)
    need_count = target_count - current_count
    
    print(f"Current crawled URLs: {current_count}")
    print(f"Target total URLs: {target_count}")
    print(f"Need to crawl: {need_count} more URLs")
    print(f"Phishing ratio: {phish_ratio*100:.1f}%")
    
    if need_count <= 0:
        print("Target already reached!")
        return []
    
    # Calculate how many phish vs benign we need
    need_phish = int(need_count * phish_ratio)
    need_benign = need_count - need_phish
    
    print(f"Will fetch: {need_phish} phishing + {need_benign} benign URLs")
    
    all_urls = []
    
    # Fetch phishing URLs
    if need_phish > 0:
        phish_urls = fetch_phishing_urls(need_phish, existing_urls)
        all_urls.extend(phish_urls)
        # Update existing_urls to prevent overlap
        for url, _, _ in phish_urls:
            existing_urls.add(url)
    
    # Fetch benign URLs
    if need_benign > 0:
        benign_urls = fetch_tranco_urls(need_benign, existing_urls)
        all_urls.extend(benign_urls)
    
    print(f"Total URLs fetched: {len(all_urls)} ({sum(1 for _, label, _ in all_urls if label == 1)} phish, {sum(1 for _, label, _ in all_urls if label == 0)} benign)")
    return all_urls


def create_temp_seed_file(urls: List[Tuple[str, int, str]], temp_file: str):
    """Create temporary CSV seed file for the crawler"""
    with open(temp_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label', 'source'])
        for url, label, source in urls:
            writer.writerow([url, label, source])


async def main():
    parser = argparse.ArgumentParser(description="Incremental Tranco crawler")
    parser.add_argument("--target-count", type=int, default=30000, 
                       help="Target total number of crawled URLs")
    parser.add_argument("--pages-jsonl", default="data/pages.jsonl",
                       help="Output JSONL file with crawled pages")
    parser.add_argument("--tranco-cache-dir", default=".tranco",
                       help="Cache directory for Tranco lists")
    parser.add_argument("--concurrency", type=int, default=12,
                       help="Number of concurrent crawlers")
    parser.add_argument("--timeout-s", type=float, default=8.0,
                       help="Timeout per URL in seconds")
    parser.add_argument("--retries", type=int, default=4,
                       help="Number of retries per URL")
    parser.add_argument("--tls-timeout", type=float, default=3.0,
                       help="TLS timeout in seconds")
    parser.add_argument("--dns-timeout", type=float, default=2.0,
                       help="DNS timeout in seconds")
    parser.add_argument("--mobile-profile", action="store_true",
                       help="Use mobile browser profile")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.pages_jsonl), exist_ok=True)
    
    # Load existing URLs
    existing_urls = load_existing_crawled_urls(args.pages_jsonl)
    print(f"Loaded {len(existing_urls)} existing URLs")
    
    # Get new URLs from Tranco
    new_urls = get_tranco_urls(args.target_count, existing_urls, args.tranco_cache_dir)
    
    if not new_urls:
        print("No new URLs to crawl!")
        return
    
    # Create temporary seed file
    temp_seed = "temp_tranco_seed.csv"
    try:
        create_temp_seed_file(new_urls, temp_seed)
        print(f"Created temporary seed file: {temp_seed}")
        
        # Run crawler with resume=True to append to existing file
        await crawl(
            input_csv=temp_seed,
            out_jsonl=args.pages_jsonl,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            block_assets=True,
            capture_external_js=False,
            retries=args.retries,
            resume=True,  # This will append to existing file
            tls_timeout=args.tls_timeout,
            dns_timeout=args.dns_timeout,
            mobile_profile=args.mobile_profile,
            use_gpu=args.gpu
        )
        
        # Check final count
        final_urls = load_existing_crawled_urls(args.pages_jsonl)
        print(f"Final crawled URL count: {len(final_urls)}")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_seed):
            os.remove(temp_seed)
            print(f"Cleaned up temporary file: {temp_seed}")


if __name__ == "__main__":
    asyncio.run(main())
