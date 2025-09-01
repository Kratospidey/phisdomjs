#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
import asyncio
import csv
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import tldextract
from phisdom.data.schema import PageRecord, ScriptItem, append_jsonl
from phisdom.data.normalize import normalize_dom, extract_scripts

# Playwright is optional until running the crawler
try:  # pragma: no cover
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover
    async_playwright = None  # type: ignore


def _hash_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


async def fetch_page(
    context,
    url: str,
    *,
    label: int,
    source: str,
    timeout_s: float = 2.0,
    disable_after_load: bool = True,
    capture_external_js: bool = True,
) -> Optional[PageRecord]:
    js_responses: Dict[str, str] = {}
    main_headers: Dict[str, Any] = {}
    main_response_url: Optional[str] = None

    async def on_response(response):  # type: ignore
        if not capture_external_js:
            return
        try:
            url_r = response.url
            ct = (response.headers or {}).get("content-type", "").lower()
            if main_response_url is None and response.request.resource_type == "document":
                main_headers.update(response.headers or {})
            if ("javascript" in ct) or url_r.endswith(".js"):
                text = await response.text()
                js_responses[url_r] = text
        except Exception:
            pass

    page = await context.new_page()
    page.on("response", on_response)
    page.set_default_timeout(max(1, int(timeout_s * 1000)))

    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=int(timeout_s * 1000))
        if resp is not None:
            main_headers.update(resp.headers or {})
            main_response_url = resp.url
    except Exception:
        await page.close()
        return None

    if disable_after_load:
        try:
            await context.set_offline(True)
        except Exception:
            pass

    try:
        html = await page.content()
    except Exception:
        html = ""

    await page.close()

    norm_html = normalize_dom(html)
    scripts_meta = extract_scripts(html)
    scripts: List[ScriptItem] = []
    for s in scripts_meta:
        src = s.get("src")
        inline = bool(s.get("inline"))
        text = s.get("text")
        if capture_external_js and (not inline) and src and src in js_responses:
            text = js_responses[src]
        scripts.append(ScriptItem(src=src, inline=inline, text=text, attrs=s.get("attrs", {})))

    ts = time.time()
    etld = tldextract.extract(url)
    etld1 = ".".join([p for p in [etld.domain, etld.suffix] if p])
    rec = PageRecord(
        id=_hash_id(url),
        url=url,
        etld1=etld1,
        timestamp=ts,
        source=source,
        label=int(label),
        html=norm_html,
        scripts=scripts,
        headers=main_headers,
    )
    return rec


async def worker(queue: asyncio.Queue, out_path: str, context, capture_external_js: bool):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        url, label, source, timeout_s = item
        try:
            print(f"[FETCH] {url} (label={label}, src={source})", flush=True)
            rec = await fetch_page(context, url, label=label, source=source, timeout_s=timeout_s, capture_external_js=capture_external_js)
            if rec is not None:
                append_jsonl(rec, out_path)
                print(f"[OK]    {url}", flush=True)
            else:
                print(f"[SKIP]  {url}", flush=True)
        except Exception as e:
            # Never let exceptions prevent task_done(), or crawl will hang
            print(f"[ERR]   {url}: {e}", flush=True)
        finally:
            queue.task_done()


async def crawl(input_csv: str, out_jsonl: str, concurrency: int, timeout_s: float, block_assets: bool, capture_external_js: bool):
    queue: asyncio.Queue = asyncio.Queue()

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"].strip()
            label = int(row.get("label", 0))
            source = row.get("source", "unknown")
            queue.put_nowait((url, label, source, timeout_s))

    if async_playwright is None:
        raise RuntimeError("playwright not installed. Install with `pip install playwright` and run `playwright install chromium`")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        async def setup_context():
            ctx = await browser.new_context()
            # Block heavy assets to speed up crawl if requested
            if block_assets:
                async def route_handler(route):  # type: ignore
                    rtype = route.request.resource_type
                    if rtype in ("image", "media", "font", "stylesheet"):
                        await route.abort()
                    else:
                        await route.continue_()
                await ctx.route("**/*", route_handler)
            return ctx

        contexts = [await setup_context() for _ in range(concurrency)]
        workers = [asyncio.create_task(worker(queue, out_jsonl, contexts[i], capture_external_js)) for i in range(concurrency)]

        # Fill queue and then signal termination
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

        # Cleanup
        for ctx in contexts:
            try:
                await ctx.close()
            except Exception:
                pass
        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Crawl pages to JSONL dataset using Playwright")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: url,label,source")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--block-assets", action="store_true", help="Block images/css/fonts/media to speed up crawling")
    parser.add_argument("--no-external-js", action="store_true", help="Do not fetch external JS bodies (faster)")
    args = parser.parse_args()
    # Fail fast if Playwright is not installed to avoid silent hangs
    if async_playwright is None:
        print("ERROR: Playwright is not installed in this environment. Install with:\n  pip install playwright\n  python -m playwright install chromium", file=sys.stderr)
        sys.exit(2)

    asyncio.run(crawl(
        args.input_csv,
        args.out_jsonl,
        args.concurrency,
        args.timeout_s,
        args.block_assets,
        capture_external_js=not args.no_external_js,
    ))


if __name__ == "__main__":
    main()
