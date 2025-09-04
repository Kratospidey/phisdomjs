#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
import asyncio
import csv
import hashlib
import time
import os
import json
from typing import Any, Dict, List, Optional

import tldextract
from phisdom.data.schema import PageRecord, ScriptItem, append_jsonl
from phisdom.data.normalize import normalize_dom, extract_scripts

# Playwright is optional until running the crawler
try:  # pragma: no cover
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover
    async_playwright = None  # type: ignore

# Optional fast JSON
try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


def _hash_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


async def fetch_page(
    context,
    url: str,
    *,
    label: int,
    source: str,
    timeout_s: float = 2.0,
    disable_after_load: bool = False,
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
    except Exception as e:
        # Ensure we close the page and propagate the error so the worker can decide on retries
        try:
            await page.close()
        finally:
            raise e

    offline_enabled = False
    if disable_after_load:
        try:
            await context.set_offline(True)
            offline_enabled = True
        except Exception:
            pass

    try:
        html = await page.content()
    except Exception:
        html = ""
    finally:
        if offline_enabled:
            try:
                await context.set_offline(False)
            except Exception:
                pass

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


async def worker(queue: asyncio.Queue, out_path: str, write_lock: asyncio.Lock, context, capture_external_js: bool, retries: int, base_timeout_s: float):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        url, label, source, timeout_s = item
        last_err: Optional[str] = None
        try:
            print(f"[FETCH] {url} (label={label}, src={source})", flush=True)
            attempt = 0
            rec: Optional[PageRecord] = None
            while True:
                try:
                    rec = await fetch_page(
                        context,
                        url,
                        label=label,
                        source=source,
                        timeout_s=(timeout_s if attempt == 0 else min(base_timeout_s * (2 ** attempt), 15.0)),
                        capture_external_js=capture_external_js,
                    )
                    last_err = None
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    rec = None

                if rec is not None:
                    # Serialize writes to avoid interleaved lines
                    async with write_lock:
                        append_jsonl(rec, out_path)
                    print(f"[OK]    {url}", flush=True)
                    break

                if attempt < max(0, retries):
                    attempt += 1
                    new_timeout = min(base_timeout_s * (2 ** attempt), 15.0)
                    print(f"[RETRY {attempt}] {url} with timeout={new_timeout:.1f}s", flush=True)
                    continue
                else:
                    reason = last_err or "no result"
                    print(f"[SKIP]  {url} - {reason}", flush=True)
                    break
        except Exception as e:
            # Never let exceptions prevent task_done(), or crawl will hang
            print(f"[ERR]   {url}: {e}", flush=True)
        finally:
            queue.task_done()


def _load_existing_urls(path: str) -> set[str]:
    urls: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    obj = orjson.loads(line)
                else:
                    obj = json.loads(line)
                u = obj.get("url")
                if isinstance(u, str):
                    urls.add(u)
            except Exception:
                # Ignore malformed lines
                continue
    return urls


async def crawl(input_csv: str, out_jsonl: str, concurrency: int, timeout_s: float, block_assets: bool, capture_external_js: bool, retries: int, resume: bool):
    queue: asyncio.Queue = asyncio.Queue()

    # If resuming and output exists, load existing URLs to skip
    existing: set[str] = set()
    if resume and os.path.exists(out_jsonl):
        try:
            existing = _load_existing_urls(out_jsonl)
            print(f"[RESUME] Loaded {len(existing)} existing URLs from {out_jsonl}", flush=True)
        except Exception as e:
            print(f"[RESUME] Could not load existing URLs from {out_jsonl}: {e}", flush=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        enq = 0
        skipped = 0
        for row in reader:
            url = row["url"].strip()
            label = int(row.get("label", 0))
            source = row.get("source", "unknown")
            if existing and url in existing:
                skipped += 1
                continue
            queue.put_nowait((url, label, source, timeout_s))
            enq += 1
    if resume:
        print(f"[QUEUE] Enqueued {enq} URLs (skipped {skipped} existing)", flush=True)

    if async_playwright is None:
        raise RuntimeError("playwright not installed. Install with `pip install playwright` and run `playwright install chromium`")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ])

        async def setup_context():
            # Use a realistic desktop UA and ignore HTTPS errors; also add small stealth tweaks
            realistic_ua = (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.6478.127 Safari/537.36"
            )
            ctx = await browser.new_context(user_agent=realistic_ua, locale="en-US", ignore_https_errors=True)
            # Basic stealth: mask webdriver and common props
            await ctx.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """
            )
            # Block heavy assets to speed up crawl if requested
            if block_assets:
                async def route_handler(route):  # type: ignore
                    rtype = route.request.resource_type
                    # Keep stylesheets to avoid breaking sites; block heavy assets only
                    if rtype in ("image", "media", "font"):
                        await route.abort()
                    else:
                        await route.continue_()
                await ctx.route("**/*", route_handler)
            return ctx

        contexts = [await setup_context() for _ in range(concurrency)]
        write_lock = asyncio.Lock()
        workers = [asyncio.create_task(
            worker(queue, out_jsonl, write_lock, contexts[i], capture_external_js, retries, timeout_s)
        ) for i in range(concurrency)]

        try:
            # Fill queue and then signal termination
            await queue.join()
        finally:
            for _ in workers:
                queue.put_nowait(None)
            # Ensure we wait for workers even on cancellation; suppress exceptions
            await asyncio.gather(*workers, return_exceptions=True)

            # Cleanup
            for ctx in contexts:
                try:
                    await ctx.close()
                except Exception:
                    pass
            try:
                await browser.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Crawl pages to JSONL dataset using Playwright")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: url,label,source")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--block-assets", action="store_true", help="Block images/fonts/media to speed up crawling")
    parser.add_argument("--no-external-js", action="store_true", help="Do not fetch external JS bodies (faster)")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries per URL on failure (default: 1)")
    # Enable resume by default; allow disabling with --no-resume
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        bool_action = None  # type: ignore
    if bool_action is not None:
        parser.add_argument("--resume", action=bool_action, default=True, help="Skip URLs already present in out-jsonl (default: True)")
    else:
        parser.add_argument("--resume", action="store_true", help="Skip URLs already present in out-jsonl")
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
        retries=max(0, args.retries),
        resume=bool(getattr(args, "resume", True)),
    ))


if __name__ == "__main__":
    main()
