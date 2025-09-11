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
import re
import math
import ssl
import socket
import contextlib
from datetime import datetime, timezone
from urllib.parse import urlparse
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import tldextract
from phisdom.data.schema import PageRecord, ScriptItem, append_jsonl
from phisdom.features import (
    extract_url_charseq,
    extract_js_charseq,
    extract_dom_graph,
    extract_text_title,
    extract_text_visible,
)
from phisdom.data.normalize import normalize_dom, extract_scripts
from bs4 import BeautifulSoup
from bs4.element import Tag

import requests
import dns.resolver  # type: ignore
from urllib.parse import urljoin

RDAP_TTL_SEC = 7 * 24 * 3600
TLS_TTL_SEC = 24 * 3600
ICON_MAX = 64 * 1024
SCRIPT_BUDGET = 128 * 1024

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    from PIL import Image  # type: ignore
    from io import BytesIO
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    BytesIO = None  # type: ignore

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

# Optional cryptography for TLS SPKI hash (best-effort)
try:  # pragma: no cover
    from cryptography import x509  # type: ignore
    from cryptography.hazmat.primitives import serialization  # type: ignore
    from cryptography.x509.oid import ExtensionOID  # type: ignore
except Exception:  # pragma: no cover
    x509 = None  # type: ignore
    serialization = None  # type: ignore
    ExtensionOID = None  # type: ignore


def _hash_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _hash64(data: bytes) -> int:
    try:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")
    except Exception:
        # Fallback: siphash-like via sha1 truncation
        return int.from_bytes(hashlib.sha1(data).digest()[:8], "big")


def _entropy(data: str) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = float(len(data))
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return round(ent, 4)


def _extract_js_string_literals(js: str) -> List[str]:
    if not js:
        return []
    # Very rough: handle single/double/backtick strings without full JS parsing
    pattern = r"(\"(?:\\.|[^\\\"])*\"|'(?:\\.|[^\\'])*'|`(?:\\.|[^\\`])*`)"
    return [m[0][1:-1] for m in re.finditer(pattern, js, flags=re.DOTALL)]


def _url_features(url: str) -> Dict[str, Any]:
    try:
        p = urlparse(url)
        host = p.hostname or ""
        is_idn = any(ord(ch) > 127 for ch in host)
        is_puny = host.startswith("xn--") or ".xn--" in host
        tld_parts = tldextract.extract(url)
        tld = tld_parts.suffix
        return {
            "tld": tld or None,
            "is_idn": bool(is_idn),
            "is_punycode": bool(is_puny),
            "has_punycode": bool(is_puny),
            "url_len": len(url),
            "num_dots": host.count('.'),
            "num_pct": url.count('%'),
            "has_at": '@' in url,
            "host_is_ip": bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host)),
            "host_hyphens": host.count('-'),
        }
    except Exception:
        return {}


def _security_headers_snapshot(headers: Dict[str, Any]) -> Dict[str, Optional[str]]:
    def get(h: str) -> Optional[str]:
        for k, v in headers.items():
            if k.lower() == h:
                return v
        return None
    return {
        "hdr_csp": get("content-security-policy"),
        "hdr_hsts": get("strict-transport-security"),
        "hdr_xfo": get("x-frame-options"),
        "hdr_refpol": get("referrer-policy"),
        "hdr_permspol": get("permissions-policy"),
        "hdr_xcto": get("x-content-type-options"),
    }


def _form_semantics(html: str, page_etld1: str, page_scheme: Optional[str]) -> Dict[str, Any]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
    except Exception:
        return {}
    forms = [f for f in soup.find_all("form") if isinstance(f, Tag)]
    pw_nodes = [n for n in soup.find_all("input", attrs={"type": re.compile("password", re.I)}) if isinstance(n, Tag)]
    pw_count = len(pw_nodes)
    email_nodes = [n for n in soup.find_all("input", attrs={"type": re.compile("email", re.I)}) if isinstance(n, Tag)]
    email_count = len(email_nodes)
    hidden_count = len([n for n in soup.find_all("input", attrs={"type": re.compile("hidden", re.I)}) if isinstance(n, Tag)])
    def _attr_str(val: Any) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            try:
                return " ".join(str(x) for x in val)
            except Exception:
                return str(val)
        return str(val)
    autocomplete_off = any(((_attr_str(f.attrs.get("autocomplete")) or "").lower() == "off") for f in forms)
    tokens = ("login", "sign in", "signin", "verify", "account", "password", "2fa")
    def _text(el):
        try:
            return el.get_text(" ").lower()
        except Exception:
            return ""
    login_tokens = sum(1 for f in forms if any(t in _text(f) for t in tokens))
    # Cross-site: any form action host etld1 different than page etld1
    cross = False
    for f in forms:
        action = _attr_str(f.attrs.get("action"))
        if not action:
            continue
        try:
            u = urlparse(action)
            if u.scheme and u.netloc:
                ext = tldextract.extract(action)
                et1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
                if et1 and page_etld1 and et1 != page_etld1:
                    cross = True
                    break
        except Exception:
            pass
    # onsubmit handlers count
    onsubmit_ct = sum(1 for f in forms if _attr_str(f.attrs.get("onsubmit")) is not None)
    # New compact form footprint and CSS signature
    def _form_fp_hash64(form: Tag) -> Optional[int]:
        try:
            names: List[str] = []
            for inp in form.find_all(["input", "select", "textarea"]):
                if not isinstance(inp, Tag):
                    continue
                t = (_attr_str(inp.get("type")) or "").strip().lower()
                n = (_attr_str(inp.get("name")) or "").strip().lower()
                names.append(f"{n}|{t}")
            if not names:
                return None
            fp = "|".join(sorted(set(names))).encode("utf-8", errors="ignore")
            return _hash64(fp)
        except Exception:
            return None

    def _form_css_sig_hash64(form: Tag) -> Optional[int]:
        try:
            classes: List[str] = []
            for sel in form.find_all(["input", "button", "a", "label"]):
                if not isinstance(sel, Tag):
                    continue
                cls = sel.get("class")
                if isinstance(cls, list):
                    classes.extend([str(c).strip().lower() for c in cls])
                elif isinstance(cls, str):
                    classes.append(cls.strip().lower())
            if not classes:
                return None
            sig = "|".join(sorted(set(classes))).encode("utf-8", errors="ignore")
            return _hash64(sig)
        except Exception:
            return None

    form_fp64: Optional[int] = None
    method_get = None
    action_xorigin = None
    action_proto_mismatch = None
    # Identify the first login-like form (with password or email)
    first_login_form: Optional[Tag] = None
    for f in forms:
        try:
            types = [str((getattr(inp, 'get', lambda *_: None)("type") or "")).lower() for inp in f.find_all("input")]
            if ("password" in types) or ("email" in types):
                first_login_form = f
                break
        except Exception:
            continue
    if first_login_form is not None:
        form_fp64 = _form_fp_hash64(first_login_form)
        # method
        m = (_attr_str(first_login_form.get("method")) or "get").strip().lower()
        method_get = (m == "get")
        # action analysis
        act = _attr_str(first_login_form.get("action")) or ""
        try:
            if act:
                u_act = urlparse(act)
                if u_act.scheme and u_act.netloc:
                    ext = tldextract.extract(act)
                    et1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
                    action_xorigin = bool(et1 and page_etld1 and et1 != page_etld1)
                    # proto mismatch compared to page scheme if known
                    if page_scheme:
                        action_proto_mismatch = (u_act.scheme.lower() != page_scheme.lower())
        except Exception:
            pass
        css_sig = _form_css_sig_hash64(first_login_form)
    else:
        css_sig = None

    return {
        "form_pw_count": pw_count,
        "form_hidden_count": hidden_count,
        "form_autocomplete_off": bool(autocomplete_off),
        "form_login_tokens": login_tokens,
        "form_cross_site": bool(cross),
        "onsubmit_handlers": onsubmit_ct,
        # New fields
        "num_pw": pw_count,
        "num_email": email_count,
        "num_hidden": hidden_count,
        "form_fp_hash64": form_fp64,
        "form_method_get": method_get,
        "action_cross_origin": action_xorigin,
        "action_proto_mismatch": action_proto_mismatch,
        "form_css_sig_hash64": css_sig,
    }


def _js_heuristics(scripts: List[ScriptItem]) -> Dict[str, Any]:
    texts: List[str] = []
    for s in scripts:
        if s.text:
            texts.append(s.text)
    all_js = "\n".join(texts)
    # Counts
    eval_ct = len(re.findall(r"\beval\s*\(", all_js)) + len(re.findall(r"new\s+Function\s*\(", all_js))
    atob_ct = len(re.findall(r"\batob\s*\(", all_js)) + len(re.findall(r"\bfromCharCode\b", all_js))
    b64_blob_ct = len(re.findall(r"[A-Za-z0-9+/]{80,}={0,2}", all_js))
    keylog_ct = len(re.findall(r"addEventListener\s*\(\s*['\"](?:key(?:down|up)|paste)['\"]", all_js))
    # Entropy on string literals if available; else on all JS concatenated
    string_literals: List[str] = []
    for js in texts:
        string_literals.extend(_extract_js_string_literals(js))
    ent_src = "\n".join(string_literals) if string_literals else all_js
    return {
        "js_entropy": _entropy(ent_src[:200000]),  # cap for cost
        "js_eval_ct": eval_ct,
        "js_atob_ct": atob_ct,
        "js_b64_blob_ct": b64_blob_ct,
        "js_keylog_listeners": keylog_ct,
    }


def _js_micro_counters(scripts: List[ScriptItem], budget: int = SCRIPT_BUDGET) -> Dict[str, Any]:
    try:
        code_parts: List[str] = []
        used = 0
        for s in scripts:
            if used >= budget:
                break
            t = s.text or ""
            if not t:
                continue
            remain = budget - used
            chunk = t[:remain]
            code_parts.append(chunk)
            used += len(chunk)
        code = "\n".join(code_parts)
        if not code:
            return {"js_eval_like": 0, "js_fromcharcode": 0, "js_atob": 0, "js_hex_ratio": 0, "js_hi_entropy_ratio": 0}
        eval_like = len(re.findall(r"\b(eval|Function)\b\s*\(", code)) + len(re.findall(r"setTimeout\s*\(\s*['\"]", code))
        fromcc = len(re.findall(r"String\\.fromCharCode", code))
        atobc = len(re.findall(r"\batob\b", code))
        hex_hits = len(re.findall(r"\\\\x[0-9A-Fa-f]{2}", code))
        hex_ratio = int(255 * (hex_hits / max(1, len(code))))
        # High-entropy window share (rough): slide 1k window, count >7.0
        step = 512
        win = 1024
        hi = 0
        total = 0
        for i in range(0, len(code), step):
            chunk = code[i:i+win]
            if not chunk:
                continue
            total += 1
            if _entropy(chunk) > 7.0:
                hi += 1
        hi_ratio = int(255 * (hi / max(1, total)))
        return {
            "js_eval_like": int(eval_like) & 0xFF,
            "js_fromcharcode": int(fromcc) & 0xFF,
            "js_atob": int(atobc) & 0xFF,
            "js_hex_ratio": int(hex_ratio) & 0xFF,
            "js_hi_entropy_ratio": int(hi_ratio) & 0xFF,
        }
    except Exception:
        return {"js_eval_like": None, "js_fromcharcode": None, "js_atob": None, "js_hex_ratio": None, "js_hi_entropy_ratio": None}


def _fingerprint_flags(html: str, scripts: List[ScriptItem]) -> Dict[str, bool]:
    hay = (html or "") + "\n" + "\n".join([s.text or "" for s in scripts])
    low = hay.lower()
    fp_canvas = ("getcontext('2d'" in low) or ("canvas.toDataURL".lower() in low) or ("measuretext" in low)
    fp_webgl = ("webglrenderingcontext" in low) or ("getcontext('webgl'" in low) or ("getcontext('experimental-webgl'" in low)
    fp_audio = ("audiocontext" in low) or ("offlineaudiocontext" in low)
    fp_font = ("measuretext" in low) or ("font" in low and "enumerat" in low)
    fp_rtc = ("rtcpeerconnection" in low) or ("getusermedia" in low) or ("enumeratedevices" in low)
    return {
        "fp_canvas": bool(fp_canvas),
        "fp_webgl": bool(fp_webgl),
        "fp_audio": bool(fp_audio),
        "fp_font_enum": bool(fp_font),
        "fp_webrtc": bool(fp_rtc),
    }


def _phash_and_colors(png_bytes: bytes) -> Tuple[Optional[int], Optional[List[int]]]:
    if Image is None or np is None or BytesIO is None:
        return None, None
    try:
        img = Image.open(BytesIO(png_bytes)).convert('L').resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        # DCT
        dct = np.fft.fft2(arr)
        mag = np.abs(dct)[:8, :8]
        med = np.median(mag)
        bits = (mag > med).astype(np.uint8)
        # pack 64 bits to int
        ph = 0
        for b in bits.flatten():
            ph = (ph << 1) | int(b)
        # dominant colors (from original small RGB)
        img_rgb = Image.open(BytesIO(png_bytes)).convert('RGB').resize((32, 32))
        arr_rgb = np.asarray(img_rgb).reshape(-1, 3)
        # K-means lite: use bincount on quantized colors
        q = (arr_rgb // 64).astype(np.int16)  # 4 levels per channel
        idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
        counts = np.bincount(idx, minlength=64)
        top3 = counts.argsort()[-3:][::-1].tolist()
        return int(ph), [int(x) for x in top3]
    except Exception:
        return None, None


def _tls_info_for_host(host: str, timeout: float = 3.0) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if not host:
        return info
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cipher = ssock.cipher()
                info["tls_version"] = getattr(ssock, "version", lambda: None)()
                # Legacy high-level cert dict for compatibility
                cert_legacy = ssock.getpeercert()
                if cert_legacy:
                    info["cert_issuer"] = ",".join("=".join(x) for t in cert_legacy.get("issuer", []) for x in t)
                    not_before_any = cert_legacy.get("notBefore")
                    not_after_any = cert_legacy.get("notAfter")
                    def _parse_dt(s: Any) -> Optional[datetime]:
                        if not isinstance(s, str):
                            return None
                        for fmt in ("%b %d %H:%M:%S %Y %Z", "%Y%m%d%H%M%SZ"):
                            try:
                                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                            except Exception:
                                continue
                        return None
                    nb = _parse_dt(not_before_any)
                    if nb:
                        info["cert_age_days"] = int((datetime.now(timezone.utc) - nb).total_seconds() / 86400)
                    san = cert_legacy.get("subjectAltName") or []
                    info["san_count"] = int(len(san)) if san else 0
                if cipher:
                    info["key_type"] = cipher[0]
                # Cryptography path for compact TLS features
                if x509 is not None and serialization is not None and ExtensionOID is not None:
                    try:
                        der = ssock.getpeercert(True)
                        if not der:
                            raise ValueError("no DER cert")
                        cert = x509.load_der_x509_certificate(der)
                        nb2 = cert.not_valid_before
                        if isinstance(nb2, datetime):
                            nb2 = nb2.replace(tzinfo=timezone.utc)
                            info["tls_not_before_days"] = int((datetime.now(timezone.utc) - nb2).total_seconds() / 86400)
                        # SAN count (best-effort)
                        san_count_precise: Optional[int] = None
                        try:
                            ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                            val = getattr(ext, "value", None)
                            gnames = getattr(val, "_general_names", None)
                            if gnames is not None:
                                san_count_precise = int(len(gnames))
                        except Exception:
                            san_count_precise = None
                        if san_count_precise is not None:
                            info["tls_san_count"] = san_count_precise
                        spki = cert.public_key().public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
                        info["tls_issuer_spki_hash64"] = _hash64(spki)
                    except Exception:
                        pass
    except Exception:
        pass
    return info


def _dhash64(png_or_bytes: bytes) -> Optional[int]:
    if Image is None or np is None or BytesIO is None:
        return None
    try:
        img = Image.open(BytesIO(png_or_bytes)).convert('L').resize((9, 8))
        arr = np.asarray(img, dtype=np.int16)
        diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()
        h = 0
        for b in diff:
            h = (h << 1) | int(b)
        return int(h)
    except Exception:
        return None


def _bitb_and_qr_flags(html: str) -> Dict[str, bool]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
    except Exception:
        return {"bitb_like_modal": False, "qr_flag": False}
    text = soup.get_text(" ").lower() if soup else ""
    # BitB heuristic: modal/container div mimicking browser chrome
    bitb = False
    def _attr_to_str(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, (list, tuple)):
            return " ".join(str(x) for x in val)
        return str(val)
    for d in soup.find_all(["div", "section", "iframe"]):
        if not isinstance(d, Tag):
            continue
        cls = _attr_to_str(d.attrs.get("class")).lower()
        style = _attr_to_str(d.attrs.get("style")).lower()
        if any(tok in cls for tok in ("modal", "dialog", "popup")) and any(tok in style for tok in ("position:fixed", "z-index")):
            # presence of close button-like content
            if any(sym in d.get_text(" ") for sym in ["×", "x", "close"]):
                bitb = True
                break
    # QR heuristic: alt text or file names mentioning qr
    qr = False
    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue
        alt = str(img.get("alt") or "").lower()
        src = str(img.get("src") or "").lower()
        if "qr" in alt or "qrcode" in alt or "qr_code" in src:
            qr = True
            break
    if not qr and ("scan qr" in text or "scan the qr" in text or "qr code" in text):
        qr = True
    return {"bitb_like_modal": bool(bitb), "qr_flag": bool(qr)}


_RDAP_CACHE_PATH = os.path.join("data", "rdap_cache.json")
_RDAP_CACHE: Dict[str, Any] = {}
_DNS_CACHE_PATH = os.path.join("data", "dns_cache.json")
_DNS_CACHE: Dict[str, Any] = {}
_TLS_CACHE: Dict[str, Any] = {}
_TLS_CACHE_PATH = os.path.join("data", "tls_cache.json")


def _load_rdap_cache() -> None:
    global _RDAP_CACHE
    try:
        if os.path.exists(_RDAP_CACHE_PATH):
            with open(_RDAP_CACHE_PATH, "r", encoding="utf-8") as f:
                _RDAP_CACHE = json.load(f)
    except Exception:
        _RDAP_CACHE = {}


def _save_rdap_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_RDAP_CACHE_PATH), exist_ok=True)
        with open(_RDAP_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_RDAP_CACHE, f)
    except Exception:
        pass


def _rdap_lookup(domain: str, timeout: float = 3.5) -> Dict[str, Any]:
    if not domain:
        return {}
    if not _RDAP_CACHE:
        _load_rdap_cache()
    now_ts = int(time.time())
    if domain in _RDAP_CACHE:
        cached = _RDAP_CACHE[domain]
        ts = int(cached.get("_ts", 0))
        if ts and (now_ts - ts) < RDAP_TTL_SEC:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    data: Dict[str, Any] = {}
    try:
        # rdap.org is a community aggregator
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout)
        if r.ok:
            j = r.json()
            events = j.get("events", [])
            created = None
            updated = None
            for e in events:
                if e.get("eventAction") in ("registration", "domain registration"):
                    created = e.get("eventDate")
                if e.get("eventAction") in ("last changed", "last update of RDAP database", "last changed"):
                    updated = e.get("eventDate")
            def _days_ago(iso: Optional[str]) -> Optional[int]:
                if not iso:
                    return None
                try:
                    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
                    return int((datetime.now(timezone.utc) - dt).total_seconds() / 86400)
                except Exception:
                    return None
            data["dns_created_days_ago"] = _days_ago(created)
            data["dns_updated_days_ago"] = _days_ago(updated)
            data["registrar"] = (j.get("registrar", {}) or {}).get("name") or None
            # nameserver count
            ns_list = j.get("nameservers", []) or []
            data["ns_count"] = len(ns_list) if ns_list else None
            # TTL/MX not available via RDAP; leave None in this lightweight pass
            data["mx_present"] = None
            data["ttl_min"] = None
            data["ttl_mean"] = None
            # Compact RDAP fields
            data["rdap_age_days"] = data.get("dns_created_days_ago")
            registrar_name = data.get("registrar") or ""
            try:
                # include IANA ID if available
                iana_id = str((j.get("registrar", {}) or {}).get("iana_id") or "")
            except Exception:
                iana_id = ""
            data["rdap_registrar_hash64"] = _hash64((registrar_name + iana_id).encode("utf-8", errors="ignore")) if registrar_name or iana_id else None
            data["rdap_ns_count"] = int(data.get("ns_count") or 0)
            js = json.dumps(j).lower()
            data["rdap_has_privacy"] = ("privacy" in js) or ("redact" in js)
    except Exception:
        pass
    # Store with timestamp
    tmp = dict(data)
    tmp["_ts"] = now_ts
    _RDAP_CACHE[domain] = tmp
    _save_rdap_cache()
    return data


def _load_dns_cache() -> None:
    global _DNS_CACHE
    try:
        if os.path.exists(_DNS_CACHE_PATH):
            with open(_DNS_CACHE_PATH, "r", encoding="utf-8") as f:
                _DNS_CACHE = json.load(f)
    except Exception:
        _DNS_CACHE = {}


def _save_dns_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_DNS_CACHE_PATH), exist_ok=True)
        with open(_DNS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_DNS_CACHE, f)
    except Exception:
        pass


def _dns_ttl_mx(domain: str, timeout: float = 2.0) -> Dict[str, Any]:
    if not domain:
        return {}
    if not _DNS_CACHE:
        _load_dns_cache()
    if domain in _DNS_CACHE:
        return _DNS_CACHE[domain]
    out: Dict[str, Any] = {"ttl_min": None, "ttl_mean": None, "mx_present": None}
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        ttls: List[int] = []
        for rrtype in ("A", "AAAA"):
            try:
                ans = resolver.resolve(domain, rrtype)
                if ans.rrset is not None and hasattr(ans.rrset, "ttl"):
                    ttls.append(int(getattr(ans.rrset, "ttl", 0)))
            except Exception:
                continue
        out["ttl_min"] = min(ttls) if ttls else None
        out["ttl_mean"] = (sum(ttls) / len(ttls)) if ttls else None
        try:
            mx_ans = resolver.resolve(domain, "MX")
            out["mx_present"] = bool(mx_ans)
        except Exception:
            out["mx_present"] = False
    except Exception:
        pass
    _DNS_CACHE[domain] = out
    _save_dns_cache()
    return out


def _tls_lookup(host: str, timeout: float = 3.0) -> Dict[str, Any]:
    if not host:
        return {}
    now_ts = int(time.time())
    # Try load from disk if empty
    try:
        if not _TLS_CACHE and os.path.exists(_TLS_CACHE_PATH):
            with open(_TLS_CACHE_PATH, "r", encoding="utf-8") as f:
                _TLS_CACHE.update(json.load(f))
    except Exception:
        pass
    ent = _TLS_CACHE.get(host)
    if ent and (now_ts - int(ent.get("_ts", 0))) < TLS_TTL_SEC:
        return {k: v for k, v in ent.items() if not k.startswith("_")}
    info = _tls_info_for_host(host, timeout=timeout)
    tmp = dict(info)
    tmp["_ts"] = now_ts
    _TLS_CACHE[host] = tmp
    try:
        os.makedirs(os.path.dirname(_TLS_CACHE_PATH), exist_ok=True)
        with open(_TLS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_TLS_CACHE, f)
    except Exception:
        pass
    return info


def _http_get_bytes_limited(url: str, max_bytes: int = ICON_MAX, timeout: float = 3.0) -> Optional[bytes]:
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if not r.ok:
                return None
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    break
                need = max_bytes - len(buf)
                if need <= 0:
                    break
                buf.extend(chunk[:need])
            return bytes(buf)
    except Exception:
        return None


async def fetch_page(
    context,
    url: str,
    *,
    label: int,
    source: str,
    timeout_s: float = 2.0,
    disable_after_load: bool = False,
    capture_external_js: bool = True,
    tls_timeout: float = 3.0,
    dns_timeout: float = 2.0,
) -> Optional[PageRecord]:
    js_responses: Dict[str, str] = {}
    main_headers: Dict[str, Any] = {}
    main_response_url: Optional[str] = None
    # Request/response tallies
    req_counts: Counter = Counter()
    etld1_contacts: Counter = Counter()
    document_resp_times: List[float] = []
    document_urls: List[str] = []

    async def on_response(response):  # type: ignore
        if not capture_external_js:
            return
        try:
            url_r = response.url
            ct = (response.headers or {}).get("content-type", "").lower()
            if main_response_url is None and response.request.resource_type == "document":
                main_headers.update(response.headers or {})
                document_resp_times.append(time.time())
            if response.request.resource_type == "document":
                document_urls.append(url_r)
            if ("javascript" in ct) or url_r.endswith(".js"):
                text = await response.text()
                js_responses[url_r] = text
        except Exception:
            pass

    def on_request(request):  # type: ignore
        try:
            rtype = request.resource_type
            req_counts[rtype] += 1
            u = request.url
            ext = tldextract.extract(u)
            et1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
            if et1:
                etld1_contacts[et1] += 1
        except Exception:
            pass

    page = await context.new_page()
    page.on("response", on_response)
    page.on("request", on_request)
    page.set_default_timeout(max(1, int(timeout_s * 1000)))

    # Commit-first navigation: quick initial load; then optional scheme flip
    try:
        resp = await page.goto(url, wait_until="commit", timeout=int(timeout_s * 1000))
    except Exception as e1:
        alt_url: Optional[str] = None
        if url.lower().startswith("http://"):
            alt_url = "https://" + url[7:]
        elif url.lower().startswith("https://"):
            alt_url = "http://" + url[8:]
        if alt_url and alt_url != url:
            try:
                print(f"[FALLBACK] switching scheme and retrying commit: {alt_url}", flush=True)
                resp = await page.goto(alt_url, wait_until="commit", timeout=int(min(timeout_s, 4.0) * 1000))
                url = alt_url
            except Exception:
                try:
                    await page.close()
                finally:
                    raise e1
        else:
            try:
                await page.close()
            finally:
                raise e1

    if resp is not None:
        try:
            main_headers.update(resp.headers or {})
            main_response_url = resp.url
        except Exception:
            pass

    offline_enabled = False
    if disable_after_load:
        try:
            await context.set_offline(True)
            offline_enabled = True
        except Exception:
            pass

    # Content and small screenshot (for pHash/colors) before toggling offline
    png_bytes: Optional[bytes] = None
    page_title: str = ""
    iframe_form_count: Optional[int] = None
    # Defaults for instrumented key-listener counters
    instr_key_total: Optional[int] = None
    instr_key_pw: Optional[int] = None
    try:
        html = await page.content()
        try:
            page_title = await page.title()
        except Exception:
            page_title = ""
        # Read instrumented key-listener counters if init script ran
        try:
            pd = await page.evaluate(
                """
                () => {
                    try{
                        if (window.__pd){
                            return { keyListeners: Number(window.__pd.keyListeners||0), keyPw: Number(window.__pd.keyPw||0) };
                        }
                    }catch(e){}
                    return null;
                }
                """
            )
            if isinstance(pd, dict):
                instr_key_total = int(pd.get("keyListeners") or 0)
                instr_key_pw = int(pd.get("keyPw") or 0)
        except Exception:
            instr_key_total = None
            instr_key_pw = None
        # Count forms in iframes safely in-page
        try:
            iframe_form_count = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('iframe')).reduce((c, f) => {
                  try { return c + (f.contentWindow?.document?.querySelectorAll('form').length || 0) } catch(e){ return c }
                }, 0)
                """
            )
        except Exception:
            iframe_form_count = None
        try:
            png_bytes = await page.screenshot(type="png", full_page=False)
        except Exception:
            png_bytes = None
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
    # URL & redirect chain
    final_url = main_response_url or url
    rec.url_final = final_url
    # Phase 1: raw URL and URL char sequence
    rec.url_raw = url
    try:
        rec.url_charseq = extract_url_charseq(final_url or url)
    except Exception:
        rec.url_charseq = None
    # Approximate redirect hops by counting document responses
    if document_resp_times:
        hops = max(0, len(document_resp_times) - 1)
        rec.redirect_hops = hops
        if len(document_resp_times) > 1:
            deltas = [int((document_resp_times[i] - document_resp_times[i-1]) * 1000) for i in range(1, len(document_resp_times))]
            rec.redirect_max_ms = float(max(deltas)) if deltas else None
    # New redirect sketch fields
    try:
        rec.redir_hops = max(0, len(document_urls) - 1) if document_urls else 0
        # cross-host hops: count transitions where host changes
        hosts = []
        for udoc in (document_urls or []):
            try:
                hosts.append(urlparse(udoc).netloc)
            except Exception:
                hosts.append(None)
        cross = 0
        for i in range(1, len(hosts)):
            if hosts[i] and hosts[i-1] and hosts[i] != hosts[i-1]:
                cross += 1
        rec.redir_cross_host = cross
    except Exception:
        pass
    # URL lexical features
    for k, v in _url_features(final_url).items():
        setattr(rec, k, v)

    # Security headers snapshot
    for k, v in _security_headers_snapshot(main_headers).items():
        setattr(rec, k, v)

    # Third-party request graph
    try:
        total_requests = sum(req_counts.values()) or 1
        # Map resource types to our buckets
        rec.req_counts_script = int(req_counts.get("script", 0))
        rec.req_counts_css = int(req_counts.get("stylesheet", 0))
        rec.req_counts_xhr = int(req_counts.get("xhr", 0) + req_counts.get("fetch", 0))
        rec.req_counts_img = int(req_counts.get("image", 0))
        rec.req_unique_etld1 = int(len(etld1_contacts)) if etld1_contacts else 0
        other = sum(c for d, c in etld1_contacts.items() if d != etld1)
        rec.req_thirdparty_ratio = round(other / float(sum(etld1_contacts.values()) or 1), 4)
    except Exception:
        pass

    # Form & login semantics
    try:
        for k, v in _form_semantics(norm_html, etld1, urlparse(final_url).scheme if final_url else None).items():
            setattr(rec, k, v)
    except Exception:
        pass
    # Iframe-related additions
    try:
        rec.top_form_count = int(len(BeautifulSoup(norm_html or "", "html.parser").find_all("form")))
    except Exception:
        pass
    if iframe_form_count is not None:
        rec.iframe_form_count = int(iframe_form_count)
        rec.iframe_login = bool((rec.iframe_form_count or 0) > 0 and (rec.num_pw or 0) > 0)

    # JS heuristics & fingerprint flags
    try:
        for k, v in _js_heuristics(scripts).items():
            setattr(rec, k, v)
        # Micro counters (u8-scaled)
        for k, v in _js_micro_counters(scripts).items():
            setattr(rec, k, v)
        for k, v in _fingerprint_flags(norm_html, scripts).items():
            setattr(rec, k, v)
    except Exception:
        pass
    # Key listener compact fields from existing heuristics
    try:
        rec.key_listeners_total = int(rec.js_keylog_listeners or 0)
        rec.key_listener_pw = bool((rec.num_pw or 0) > 0 and (rec.key_listeners_total or 0) > 0)
    except Exception:
        pass
    # Prefer instrumented counts if available
    try:
        if 'instr_key_total' in locals() and instr_key_total is not None:
            rec.key_listeners_total = int(max(0, min(255, instr_key_total)))
        if 'instr_key_pw' in locals() and instr_key_pw is not None:
            # True if any password/email field had a listener registered
            rec.key_listener_pw = bool((instr_key_pw or 0) > 0)
    except Exception:
        pass

    # Visual-lite (pHash, colors)
    if png_bytes:
        ph, colors = _phash_and_colors(png_bytes)
        rec.phash64 = ph
        rec.logo_dom_color3 = colors
    # Favicon/Logo hashes best-effort (size-guarded)
    try:
        fav_url = None
        soup = BeautifulSoup(html or "", "html.parser")
        parsed = urlparse(final_url)
        base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        icons: List[str] = []
        cross_flags: Dict[str, bool] = {}
        if soup:
            for l in soup.find_all("link"):
                if not isinstance(l, Tag):
                    continue
                try:
                    rel_attr = l.get("rel")
                    rel = " ".join(rel_attr) if isinstance(rel_attr, list) else str(rel_attr or "")
                    if rel and re.search(r"icon", rel, re.I):
                        href_attr = l.get("href")
                        href = (href_attr[0] if isinstance(href_attr, list) and href_attr else href_attr) or ""
                        href = str(href)
                        if not href:
                            continue
                        full = href if href.startswith("http") else (urljoin(base + "/" if base else "", href))
                        if full:
                            icons.append(full)
                            try:
                                cross_flags[full] = (urlparse(full).netloc != (urlparse(base).netloc if base else ""))
                            except Exception:
                                cross_flags[full] = False
                except Exception:
                    continue
        if not icons and base:
            icons = [base + "/favicon.ico"]
            cross_flags[icons[0]] = False
        rec.fav_rel_count = int(len(icons)) if icons else 0
        # Try up to 3 icons
        fav_dh: Optional[int] = None
        fav_cross = False
        for uicon in icons[:3]:
            buf = _http_get_bytes_limited(uicon, max_bytes=ICON_MAX)
            if buf:
                fav_dh = _dhash64(buf)
                fav_cross = bool(cross_flags.get(uicon, False))
                break
        if fav_dh is not None:
            rec.favicon_dhash64 = fav_dh
            rec.favicon_dhash = fav_dh  # legacy mirror
            rec.fav_cross_origin = fav_cross

        # Logo pHash: candidates images mentioning logo/brand
        logo_hash: Optional[int] = None
        logo_from_alt = False
        if soup:
            candidates: List[Tuple[str, int, bool]] = []  # url, score, fromAlt
            for img in soup.find_all("img"):
                if not isinstance(img, Tag):
                    continue
                try:
                    alt_attr = img.get("alt")
                    src_attr = img.get("src")
                    alt = str((alt_attr[0] if isinstance(alt_attr, list) and alt_attr else alt_attr) or "")
                    src = str((src_attr[0] if isinstance(src_attr, list) and src_attr else src_attr) or "")
                    if re.search(r"logo|brand", alt, re.I) or re.search(r"logo|brand", src, re.I):
                        # Score from width*height attrs if present
                        try:
                            w_attr = img.get("width")
                            h_attr = img.get("height")
                            w = int(str((w_attr[0] if isinstance(w_attr, list) and w_attr else w_attr) or 0))
                            h = int(str((h_attr[0] if isinstance(h_attr, list) and h_attr else h_attr) or 0))
                            score = (w * h) if (w and h) else 0
                        except Exception:
                            score = 0
                        href = src if src.startswith("http") else (urljoin(base + "/" if base else "", src))
                        candidates.append((href, score, bool(re.search(r"logo|brand", alt, re.I))))
                except Exception:
                    continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            for href, _score, fromAlt in candidates[:3]:
                buf = _http_get_bytes_limited(href, max_bytes=ICON_MAX)
                if buf:
                    ph, _ = _phash_and_colors(buf)
                    if ph is not None:
                        logo_hash = ph
                        logo_from_alt = fromAlt
                        break
        if logo_hash is not None:
            rec.logo_phash64 = logo_hash
            rec.logo_from_alt_or_name = logo_from_alt
    except Exception:
        pass

    # TLS/Certificate best-effort (HTTPS only)
    try:
        parsed = urlparse(final_url)
        if parsed.scheme.lower() == "https" and parsed.hostname:
            tls = _tls_lookup(parsed.hostname, timeout=tls_timeout)
            for k, v in tls.items():
                setattr(rec, k, v)
    except Exception:
        pass

    # Title↔host Jaccard and client-side redirect markers
    try:
        host = urlparse(final_url).hostname or ""
        host_toks = set([t for t in re.split(r"[^a-z0-9]+", host.lower()) if t])
        title_toks = set([t for t in re.split(r"[^a-z0-9]+", (page_title or "").lower()) if t])
        j = (len(host_toks & title_toks) / float(max(1, len(host_toks | title_toks))))
        rec.title_host_jaccard_q8 = int(round(j * 255))
        # Phase 1: text fields (title/visible) and DOM graph
        rec.text_title = extract_text_title(norm_html)
        rec.text_visible = extract_text_visible(norm_html)
        rec.dom_graph = extract_dom_graph(norm_html)
    except Exception:
        pass
    try:
        # has_meta_refresh via soup
        soup2 = BeautifulSoup(html or "", "html.parser")
        rec.has_meta_refresh = bool(soup2.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}))
    except Exception:
        pass
    try:
        # has_js_loc_replace via combined JS code
        js_code = "\n".join([(s.text or "") for s in scripts])
        rec.has_js_loc_replace = bool(re.search(r"(?:window\.|document\.)?location\.replace\s*\(", js_code))
        # Phase 1: JS char sequence from concatenated code (cap inside extractor)
        try:
            rec.js_charseq = extract_js_charseq(js_code)
        except Exception:
            rec.js_charseq = None
    except Exception:
        pass

    # RDAP lite per eTLD+1 + DNS TTL/MX
    try:
        ext = tldextract.extract(final_url)
        domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
        if domain:
            rd = _rdap_lookup(domain)
            for k, v in rd.items():
                setattr(rec, k, v)
            dns = _dns_ttl_mx(domain, timeout=dns_timeout)
            for k, v in dns.items():
                setattr(rec, k, v)
    except Exception:
        pass
    return rec


async def worker(worker_id: int, queue: asyncio.Queue, out_path: str, write_lock: asyncio.Lock,
                 context, setup_context_fn, capture_external_js: bool, retries: int, base_timeout_s: float,
                 tls_timeout: float, dns_timeout: float, do_mobile_profile: bool,
                 inflight: Dict[int, Tuple[str, float]], stats: Dict[str, int],
                 per_url_cap_s: float,
                 blocked_etld1: set[str], circuit_counts: Dict[str, int], circuit_lock: asyncio.Lock,
                 adaptive_block_after: int):
    """Worker that enforces a hard per-URL cap across retries and records progress."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        url, label, source, timeout_s = item  # timeout_s retained for first attempt compatibility
        # Early skip if domain already circuit-blocked
        try:
            ext = tldextract.extract(url)
            etld1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
        except Exception:
            etld1 = None
        if etld1 and etld1 in blocked_etld1:
            async with write_lock:
                stats["done"] += 1
                stats["skip"] += 1
            queue.task_done()
            continue
        last_err: Optional[str] = None
        start = time.monotonic()
        inflight[worker_id] = (url, start)
        outcome = "skip"  # default unless successful
        try:
            print(f"[FETCH] {url} (label={label}, src={source})", flush=True)
            attempt = 0
            rec: Optional[PageRecord] = None
            deadline = start + max(1.0, per_url_cap_s)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_err = f"per-url cap {per_url_cap_s:.1f}s exceeded"
                    break
                # exponential backoff per original logic but clamped to remaining budget and 15s
                base_to = (timeout_s if attempt == 0 else min(base_timeout_s * (2 ** attempt), 15.0))
                this_timeout = min(base_to, 15.0, max(1.0, remaining))
                try:
                    # Enforce a hard asyncio-level timeout separate from page timeouts
                    rec = await asyncio.wait_for(
                        fetch_page(
                            context,
                            url,
                            label=label,
                            source=source,
                            timeout_s=this_timeout,
                            capture_external_js=capture_external_js,
                            tls_timeout=tls_timeout,
                            dns_timeout=dns_timeout,
                        ),
                        timeout=max(1.0, remaining)
                    )
                    last_err = None
                except Exception as e:
                    if isinstance(e, asyncio.TimeoutError):
                        last_err = f"hard timeout >={per_url_cap_s:.1f}s"
                        # Recycle context to nuke stuck page(s)
                        with contextlib.suppress(Exception):
                            await context.close()
                        try:
                            context = await setup_context_fn()
                        except Exception:
                            pass
                    else:
                        last_err = f"{type(e).__name__}: {e}"
                    rec = None

                if rec is not None:
                    # Optional mobile-profile pass (only if meaningful time left)
                    remaining2 = deadline - time.monotonic()
                    if do_mobile_profile and remaining2 > 2.0:
                        try:
                            realistic_mobile = (
                                "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36"
                            )
                            mctx = await context.browser.new_context(
                                user_agent=realistic_mobile,
                                viewport={"width": 375, "height": 667},
                                device_scale_factor=2,
                                is_mobile=True,
                                ignore_https_errors=True,
                            )
                            mpage = await mctx.new_page()
                            await mpage.goto(url, wait_until="domcontentloaded", timeout=int(min(this_timeout, remaining2) * 1000))
                            mhtml = await mpage.content()
                            await mpage.close()
                            await mctx.close()
                            rec.cloak_delta_domlen = abs(len(rec.html) - len(normalize_dom(mhtml))) if rec.html else len(normalize_dom(mhtml))
                            rec.cloak_profile_mismatch = bool((rec.headers.get("content-type", "").lower().startswith("text/html")) and (rec.cloak_delta_domlen or 0) > 500)
                        except Exception:
                            pass
                    async with write_lock:
                        append_jsonl(rec, out_path)
                    outcome = "ok"
                    print(f"[OK]    {url}", flush=True)
                    break

                # retry path
                if attempt < max(0, retries):
                    attempt += 1
                    print(f"[RETRY {attempt}] {url} with timeout={this_timeout:.1f}s (remaining cap ~{max(0.0, deadline - time.monotonic()):.1f}s)", flush=True)
                    continue
                else:
                    reason = last_err or "no result"
                    print(f"[SKIP]  {url} - {reason}", flush=True)
                    break
        except Exception as e:
            print(f"[ERR]   {url}: {e}", flush=True)
        finally:
            async with write_lock:
                stats["done"] += 1
                stats[outcome] += 1
            inflight.pop(worker_id, None)
            # Adaptive circuit breaker: count timeout/cap events and block noisy eTLD+1
            if adaptive_block_after > 0 and last_err and ("hard timeout" in last_err or "per-url cap" in last_err):
                if etld1:
                    async with circuit_lock:
                        circuit_counts[etld1] = circuit_counts.get(etld1, 0) + 1
                        c = circuit_counts[etld1]
                        if c >= adaptive_block_after and etld1 not in blocked_etld1:
                            blocked_etld1.add(etld1)
                            print(f"[CIRCUIT] auto-blocking {etld1} after {c} timeouts", flush=True)
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


async def crawl(input_csv: str, out_jsonl: str, concurrency: int, timeout_s: float, block_assets: bool,
                capture_external_js: bool, retries: int, resume: bool, tls_timeout: float, dns_timeout: float,
                mobile_profile: bool, use_gpu: bool, per_url_cap_s: float = 45.0, progress_interval_s: float = 10.0,
                skip_numeric_sld_min_digits: int = 0, skip_numeric_allow: str = "", skip_suffix: str = "",
                adaptive_block_after: int = 0):
    """Crawl entry point with per-URL cap and live progress monitoring."""
    queue: asyncio.Queue = asyncio.Queue()

    existing: set[str] = set()
    if resume and os.path.exists(out_jsonl):
        try:
            existing = _load_existing_urls(out_jsonl)
            print(f"[RESUME] Loaded {len(existing)} existing URLs from {out_jsonl}", flush=True)
        except Exception as e:
            print(f"[RESUME] Could not load existing URLs from {out_jsonl}: {e}", flush=True)

    total_enq = 0
    skipped_pre = 0
    skipped_pattern = 0
    allow_numeric = {s for s in (skip_numeric_allow.split(",") if skip_numeric_allow else []) if s}
    suffixes = {s for s in (skip_suffix.split(",") if skip_suffix else []) if s}
    numeric_re = re.compile(r"^[0-9]+$") if skip_numeric_sld_min_digits > 0 else None
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"].strip()
            label = int(row.get("label", 0))
            source = row.get("source", "unknown")
            if existing and url in existing:
                skipped_pre += 1
                continue
            # Pattern-based skipping
            do_skip = False
            if numeric_re is not None or suffixes:
                try:
                    ext = tldextract.extract(url)
                    sld = ext.domain or ""
                except Exception:
                    sld = ""
                if sld:
                    if numeric_re is not None and numeric_re.match(sld) and sld not in allow_numeric and len(sld) >= skip_numeric_sld_min_digits:
                        do_skip = True
                    if not do_skip and suffixes and any(sld.endswith(sf) for sf in suffixes):
                        do_skip = True
            if do_skip:
                skipped_pattern += 1
                continue
            queue.put_nowait((url, label, source, timeout_s))
            total_enq += 1
    if resume:
        print(f"[QUEUE] Enqueued {total_enq} URLs (skipped {skipped_pre} existing)", flush=True)
    if skipped_pattern:
        print(f"[SKIP] Prefilter skipped {skipped_pattern} URLs (numeric/suffix rules)", flush=True)

    if async_playwright is None:
        raise RuntimeError("playwright not installed. Install with `pip install playwright` and run `playwright install chromium`")

    # shared monitoring state
    inflight: Dict[int, Tuple[str, float]] = {}
    stats: Dict[str, int] = {"done": 0, "ok": 0, "skip": 0}

    async with async_playwright() as pw:
        async def setup_context():
            realistic_ua = (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.6478.127 Safari/537.36"
            )
            ctx = await browser.new_context(user_agent=realistic_ua, locale="en-US", ignore_https_errors=True)
            await ctx.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """
            )
            await ctx.add_init_script(
                """
                (function(){
                    try{
                        window.__pd = window.__pd || {keyListeners:0,keyPw:0};
                        const origAdd = EventTarget.prototype.addEventListener;
                        EventTarget.prototype.addEventListener = function(type, listener, options){
                            try{
                                const t = String(type||'');
                                if (/^(?:key(?:down|up)|input)$/i.test(t)){
                                    window.__pd.keyListeners = (window.__pd.keyListeners||0)+1;
                                    try{
                                        const et = (this && this.type) ? String(this.type).toLowerCase() : '';
                                        if (et === 'password' || et === 'email'){
                                            window.__pd.keyPw = (window.__pd.keyPw||0)+1;
                                        }
                                    }catch(e){}
                                }
                            }catch(e){}
                            return origAdd.call(this, type, listener, options);
                        };
                    }catch(e){}
                })();
                """
            )
            if block_assets:
                async def route_handler(route):  # type: ignore
                    rtype = route.request.resource_type
                    # Expanded block list to suppress long-lived or heavy resources
                    if rtype in ("image","media","font","websocket","eventsource","stylesheet","fetch","xhr","preload","prefetch","beacon"):
                        await route.abort()
                    else:
                        await route.continue_()
                await ctx.route("**/*", route_handler)
            return ctx

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]
        if use_gpu:
            launch_args.extend([
                "--enable-gpu-rasterization",
                "--enable-oop-rasterization",
                "--ignore-gpu-blocklist",
            ])
        browser = await pw.chromium.launch(headless=True, args=launch_args)

        contexts = [await setup_context() for _ in range(concurrency)]
        write_lock = asyncio.Lock()

        async def monitor():
            while True:
                try:
                    in_flight = len(inflight)
                    done = stats["done"]
                    ok = stats["ok"]
                    skip = stats["skip"]
                    q_left = queue.qsize()
                    slots = []
                    for wid, (u, st) in list(inflight.items()):
                        elapsed = time.monotonic() - st
                        host = urlparse(u).netloc if "://" in u else u[:40]
                        slots.append(f"w{wid}:{elapsed:4.1f}s {host[:30]}")
                    print(f"[PROGRESS] done={done}/{total_enq} ok={ok} skip={skip} inflight={in_flight} qleft={q_left}  {' | '.join(slots)}", flush=True)
                    await asyncio.sleep(progress_interval_s)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(progress_interval_s)

        monitor_task = asyncio.create_task(monitor())

        # Circuit breaker shared data
        blocked_etld1: set[str] = set()
        circuit_counts: Dict[str, int] = {}
        circuit_lock = asyncio.Lock()
        workers = [asyncio.create_task(
            worker(i, queue, out_jsonl, write_lock, contexts[i], setup_context, capture_external_js, retries, timeout_s,
                   tls_timeout, dns_timeout, mobile_profile, inflight, stats, per_url_cap_s,
                   blocked_etld1, circuit_counts, circuit_lock, adaptive_block_after)
        ) for i in range(concurrency)]

        try:
            await queue.join()
        finally:
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers, return_exceptions=True)
            monitor_task.cancel()
            with contextlib.suppress(Exception):
                await monitor_task
            for ctx in contexts:
                with contextlib.suppress(Exception):
                    await ctx.close()
            with contextlib.suppress(Exception):
                await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Crawl pages to JSONL dataset using Playwright")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: url,label,source")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--block-assets", action="store_true", help="Block images/fonts/media to speed up crawling")
    parser.add_argument("--no-external-js", action="store_true", help="Do not fetch external JS bodies (faster)")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries per URL on failure (default: 1)")
    parser.add_argument("--tls-timeout", type=float, default=3.0, help="Timeout seconds for TLS metadata fetch (per host)")
    parser.add_argument("--dns-timeout", type=float, default=2.0, help="Timeout seconds for DNS TTL/MX lookup (per domain)")
    parser.add_argument("--mobile-profile", action="store_true", help="Do a second quick mobile-profile load to compute cloaking delta")
    parser.add_argument("--gpu", action="store_true", help="Use GPU-friendly Chromium flags if available")
    # Enable resume by default; allow disabling with --no-resume
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        bool_action = None  # type: ignore
    if bool_action is not None:
        parser.add_argument("--resume", action=bool_action, default=True, help="Skip URLs already present in out-jsonl (default: True)")
    else:
        parser.add_argument("--resume", action="store_true", help="Skip URLs already present in out-jsonl")
    # Numeric / pattern skipping + adaptive blocking
    parser.add_argument("--skip-numeric-sld-min-digits", type=int, default=8, help="Skip SLDs that are all digits and at least this many digits (0 disables)")
    parser.add_argument("--skip-numeric-allow", type=str, default="360,163,12306", help="Comma list of numeric SLD tokens to always allow")
    parser.add_argument("--skip-suffix", type=str, default="", help="Comma list of SLD suffixes to skip (e.g., -cdn,-cache)")
    parser.add_argument("--adaptive-block-after", type=int, default=3, help="Auto-block eTLD+1 after this many timeout/cap events (0 disables)")
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
        tls_timeout=float(args.tls_timeout),
        dns_timeout=float(args.dns_timeout),
        mobile_profile=bool(getattr(args, "mobile_profile", False)),
        use_gpu=bool(getattr(args, "gpu", False)),
        skip_numeric_sld_min_digits=int(getattr(args, "skip_numeric_sld_min_digits", 0)),
        skip_numeric_allow=str(getattr(args, "skip_numeric_allow", "")),
        skip_suffix=str(getattr(args, "skip_suffix", "")),
        adaptive_block_after=int(getattr(args, "adaptive_block_after", 0)),
    ))


if __name__ == "__main__":
    main()
