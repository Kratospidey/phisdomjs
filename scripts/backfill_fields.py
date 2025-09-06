#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone

import tldextract
from bs4 import BeautifulSoup, Tag

import requests
import ssl
import socket
import dns.resolver  # type: ignore
from phisdom.features import (
    extract_url_charseq,
    extract_js_charseq,
    extract_dom_graph,
    extract_text_title,
    extract_text_visible,
)

try:
    # Optional: richer TLS parsing for SPKI hash
    from cryptography import x509  # type: ignore
    from cryptography.hazmat.primitives import hashes, serialization  # type: ignore
    from cryptography.hazmat.backends import default_backend  # type: ignore
except Exception:  # pragma: no cover
    x509 = None  # type: ignore
    default_backend = None  # type: ignore
    serialization = None  # type: ignore


RDAP_CACHE_PATH = os.path.join("data", "rdap_cache.json")
DNS_CACHE_PATH = os.path.join("data", "dns_cache.json")


def _hash64_bytes(b: bytes) -> int:
    try:
        return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")
    except Exception:
        h = hashlib.sha256(b).digest()[:8]
        return int.from_bytes(h, "big")


def _hash64_str(s: str) -> int:
    return _hash64_bytes(s.encode("utf-8", errors="ignore"))


def load_cache(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_cache(path: str, cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def url_features(url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        p = urlparse(url)
        host = p.hostname or ""
        is_idn = any(ord(ch) > 127 for ch in host)
        is_puny = host.startswith("xn--") or ".xn--" in host
        host_hyphens = host.count("-")
        tld_parts = tldextract.extract(url)
        tld = tld_parts.suffix
        out.update({
            "tld": tld or None,
            "is_idn": bool(is_idn),
            "is_punycode": bool(is_puny),
            "has_punycode": bool(is_puny),
            "url_len": len(url),
            "num_dots": host.count('.'),
            "num_pct": url.count('%'),
            "has_at": '@' in url,
            "host_is_ip": bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host)),
            "host_hyphens": host_hyphens,
        })
    except Exception:
        pass
    return out


def security_headers_snapshot(headers: Dict[str, Any]) -> Dict[str, Optional[str]]:
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


def form_semantics(html: str, page_etld1: str, page_scheme: Optional[str] = None) -> Dict[str, Any]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
    except Exception:
        return {}
    forms = [f for f in soup.find_all("form") if isinstance(f, Tag)]
    pw_count = len(soup.find_all("input", attrs={"type": re.compile("password", re.I)}))
    email_count = len(soup.find_all("input", attrs={"type": re.compile("email", re.I)}))
    hidden_count = len(soup.find_all("input", attrs={"type": re.compile("hidden", re.I)}))
    def _attr_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return " ".join(str(x) for x in v)
        return str(v)
    autocomplete_off = any((_attr_str(f.attrs.get("autocomplete")).lower() == "off") for f in forms if isinstance(f, Tag))
    tokens = ("login", "sign in", "signin", "verify", "account", "password", "2fa")
    def _text(el):
        try:
            return el.get_text(" ").lower()
        except Exception:
            return ""
    login_tokens = sum(1 for f in forms if any(t in _text(f) for t in tokens))
    cross = False
    method_get = False
    proto_mismatch = False
    for f in forms:
        if not isinstance(f, Tag):
            continue
        action = _attr_str(f.attrs.get("action"))
        method = _attr_str(f.attrs.get("method")).lower()
        if method == "get":
            method_get = True
        if not action:
            continue
        try:
            u = urlparse(action)
            if u.scheme and u.netloc:
                ext = tldextract.extract(action)
                et1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
                if et1 and page_etld1 and et1 != page_etld1:
                    cross = True
            if page_scheme and u.scheme and u.scheme.lower() != page_scheme.lower():
                proto_mismatch = True
        except Exception:
            pass
    onsubmit_ct = sum(1 for f in forms if isinstance(f, Tag) and _attr_str(f.attrs.get("onsubmit")) != "")
    # Compact form fingerprint and CSS signature (approximate)
    try:
        fp_basis = f"pw:{pw_count}|email:{email_count}|hid:{hidden_count}|get:{int(method_get)}|cross:{int(cross)}|pmis:{int(proto_mismatch)}|acoff:{int(autocomplete_off)}|ons:{onsubmit_ct}|ltok:{login_tokens}"
        form_fp_hash64 = _hash64_str(fp_basis)
    except Exception:
        form_fp_hash64 = None
    try:
        # Gather class names across forms/inputs as a coarse CSS signature
        classes: List[str] = []
        for el in soup.find_all(["form", "input", "button"]):
            if not isinstance(el, Tag):
                continue
            cls = el.get("class")
            if isinstance(cls, list):
                classes.extend([str(c) for c in cls])
            elif isinstance(cls, str):
                classes.append(cls)
        css_sig = "|".join(sorted(set(classes))[:50])
        form_css_sig_hash64 = _hash64_str(css_sig) if css_sig else None
    except Exception:
        form_css_sig_hash64 = None
    return {
        # legacy/basic
        "form_pw_count": pw_count,
        "form_hidden_count": hidden_count,
        "form_autocomplete_off": bool(autocomplete_off),
        "form_login_tokens": login_tokens,
        "form_cross_site": bool(cross),
        "onsubmit_handlers": onsubmit_ct,
        # compact
        "num_pw": pw_count,
        "num_email": email_count,
        "num_hidden": hidden_count,
        "form_method_get": bool(method_get),
        "action_cross_origin": bool(cross),
        "action_proto_mismatch": bool(proto_mismatch),
        "form_fp_hash64": form_fp_hash64,
        "form_css_sig_hash64": form_css_sig_hash64,
        "top_form_count": len(forms),
    }


def js_heuristics(scripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts: List[str] = []
    for s in scripts:
        t = s.get("text")
        if t:
            texts.append(str(t))
    all_js = "\n".join(texts)
    eval_ct = len(re.findall(r"\beval\s*\(", all_js)) + len(re.findall(r"new\s+Function\s*\(", all_js))
    atob_ct = len(re.findall(r"\batob\s*\(", all_js)) + len(re.findall(r"\bfromCharCode\b", all_js))
    b64_blob_ct = len(re.findall(r"[A-Za-z0-9+/]{80,}={0,2}", all_js))
    keylog_ct = len(re.findall(r"addEventListener\s*\(\s*['\"](?:key(?:down|up)|paste)['\"]", all_js))
    def _entropy(s: str) -> float:
        if not s:
            return 0.0
        from collections import Counter
        import math
        c = Counter(s)
        n = float(len(s))
        ent = 0.0
        for v in c.values():
            p = v / n
            ent -= p * math.log2(p)
        return round(ent, 4)
    return {
        "js_entropy": _entropy(all_js[:200000]),
        "js_eval_ct": eval_ct,
        "js_atob_ct": atob_ct,
        "js_b64_blob_ct": b64_blob_ct,
        "js_keylog_listeners": keylog_ct,
    }


def js_micro_counters(scripts: List[Dict[str, Any]]) -> Dict[str, int]:
    texts: List[str] = []
    for s in scripts:
        t = s.get("text")
        if t:
            texts.append(str(t))
    js = "\n".join(texts)
    n = max(1, len(js))
    eval_like = len(re.findall(r"\beval\s*\(|new\s+Function\s*\(", js))
    from_cc = len(re.findall(r"fromCharCode", js, flags=re.I))
    atob_ct = len(re.findall(r"\batob\s*\(", js))
    hex_chars = len(re.findall(r"[0-9a-fA-F]", js))
    hex_ratio = int(min(255, int(255.0 * (hex_chars / float(n)))))
    # Simple high-entropy proxy: scale overall entropy (0..8 bits) to 0..255
    def _entropy(s: str) -> float:
        if not s:
            return 0.0
        from collections import Counter
        import math
        c = Counter(s)
        n = float(len(s))
        ent = 0.0
        for v in c.values():
            p = v / n
            ent -= p * math.log2(p)
        return ent
    hi_ent = int(min(255, int((_entropy(js[:200000]) / 8.0) * 255.0)))
    return {
        "js_eval_like": int(min(255, eval_like)),
        "js_hex_ratio": hex_ratio,
        "js_fromcharcode": int(min(255, from_cc)),
        "js_hi_entropy_ratio": hi_ent,
        "js_atob": int(min(255, atob_ct)),
    }


def fingerprint_flags(html: str, scripts: List[Dict[str, Any]]) -> Dict[str, bool]:
    hay = (html or "") + "\n" + "\n".join([str(s.get("text") or "") for s in scripts])
    low = hay.lower()
    fp_canvas = ("getcontext('2d'" in low) or ("canvas.todataurl" in low) or ("measuretext" in low)
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


def title_host_jaccard_q8(html: str, url: str) -> Optional[int]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        title = soup.title.get_text(" ") if soup.title else ""
    except Exception:
        title = ""
    host = urlparse(url).hostname or ""
    def toks(s: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
    a = set(toks(title))
    b = set(toks(host))
    if not a or not b:
        return 0
    j = len(a & b) / float(len(a | b))
    return int(max(0, min(255, int(round(j * 255)))))


def tls_info_for_host(host: str, timeout: float = 3.0) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if not host:
        return info
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cipher = ssock.cipher()
                info["tls_version"] = getattr(ssock, "version", lambda: None)()
                # Try binary cert for richer parsing
                cert_bin = None
                try:
                    cert_bin = ssock.getpeercert(binary_form=True)
                except Exception:
                    cert_bin = None
                if cert_bin and x509 is not None and default_backend is not None:
                    try:
                        cert = x509.load_der_x509_certificate(cert_bin, default_backend())
                        issuer = cert.issuer.rfc4514_string()
                        info["cert_issuer"] = issuer
                        nb = cert.not_valid_before
                        if nb:
                            info["cert_age_days"] = int((datetime.now(timezone.utc) - nb.replace(tzinfo=timezone.utc)).total_seconds() / 86400)
                        try:
                            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
                            try:
                                info["san_count"] = len(san._general_names)  # type: ignore[attr-defined]
                            except Exception:
                                info["san_count"] = None
                        except Exception:
                            pass
                        # SPKI hash
                        try:
                            if serialization is not None:
                                spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
                                info["tls_issuer_spki_hash64"] = _hash64_bytes(spki)
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    cert = ssock.getpeercert()
                    if cert:
                        info["cert_issuer"] = ",".join("=".join(x) for t in cert.get("issuer", []) for x in t)
                        not_before = cert.get("notBefore")
                        def _parse_dt(s: Optional[str]) -> Optional[datetime]:
                            if not s:
                                return None
                            for fmt in ("%b %d %H:%M:%S %Y %Z", "%Y%m%d%H%M%SZ"):
                                try:
                                    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                                except Exception:
                                    continue
                            return None
                        nb = _parse_dt(not_before if isinstance(not_before, str) else None)
                        if nb:
                            info["cert_age_days"] = int((datetime.now(timezone.utc) - nb).total_seconds() / 86400)
                        san = cert.get("subjectAltName") or []
                        info["san_count"] = int(len(san)) if san else 0
                if cipher:
                    info["key_type"] = cipher[0]
    except Exception:
        pass
    return info


def rdap_lookup(domain: str, timeout: float, cache: Dict[str, Any]) -> Dict[str, Any]:
    if not domain:
        return {}
    if domain in cache:
        return cache[domain]
    data: Dict[str, Any] = {}
    try:
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
            ns_list = j.get("nameservers", []) or []
            data["ns_count"] = len(ns_list) if ns_list else None
            data["mx_present"] = None
            data["ttl_min"] = None
            data["ttl_mean"] = None
    except Exception:
        pass
    cache[domain] = data
    return data


def dns_ttl_mx(domain: str, timeout: float, cache: Dict[str, Any]) -> Dict[str, Any]:
    if not domain:
        return {}
    if domain in cache:
        return cache[domain]
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
    cache[domain] = out
    return out


def favicon_info(html: str, base_url: str, timeout: float = 3.0) -> Dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
        from io import BytesIO
        import numpy as np  # type: ignore
    except Exception:
        return {}
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        # gather icons
        icons: List[str] = []
        cross: Dict[str, bool] = {}
        for l in soup.find_all("link"):
            if not isinstance(l, Tag):
                continue
            try:
                rel = l.get("rel")
                rels = " ".join(rel) if isinstance(rel, list) else str(rel or "")
                if rels and re.search(r"icon", rels, re.I):
                    href = l.get("href")
                    if isinstance(href, list):
                        href = href[0] if href else None
                    href = str(href or "")
                    if not href:
                        continue
                    full = href if href.startswith("http") else (urljoin(base + "/" if base else "", href))
                    if full:
                        icons.append(full)
                        try:
                            cross[full] = (urlparse(full).netloc != (urlparse(base).netloc if base else ""))
                        except Exception:
                            cross[full] = False
            except Exception:
                continue
        if not icons and base:
            icons = [base + "/favicon.ico"]
            cross[icons[0]] = False
        fav_rel_count = len(icons)
        fav_dhash: Optional[int] = None
        fav_cross = False
        if icons:
            u = icons[0]
            r = requests.get(u, timeout=timeout)
            if r.ok and r.content:
                img = Image.open(BytesIO(r.content)).convert('L').resize((9, 8))
                arr = np.asarray(img, dtype=np.int16)
                diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()
                h = 0
                for b in diff:
                    h = (h << 1) | int(b)
                fav_dhash = int(h)
                fav_cross = bool(cross.get(u, False))
        out: Dict[str, Any] = {}
        if fav_dhash is not None:
            out["favicon_dhash"] = fav_dhash
            out["favicon_dhash64"] = fav_dhash
        out["fav_rel_count"] = fav_rel_count
        out["fav_cross_origin"] = fav_cross
        return out
    except Exception:
        return {}


def logo_phash(html: str, base_url: str, timeout: float = 3.0) -> Dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
        from io import BytesIO
        import numpy as np  # type: ignore
    except Exception:
        return {}
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        cand: List[Tuple[str, bool]] = []
        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue
            try:
                alt = str(img.get("alt") or "")
                name = str(img.get("name") or "")
                src_val = img.get("src")
                src = str(src_val[0] if isinstance(src_val, list) and src_val else src_val or "")
                text = (str(alt) + " " + str(name)).lower()
                if "logo" in text or "brand" in text:
                    full = src if isinstance(src, str) and src.startswith("http") else (urljoin(base + "/" if base else "", str(src)))
                    if full:
                        from_alt = ("logo" in text or "brand" in text)
                        cand.append((str(full), bool(from_alt)))
            except Exception:
                continue
        if not cand:
            return {}
        u, from_alt = cand[0]
        r = requests.get(u, timeout=timeout)
        if not r.ok or not r.content:
            return {}
        # Compute simple pHash (8x8 DCT of grayscale 32x32)
        img = Image.open(BytesIO(r.content)).convert('L').resize((32, 32))
        arr = np.asarray(img, dtype=np.float32)
        dct = np.fft.fft2(arr)
        low = dct[:8, :8].real
        med = float(np.median(low))
        bits = (low > med).astype(np.uint8).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return {"logo_phash64": int(h), "logo_from_alt_or_name": bool(from_alt)}
    except Exception:
        return {}


def enrich_record(r: Dict[str, Any], *, allow_network: bool, tls_timeout: float, dns_timeout: float, rdap_cache: Dict[str, Any], dns_cache: Dict[str, Any]) -> Dict[str, Any]:
    # Use url_final if present else url
    final_url = r.get("url_final") or r.get("url")
    if isinstance(final_url, str):
        # URL lexicals
        for k, v in url_features(final_url).items():
            r.setdefault(k, v)
        # Basic: set url_final if missing
        r.setdefault("url_final", final_url)
        # Phase 1: raw url and char sequence
        r.setdefault("url_raw", r.get("url") or final_url)
        try:
            if "url_charseq" not in r:
                r["url_charseq"] = extract_url_charseq(final_url)
        except Exception:
            pass
        # Title-host overlap
        if "title_host_jaccard_q8" not in r and isinstance(r.get("html"), str):
            tj = title_host_jaccard_q8(r.get("html") or "", final_url)
            if tj is not None:
                r["title_host_jaccard_q8"] = tj
        # DNS/RDAP/TLS
        if allow_network:
            ext = tldextract.extract(final_url)
            et1 = ".".join([p for p in [ext.domain, ext.suffix] if p])
            host = urlparse(final_url).hostname or ""
            if et1:
                rd = rdap_lookup(et1, timeout=tls_timeout, cache=rdap_cache)
                for k, v in rd.items():
                    if v is not None:
                        r.setdefault(k, v)
                dn = dns_ttl_mx(et1, timeout=dns_timeout, cache=dns_cache)
                for k, v in dn.items():
                    if v is not None:
                        r.setdefault(k, v)
            if host and final_url.lower().startswith("https"):
                ti = tls_info_for_host(host, timeout=tls_timeout)
                for k, v in ti.items():
                    if v is not None:
                        r.setdefault(k, v)
    # Redirect sketch best-effort
        try:
            html = r.get("html") or ""
            # meta refresh
            has_meta = False
            try:
                soup = BeautifulSoup(html, "html.parser")
                for m in soup.find_all("meta"):
                    if not isinstance(m, Tag):
                        continue
                    http_val = m.get("http-equiv") or m.get("http_equiv") or ""
                    http = str(http_val).lower()
                    if http == "refresh":
                        has_meta = True
                        break
            except Exception:
                has_meta = False
            r.setdefault("has_meta_refresh", bool(has_meta))
            # JS location replace
            scripts = r.get("scripts") or []
            jstext = "\n".join([str(s.get("text") or "") for s in scripts]) if isinstance(scripts, list) else ""
            has_loc = bool(re.search(r"location\s*\.(?:replace|assign)\s*\(", jstext, re.I))
            r.setdefault("has_js_loc_replace", has_loc)
            # Map redirect_hops to redir_hops if present
            if "redirect_hops" in r and "redir_hops" not in r:
                try:
                    r["redir_hops"] = int(r.get("redirect_hops") or 0)
                except Exception:
                    pass
        except Exception:
            pass
    # Headers snapshot
    headers = r.get("headers") or {}
    if isinstance(headers, dict):
        for k, v in security_headers_snapshot(headers).items():
            if v is not None:
                r.setdefault(k, v)
    # HTML-derived
    html = r.get("html") or ""
    etld1 = str(r.get("etld1") or "")
    if isinstance(html, str):
        # Phase 1: text fields and DOM graph
        try:
            r.setdefault("text_title", extract_text_title(html))
            r.setdefault("text_visible", extract_text_visible(html))
            if "dom_graph" not in r:
                r["dom_graph"] = extract_dom_graph(html)
        except Exception:
            pass
        # Form
        f = form_semantics(html, etld1, urlparse(final_url).scheme if isinstance(final_url, str) else None)
        for k, v in f.items():
            if v is not None:
                r.setdefault(k, v)
        scripts = r.get("scripts") or []
        if isinstance(scripts, list):
            # JS char sequence from concatenated code
            try:
                if "js_charseq" not in r:
                    code = "\n".join(str(s.get("text") or "") for s in scripts)
                    r["js_charseq"] = extract_js_charseq(code)
            except Exception:
                pass
            for k, v in js_heuristics(scripts).items():
                if v is not None:
                    r.setdefault(k, v)
            for k, v in js_micro_counters(scripts).items():
                r.setdefault(k, v)
            for k, v in fingerprint_flags(html, scripts).items():
                r.setdefault(k, v)
        # Key listener compact best-effort from heuristics
        try:
            if "key_listeners_total" not in r and isinstance(r.get("js_keylog_listeners"), (int, float)):
                r["key_listeners_total"] = int(r.get("js_keylog_listeners") or 0)
            if "key_listener_pw" not in r:
                r["key_listener_pw"] = bool((r.get("num_pw") or 0) > 0 and (r.get("key_listeners_total") or 0) > 0)
        except Exception:
            pass
        # Visuals (network optional)
        if allow_network and isinstance(final_url, str):
            fav = favicon_info(html, final_url)
            for k, v in fav.items():
                r.setdefault(k, v)
            logo = logo_phash(html, final_url)
            for k, v in logo.items():
                r.setdefault(k, v)
        # TLS compact mapping
        try:
            cad = r.get("cert_age_days")
            sc = r.get("san_count")
            if "tls_not_before_days" not in r and isinstance(cad, (int, float)):
                r["tls_not_before_days"] = int(cad)
            if "tls_san_count" not in r and isinstance(sc, (int, float)):
                r["tls_san_count"] = int(sc)
        except Exception:
            pass
        # RDAP compact mapping
        try:
            reg = str(r.get("registrar") or "")
            dcd = r.get("dns_created_days_ago")
            nsc = r.get("ns_count")
            if "rdap_age_days" not in r and isinstance(dcd, (int, float)):
                r["rdap_age_days"] = int(dcd)
            if "rdap_ns_count" not in r and isinstance(nsc, (int, float)):
                r["rdap_ns_count"] = int(nsc)
            if reg and "rdap_registrar_hash64" not in r:
                r["rdap_registrar_hash64"] = _hash64_str(reg)
                low = reg.lower()
                r.setdefault("rdap_has_privacy", any(x in low for x in ("privacy", "whoisguard", "proxy", "redacted")))
        except Exception:
            pass
    return r


def process_file(path: str, *, overwrite: bool, allow_network: bool, tls_timeout: float, dns_timeout: float) -> None:
    if not os.path.exists(path):
        return
    rdap_cache = load_cache(RDAP_CACHE_PATH)
    dns_cache = load_cache(DNS_CACHE_PATH)
    tmp_path = path + ".tmp"
    out = open(tmp_path, "w", encoding="utf-8")
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj = enrich_record(obj, allow_network=allow_network, tls_timeout=tls_timeout, dns_timeout=dns_timeout, rdap_cache=rdap_cache, dns_cache=dns_cache)
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")
            n += 1
    out.close()
    save_cache(RDAP_CACHE_PATH, rdap_cache)
    save_cache(DNS_CACHE_PATH, dns_cache)
    if overwrite:
        os.replace(tmp_path, path)
    else:
        backup = path + ".bak"
        os.replace(path, backup)
        os.replace(tmp_path, path)
        print(f"[BACKFILL] original saved to {backup}")
    print(f"[BACKFILL] updated {n} records in {path}")


def main():
    ap = argparse.ArgumentParser(description="Backfill lightweight fields into existing JSONL records")
    ap.add_argument("--inputs", nargs="+", default=["data/pages.jsonl", "data/pages_train.jsonl", "data/pages_val.jsonl", "data/pages_test.jsonl"], help="One or more JSONL files to backfill")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files in place (default creates .bak)")
    ap.add_argument("--network", action="store_true", help="Allow network lookups (RDAP/DNS/TLS/favicon)")
    ap.add_argument("--tls-timeout", type=float, default=3.0)
    ap.add_argument("--dns-timeout", type=float, default=2.0)
    args = ap.parse_args()
    for p in args.inputs:
        process_file(p, overwrite=bool(args.overwrite), allow_network=bool(args.network), tls_timeout=float(args.tls_timeout), dns_timeout=float(args.dns_timeout))


if __name__ == "__main__":
    main()
