from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Iterator
import json

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


@dataclass
class ScriptItem:
    # Either inline or external JS payload
    src: Optional[str]  # URL if external, else None
    inline: bool
    text: Optional[str]  # script source text (may be None if not captured)
    attrs: Dict[str, Any]


@dataclass
class PageRecord:
    id: str
    url: str
    etld1: str
    timestamp: float
    source: str
    label: int  # 1=phish, 0=benign
    html: str
    scripts: List[ScriptItem]
    headers: Dict[str, Any]

    # Lightweight signals (all optional; storage-light)
    # URL & redirect
    url_final: Optional[str] = None
    redirect_hops: Optional[int] = None
    redirect_max_ms: Optional[float] = None
    # New compact redirect sketch
    redir_hops: Optional[int] = None  # u8
    redir_cross_host: Optional[int] = None  # u8
    has_meta_refresh: Optional[bool] = None
    has_js_loc_replace: Optional[bool] = None
    tld: Optional[str] = None
    is_idn: Optional[bool] = None
    is_punycode: Optional[bool] = None
    has_punycode: Optional[bool] = None  # duplicate explicit flag (u1)
    url_len: Optional[int] = None
    num_dots: Optional[int] = None
    num_pct: Optional[int] = None
    has_at: Optional[bool] = None
    host_is_ip: Optional[bool] = None
    host_hyphens: Optional[int] = None  # u8

    # DNS / RDAP / WHOIS lite
    dns_created_days_ago: Optional[int] = None
    dns_updated_days_ago: Optional[int] = None
    registrar: Optional[str] = None
    ns_count: Optional[int] = None
    mx_present: Optional[bool] = None
    ttl_min: Optional[int] = None
    ttl_mean: Optional[float] = None
    # New RDAP compact fields
    rdap_age_days: Optional[int] = None  # u16
    rdap_registrar_hash64: Optional[int] = None  # u64
    rdap_ns_count: Optional[int] = None  # u8
    rdap_has_privacy: Optional[bool] = None

    # TLS/Certificate (HTTPS)
    tls_version: Optional[str] = None
    cert_issuer: Optional[str] = None
    cert_age_days: Optional[int] = None
    san_count: Optional[int] = None
    key_type: Optional[str] = None
    # New TLS compact fields
    tls_not_before_days: Optional[int] = None  # u16
    tls_san_count: Optional[int] = None  # u8
    tls_issuer_spki_hash64: Optional[int] = None  # u64

    # HTTP security headers snapshot
    hdr_csp: Optional[str] = None
    hdr_hsts: Optional[str] = None
    hdr_xfo: Optional[str] = None
    hdr_refpol: Optional[str] = None
    hdr_permspol: Optional[str] = None
    hdr_xcto: Optional[str] = None

    # Third-party request graph
    req_unique_etld1: Optional[int] = None
    req_thirdparty_ratio: Optional[float] = None
    req_counts_script: Optional[int] = None
    req_counts_css: Optional[int] = None
    req_counts_xhr: Optional[int] = None
    req_counts_img: Optional[int] = None

    # Form & login semantics
    form_pw_count: Optional[int] = None
    form_cross_site: Optional[bool] = None
    form_login_tokens: Optional[int] = None
    form_hidden_count: Optional[int] = None
    form_autocomplete_off: Optional[bool] = None
    onsubmit_handlers: Optional[int] = None
    # New compact form/action features
    form_fp_hash64: Optional[int] = None  # u64
    num_pw: Optional[int] = None  # u8
    num_email: Optional[int] = None  # u8
    num_hidden: Optional[int] = None  # u8
    form_method_get: Optional[bool] = None
    action_cross_origin: Optional[bool] = None
    action_proto_mismatch: Optional[bool] = None
    iframe_login: Optional[bool] = None
    top_form_count: Optional[int] = None  # u8
    iframe_form_count: Optional[int] = None  # u8
    form_css_sig_hash64: Optional[int] = None  # u64

    # JavaScript obfuscation & behaviors
    js_entropy: Optional[float] = None
    js_eval_ct: Optional[int] = None
    js_atob_ct: Optional[int] = None
    js_b64_blob_ct: Optional[int] = None
    js_keylog_listeners: Optional[int] = None
    # New micro-counters (u8-scaled)
    js_eval_like: Optional[int] = None  # u8
    js_hex_ratio: Optional[int] = None  # u8
    js_fromcharcode: Optional[int] = None  # u8
    js_hi_entropy_ratio: Optional[int] = None  # u8
    js_atob: Optional[int] = None  # u8
    key_listener_pw: Optional[bool] = None
    key_listeners_total: Optional[int] = None  # u8
    title_host_jaccard_q8: Optional[int] = None  # u8

    # Fingerprinting hints
    fp_canvas: Optional[bool] = None
    fp_webgl: Optional[bool] = None
    fp_audio: Optional[bool] = None
    fp_font_enum: Optional[bool] = None
    fp_webrtc: Optional[bool] = None

    # Visual-lite
    phash64: Optional[int] = None  # 64-bit integer
    favicon_dhash: Optional[int] = None  # 64-bit integer (dHash) [legacy]
    favicon_dhash64: Optional[int] = None  # new explicit name
    fav_rel_count: Optional[int] = None  # u8
    fav_cross_origin: Optional[bool] = None
    logo_phash64: Optional[int] = None  # u64
    logo_from_alt_or_name: Optional[bool] = None
    logo_dom_color3: Optional[List[int]] = None  # top-3 dominant color indexes/small ints

    # Cloaking / BitB / QR
    bitb_like_modal: Optional[bool] = None
    qr_flag: Optional[bool] = None
    cloak_delta_domlen: Optional[int] = None
    cloak_profile_mismatch: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop None optional fields to keep storage light (top-level only)
        d = {k: v for k, v in d.items() if v is not None}
        return d


def dumps(obj: Any) -> str:
    if orjson is not None:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False)


def dump_jsonl(records: List[PageRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(dumps(r.to_dict()))
            f.write("\n")


def append_jsonl(record: PageRecord, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(dumps(record.to_dict()))
        f.write("\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if orjson is not None:
                rows.append(orjson.loads(line))
            else:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file one-by-one.

    This avoids holding the entire dataset in memory. Each yielded dict is
    independent; callers should avoid accumulating the full stream unless
    necessary.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    yield orjson.loads(line)  # type: ignore[misc]
                else:
                    yield json.loads(line)
            except Exception:
                # Skip malformed lines to be robust in long streams
                continue
