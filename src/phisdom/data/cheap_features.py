from __future__ import annotations
from typing import Any, Dict, List


CHEAP_FEATURES: List[str] = [
    # URL/redirect
    "redirect_hops", "redirect_max_ms", "url_len", "num_dots", "num_pct", "has_at", "host_is_ip",
    # New tiny lexicals
    "host_hyphens", "has_punycode",
    # Redirect sketch
    "redir_hops", "redir_cross_host", "has_meta_refresh", "has_js_loc_replace",
    # DNS/RDAP
    "dns_created_days_ago", "dns_updated_days_ago", "ns_count", "mx_present", "ttl_min", "ttl_mean",
    # New RDAP compact
    "rdap_age_days", "rdap_registrar_hash64", "rdap_ns_count", "rdap_has_privacy",
    # TLS
    "cert_age_days", "san_count",
    # New TLS compact
    "tls_not_before_days", "tls_san_count", "tls_issuer_spki_hash64",
    # Request graph
    "req_unique_etld1", "req_thirdparty_ratio", "req_counts_script", "req_counts_css", "req_counts_xhr", "req_counts_img",
    # Form semantics
    "form_pw_count", "form_cross_site", "form_login_tokens", "form_hidden_count", "form_autocomplete_off", "onsubmit_handlers",
    # New compact form/action
    "form_fp_hash64", "num_pw", "num_email", "num_hidden", "form_method_get", "action_cross_origin", "action_proto_mismatch", "iframe_login", "top_form_count", "iframe_form_count", "form_css_sig_hash64",
    # JS heuristics
    "js_entropy", "js_eval_ct", "js_atob_ct", "js_b64_blob_ct", "js_keylog_listeners",
    # New micro-counters
    "js_eval_like", "js_hex_ratio", "js_fromcharcode", "js_hi_entropy_ratio", "js_atob", "key_listener_pw", "key_listeners_total",
    # Fingerprinting
    "fp_canvas", "fp_webgl", "fp_audio", "fp_font_enum", "fp_webrtc",
    # Visual-lite
    "favicon_dhash64", "fav_rel_count", "fav_cross_origin", "logo_phash64", "logo_from_alt_or_name",
    # Title↔host
    "title_host_jaccard_q8",
]


def row_to_features(r: Dict[str, Any], use_features: bool) -> List[float]:
    feats: List[float] = []
    if not use_features:
        return feats
    for name in CHEAP_FEATURES:
        v = r.get(name)
        if isinstance(v, bool):
            feats.append(1.0 if v else 0.0)
        elif v is None:
            feats.append(0.0)
        else:
            try:
                feats.append(float(str(v)))
            except Exception:
                feats.append(0.0)
    return feats
