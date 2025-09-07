from __future__ import annotations
"""
Deterministic, storage-light feature extractors for Phase 1 schema fields.

Adds:
- URL char sequence (for char-CNN)
- JS char sequence (for char-CNN)
- DOM graph compact representation (hashed tag/class features, depth, text bins)
- Visible text and title extraction (capped)

All outputs are capped and use small integer encodings to keep JSONL size low.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import hashlib
import re
import os


# Fixed small alphabets with 0=PAD, 1=UNK
_DEFAULT_URL_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"  # 26
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26
    "0123456789"                  # 10
    ":/?#[]@!$&'()*+,;=%._-~"     # RFC3986 + common URL chars
)
_DEFAULT_JS_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"  # 26
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26
    "0123456789"                  # 10
    "{}[]()<>;:,.!@#$%^&*-_=+|\\/\"'`?\n\r\t "  # structural + whitespace
)

_URL_IDX = {ch: i + 2 for i, ch in enumerate(_DEFAULT_URL_ALPHABET)}  # 0=pad,1=unk
_JS_IDX = {ch: i + 2 for i, ch in enumerate(_DEFAULT_JS_ALPHABET)}


def get_js_vocab() -> Dict[str, Union[str, Dict[str, int], Dict[int, str]]]:
    """Return JS vocab dictionaries: {'alphabet': str, 'idx': {ch->id}, 'inv': {id->ch}}"""
    inv = {v: k for k, v in _JS_IDX.items()}
    return {"alphabet": _DEFAULT_JS_ALPHABET, "idx": _JS_IDX, "inv": inv}


def get_url_vocab() -> Dict[str, Union[str, Dict[str, int], Dict[int, str]]]:
    inv = {v: k for k, v in _URL_IDX.items()}
    return {"alphabet": _DEFAULT_URL_ALPHABET, "idx": _URL_IDX, "inv": inv}


def _encode_chars(text: str, *, alphabet: str, idx: Dict[str, int], max_len: int) -> List[int]:
    if not isinstance(text, str):
        return []
    out: List[int] = []
    for ch in text[:max_len]:
        out.append(idx.get(ch, 1))  # 1=UNK
    return out


def extract_url_charseq(url: str, max_len: int = 256) -> List[int]:
    """Encode raw URL into fixed small alphabet indexes, capped length."""
    return _encode_chars(url or "", alphabet=_DEFAULT_URL_ALPHABET, idx=_URL_IDX, max_len=max_len)


def extract_js_charseq(js_text: str, max_len: int = 2048) -> List[int]:
    """Encode JavaScript text to small alphabet indexes with cap."""
    return _encode_chars(js_text or "", alphabet=_DEFAULT_JS_ALPHABET, idx=_JS_IDX, max_len=max_len)


def decode_js_charseq(seq: List[int]) -> str:
    inv = {v: k for k, v in _JS_IDX.items()}
    return "".join(inv.get(int(t), "") for t in (seq or []))


def _hash32(s: str) -> int:
    d = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=4).digest()
    return int.from_bytes(d, "big", signed=False)


def _text_len_bin(text: str) -> int:
    n = len(text or "")
    if n == 0:
        return 0
    if n <= 10:
        return 1
    if n <= 50:
        return 2
    if n <= 200:
        return 3
    return 4


def extract_dom_graph(html: str, max_nodes: int = 256) -> Dict[str, Any]:
    """
    Build a compact DOM tree graph from HTML:
    - nodes: list of {t: tag_hash32, c: classes_hash32, d: depth (u8), x: text_len_bin (0..4)}
    - edges: list of [parent_idx, child_idx]
    Limits to max_nodes in pre-order traversal for storage-light representation.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    nodes: List[Dict[str, int]] = []
    edges: List[List[int]] = []

    def _classes_hash(tag: Tag) -> int:
        try:
            cls = tag.get("class")
            if not cls:
                return 0
            if isinstance(cls, list):
                s = "|".join(sorted(str(c).lower() for c in cls))
            else:
                s = str(cls).lower()
            return _hash32(s) if s else 0
        except Exception:
            return 0

    # Pre-order traversal with cap
    stack: List[Tuple[Optional[int], Tag, int]] = []  # (parent_idx, node, depth)
    root = soup
    for child in root.children:
        if isinstance(child, Tag):
            stack.append((None, child, 0))

    while stack and len(nodes) < max_nodes:
        parent_idx, tag, depth = stack.pop()
        try:
            tname = tag.name or ""
            t_hash = _hash32(tname)
            c_hash = _classes_hash(tag)
            # visible text immediate length (stripped) without children text accumulation
            own_text_parts: List[str] = []
            for t in tag.contents:
                if isinstance(t, NavigableString):
                    own_text_parts.append(str(t))
            x_bin = _text_len_bin(re.sub(r"\s+", " ", "".join(own_text_parts).strip()))
            cur_idx = len(nodes)
            nodes.append({"t": t_hash, "c": c_hash, "d": int(max(0, min(255, depth))), "x": x_bin})
            if parent_idx is not None:
                edges.append([parent_idx, cur_idx])
            # push children in reverse order to maintain document order on pop
            children_tags = [c for c in tag.children if isinstance(c, Tag)]
            for ch in reversed(children_tags):
                if len(nodes) + len(stack) >= max_nodes:
                    break
                stack.append((cur_idx, ch, depth + 1))
        except Exception:
            continue

    return {"n": len(nodes), "nodes": nodes, "edges": edges}


def extract_text_title(html: str, max_chars: int = 256) -> str:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        t = soup.title.get_text(" ") if soup.title else ""
        t = re.sub(r"\s+", " ", t).strip()
        return t[:max_chars]
    except Exception:
        return ""


def extract_text_visible(html: str, max_chars: int = 4000) -> str:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        # Remove script/style/noscript and hidden elements (style display:none or hidden attr)
        for el in soup(["script", "style", "noscript"]):
            el.decompose()
        # Heuristic: drop nodes explicitly hidden
        for el in list(soup.find_all(True)):
            if not isinstance(el, Tag):
                continue
            try:
                style = str(el.attrs.get("style") or "").lower()
                hidden = ("hidden" in el.attrs) or el.has_attr("hidden")
                if ("display:none" in style) or hidden:
                    el.decompose()
            except Exception:
                continue
        text = soup.get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# --- Simple JS canonicalization / obfuscation helpers (Phase 6) ---

def js_minify_whitespace(s: str) -> str:
    try:
        # remove /* */ comments
        s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
        # remove // comments
        s = re.sub(r"//.*", " ", s)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    except Exception:
        return s


def js_hex_escape_subset(s: str, prob: float = 0.05, seed: Optional[int] = None) -> str:
    import random
    rng = random.Random(seed)
    out = []
    for ch in s:
        if ch.isalnum() and rng.random() < prob:
            out.append("\\x%02x" % (ord(ch) & 0xFF))
        else:
            out.append(ch)
    return "".join(out)


def js_split_string_concat(s: str, prob: float = 0.05, seed: Optional[int] = None) -> str:
    import re as _re
    import random
    rng = random.Random(seed)
    def split_token(m):
        q = m.group(1)
        body = m.group(2)
        if len(body) < 4 or rng.random() >= prob:
            return m.group(0)
        k = rng.randint(1, len(body) - 1)
        return f"{q}{body[:k]}{q}+{q}{body[k:]}{q}"
    return _re.sub(r"([\'\"])([^\1]{2,}?)\1", split_token, s)


def _dedupe_preserve(s: str) -> str:
    seen = set(); out = []
    for ch in s:
        if ch not in seen:
            seen.add(ch); out.append(ch)
    return "".join(out)

def _load_alpha(env_chars: str, env_file: str, default: str, env_extra: str | None = None, extra_file_env: str | None = None) -> str:
    s = os.getenv(env_chars)
    if s:
        base = s.replace("\\n", "\n").replace("\\t", "\t")
    else:
        path = os.getenv(env_file)
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    base = fh.read()
            except Exception:
                base = default
        else:
            base = default
    base = _dedupe_preserve(base)
    if env_extra:
        extra = os.getenv(env_extra)
        if extra:
            base = _dedupe_preserve(base + extra.replace("\\n", "\n").replace("\\t", "\t"))
    if extra_file_env:
        ep = os.getenv(extra_file_env)
        if ep:
            try:
                with open(ep, "r", encoding="utf-8") as fh:
                    extra2 = fh.read()
                base = _dedupe_preserve(base + extra2.replace("\\n", "\n").replace("\\t", "\t"))
            except Exception:
                pass
    return base

_URL_ALPHABET = _load_alpha(
    "PHISDOM_URL_ALPHABET", "PHISDOM_URL_ALPHA_FILE", _DEFAULT_URL_ALPHABET,
    "PHISDOM_URL_ALPHABET_EXTRA", "PHISDOM_URL_ALPHA_EXTRA_FILE"
)
_JS_ALPHABET = _load_alpha(
    "PHISDOM_JS_ALPHABET", "PHISDOM_JS_ALPHA_FILE", _DEFAULT_JS_ALPHABET,
    "PHISDOM_JS_ALPHABET_EXTRA", "PHISDOM_JS_ALPHA_EXTRA_FILE"
)

