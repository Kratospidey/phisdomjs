from collections import Counter
import math
import re
from typing import Dict, List, Tuple

SAFE_EPS = 1e-12


def _char_ngrams(text: str, n: int) -> List[str]:
    if n <= 0:
        return []
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def extract_url_features(url: str, ngram: int = 3, vocab_cap: int = 500) -> Dict[str, float]:
    """
    Lightweight URL features suitable as a baseline:
    - Normalized char n-gram counts (top-K by frequency within the URL, capped)
    - Length, digit ratio, symbol ratio, subdomain depth, and entropy-like features
    """
    url = (url or "").strip()
    host_match = re.search(r"^[a-z]+://([^/]+)/?", url, flags=re.I)
    host = host_match.group(1).lower() if host_match else ""

    # Basic stats
    length = float(len(url))
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    symbols = length - digits - letters
    digit_ratio = digits / length if length else 0.0
    symbol_ratio = symbols / length if length else 0.0
    entropy = _shannon_entropy(url)

    # Subdomain depth heuristic (split by dots)
    sub_depth = max(0, host.count(".") - 1) if host else 0

    # N-grams over the full URL string
    grams = _char_ngrams(url.lower(), max(1, ngram))
    counts = Counter(grams)
    # Keep top-K grams within this URL to limit feature size per sample
    most_common = counts.most_common(vocab_cap)
    total_grams = sum(c for _, c in most_common) or 1.0

    feats: Dict[str, float] = {
        "len": length,
        "digit_ratio": float(digit_ratio),
        "symbol_ratio": float(symbol_ratio),
        "entropy": float(entropy),
        "sub_depth": float(sub_depth),
    }
    for g, c in most_common:
        feats[f"ng{ngram}:{g}"] = c / total_grams

    return feats


def vectorize_feature_dict(feats: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """
    Deterministically order a sparse feature dict to a pair of (keys, values) lists.
    Useful for consistent hashing or feeding simple models.
    """
    items = sorted(feats.items())
    keys = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    return keys, vals
