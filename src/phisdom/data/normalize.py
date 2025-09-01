from __future__ import annotations
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import re


def normalize_dom(html: str) -> str:
    """
    A minimal DOM normalizer:
    - Parses HTML with html5lib-like behavior (via html.parser fallback)
    - Removes excess whitespace
    - Lower-cases tag names and sorts attributes for stability
    - Leaves text content intact
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(True):
        # Sort attributes for stability
        if tag.attrs:
            items = sorted((k, " ".join(v) if isinstance(v, list) else v) for k, v in tag.attrs.items())
            tag.attrs.clear()
            for k, v in items:
                tag.attrs[k.lower()] = v
    # Compact whitespace in text nodes
    for text_node in soup.find_all(string=True):
        # Skip script/style content; keep as-is
        if text_node.parent and text_node.parent.name in ("script", "style"):
            continue
        s = re.sub(r"\s+", " ", str(text_node)).strip()
        text_node.replace_with(s)
    # Return a canonical-ish string
    return soup.decode()


def extract_scripts(html: str) -> List[Dict[str, Any]]:
    """
    Extract <script> tags with basic metadata.
    Returns list of dicts: {src, inline, text, attrs}
    Note: external script text is None here; the crawler may populate it
    from captured network responses.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    items: List[Dict[str, Any]] = []
    for s in soup.find_all("script"):
        src = s.get("src")
        inline = src is None
        text = None
        if inline:
            # Keep raw text as-is
            text = s.string if s.string is not None else s.get_text()
        attrs = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in s.attrs.items() if k != "src"}
        items.append({
            "src": src,
            "inline": inline,
            "text": text,
            "attrs": attrs,
        })
    return items
