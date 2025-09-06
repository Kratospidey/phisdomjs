from __future__ import annotations

from phisdom.features.extractors import js_minify_whitespace, js_hex_escape_subset, js_split_string_concat, extract_js_charseq


def test_js_minify_whitespace_basic():
    s = "// comment\nvar x = 1;  /* block */\n\n  x += 2;"
    out = js_minify_whitespace(s)
    assert "comment" not in out
    assert "block" not in out
    assert "var x = 1; x += 2;" == out


def test_js_hex_escape_deterministic():
    s = "abcDEF123"
    out1 = js_hex_escape_subset(s, prob=0.5, seed=123)
    out2 = js_hex_escape_subset(s, prob=0.5, seed=123)
    assert out1 == out2


def test_js_split_string_concat_deterministic_and_encodable():
    s = 'const a = "abcdef"; const b = \"g\";'
    out1 = js_split_string_concat(s, prob=1.0, seed=7)
    out2 = js_split_string_concat(s, prob=1.0, seed=7)
    assert out1 == out2
    # Should remain encodable by extract_js_charseq without errors
    ids = extract_js_charseq(out1, max_len=128)
    assert isinstance(ids, list) and len(ids) > 0
