from __future__ import annotations
import pytest
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from phisdom.data.new_heads import UrlSeqDataset, JsSeqDataset, DomGraphDataset, PaddedSeqCollator, DomGraphCollator

if torch is None:
    pytest.skip("torch not installed; skipping new head tests", allow_module_level=True)

from phisdom.models.heads import UrlCharCNN, JsCharCNN, DomGCN


def _mk_tmp_jsonl(tmp_path):
    p = tmp_path / "mini.jsonl"
    rows = [
        {"id": "a", "label": 1, "url_charseq": [2,3,4], "js_charseq": [5,6,7,8], "dom_graph": {"n": 2, "nodes": [{"t": 10, "c": 11, "d": 0, "x": 1}, {"t": 12, "c": 13, "d": 1, "x": 0}], "edges": [[0,1]]}},
        {"id": "b", "label": 0, "url_charseq": [2], "js_charseq": [6], "dom_graph": {"n": 1, "nodes": [{"t": 14, "c": 15, "d": 0, "x": 2}], "edges": []}},
    ]
    with p.open("w", encoding="utf-8") as f:
        import json
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_seq_datasets_and_collators(tmp_path):
    path = _mk_tmp_jsonl(tmp_path)
    url_ds = UrlSeqDataset(str(path))
    js_ds = JsSeqDataset(str(path))
    assert len(url_ds) == 2 and len(js_ds) == 2
    coll = PaddedSeqCollator(pad_idx=0)
    batch = [url_ds[0], url_ds[1]]
    out = coll(batch)
    assert out["input_ids"].shape[0] == 2
    assert out["input_ids"].shape[1] >= 1
    model = UrlCharCNN()
    logits = model(out["input_ids"], out.get("attention_mask"))
    assert logits.shape == (2,)


def test_dom_graph_dataset_and_collator(tmp_path):
    path = _mk_tmp_jsonl(tmp_path)
    ds = DomGraphDataset(str(path))
    assert len(ds) == 2
    coll = DomGraphCollator()
    batch = coll([ds[0], ds[1]])
    assert batch["node_feats_raw"].shape[-1] == 4
    model = DomGCN()
    logits = model(batch["node_feats_raw"], batch["edge_index"], batch["batch_index"])
    assert logits.shape == (2,)


def test_js_head_forward(tmp_path):
    path = _mk_tmp_jsonl(tmp_path)
    ds = JsSeqDataset(str(path))
    coll = PaddedSeqCollator(pad_idx=0)
    batch = coll([ds[0], ds[1]])
    model = JsCharCNN()
    logits = model(batch["input_ids"], batch.get("attention_mask"))
    assert logits.shape == (2,)
