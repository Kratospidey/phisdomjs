# save as sanity_counts.py; run: PYTHONPATH=src python sanity_counts.py
from phisdom.data.new_heads import UrlSeqDataset, JsSeqDataset, DomGraphDataset
from phisdom.data.new_heads import TextSeqDataset, CheapFeaturesDataset

VAL = "data/pages_val.jsonl"
TEST = "data/pages_test.jsonl"

for name, DS in [
    ("URL", UrlSeqDataset),
    ("JS", JsSeqDataset),
    ("DOM-GCN", DomGraphDataset),
    ("TEXT", TextSeqDataset),
    ("CHEAP", CheapFeaturesDataset),
]:
    for split, path in [("val", VAL), ("test", TEST)]:
        ds = DS(path)
        n = len(ds)
        pos = sum(int(ds[i]["label"]) for i in range(n))
        neg = n - pos
        print(f"{name:7s} {split:4s}: total={n}, pos={pos}, neg={neg}")
