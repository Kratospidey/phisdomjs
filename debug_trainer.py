#!/usr/bin/env python
"""Debug script to isolate the Trainer issue"""

from src.phisdom.data.loader import JsonlPhishDataset, MarkupLMDataCollator
from transformers import AutoModelForSequenceClassification, MarkupLMProcessor, TrainingArguments, Trainer
import os
import torch
from torch.utils.data import Dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DebugDataset(Dataset):
    def __init__(self):
        self.data = [
            {"id": "test1", "html": "<p>Test HTML 1</p>", "label": 0},
            {"id": "test2", "html": "<p>Test HTML 2</p>", "label": 1},
            {"id": "test3", "html": "<p>Test HTML 3</p>", "label": 0},
            {"id": "test4", "html": "<p>Test HTML 4</p>", "label": 1},
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DebugCollator:
    def __call__(self, batch):
        print(f"[DEBUG COLLATOR] Batch size: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"[DEBUG COLLATOR] Item {i}: {item}")
        return {"dummy": torch.tensor([1])}

# Test 1: Simple dataset with debug collator
print("=== TEST 1: Debug Dataset with Debug Collator ===")
debug_ds = DebugDataset()
debug_collator = DebugCollator()

targs = TrainingArguments(
    output_dir="/tmp/debug",
    per_device_eval_batch_size=2,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
)

# Create a dummy model for testing
model = torch.nn.Linear(1, 2)
trainer = Trainer(model=model, args=targs, data_collator=debug_collator)

try:
    result = trainer.predict(debug_ds)
    print("Debug dataset test passed!")
except Exception as e:
    print(f"Debug dataset test failed: {e}")

# Test 2: JsonlPhishDataset with debug collator
print("\n=== TEST 2: JsonlPhishDataset with Debug Collator ===")
val_ds = JsonlPhishDataset('data/pages_val.jsonl')
print(f"JsonlPhishDataset size: {len(val_ds)}")

try:
    result = trainer.predict(val_ds)
    print("JsonlPhishDataset test passed!")
except Exception as e:
    print(f"JsonlPhishDataset test failed: {e}")
