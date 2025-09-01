#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from phisdom.data.js import JsonlJsDataset
from phisdom.metrics import pr_auc as pr_auc_fn, roc_auc as roc_auc_fn


class MeanPooler(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / counts


class T5EncoderClassifier(nn.Module):
    def __init__(self, base: T5EncoderModel, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.encoder = base
        self.pool = MeanPooler()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(enc_out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = (e[:, 1] / (e[:, 0] + e[:, 1])).clip(1e-6, 1 - 1e-6)
    else:
        probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    y_true = labels.astype(int).tolist()
    y_score = probs.tolist()
    return {"pr_auc": pr_auc_fn(y_true, y_score), "roc_auc": roc_auc_fn(y_true, y_score)}


def main():
    parser = argparse.ArgumentParser(description="Train CodeT5+ (encoder) JS classifier")
    parser.add_argument("--train-jsonl", default="data/pages_train.jsonl")
    parser.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    parser.add_argument("--output-dir", default="artifacts/js_codet5p")
    parser.add_argument("--model-name", default="Salesforce/codet5p-220m")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility; default is random")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base = T5EncoderModel.from_pretrained(args.model_name)
    hidden = base.config.d_model
    model = T5EncoderClassifier(base, hidden)

    def tok_batch(batch):
        texts = [r["text"] for r in batch]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        labels = torch.tensor([int(r["label"]) for r in batch], dtype=torch.long)
        enc["labels"] = labels
        return enc

    train_ds = JsonlJsDataset(args.train_jsonl)
    val_ds = JsonlJsDataset(args.val_jsonl)

    use_cuda = torch.cuda.is_available()
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    fp16_ok = use_cuda
    if use_cuda:
        try:
            gcount = torch.cuda.device_count()
            gnames = [torch.cuda.get_device_name(i) for i in range(gcount)]
        except Exception:
            gcount, gnames = (0, [])
        prec = "bf16" if bf16_ok else ("fp16" if fp16_ok else "fp32")
        print(f"[INFO] Using CUDA (gpus={gcount}): {gnames} | precision={prec}")
    else:
        print("[INFO] Using CPU | precision=fp32")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
    save_safetensors=False,
    save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        fp16=fp16_ok and not bf16_ok,
        bf16=bf16_ok,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=tok_batch,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save encoder + tokenizer + classifier head weights
    base.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    clf_state = {"weight": model.classifier.weight.detach().cpu(), "bias": model.classifier.bias.detach().cpu()}
    torch.save(clf_state, os.path.join(args.output_dir, "classifier.pt"))
    with open(os.path.join(args.output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args)}, f)


if __name__ == "__main__":
    main()
