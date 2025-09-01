#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
from typing import Any, Dict, cast

import yaml
from transformers import (
    AutoModelForSequenceClassification,
    MarkupLMProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging
from torch.utils.data import Dataset as TorchDataset
import logging as py_logging

from phisdom.data.loader import JsonlPhishDataset, MarkupLMDataCollator
import torch
from phisdom.metrics import pr_auc as pr_auc_fn, roc_auc as roc_auc_fn


def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    # Binary probability for class 1
    if logits.ndim == 2 and logits.shape[1] == 2:
        # stable softmax for p1
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = (e[:, 1] / (e[:, 0] + e[:, 1])).clip(1e-6, 1 - 1e-6)
    else:
        probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    y_true = labels.astype(int).tolist()
    y_score = probs.tolist()
    return {
        "pr_auc": pr_auc_fn(y_true, y_score),
        "roc_auc": roc_auc_fn(y_true, y_score),
    }


def main():
    parser = argparse.ArgumentParser(description="Train MarkupLM DOM head on JSONL dataset")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "artifacts/markup_run")
    os.makedirs(out_dir, exist_ok=True)

    # Reduce noisy HF warnings (e.g., "Some weights ... not initialized" during head init)
    hf_logging.set_verbosity_error()
    # Also force Python loggers used inside Transformers to ERROR
    py_logging.getLogger("transformers").setLevel(py_logging.ERROR)
    py_logging.getLogger("transformers.modeling_utils").setLevel(py_logging.ERROR)
    # Use explicit seed only if provided; otherwise keep run stochastic
    if "seed" in cfg and cfg["seed"] is not None:
        set_seed(int(cfg["seed"]))

    model_name = cfg.get("model_name", "microsoft/markuplm-base")
    max_length = int(cfg.get("max_length", 512))
    batch_size = int(cfg.get("batch_size", 4))
    num_epochs = float(cfg.get("num_epochs", 1))
    lr = float(cfg.get("learning_rate", 3e-5))

    train_path = cfg["train_jsonl"]
    val_path = cfg.get("val_jsonl", train_path)

    processor = MarkupLMProcessor.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # Set label metadata
    model.config.id2label = {0: "benign", 1: "phish"}
    model.config.label2id = {"benign": 0, "phish": 1}
    model.config.problem_type = "single_label_classification"

    train_ds = JsonlPhishDataset(train_path)
    val_ds = JsonlPhishDataset(val_path)
    collator = MarkupLMDataCollator(processor=processor, max_length=max_length)

    # Auto mixed-precision on CUDA if available
    use_cuda = torch.cuda.is_available()
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    fp16_ok = use_cuda

    # Log device/precision info for visibility during e2e runs
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

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        fp16=fp16_ok and not bf16_ok,
        bf16=bf16_ok,
    # Keep raw 'html' in batch for custom collator
    remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=cast(TorchDataset, train_ds),  # type: ignore[arg-type]
        eval_dataset=cast(TorchDataset, val_ds),  # type: ignore[arg-type]
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    # Save a tiny run manifest
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "labels": ["benign", "phish"]}, f)


if __name__ == "__main__":
    main()
