#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    MarkupLMProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
import logging as py_logging
import torch

from phisdom.data.loader import JsonlPhishDataset, MarkupLMDataCollator
from phisdom.calibration import TemperatureScaler
from phisdom.metrics import pr_auc_safe, roc_auc_safe, fpr_at_tpr
from phisdom.utils.prediction_standardizer import (
    standardize_prediction_format,
    save_standardized_predictions
)


def main():
    # Silence noisy HF warnings (e.g., head init) to keep logs readable
    hf_logging.set_verbosity_error()
    py_logging.getLogger("transformers").setLevel(py_logging.ERROR)
    py_logging.getLogger("transformers.modeling_utils").setLevel(py_logging.ERROR)
    parser = argparse.ArgumentParser(description="Evaluate MarkupLM model, compute metrics and calibration")
    parser.add_argument("--model-dir", required=True, help="Directory with saved model + processor")
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-html-chars", type=int, default=800000, help="Truncate each HTML document to this many chars before tokenization (controls memory). Use -1 to disable.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, limit number of examples per split for a fast smoke test.")
    parser.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
    parser.add_argument("--tag", default="", help="Optional suffix tag for output prediction/calibration files (e.g. _full)")
    args = parser.parse_args()

    # Guard: require a fine-tuned local model directory (avoid loading base Hub ID)
    if not os.path.isdir(args.model_dir):
        raise SystemExit(
            f"[ERROR] --model-dir must be a fine-tuned local directory (got '{args.model_dir}').\n"
            "Run training first (scripts/train_markup.py) and pass its output dir, e.g. artifacts/markup_run."
        )
    has_weights = any(
        os.path.exists(os.path.join(args.model_dir, name))
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if not has_weights:
        raise SystemExit(
            f"[ERROR] No model weights found in '{args.model_dir}'. Expected model.safetensors or pytorch_model.bin.\n"
            "Run training first (scripts/train_markup.py)."
        )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    processor = MarkupLMProcessor.from_pretrained(args.model_dir)

    val_ds = JsonlPhishDataset(args.val_jsonl)
    test_ds = JsonlPhishDataset(args.test_jsonl)

    max_html_chars = None if args.max_html_chars is not None and args.max_html_chars < 0 else args.max_html_chars
    collator = MarkupLMDataCollator(processor=processor, max_length=args.max_length, max_html_chars=max_html_chars)

    if args.limit and args.limit > 0:
        # Simple slice by rebuilding reduced index lists
        val_ds._index = val_ds._index[: args.limit]  # type: ignore[attr-defined]
        test_ds._index = test_ds._index[: args.limit]  # type: ignore[attr-defined]
        print(f"[INFO] Limiting evaluation to first {args.limit} examples per split for smoke test")
    # Use Trainer but keep raw sample keys so collator sees 'html'
    targs = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "_eval_tmp"),
        per_device_eval_batch_size=4,
        dataloader_drop_last=False,
        remove_unused_columns=False,  # CRITICAL: keep 'html' for collator
    report_to=[],
    logging_strategy="no",
    )
    trainer = Trainer(model=model, args=targs, data_collator=collator)

    # Log device info
    if torch.cuda.is_available():
        try:
            gcount = torch.cuda.device_count()
            gnames = [torch.cuda.get_device_name(i) for i in range(gcount)]
        except Exception:
            gcount, gnames = (0, [])
        print(f"[INFO] Eval using CUDA (gpus={gcount}): {gnames}")
    else:
        print("[INFO] Eval using CPU")

    # Predict on val using Trainer
    val_out = trainer.predict(val_ds)  # type: ignore[arg-type]
    logits_val = val_out.predictions
    if isinstance(logits_val, tuple):
        logits_val = logits_val[0]
    if logits_val.ndim == 2 and logits_val.shape[1] == 2:
        z_val = logits_val[:, 1] - logits_val[:, 0]
    else:
        z_val = logits_val.reshape(-1)
    y_val = np.array([val_ds[i]["label"] for i in range(len(val_ds))], dtype=int)

    ts = TemperatureScaler(is_logit=True)
    T = ts.fit(y_val.tolist(), z_val.tolist())

    # Predict on test
    test_out = trainer.predict(test_ds)  # type: ignore[arg-type]
    logits_test = test_out.predictions
    if isinstance(logits_test, tuple):
        logits_test = logits_test[0]
    if logits_test.ndim == 2 and logits_test.shape[1] == 2:
        z_test = logits_test[:, 1] - logits_test[:, 0]
    else:
        z_test = logits_test.reshape(-1)
    y_test = np.array([test_ds[i]["label"] for i in range(len(test_ds))], dtype=int)

    # Calibrated probabilities
    p_val = np.array(ts.transform(z_val.tolist()))
    p_test = np.array(ts.transform(z_test.tolist()))

    # Metrics
    pr = pr_auc_safe(y_test.tolist(), p_test.tolist())
    roc = roc_auc_safe(y_test.tolist(), p_test.tolist())

    thresholds = {}
    for tpr in args.tpr:
        fpr, thr = fpr_at_tpr(y_test.tolist(), p_test.tolist(), tpr)
        thresholds[str(tpr)] = {"fpr": fpr, "threshold": thr}

    # Save calibration and metrics
    cal = {
        "temperature": T,
        "metrics": {"pr_auc": pr, "roc_auc": roc},
        "thresholds": thresholds,
    }
    tag = args.tag
    def _with_tag(base: str) -> str:
        if not tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}{tag}{ext}" if ext else f"{base}{tag}"

    with open(os.path.join(args.model_dir, _with_tag("calibration.json")), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    # Save predictions using standardized format
    def save_preds_standardized(split_name: str, ds: JsonlPhishDataset, probs: np.ndarray):
        ids = [ds[i].get("id", str(i)) for i in range(len(ds))]
        labels = np.array([ds[i].get("label", 0) for i in range(len(ds))], dtype=int)
        
        # Use unified model label 'dom' for downstream fusion consistency
        preds, metadata = standardize_prediction_format(
            ids, labels, probs, "dom", split_name, auto_flip=True
        )
        # Temporarily write to tag-suffixed filename if provided
        if tag:
            # Write predictions manually replicating save_standardized_predictions but with tag
            pred_path = os.path.join(args.model_dir, _with_tag(f"preds_{split_name}.jsonl"))
            with open(pred_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    f.write(json.dumps(pred))
                    f.write("\n")
            meta_path = os.path.join(args.model_dir, _with_tag(f"preds_{split_name}_metadata.json"))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        else:
            save_standardized_predictions(preds, metadata, args.model_dir, split_name)

    save_preds_standardized("val", val_ds, p_val)
    save_preds_standardized("test", test_ds, p_test)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
