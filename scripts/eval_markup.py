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
from typing import cast
from torch.utils.data import Dataset as TorchDataset
import torch

from phisdom.data.loader import JsonlPhishDataset, MarkupLMDataCollator
from phisdom.calibration import TemperatureScaler
from phisdom.metrics import pr_auc, roc_auc, fpr_at_tpr


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
    parser.add_argument("--tpr", type=float, nargs="*", default=[0.95, 0.90])
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

    collator = MarkupLMDataCollator(processor=processor, max_length=args.max_length)

    # Dummy trainer for prediction loop
    targs = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "_eval_tmp"),
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
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

    # Predict on val for calibration
    val_out = trainer.predict(cast(TorchDataset, val_ds))  # type: ignore[arg-type]
    logits_val = val_out.predictions
    if isinstance(logits_val, tuple):
        logits_val = logits_val[0]
    # Binary logits: if 2-dim, use logit for class 1 as z = log(p1/p0)
    if logits_val.ndim == 2 and logits_val.shape[1] == 2:
        z_val = logits_val[:, 1] - logits_val[:, 0]
    else:
        z_val = logits_val.reshape(-1)
    # Extract labels without relying on deprecated `.rows`
    y_val = np.array([val_ds[i]["label"] for i in range(len(val_ds))], dtype=int)

    ts = TemperatureScaler(is_logit=True)
    T = ts.fit(y_val.tolist(), z_val.tolist())

    # Predict on test
    test_out = trainer.predict(cast(TorchDataset, test_ds))  # type: ignore[arg-type]
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
    pr = pr_auc(y_test.tolist(), p_test.tolist())
    roc = roc_auc(y_test.tolist(), p_test.tolist())

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
    with open(os.path.join(args.model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    # Save predictions
    def dump_preds(path: str, ds: JsonlPhishDataset, probs: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for i, p in enumerate(probs.tolist()):
                r = ds[i]
                obj = {"id": r.get("id", str(i)), "label": int(r.get("label", 0)), "prob": float(p)}
                f.write(json.dumps(obj))
                f.write("\n")

    dump_preds(os.path.join(args.model_dir, "preds_val.jsonl"), val_ds, p_val)
    dump_preds(os.path.join(args.model_dir, "preds_test.jsonl"), test_ds, p_test)

    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
