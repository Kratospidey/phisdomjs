#!/usr/bin/env python
"""
Prediction format standardization utilities.
Ensures all heads save predictions in consistent formats for fusion.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score


def standardize_prediction_format(
    ids: List[str],
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    split: str,
    auto_flip: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Standardize prediction format and optionally auto-correct orientation.
    
    Args:
        ids: List of sample IDs
        labels: True labels (1=phish, 0=benign)
        probs: Model probabilities 
        model_name: Name of the model/head
        split: Data split name (train/val/test)
        auto_flip: Whether to auto-correct probability orientation
        
    Returns:
        Tuple of (predictions_list, metadata)
    """
    # Convert to numpy for easier manipulation
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    
    # Auto-flip detection and correction
    flipped = False
    auc = None
    if auto_flip and len(labels) > 1 and len(set(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs)
            if auc < 0.5:
                probs = 1.0 - probs
                auc = 1.0 - auc
                flipped = True
                print(f"[WARN] Auto-flipped {model_name} {split} predictions (AUC was {1.0-auc:.3f}, now {auc:.3f})")
        except Exception as e:
            print(f"[WARN] Could not compute AUC for auto-flip: {e}")
    
    # Build standardized predictions
    predictions = []
    for id_, label, prob in zip(ids, labels.tolist(), probs.tolist()):
        predictions.append({
            "id": str(id_),
            "label": int(label),
            "prob": float(prob),  # P(phish)
            "split": split,
            "model": model_name
        })
    
    # Metadata
    metadata = {
        "model": model_name,
        "split": split,
        "num_samples": len(predictions),
        "flipped": flipped,
        "auc": auc if auc is not None else float('nan'),
        "schema_version": "1.0"
    }
    
    return predictions, metadata


def save_standardized_predictions(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: str,
    split: str
) -> None:
    """Save predictions and metadata in standardized format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    pred_path = os.path.join(output_dir, f"preds_{split}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred))
            f.write("\n")
    
    # Save metadata
    meta_path = os.path.join(output_dir, f"preds_{split}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def validate_prediction_format(pred_path: str) -> Dict[str, Any]:
    """
    Validate prediction file format and return diagnostics.
    
    Returns:
        Dictionary with validation results and diagnostics
    """
    diagnostics = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "num_predictions": 0,
        "unique_ids": 0,
        "label_distribution": {},
        "prob_range": None,
        "schema_version": None
    }
    
    if not os.path.exists(pred_path):
        diagnostics["valid"] = False
        diagnostics["errors"].append(f"File does not exist: {pred_path}")
        return diagnostics
    
    ids_seen = set()
    labels = []
    probs = []
    
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError as e:
                    diagnostics["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check required fields
                if "id" not in pred:
                    diagnostics["errors"].append(f"Line {line_num}: Missing 'id' field")
                if "label" not in pred:
                    diagnostics["errors"].append(f"Line {line_num}: Missing 'label' field")
                if "prob" not in pred:
                    diagnostics["errors"].append(f"Line {line_num}: Missing 'prob' field")
                
                # Collect data for validation
                if "id" in pred:
                    ids_seen.add(str(pred["id"]))
                if "label" in pred:
                    labels.append(pred["label"])
                if "prob" in pred:
                    probs.append(pred["prob"])
                
                diagnostics["num_predictions"] += 1
    
    except Exception as e:
        diagnostics["valid"] = False
        diagnostics["errors"].append(f"Failed to read file: {e}")
        return diagnostics
    
    # Compute diagnostics
    diagnostics["unique_ids"] = len(ids_seen)
    if diagnostics["num_predictions"] != diagnostics["unique_ids"]:
        diagnostics["warnings"].append("Duplicate IDs found")
    
    if labels:
        unique_labels, counts = np.unique(labels, return_counts=True)
        diagnostics["label_distribution"] = dict(zip(unique_labels.tolist(), counts.tolist()))
    
    if probs:
        diagnostics["prob_range"] = [float(np.min(probs)), float(np.max(probs))]
        
        # Check for suspicious probability ranges
        if np.min(probs) < 0 or np.max(probs) > 1:
            diagnostics["warnings"].append("Probabilities outside [0,1] range")
    
    # Mark as invalid if any errors
    if diagnostics["errors"]:
        diagnostics["valid"] = False
    
    return diagnostics


def load_and_validate_predictions(pred_path: str) -> Tuple[Dict[str, Tuple[int, float]], Dict[str, Any]]:
    """
    Load predictions with validation.
    
    Returns:
        Tuple of (id_to_label_prob, diagnostics)
    """
    diagnostics = validate_prediction_format(pred_path)
    
    predictions = {}
    if diagnostics["valid"]:
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pred = json.loads(line)
                predictions[str(pred["id"])] = (int(pred["label"]), float(pred["prob"]))
    
    return predictions, diagnostics
