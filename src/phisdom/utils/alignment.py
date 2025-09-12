#!/usr/bin/env python
"""
Robust data alignment utilities for fusion.
Fixes the sequential intersection problem that drops too many samples.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

from ..data.schema import load_jsonl
from ..data.cheap_features import row_to_features, CHEAP_FEATURES


class AlignmentStrategy:
    """Base class for alignment strategies."""
    
    def align(
        self,
        jsonl_path: str,
        head_dirs: Dict[str, str],
        required_heads: Optional[Set[str]] = None,
        use_cheap_features: bool = True,
        head_tag: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Align predictions from multiple heads.
        
        Args:
            jsonl_path: Path to ground truth JSONL
            head_dirs: Map from head name to directory containing predictions
            required_heads: If specified, only include samples with all required heads
            use_cheap_features: Whether to include cheap features
            head_tag: Optional suffix for prediction files (e.g., "_full").
            
        Returns:
            Tuple of (features, labels, sample_ids, feature_names)
        """
        raise NotImplementedError


class InnerJoinAlignment(AlignmentStrategy):
    """Inner join alignment - only samples present in ALL heads."""
    
    def align(
        self,
        jsonl_path: str,
        head_dirs: Dict[str, str],
        required_heads: Optional[Set[str]] = None,
        use_cheap_features: bool = True,
        head_tag: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        
        # Determine which heads to require
        if required_heads is None:
            required_heads = set(head_dirs.keys())
        
        # Load predictions from each head
        split = self._extract_split_name(jsonl_path)
        head_predictions = {}
        
        for head_name, head_dir in head_dirs.items():
            pred_path = os.path.join(head_dir, f"preds_{split}{head_tag}.jsonl")
            preds = self._load_predictions(pred_path)
            head_predictions[head_name] = preds
            print(f"Loaded {len(preds)} predictions for {head_name}")
        
        # Load ground truth
        rows = load_jsonl(jsonl_path)
        gold_labels = {}
        id2row = {}
        
        for row in rows:
            id_ = self._extract_id(row)
            gold_labels[id_] = int(row.get("label", 0))
            id2row[id_] = row
        
        print(f"Loaded {len(gold_labels)} gold labels")
        
        # Find intersection based on required heads
        common_ids = set(gold_labels.keys())
        for head_name in required_heads:
            if head_name in head_predictions:
                head_ids = set(head_predictions[head_name].keys())
                common_ids &= head_ids
                print(f"After intersecting with {head_name}: {len(common_ids)} common IDs")
        
        if not common_ids:
            print("WARNING: No common IDs found!")
            # Return empty matrices with expected feature dimension (num heads [+ cheap later])
            feat_dim = len(head_dirs)
            return np.zeros((0, feat_dim), dtype=float), np.zeros((0,), dtype=int), [], list(head_dirs.keys())
        
        # Build feature matrix
        features = []
        labels = []
        sample_ids = []
        feature_names = list(head_dirs.keys())
        
        for sample_id in sorted(common_ids):
            # Head predictions
            head_features = []
            for head_name in head_dirs.keys():
                if head_name in head_predictions and sample_id in head_predictions[head_name]:
                    head_features.append(head_predictions[head_name][sample_id][1])  # prob
                else:
                    head_features.append(0.0)  # missing head gets 0
            
            # Cheap features if requested
            if use_cheap_features and sample_id in id2row:
                cheap_feats = row_to_features(id2row[sample_id], True)
                head_features.extend(cheap_feats)
                if len(feature_names) == len(head_dirs):  # only add names once
                    feature_names.extend(CHEAP_FEATURES)
            
            features.append(head_features)
            labels.append(gold_labels[sample_id])
            sample_ids.append(sample_id)
        
        print(f"Final aligned dataset: {len(sample_ids)} examples with {len(features[0]) if features else 0} features")
        
        return (
            np.array(features, dtype=float),
            np.array(labels, dtype=int),
            sample_ids,
            feature_names
        )
    
    def _extract_split_name(self, jsonl_path: str) -> str:
        """Extract split name from JSONL path.
        Supports patterns like pages_val.jsonl, pages_val_full.jsonl, test_data.jsonl.
        Defaults to 'val' if 'val' appears in name, else 'test'.
        """
        basename = os.path.basename(jsonl_path)
        if basename.startswith("pages_") and basename.endswith("_full.jsonl"):
            core = basename[len("pages_") : -len(".jsonl")]  # pages_val_full -> val_full
            return core
        if basename.startswith("pages_") and basename.endswith(".jsonl"):
            # pages_val.jsonl -> val
            return basename[len("pages_") : -len(".jsonl")]
        name = basename.replace('.jsonl','')
        if 'val' in name:
            return 'val'
        return 'test'
    
    def _extract_id(self, row: Dict[str, Any]) -> str:
        """Extract stable ID from row."""
        for key in ["id", "uid", "sha1", "url_hash"]:
            if key in row and row[key] is not None:
                return str(row[key])
        raise KeyError(f"No stable ID found in row: {list(row.keys())}")
    
    def _load_predictions(self, pred_path: str) -> Dict[str, Tuple[int, float]]:
        """Load predictions from JSONL file."""
        predictions = {}
        if not os.path.exists(pred_path):
            print(f"WARNING: Prediction file missing: {pred_path}")
            # Fallback: try any preds_*.jsonl in same directory
            head_dir = os.path.dirname(pred_path)
            try:
                import glob
                cands = sorted(glob.glob(os.path.join(head_dir, "preds_*.jsonl")))
                if cands:
                    pred_path = cands[0]
                else:
                    return predictions
            except Exception:
                return predictions
        
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    pred = json.loads(line)
                    id_ = str(pred.get("id", ""))
                    label = int(pred.get("label", 0))
                    prob = float(pred.get("prob", 0.0))
                    predictions[id_] = (label, prob)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"WARNING: Skipping malformed prediction: {e}")
        
        return predictions


class CoverageMaximizingAlignment(AlignmentStrategy):
    """
    Alignment that maximizes sample coverage by using all available head combinations.
    Uses imputation for missing heads.
    """
    
    def __init__(self, min_heads: int = 2, imputation_value: float = 0.5):
        self.min_heads = min_heads
        self.imputation_value = imputation_value
    
    def align(
        self,
        jsonl_path: str,
        head_dirs: Dict[str, str],
        required_heads: Optional[Set[str]] = None,
        use_cheap_features: bool = True,
        head_tag: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        
        # Load all predictions
        split = self._extract_split_name(jsonl_path)
        head_predictions = {}
        
        for head_name, head_dir in head_dirs.items():
            pred_path = os.path.join(head_dir, f"preds_{split}{head_tag}.jsonl")
            preds = self._load_predictions(pred_path)
            head_predictions[head_name] = preds
            print(f"Loaded {len(preds)} predictions for {head_name}")
        
        # Load ground truth
        rows = load_jsonl(jsonl_path)
        gold_labels = {}
        id2row = {}
        
        for row in rows:
            id_ = self._extract_id(row)
            gold_labels[id_] = int(row.get("label", 0))
            id2row[id_] = row
        
        print(f"Loaded {len(gold_labels)} gold labels")
        
        # Find samples with at least min_heads predictions
        sample_coverage = defaultdict(int)
        for head_name, preds in head_predictions.items():
            for sample_id in preds.keys():
                if sample_id in gold_labels:
                    sample_coverage[sample_id] += 1
        
        valid_samples = {
            sample_id for sample_id, count in sample_coverage.items()
            if count >= self.min_heads
        }
        
        print(f"Found {len(valid_samples)} samples with at least {self.min_heads} head predictions")
        
        if not valid_samples:
            print("WARNING: No samples meet minimum head requirements!")
            return np.array([]), np.array([]), [], []
        
        # Build feature matrix with imputation
        features = []
        labels = []
        sample_ids = []
        feature_names = list(head_dirs.keys())
        
        for sample_id in sorted(valid_samples):
            # Head predictions with imputation
            head_features = []
            for head_name in head_dirs.keys():
                if head_name in head_predictions and sample_id in head_predictions[head_name]:
                    head_features.append(head_predictions[head_name][sample_id][1])  # prob
                else:
                    head_features.append(self.imputation_value)  # impute missing
            
            # Cheap features if requested
            if use_cheap_features and sample_id in id2row:
                cheap_feats = row_to_features(id2row[sample_id], True)
                head_features.extend(cheap_feats)
                if len(feature_names) == len(head_dirs):  # only add names once
                    feature_names.extend(CHEAP_FEATURES)
            
            features.append(head_features)
            labels.append(gold_labels[sample_id])
            sample_ids.append(sample_id)
        
        print(f"Final aligned dataset: {len(sample_ids)} examples with {len(features[0]) if features else 0} features")
        
        return (
            np.array(features, dtype=float),
            np.array(labels, dtype=int),
            sample_ids,
            feature_names
        )
    
    def _extract_split_name(self, jsonl_path: str) -> str:
        """Extract split name from JSONL path."""
        basename = os.path.basename(jsonl_path)
        # Handle extended split naming: pages_val_full.jsonl -> val_full
        if basename.startswith("pages_") and basename.endswith("_full.jsonl"):
            core = basename[len("pages_") : -len(".jsonl")]
            return core
        if "_" in basename:
            return basename.split("_")[-1].replace(".jsonl", "")
        return "test"  # fallback
    
    def _extract_id(self, row: Dict[str, Any]) -> str:
        """Extract stable ID from row."""
        for key in ["id", "uid", "sha1", "url_hash"]:
            if key in row and row[key] is not None:
                return str(row[key])
        raise KeyError(f"No stable ID found in row: {list(row.keys())}")
    
    def _load_predictions(self, pred_path: str) -> Dict[str, Tuple[int, float]]:
        """Load predictions from JSONL file."""
        predictions = {}
        
        if not os.path.exists(pred_path):
            print(f"WARNING: Prediction file missing: {pred_path}")
            head_dir = os.path.dirname(pred_path)
            try:
                import glob
                cands = sorted(glob.glob(os.path.join(head_dir, "preds_*.jsonl")))
                if cands:
                    pred_path = cands[0]
                else:
                    return predictions
            except Exception:
                return predictions
        
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    pred = json.loads(line)
                    id_ = str(pred.get("id", ""))
                    label = int(pred.get("label", 0))
                    prob = float(pred.get("prob", 0.0))
                    predictions[id_] = (label, prob)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"WARNING: Skipping malformed prediction: {e}")
        
        return predictions


def get_alignment_strategy(strategy: str = "inner_join", **kwargs) -> AlignmentStrategy:
    """Factory function for alignment strategies."""
    if strategy == "inner_join":
        return InnerJoinAlignment()
    elif strategy == "coverage_max":
        return CoverageMaximizingAlignment(**kwargs)
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")
