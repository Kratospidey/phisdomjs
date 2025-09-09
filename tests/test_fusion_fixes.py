#!/usr/bin/env python
"""
Comprehensive tests for fusion system fixes.
Tests all critical components to prevent failures.
"""
from __future__ import annotations
import os
import json
import tempfile
import shutil
from typing import Dict, List, Any
import numpy as np
import pytest

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phisdom.utils.prediction_standardizer import (
    standardize_prediction_format,
    save_standardized_predictions,
    validate_prediction_format,
    load_and_validate_predictions
)
from phisdom.utils.alignment import (
    InnerJoinAlignment,
    CoverageMaximizingAlignment,
    get_alignment_strategy
)


class TestPredictionStandardizer:
    """Test prediction format standardization utilities."""
    
    def test_standardize_format_basic(self):
        """Test basic prediction standardization."""
        ids = ["sample1", "sample2", "sample3"]
        labels = np.array([0, 1, 0])
        probs = np.array([0.2, 0.8, 0.3])
        
        preds, metadata = standardize_prediction_format(
            ids, labels, probs, "test_model", "val", auto_flip=False
        )
        
        assert len(preds) == 3
        assert preds[0]["id"] == "sample1"
        assert preds[0]["label"] == 0
        assert preds[0]["prob"] == 0.2
        assert preds[0]["split"] == "val"
        assert preds[0]["model"] == "test_model"
        
        assert metadata["model"] == "test_model"
        assert metadata["split"] == "val"
        assert metadata["num_samples"] == 3
        assert not metadata["flipped"]
    
    def test_auto_flip_detection(self):
        """Test automatic probability orientation correction."""
        ids = ["s1", "s2", "s3", "s4"]
        labels = np.array([0, 0, 1, 1])  # balanced for reliable AUC
        probs = np.array([0.8, 0.9, 0.2, 0.1])  # inverted (low probs for phish)
        
        preds, metadata = standardize_prediction_format(
            ids, labels, probs, "test_model", "test", auto_flip=True
        )
        
        assert metadata["flipped"]
        assert preds[2]["prob"] == 0.8  # was 0.2, should be flipped
        assert preds[3]["prob"] == 0.9  # was 0.1, should be flipped
    
    def test_save_and_load_standardized(self):
        """Test saving and loading standardized predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ids = ["test1", "test2"]
            labels = np.array([0, 1])
            probs = np.array([0.3, 0.7])
            
            preds, metadata = standardize_prediction_format(
                ids, labels, probs, "test_model", "val", auto_flip=False
            )
            save_standardized_predictions(preds, metadata, tmpdir, "val")
            
            # Check files were created
            pred_path = os.path.join(tmpdir, "preds_val.jsonl")
            meta_path = os.path.join(tmpdir, "preds_val_metadata.json")
            assert os.path.exists(pred_path)
            assert os.path.exists(meta_path)
            
            # Validate format
            diagnostics = validate_prediction_format(pred_path)
            assert diagnostics["valid"]
            assert diagnostics["num_predictions"] == 2
            
            # Load back and verify
            loaded_preds, load_diag = load_and_validate_predictions(pred_path)
            assert load_diag["valid"]
            assert len(loaded_preds) == 2
            assert "test1" in loaded_preds
            assert loaded_preds["test1"] == (0, 0.3)
    
    def test_validation_invalid_format(self):
        """Test validation with invalid prediction format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid prediction file
            pred_path = os.path.join(tmpdir, "invalid_preds.jsonl")
            with open(pred_path, "w") as f:
                f.write('{"missing_required_fields": true}\n')
                f.write('invalid json line\n')
            
            diagnostics = validate_prediction_format(pred_path)
            assert not diagnostics["valid"]
            assert len(diagnostics["errors"]) > 0


class TestAlignmentStrategies:
    """Test data alignment strategies."""
    
    def setup_method(self):
        """Set up test data."""
        self.tmpdir = tempfile.mkdtemp()
        
        # Create test JSONL data
        test_data = [
            {"id": "sample1", "label": 0, "url": "http://example.com"},
            {"id": "sample2", "label": 1, "url": "http://phish.com"},
            {"id": "sample3", "label": 0, "url": "http://good.com"},
        ]
        
        self.jsonl_path = os.path.join(self.tmpdir, "test_data.jsonl")
        with open(self.jsonl_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Create mock head prediction directories
        self.head_dirs = {}
        for head_name in ["head1", "head2", "head3"]:
            head_dir = os.path.join(self.tmpdir, head_name)
            os.makedirs(head_dir)
            self.head_dirs[head_name] = head_dir
            
            # Create predictions (head3 missing sample3 to test alignment)
            preds = [
                {"id": "sample1", "label": 0, "prob": 0.2 + np.random.random() * 0.1},
                {"id": "sample2", "label": 1, "prob": 0.8 + np.random.random() * 0.1},
            ]
            if head_name != "head3":  # head3 missing sample3
                preds.append({"id": "sample3", "label": 0, "prob": 0.3 + np.random.random() * 0.1})
            
            pred_path = os.path.join(head_dir, "preds_data.jsonl")
            with open(pred_path, "w") as f:
                for pred in preds:
                    f.write(json.dumps(pred) + "\n")
    
    def teardown_method(self):
        """Clean up test data."""
        shutil.rmtree(self.tmpdir)
    
    def test_inner_join_alignment(self):
        """Test inner join alignment strategy."""
        aligner = InnerJoinAlignment()
        
        X, y, ids, feature_names = aligner.align(
            self.jsonl_path, self.head_dirs, use_cheap_features=False
        )
        
        # Should only have samples present in ALL heads
        assert len(ids) == 2  # sample1, sample2 (sample3 missing from head3)
        assert X.shape == (2, 3)  # 2 samples, 3 heads
        assert len(feature_names) == 3
        assert set(ids) == {"sample1", "sample2"}
    
    def test_coverage_maximizing_alignment(self):
        """Test coverage maximizing alignment with imputation."""
        aligner = CoverageMaximizingAlignment(min_heads=2)
        
        X, y, ids, feature_names = aligner.align(
            self.jsonl_path, self.head_dirs, use_cheap_features=False
        )
        
        # Should include all samples with at least 2 heads
        assert len(ids) == 3  # all samples (sample3 has 2/3 heads)
        assert X.shape == (3, 3)  # 3 samples, 3 heads
        
        # Check imputation for missing head3 prediction for sample3
        sample3_idx = ids.index("sample3")
        # head3 feature should be imputed (default 0.5)
        assert X[sample3_idx, 2] == 0.5
    
    def test_alignment_factory(self):
        """Test alignment strategy factory."""
        inner_aligner = get_alignment_strategy("inner_join")
        assert isinstance(inner_aligner, InnerJoinAlignment)
        
        coverage_aligner = get_alignment_strategy("coverage_max", min_heads=3)
        assert isinstance(coverage_aligner, CoverageMaximizingAlignment)
        assert coverage_aligner.min_heads == 3
        
        with pytest.raises(ValueError):
            get_alignment_strategy("invalid_strategy")


class TestFusionModelRobustness:
    """Test fusion model robustness fixes."""
    
    def test_lazy_linear_cheap_features(self):
        """Test that LazyLinear handles variable cheap feature dimensions."""
        import torch
        from phisdom.models.fusion import CrossModalTransformerFusion
        
        # Create model with lazy cheap features
        model = CrossModalTransformerFusion(
            d_model=128, n_heads=4, n_layers=1,
            use_cheap=True, cheap_dim=None  # None = lazy
        )
        
        # Test with 74-dim cheap features (the problematic case)
        batch = {
            "cheap_features": torch.randn(4, 74),  # batch_size=4, cheap_dim=74
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.float)
        }
        
        # Should not crash
        logits = model(batch)
        assert logits.shape == (4, 1)
        
        # Test with different dimension
        batch2 = {
            "cheap_features": torch.randn(2, 32),  # different dim
            "labels": torch.tensor([0, 1], dtype=torch.float)
        }
        
        # Should still work (LazyLinear adapts)
        logits2 = model(batch2)
        assert logits2.shape == (2, 1)
    
    def test_fixed_dimension_validation(self):
        """Test that fixed dimension validation works."""
        import torch
        from phisdom.models.fusion import CrossModalTransformerFusion
        
        # Create model with fixed cheap dimension
        model = CrossModalTransformerFusion(
            d_model=128, n_heads=4, n_layers=1,
            use_cheap=True, cheap_dim=32  # fixed dimension
        )
        
        # Test with correct dimension
        batch = {
            "cheap_features": torch.randn(4, 32),
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.float)
        }
        
        logits = model(batch)
        assert logits.shape == (4, 1)
        
        # Test with wrong dimension - should raise error
        batch_wrong = {
            "cheap_features": torch.randn(4, 74),  # wrong dimension
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.float)
        }
        
        with pytest.raises(RuntimeError):  # matmul shape mismatch
            model(batch_wrong)
    
    def test_dimension_validation_at_init(self):
        """Test that invalid cheap_dim raises error at initialization."""
        import torch
        from phisdom.models.fusion import CrossModalTransformerFusion
        
        with pytest.raises(ValueError):
            CrossModalTransformerFusion(cheap_dim=-1)  # negative
        
        with pytest.raises(ValueError):
            CrossModalTransformerFusion(cheap_dim=0)   # zero


def test_end_to_end_fusion_pipeline():
    """Test complete fusion pipeline end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock data and predictions
        test_data = [
            {"id": f"sample{i}", "label": i % 2, "url": f"http://example{i}.com"}
            for i in range(10)
        ]
        
        jsonl_path = os.path.join(tmpdir, "test_data.jsonl")
        with open(jsonl_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Create mock head predictions
        head_dirs = {}
        for head_name in ["url_head", "js_head"]:
            head_dir = os.path.join(tmpdir, head_name)
            os.makedirs(head_dir)
            head_dirs[head_name] = head_dir
            
            # Generate realistic predictions
            for split in ["val", "test"]:
                preds = []
                for i, item in enumerate(test_data):
                    # Add some noise to make predictions realistic
                    base_prob = 0.8 if item["label"] == 1 else 0.2
                    prob = max(0.0, min(1.0, base_prob + np.random.normal(0, 0.1)))
                    preds.append({"id": item["id"], "label": item["label"], "prob": prob})
                
                pred_path = os.path.join(head_dir, f"preds_{split}.jsonl")
                with open(pred_path, "w") as f:
                    for pred in preds:
                        f.write(json.dumps(pred) + "\n")
        
        # Test alignment
        aligner = InnerJoinAlignment()
        X, y, ids, feature_names = aligner.align(
            jsonl_path, head_dirs, use_cheap_features=False
        )
        
        assert X.shape[0] == len(test_data)  # all samples should align
        assert X.shape[1] == 2  # 2 heads
        assert len(ids) == len(test_data)
        
        # Test logistic fusion (mock sklearn)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            if len(set(y)) > 1:  # ensure not one-class
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                clf = LogisticRegression(class_weight="balanced")
                clf.fit(X_scaled, y)
                
                probs = clf.predict_proba(X_scaled)[:, 1]
                assert len(probs) == len(y)
                assert all(0 <= p <= 1 for p in probs)
        except ImportError:
            print("Sklearn not available, skipping logistic fusion test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
