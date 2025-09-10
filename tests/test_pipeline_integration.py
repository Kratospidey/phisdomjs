#!/usr/bin/env python
"""
Integration tests for the complete phisdom pipeline.
Tests all critical failure points to ensure robustness.
"""
from __future__ import annotations
import os
import json
import tempfile
import shutil
import subprocess
import sys
from typing import Dict, List, Any
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tmpdir = tempfile.mkdtemp()
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        
        # Create minimal test data
        self.test_data = [
            {
                "id": f"test_sample_{i}",
                "label": i % 2,
                "url": f"http://{'phish' if i % 2 else 'legit'}{i}.com",
                "url_charseq": list(range(50)),  # fake char sequence
                "js_charseq": list(range(30)),   # fake JS sequence
                "html": f"<html><body>Test page {i}</body></html>",
                "text_title": f"Test Page {i}",
                "text_visible": f"This is test content for page {i}",
                "dom_graph": {"n": 3, "nodes": [{"tag": "html"}, {"tag": "body"}, {"tag": "text"}], "edges": [[0,1], [1,2]]},
                # Add cheap features (matching CHEAP_FEATURES length)
                "redirect_hops": 0,
                "url_len": 20 + i,
                "num_dots": 1,
                "has_at": False,
                # ... add more cheap features to match expected count
            }
            for i in range(20)
        ]
        
        # Split into train/val/test
        self.train_data = self.test_data[:12]
        self.val_data = self.test_data[12:16]
        self.test_data_split = self.test_data[16:20]
        
        # Save to files
        for split_name, split_data in [
            ("train", self.train_data), 
            ("val", self.val_data), 
            ("test", self.test_data_split)
        ]:
            split_path = os.path.join(self.tmpdir, f"pages_{split_name}.jsonl")
            with open(split_path, "w") as f:
                for item in split_data:
                    f.write(json.dumps(item) + "\n")
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.tmpdir)
    
    def test_fusion_model_dimension_handling(self):
        """Test that fusion model handles various cheap feature dimensions."""
        import torch
        from phisdom.models.fusion import CrossModalTransformerFusion
        from phisdom.data.cheap_features import CHEAP_FEATURES
        
        # Test with actual cheap features length
        actual_cheap_dim = len(CHEAP_FEATURES)
        
        # Model with lazy adaptation
        model_lazy = CrossModalTransformerFusion(
            d_model=64, n_heads=2, n_layers=1,
            use_cheap=True, cheap_dim=None
        )
        
        batch = {
            "cheap_features": torch.randn(4, actual_cheap_dim),
            "url_input_ids": torch.randint(0, 100, (4, 32)),
            "js_input_ids": torch.randint(0, 100, (4, 32)),
        }
        
        # Should work without crashes
        logits = model_lazy(batch)
        assert logits.shape == (4, 1)
        
        # Model with explicit dimension
        model_explicit = CrossModalTransformerFusion(
            d_model=64, n_heads=2, n_layers=1,
            use_cheap=True, cheap_dim=actual_cheap_dim
        )
        
        logits2 = model_explicit(batch)
        assert logits2.shape == (4, 1)
    
    def test_prediction_standardization_pipeline(self):
        """Test that prediction standardization works across the pipeline."""
        from phisdom.utils.prediction_standardizer import (
            standardize_prediction_format,
            validate_prediction_format,
            save_standardized_predictions
        )
        import numpy as np
        
        # Create mock predictions
        ids = [f"sample_{i}" for i in range(10)]
        labels = np.array([i % 2 for i in range(10)])
        
        # Test normal orientation
        probs_normal = np.array([0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])
        preds_normal, meta_normal = standardize_prediction_format(
            ids, labels, probs_normal, "test_head", "test", auto_flip=True
        )
        assert not meta_normal["flipped"]
        
        # Test inverted orientation (should auto-flip)
        probs_inverted = 1.0 - probs_normal
        preds_inverted, meta_inverted = standardize_prediction_format(
            ids, labels, probs_inverted, "test_head", "test", auto_flip=True
        )
        assert meta_inverted["flipped"]
        
        # Save and validate
        test_dir = os.path.join(self.tmpdir, "test_head_output")
        save_standardized_predictions(preds_normal, meta_normal, test_dir, "test")
        
        pred_path = os.path.join(test_dir, "preds_test.jsonl")
        diagnostics = validate_prediction_format(pred_path)
        assert diagnostics["valid"]
        assert diagnostics["num_predictions"] == 10
    
    def test_alignment_robustness(self):
        """Test that alignment handles missing predictions gracefully."""
        from phisdom.utils.alignment import InnerJoinAlignment, CoverageMaximizingAlignment
        
        # Create test JSONL
        jsonl_path = os.path.join(self.tmpdir, "test_alignment.jsonl")
        with open(jsonl_path, "w") as f:
            for item in self.val_data:
                f.write(json.dumps(item) + "\n")
        
        # Create head directories with missing predictions
        head_dirs = {}
        for i, head_name in enumerate(["head_a", "head_b", "head_c"]):
            head_dir = os.path.join(self.tmpdir, head_name)
            os.makedirs(head_dir)
            head_dirs[head_name] = head_dir
            
            # Each head missing different samples
            preds = []
            for j, item in enumerate(self.val_data):
                if (i + j) % 3 != 0:  # Skip some predictions
                    preds.append({
                        "id": item["id"],
                        "label": item["label"],
                        "prob": 0.5 + 0.3 * (1 if item["label"] else -1)
                    })
            
            pred_path = os.path.join(head_dir, "preds_alignment.jsonl")
            with open(pred_path, "w") as f:
                for pred in preds:
                    f.write(json.dumps(pred) + "\n")
        
        # Test inner join (strict)
        inner_aligner = InnerJoinAlignment()
        X_inner, y_inner, ids_inner, names_inner = inner_aligner.align(
            jsonl_path, head_dirs, use_cheap_features=False
        )
        
        # Should have fewer samples due to strict intersection
        assert len(ids_inner) < len(self.val_data)
        assert X_inner.shape[1] == 3  # 3 heads
        
        # Test coverage maximizing (with imputation)
        coverage_aligner = CoverageMaximizingAlignment(min_heads=2)
        X_cov, y_cov, ids_cov, names_cov = coverage_aligner.align(
            jsonl_path, head_dirs, use_cheap_features=False
        )
        
        # Should have more samples than inner join
        assert len(ids_cov) > len(ids_inner)
        assert X_cov.shape[1] == 3  # 3 heads
    
    def test_cascade_graceful_failure(self):
        """Test that cascade handles missing fusion predictions gracefully."""
        import sys
        import subprocess
        from io import StringIO
        # Create mock URL predictions (cheap_mlp removed)
        for head_name in ["url_head"]:
            head_dir = os.path.join(self.tmpdir, head_name)
            os.makedirs(head_dir, exist_ok=True)
            for split in ["val", "test"]:
                split_data = self.val_data if split == "val" else self.test_data_split
                preds = [{
                    "id": item["id"],
                    "label": item["label"],
                    "prob": 0.5 + 0.3 * (1 if item["label"] else -1)
                } for item in split_data]
                pred_path = os.path.join(head_dir, f"preds_{split}.jsonl")
                with open(pred_path, "w") as f:
                    for pred in preds:
                        f.write(json.dumps(pred) + "\n")
        
        # Create cascade output directory
        cascade_dir = os.path.join(self.tmpdir, "cascade_test")
        os.makedirs(cascade_dir)
        
        # Test cascade with missing fusion directory
        fake_fusion_dir = os.path.join(self.tmpdir, "nonexistent_fusion")
        
        # Mock sys.argv for cascade script
        old_argv = sys.argv
        old_stdout = sys.stdout  # Define old_stdout in proper scope
        try:
            sys.argv = [
                "cascade.py",
                "--url-dir", os.path.join(self.tmpdir, "url_head"),
                # removed --cheap-dir
                "--fusion-dir", fake_fusion_dir,
                "--val-jsonl", os.path.join(self.tmpdir, "pages_val.jsonl"),
                "--test-jsonl", os.path.join(self.tmpdir, "pages_test.jsonl"),
                "--out-dir", cascade_dir
            ]
            
            # Capture output
            sys.stdout = StringIO()
            
            # Run cascade script as subprocess instead of importing
            cascade_script = os.path.join(os.path.dirname(self.tmpdir), "scripts", "cascade.py")
            if not os.path.exists(cascade_script):
                cascade_script = "scripts/cascade.py"
            
            result = subprocess.run([
                sys.executable, cascade_script,
                "--url-dir", os.path.join(self.tmpdir, "url_head"),
                # removed --cheap-dir
                "--fusion-dir", fake_fusion_dir,
                "--val-jsonl", os.path.join(self.tmpdir, "pages_val.jsonl"),
                "--test-jsonl", os.path.join(self.tmpdir, "pages_test.jsonl"),
                "--out-dir", cascade_dir
            ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(self.tmpdir)))
            # Check that cascade completed (may have warnings but shouldn't crash)
            # The cascade should handle missing fusion gracefully
            
            # Check that cascade.json was created with error info or dummy results
            cascade_result_path = os.path.join(cascade_dir, "cascade.json")
            if os.path.exists(cascade_result_path):
                with open(cascade_result_path) as f:
                    result_data = json.load(f)
                # Should have either error info or dummy results
                assert "error" in result_data or "coverage" in result_data
            else:
                # If no result file, check that process indicated graceful handling
                print(f"Cascade stderr: {result.stderr}")
                # Should not have crashed with unhandled exception
                assert "Traceback" not in result.stderr or "gracefully" in result.stderr.lower()
            
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout


def test_cheap_features_consistency():
    """Test that cheap features are consistently computed."""
    from phisdom.data.cheap_features import CHEAP_FEATURES, row_to_features
    
    # Test with minimal row
    minimal_row = {"id": "test", "url_len": 25, "has_at": True}
    features = row_to_features(minimal_row, True)
    
    assert len(features) == len(CHEAP_FEATURES)
    assert isinstance(features[CHEAP_FEATURES.index("url_len")], float)
    assert features[CHEAP_FEATURES.index("has_at")] == 1.0  # True -> 1.0
    
    # Test with empty row
    empty_features = row_to_features({}, True)
    assert len(empty_features) == len(CHEAP_FEATURES)
    assert all(f == 0.0 for f in empty_features)  # All should be 0.0


def test_model_weight_initialization():
    """Test that fusion model weights initialize properly."""
    import torch
    from phisdom.models.fusion import CrossModalTransformerFusion
    
    torch.manual_seed(42)  # Deterministic initialization
    
    model = CrossModalTransformerFusion(
        d_model=128, n_heads=4, n_layers=2,
        use_url=True, use_js=True, use_text=True, use_dom=True, use_cheap=True,
        cheap_dim=None  # lazy
    )
    
    # Check that type embeddings are initialized
    assert model.type_emb.weight.requires_grad
    assert not torch.allclose(model.type_emb.weight, torch.zeros_like(model.type_emb.weight))
    
    # Check that classifier is initialized
    assert model.cls.weight.requires_grad
    assert model.cls.bias.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
