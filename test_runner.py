#!/usr/bin/env python
"""
Test runner for phisdom pipeline.
Runs all tests to ensure system robustness.
"""
from __future__ import annotations
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_basic_tests():
    """Run basic unit tests."""
    print("=" * 60)
    print("RUNNING BASIC UNIT TESTS")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent
    test_files = [
        "tests/test_fusion_fixes.py",
        "tests/test_calibration.py", 
        "tests/test_metrics.py",
        "tests/test_xfusion_smoke.py"
    ]
    
    for test_file in test_files:
        test_path = repo_root / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(test_path), "-v"
                ], cwd=repo_root, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} PASSED")
                else:
                    print(f"❌ {test_file} FAILED")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    return False
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} TIMEOUT")
                return False
            except Exception as e:
                print(f"💥 {test_file} ERROR: {e}")
                return False
        else:
            print(f"⚠️  {test_file} not found, skipping")
    
    return True


def test_fusion_model_smoke():
    """Test fusion model with different configurations."""
    print("=" * 60)
    print("TESTING FUSION MODEL ROBUSTNESS")
    print("=" * 60)
    
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from phisdom.models.fusion import CrossModalTransformerFusion
        
        # Test 1: Lazy cheap features (the fix for the original crash)
        print("Test 1: Lazy cheap features...")
        model_lazy = CrossModalTransformerFusion(
            d_model=128, n_heads=4, n_layers=1,
            use_cheap=True, cheap_dim=None  # This is the key fix
        )
        
        # Simulate the problematic 74-dimension case
        batch = {
            "cheap_features": torch.randn(8, 74),  # The exact failing case
            "url_input_ids": torch.randint(0, 100, (8, 32)),
            "js_input_ids": torch.randint(0, 100, (8, 32)),
        }
        
        logits = model_lazy(batch)
        assert logits.shape == (8, 1)
        print("✅ Lazy cheap features test PASSED")
        
        # Test 2: Different batch sizes and dimensions (create new model for each dimension)
        print("Test 2: Variable dimensions...")
        for batch_size in [1, 4, 16]:
            for cheap_dim in [32, 74, 128]:
                # Create fresh model for each dimension test
                model_var = CrossModalTransformerFusion(cheap_dim=None, token_vocab_size=100)
                batch_var = {
                    "cheap_features": torch.randn(batch_size, cheap_dim),
                    "url_input_ids": torch.randint(0, 100, (batch_size, 32)),
                }
                logits_var = model_var(batch_var)
                assert logits_var.shape == (batch_size, 1)
        print("✅ Variable dimensions test PASSED")
        
        # Test 3: Missing modalities (robustness)
        print("Test 3: Missing modalities...")
        batch_minimal = {"cheap_features": torch.randn(4, 74)}
        logits_minimal = model_lazy(batch_minimal)
        assert logits_minimal.shape == (4, 1)
        print("✅ Missing modalities test PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Fusion model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_standardization():
    """Test prediction format standardization."""
    print("=" * 60)
    print("TESTING PREDICTION STANDARDIZATION") 
    print("=" * 60)
    
    try:
        import numpy as np
        import tempfile
        import json
        
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from phisdom.utils.prediction_standardizer import (
            standardize_prediction_format,
            validate_prediction_format,
            save_standardized_predictions
        )
        
        # Test auto-flip detection
        print("Test 1: Auto-flip detection...")
        ids = ["s1", "s2", "s3", "s4"]
        labels = np.array([0, 0, 1, 1])
        probs_wrong = np.array([0.9, 0.8, 0.2, 0.1])  # Inverted
        
        preds, meta = standardize_prediction_format(
            ids, labels, probs_wrong, "test_head", "test", auto_flip=True
        )
        
        assert meta["flipped"], "Should have detected and flipped wrong orientation"
        assert preds[2]["prob"] > 0.5, "Phish sample should have high prob after flip"
        print("✅ Auto-flip detection PASSED")
        
        # Test validation
        print("Test 2: Format validation...")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_standardized_predictions(preds, meta, tmpdir, "test")
            
            pred_path = os.path.join(tmpdir, "preds_test.jsonl")
            diagnostics = validate_prediction_format(pred_path)
            
            assert diagnostics["valid"], f"Validation failed: {diagnostics['errors']}"
            assert diagnostics["num_predictions"] == 4
            print("✅ Format validation PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction standardization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment_strategies():
    """Test data alignment strategies."""
    print("=" * 60)
    print("TESTING ALIGNMENT STRATEGIES")
    print("=" * 60)
    
    try:
        import numpy as np
        import tempfile
        import json
        
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from phisdom.utils.alignment import InnerJoinAlignment, CoverageMaximizingAlignment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            test_data = [
                {"id": f"sample{i}", "label": i % 2}
                for i in range(10)
            ]
            
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")
            
            # Create head predictions (with some missing)
            head_dirs = {}
            for head_name in ["head1", "head2", "head3"]:
                head_dir = os.path.join(tmpdir, head_name)
                os.makedirs(head_dir)
                head_dirs[head_name] = head_dir
                
                # head3 missing last 3 samples
                n_samples = 7 if head_name == "head3" else 10
                preds = [
                    {"id": f"sample{i}", "label": i % 2, "prob": 0.3 + 0.4 * (i % 2)}
                    for i in range(n_samples)
                ]
                
                with open(os.path.join(head_dir, "preds_test.jsonl"), "w") as f:
                    for pred in preds:
                        f.write(json.dumps(pred) + "\n")
            
            # Test inner join
            print("Test 1: Inner join alignment...")
            inner_aligner = InnerJoinAlignment()
            X_inner, y_inner, ids_inner, names_inner = inner_aligner.align(
                jsonl_path, head_dirs, use_cheap_features=False
            )
            
            assert len(ids_inner) == 7, f"Expected 7 samples, got {len(ids_inner)}"  # Only samples 0-6
            assert X_inner.shape == (7, 3), f"Expected (7,3), got {X_inner.shape}"
            print("✅ Inner join alignment PASSED")
            
            # Test coverage maximizing
            print("Test 2: Coverage maximizing alignment...")
            cov_aligner = CoverageMaximizingAlignment(min_heads=2)
            X_cov, y_cov, ids_cov, names_cov = cov_aligner.align(
                jsonl_path, head_dirs, use_cheap_features=False
            )
            
            assert len(ids_cov) == 10, f"Expected 10 samples, got {len(ids_cov)}"  # All samples
            assert X_cov.shape == (10, 3), f"Expected (10,3), got {X_cov.shape}"
            
            # Check imputation for missing samples
            for i in range(7, 10):  # samples missing from head3
                sample_idx = ids_cov.index(f"sample{i}")
                assert X_cov[sample_idx, 2] == 0.5, "Missing head should be imputed with 0.5"
            
            print("✅ Coverage maximizing alignment PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Alignment strategies test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cheap_features_consistency():
    """Test cheap features computation consistency."""
    print("=" * 60)
    print("TESTING CHEAP FEATURES CONSISTENCY")
    print("=" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from phisdom.data.cheap_features import CHEAP_FEATURES, row_to_features
        
        print(f"Number of cheap features defined: {len(CHEAP_FEATURES)}")
        
        # Test with various inputs
        test_cases = [
            {},  # Empty
            {"url_len": 25, "has_at": True, "redirect_hops": 2},  # Partial
            {feat: i for i, feat in enumerate(CHEAP_FEATURES)},  # Complete
        ]
        
        for i, test_row in enumerate(test_cases):
            features = row_to_features(test_row, True)
            assert len(features) == len(CHEAP_FEATURES), f"Case {i}: Expected {len(CHEAP_FEATURES)} features, got {len(features)}"
            assert all(isinstance(f, (int, float)) for f in features), f"Case {i}: All features should be numeric"
            print(f"✅ Cheap features case {i+1} PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Cheap features test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run phisdom pipeline tests")
    parser.add_argument("--quick", action="store_true", help="Run only critical tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("🧪 PHISDOM PIPELINE ROBUSTNESS TESTS")
    print("=" * 60)
    
    tests = [
        ("Fusion Model Smoke Test", test_fusion_model_smoke),
        ("Prediction Standardization", test_prediction_standardization),
        ("Alignment Strategies", test_alignment_strategies),
        ("Cheap Features Consistency", test_cheap_features_consistency),
    ]
    
    if not args.quick:
        tests.insert(0, ("Basic Unit Tests", run_basic_tests))
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline is robust!")
        return 0
    else:
        print("💥 SOME TESTS FAILED - Fix issues before deployment!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
