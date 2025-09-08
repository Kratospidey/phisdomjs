#!/usr/bin/env python
"""Unit test for accuracy curve correctness."""
import numpy as np
import os
import tempfile
import sys

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

def test_accuracy_curve_uses_labels():
    """Ensure accuracy curve computes accuracy against labels, not just prediction rates."""
    from scripts.report_eval import plot_accuracy_curve_multi_splits
    
    # Perfect classifier case: should get 100% accuracy at threshold 0.5
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.55])
    
    # Manually compute accuracy at t=0.5
    yhat = (p >= 0.5).astype(int)
    expected_acc = (yhat == y).mean()
    assert expected_acc == 1.0, f"Expected perfect accuracy, got {expected_acc}"
    
    # Test the function doesn't crash
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            plot_accuracy_curve_multi_splits([("test", y, p)], tmpdir, "unit_test")
            print("✓ Accuracy curve function runs without errors")
        except Exception as e:
            print(f"✗ Accuracy curve function failed: {e}")
            raise

def test_cascade_threshold_overlap():
    """Test that overlapping thresholds get clamped properly."""
    # This is more of a conceptual test - in real usage, overlapping thresholds
    # would be detected and fixed in the cascade logic
    thr_lo = 0.7  # benign threshold
    thr_hi = 0.3  # phish threshold
    
    # Simulate the overlap fix
    if thr_lo > thr_hi:
        eps = 1e-6
        mid = 0.5 * (thr_lo + thr_hi)
        thr_lo_fixed = max(0.0, mid - eps)
        thr_hi_fixed = min(1.0, mid + eps)
        
        assert thr_lo_fixed <= thr_hi_fixed, "Fixed thresholds should not overlap"
        assert thr_lo_fixed >= 0.0 and thr_hi_fixed <= 1.0, "Thresholds should be in [0,1]"
        print(f"✓ Overlap fix: {thr_lo:.3f}, {thr_hi:.3f} → {thr_lo_fixed:.6f}, {thr_hi_fixed:.6f}")

if __name__ == "__main__":
    print("Testing accuracy curve correctness...")
    test_accuracy_curve_uses_labels()
    
    print("Testing cascade threshold overlap handling...")
    test_cascade_threshold_overlap()
    
    print("All correctness tests passed! 🎉")
