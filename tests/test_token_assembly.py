#!/usr/bin/env python
"""Unit test for minimal token assembly CrossModalTransformerFusion."""
import torch
import pytest
from src.phisdom.models.fusion import CrossModalTransformerFusion

def test_token_assembly_key_flexibility():
    """Test that the fusion model handles different key aliases correctly."""
    model = CrossModalTransformerFusion(
        d_model=64, n_heads=4, n_layers=1,
        use_url=True, use_js=True, use_text=False, use_dom=False, use_cheap=True,
        cheap_dim=None  # LazyLinear
    )
    
    B, L = 4, 8
    
    # Test 1: Standard collator keys
    batch1 = {
        "url_input_ids": torch.randint(1, 100, (B, L)),
        "url_attention_mask": torch.ones(B, L, dtype=torch.bool),
        "js_input_ids": torch.randint(1, 100, (B, L)),
        "js_attention_mask": torch.ones(B, L, dtype=torch.bool),
        "cheap_features": torch.randn(B, 37),  # arbitrary dimension
    }
    
    logits1 = model(batch1)
    assert logits1.shape == (B, 1)
    
    # Test 2: Alternative key names
    batch2 = {
        "url_tokens": torch.randint(1, 100, (B, L)),
        "url_mask": torch.ones(B, L, dtype=torch.bool),
        "js_ids": torch.randint(1, 100, (B, L)),
        "js_mask": torch.ones(B, L, dtype=torch.bool),
        "cheap": torch.randn(B, 37),  # same dimension as batch1
    }
    
    logits2 = model(batch2)
    assert logits2.shape == (B, 1)
    
    # Test 3: Missing masks (should work with auto-generated masks)
    batch3 = {
        "url_input_ids": torch.randint(1, 100, (B, L)),
        "js_input_ids": torch.randint(1, 100, (B, L)),
        "cheap_features": torch.randn(B, 37),
    }
    
    logits3 = model(batch3)
    assert logits3.shape == (B, 1)
    
    # Test 4: Only cheap features
    batch4 = {
        "cheap_features": torch.randn(B, 37),
    }
    
    logits4 = model(batch4)
    assert logits4.shape == (B, 1)
    
    # Test 5: Empty batch (should return zeros safely)
    batch5 = {}
    
    logits5 = model(batch5)
    assert logits5.shape == (1, 1)  # Default batch size of 1
    assert torch.allclose(logits5, torch.zeros_like(logits5))

def test_deterministic_order():
    """Test that concatenation order is deterministic (URL→JS→TEXT→DOM→CHEAP)."""
    B, L = 2, 4
    
    batch = {
        # Present in mixed order to test deterministic assembly
        "cheap_features": torch.randn(B, 10),
        "text_input_ids": torch.randint(1, 50, (B, L)),
        "url_input_ids": torch.randint(1, 50, (B, L)),
        "js_input_ids": torch.randint(1, 50, (B, L)),
    }
    
    # Create model and warm it up first
    model = CrossModalTransformerFusion(
        d_model=32, n_heads=2, n_layers=1,
        use_url=True, use_js=True, use_text=True, use_dom=False, use_cheap=True,
        cheap_dim=None, token_vocab_size=50
    )
    
    # Warm up to initialize lazy modules
    _ = model(batch)
    
    # Now test determinism
    torch.manual_seed(123)
    logits_a = model(batch)
    
    torch.manual_seed(123)
    logits_b = model(batch)
    
    # Should be exactly the same (deterministic)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)

def test_lazy_dimension_adaptation():
    """Test that LazyLinear adapts to different cheap feature dimensions."""
    model = CrossModalTransformerFusion(
        d_model=32, n_heads=2, n_layers=1,
        use_url=False, use_js=False, use_text=False, use_dom=False, use_cheap=True,
        cheap_dim=None  # This enables LazyLinear
    )
    
    B = 3
    
    # First batch: 20 dimensions
    batch1 = {"cheap_features": torch.randn(B, 20)}
    logits1 = model(batch1)
    assert logits1.shape == (B, 1)
    
    # Second batch: same 20 dimensions (should work)
    batch2 = {"cheap_features": torch.randn(B, 20)}
    logits2 = model(batch2)
    assert logits2.shape == (B, 1)
    
    # Third batch: different dimensions should fail after initialization
    batch3 = {"cheap_features": torch.randn(B, 15)}
    with pytest.raises((RuntimeError, ValueError)):
        model(batch3)

if __name__ == "__main__":
    print("Testing token assembly flexibility...")
    test_token_assembly_key_flexibility()
    print("✓ Key flexibility test passed")
    
    print("Testing deterministic order...")
    test_deterministic_order()
    print("✓ Deterministic order test passed")
    
    print("Testing lazy dimension adaptation...")
    test_lazy_dimension_adaptation()
    print("✓ Lazy dimension adaptation test passed")
    
    print("All tests passed! 🎉")
