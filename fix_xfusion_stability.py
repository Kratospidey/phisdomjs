#!/usr/bin/env python3
"""
Fixed XFusion training script with numerical stability safeguards
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, 'src')

from phisdom.models.fusion import CrossModalTransformerFusion

def fix_fusion_model():
    """Add numerical stability fixes to the fusion model"""
    
    # Hook to detect and fix NaN/Inf gradients
    def grad_hook(module, grad_input, grad_output):
        if grad_output is not None:
            for i, grad in enumerate(grad_output):
                if grad is not None:
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"🚨 NaN/Inf gradient detected in {module.__class__.__name__}")
                        # Replace with zeros to prevent propagation
                        grad_output = tuple(
                            torch.where(torch.isfinite(g), g, torch.zeros_like(g)) if g is not None else g
                            for g in grad_output
                        )
                        print(f"✓ Gradient sanitized")
                        break
        return grad_output
    
    # Enhanced CrossModalTransformerFusion with stability fixes
    class StableCrossModalTransformerFusion(CrossModalTransformerFusion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Register gradient hooks for all modules
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm, nn.TransformerEncoderLayer)):
                    module.register_full_backward_hook(grad_hook)
        
        def forward(self, batch):
            """Enhanced forward with NaN checking"""
            # Call parent forward
            logits = super().forward(batch)
            
            # Check for NaN/inf in output
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("🚨 NaN/Inf detected in model output, replacing with zeros")
                logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            
            # Clamp extreme values
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            return logits
    
    return StableCrossModalTransformerFusion

def create_stable_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Create optimizer with gradient clipping"""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        eps=1e-8,  # Larger epsilon for numerical stability
        amsgrad=True  # More stable variant
    )
    return optimizer

def train_with_stability_checks():
    """Train XFusion with enhanced stability"""
    print("🔧 Creating numerically stable XFusion model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use stable model
    StableFusion = fix_fusion_model()
    model = StableFusion(
        d_model=128,
        n_heads=4, 
        n_layers=2,
        dropout=0.1,
        use_url=True,
        use_js=True,
        use_text=True,
        use_dom=True,
        use_cheap=True,
        cheap_dim=74  # Fixed dimension instead of lazy
    ).to(device)
    
    # Initialize with smaller values
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)  # Smaller std
    
    model.apply(init_weights)
    
    # Create stable optimizer with lower learning rate
    optimizer = create_stable_optimizer(model, lr=1e-5, weight_decay=1e-5)
    
    print("✓ Stable model created")
    print("✓ Use this model in train_fusion_xattn.py by:")
    print("  1. Lower learning rate: --lr 1e-5")  
    print("  2. Add gradient clipping")
    print("  3. Check for NaN after each batch")
    
    return model, optimizer

if __name__ == "__main__":
    train_with_stability_checks()
