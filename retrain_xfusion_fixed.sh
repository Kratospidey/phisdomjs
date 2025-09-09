#!/bin/bash

# XFusion Retraining with Cheap Feature Fix
# This script retrains XFusion with the numerical stability fixes

set -e

echo "🚀 Starting XFusion retraining with cheap feature normalization fix..."

# Backup the corrupted model
if [ -f "artifacts/fusion_xattn/model.pt" ]; then
    echo "📦 Backing up corrupted model..."
    cp artifacts/fusion_xattn/model.pt artifacts/fusion_xattn/model_corrupted_backup.pt
    echo "✅ Backup saved to artifacts/fusion_xattn/model_corrupted_backup.pt"
fi

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/fusion_xattn

# Train with stable parameters and cheap feature normalization
echo "🏋️ Training XFusion with normalization fix..."
python train_fusion_xattn_fixed.py \
    --train-jsonl data/pages_train.jsonl \
    --val-jsonl data/pages_val.jsonl \
    --test-jsonl data/pages_test.jsonl \
    --out-dir artifacts/fusion_xattn \
    --lr 5e-5 \
    --batch-size 8 \
    --epochs 10 \
    --weight-decay 1e-5 \
    --seed 42

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "✅ XFusion training completed successfully!"
    
    # Validate the new model
    echo "🔍 Validating the retrained model..."
    python -c "
import torch
import sys
sys.path.insert(0, 'src')
from phisdom.models.fusion import CrossModalTransformerFusion

print('Loading retrained model...')
try:
    model = torch.load('artifacts/fusion_xattn/model.pt', map_location='cpu')
    print('✅ Model loaded successfully')
    
    # Check for NaN parameters
    nan_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f'❌ {name}: has NaN/Inf values')
            nan_params += 1
        else:
            print(f'✅ {name}: OK')
    
    if nan_params == 0:
        print(f'🎉 All {total_params} parameters are clean!')
    else:
        print(f'💥 {nan_params}/{total_params} parameters have NaN/Inf')
        
except Exception as e:
    print(f'❌ Failed to load model: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "🎉 XFusion model is healthy and ready for use!"
        echo ""
        echo "Next steps:"
        echo "  1. Update Makefile to use train_fusion_xattn_fixed.py"
        echo "  2. Run 'make eval-xfusion' to test performance"
        echo "  3. Run 'make report' to generate plots"
    else
        echo "❌ Model validation failed"
        exit 1
    fi
else
    echo "❌ XFusion training failed"
    exit 1
fi
