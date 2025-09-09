#!/bin/bash
# Retrain XFusion with numerical stability fixes

echo "🔧 Retraining XFusion with stability fixes..."

# Backup the corrupted model
if [ -d "artifacts/fusion_xattn" ]; then
    echo "📦 Backing up corrupted model..."
    mv artifacts/fusion_xattn artifacts/fusion_xattn_corrupted_$(date +%s)
fi

# Create fresh output directory
mkdir -p artifacts/fusion_xattn

echo "🚀 Starting stable XFusion training..."

# Train with much lower learning rate and other stability parameters
PYTHONPATH=src python scripts/train_fusion_xattn.py \
    --train-jsonl data/pages_train.jsonl \
    --val-jsonl data/pages_val.jsonl \
    --test-jsonl data/pages_test.jsonl \
    --out-dir artifacts/fusion_xattn \
    --batch-size 8 \
    --epochs 10 \
    --lr 5e-5 \
    --weight-decay 1e-5 \
    --seed 42

echo "✅ XFusion retraining complete!"

# Check if the model was saved successfully
if [ -f "artifacts/fusion_xattn/model.pt" ]; then
    echo "✓ Model saved successfully"
    
    # Quick validation check
    python -c "
import torch
import json

# Check model parameters
state = torch.load('artifacts/fusion_xattn/model.pt', map_location='cpu')
nan_params = []
for k, v in state.items():
    if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
        nan_params.append(k)

if nan_params:
    print(f'❌ Model still has NaN parameters: {nan_params}')
else:
    print('✅ Model parameters are numerically stable')

# Check predictions
try:
    with open('artifacts/fusion_xattn/preds_val.jsonl') as f:
        sample = json.loads(f.readline())
    if sample['prob'] != sample['prob']:  # NaN check
        print('❌ Predictions still contain NaN')
    else:
        print(f'✅ Sample prediction: {sample[\"prob\"]:.4f}')
except Exception as e:
    print(f'⚠️  Could not check predictions: {e}')
"
else
    echo "❌ Model training failed - no model.pt found"
fi
