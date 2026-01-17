#!/bin/bash
# Quick test to verify AA-CLIP can be loaded in the ADer environment

cd /mnt/task_runtime/ADer

echo "Testing AA-CLIP integration..."
echo "================================"
echo ""

# Test dry-run to see if imports work
echo "Running config validation..."
python -c "
import sys
sys.path.insert(0, '.')

# Import config
from configs.benchmark.aa_clip.aaclip_mvtec_518 import cfg

config = cfg()
print('✓ Config loaded successfully')
print(f'  Trainer: {config.trainer.name}')
print(f'  Model: {config.model.name}')
print(f'  Epochs: {config.text_epochs} + {config.image_epochs} = {config.epoch_full}')
print(f'  Image size: {config.size}')
print(f'  Batch size: {config.batch_train}')
print('')
print('Integration ready! Run with:')
print('  CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train')
"
