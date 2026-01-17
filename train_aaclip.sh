#!/bin/bash
# AA-CLIP Training Script for ADer

cd /mnt/task_runtime/ADer

echo "=========================================="
echo "AA-CLIP Training on MVTec-AD"
echo "=========================================="
echo ""

# Check if checkpoint exists
CKPT_PATH="/mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: CLIP checkpoint not found at $CKPT_PATH"
    echo "Please ensure the checkpoint is available."
    exit 1
fi

echo "✓ CLIP checkpoint found (1.6GB)"
echo ""

# Check if data exists
DATA_PATH="data/mvtec"
if [ ! -d "$DATA_PATH" ]; then
    echo "WARNING: MVTec data not found at $DATA_PATH"
    echo "Make sure dataset is available before training."
fi

echo "Configuration:"
echo "  - Model: AA-CLIP (AdaptedCLIP)"
echo "  - Dataset: MVTec-AD"
echo "  - Image size: 518x518"
echo "  - Batch size: 2"
echo "  - Stage 1 (text): 5 epochs @ LR 0.00001"
echo "  - Stage 2 (image): 20 epochs @ LR 0.0005"
echo "  - Total: 25 epochs"
echo ""

# Prompt user
read -p "Start training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    echo ""

    # Run training (correct format: dots, no .py extension)
    CUDA_VISIBLE_DEVICES=0 python run.py \
        -c configs.benchmark.aa_clip.aaclip_mvtec_518 \
        -m train
else
    echo "Training cancelled."
    echo ""
    echo "To run manually:"
    echo "  CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train"
fi
