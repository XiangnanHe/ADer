# AA-CLIP Integration for ADer

This directory contains the configuration and implementation for running AA-CLIP (Adapting CLIP for Anomaly Detection) within the ADer benchmark framework.

## Quick Start

```bash
# Train AA-CLIP on MVTec-AD dataset
conda activate ader
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/benchmark/aa_clip/aaclip_mvtec_518 -m train

# Test only (requires trained model)
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/benchmark/aa_clip/aaclip_mvtec_518 -m test
```

## Architecture

AA-CLIP uses a two-stage training approach:

### Stage 1: Text Adapter Training (5 epochs)
- Trains learnable text adapter modules to generate anomaly-aware text embeddings
- Uses multi-level patch features from frozen CLIP encoder (layers 6, 12, 18, 24)
- Features are normalized and enhanced with class token residuals
- Loss: Segmentation loss (Dice + Focal) + Orthogonal constraint on normal/anomaly embeddings

### Stage 2: Image Adapter Training (20 epochs)
- Trains learnable image adapter modules to align visual features with text embeddings
- Text embeddings from Stage 1 are frozen
- Loss: Image-level classification (Cross-Entropy) + Pixel-level segmentation

## Key Implementation Details

### Model Configuration
- **Base Model**: ViT-L-14 (CLIP Large with 14×14 patch size)
- **Input Resolution**: 518×518 (high resolution for fine-grained anomalies)
- **Patch Dropout**: Disabled (maintains full 37×37 spatial resolution)
- **Positional Embeddings**: Interpolated from 336px → 518px using bicubic interpolation

### Training Configuration
- **Batch Size**: 2 (due to high resolution)
- **Text Adapter LR**: 1e-5
- **Image Adapter LR**: 5e-4
- **Optimizer**: Adam with β=(0.9, 0.999)
- **Feature Levels**: [6, 12, 18, 24] (multi-scale features)

### Critical Implementation Details

1. **Feature Normalization**: All patch features are L2-normalized before computing similarity
2. **Class Token Residual**: Normalized class token is added to patch features for global context
3. **Multi-level Supervision**: Stage 1 uses features from 4 different layers
4. **Domain-specific Post-processing**: Industrial domain uses σ=1, kernel=7 for Gaussian smoothing

## Files

- `aaclip_mvtec_518.py` - Main configuration for MVTec-AD benchmark
- `/mnt/task_runtime/ADer/model/aaclip.py` - Model wrapper with namespace isolation
- `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py` - Two-stage trainer implementation

## Expected Performance

On MVTec-AD dataset, AA-CLIP should achieve:
- Image-level AUROC: ~95-98%
- Pixel-level AUROC: ~95-97%
- Pixel-level AUPRO: ~90-95%

## Troubleshooting

### Low Metrics
- Ensure features are normalized (line 148 in trainer)
- Verify class token is added (line 150 in trainer)
- Check that all 4 feature levels are used (line 169-183 in trainer)

### Training Errors
- **ZeroDivisionError in FocalLoss**: Fixed in Stage 2 by using CrossEntropyLoss instead
- **Shape mismatch in similarity map**: Ensure patch dropout is disabled
- **Module import conflicts**: Uses isolated `_aaclip_model` package structure

## Comparison with Other Methods

After training completes, compare results using:

```bash
# Compare all runs in the runs/ directory
python tools/compare_runs.py

# Generate paper-formatted table
python tools/csv_to_paper_table.py runs/comparison.csv
```

## References

- Original AA-CLIP: https://github.com/xcyao00/AA-CLIP
- Paper: "Adapting CLIP for Anomaly Detection via Few-Shot Learning"
- ADer Framework: https://github.com/zhangzjn/ADer
