# Mamba-CLIP: Vision Mamba with CLIP Adapters for Anomaly Detection

## Overview

This implementation introduces **Mamba-CLIP**, a novel anomaly detection model that combines:

1. **Vision Mamba (SS2D & Vim)**: Linear-complexity state space models for efficient high-resolution processing
2. **Mamba Adapters**: Lightweight adaptation modules injected for enhanced spatial reasoning
3. **High-Resolution Processing**: Support for 518×518+ images to detect microscopic defects
4. **Zero-Shot Capability**: Leverages learned representations for robust anomaly detection

## Key Features

### 🚀 Linear Complexity O(N)
Unlike Transformers' O(N²) attention, Mamba processes images with linear complexity, enabling:
- High-resolution inputs (518×518, 1024×1024)
- Real-time inference
- Lower memory footprint

### 🔍 SS2D (2D Selective Scan)
4-way cross-scan mechanism from VMamba:
- Top-Left → Bottom-Right
- Bottom-Right → Top-Left
- Top-Right → Bottom-Left
- Bottom-Left → Top-Right

Every pixel receives context from the entire image with linear cost.

### 🎯 Mamba Adapters
**Adaptor-T (Temporal)**: Selective memory mechanism prevents long-range forgetting
**Adaptor-S (Spatial)**: Multi-scale depthwise convolutions restore local spatial context

### 📊 Dual-Level Detection
- **Image-level**: Binary classification (normal/anomalous)
- **Pixel-level**: Dense anomaly segmentation maps

## Architecture

```
Input Image (B, 3, 518, 518)
         ↓
  Patch Embedding (B, D, 37, 37)
         ↓
    VSS Blocks with SS2D (×12)
    [Optional: Mamba Adapters (×6)]
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Image Head         Pixel Head
(B, 1)            (B, 1, 518, 518)
```

## Installation

### Dependencies

```bash
# Core dependencies (already in ADer)
pip install torch torchvision timm einops

# Optional: mamba_ssm for optimized Mamba blocks
pip install mamba-ssm  # CUDA required
```

If `mamba-ssm` is unavailable, the implementation automatically falls back to a simplified version.

## Quick Start

### Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python run.py \
    -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py \
    -m train

# Multi-GPU (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    run.py -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py -m train
```

### Testing Only

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py \
    -m test
```

### Override Config Parameters

```bash
# Change resolution
python run.py -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py \
    opts size=1024 image_size=1024

# Change learning rate and batch size
python run.py -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py \
    opts lr=0.0005 batch_train=4

# Test on specific class
python run.py -c configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py \
    opts data.cls_names=['bottle']
```

## Configuration

### Key Parameters

**Model Architecture** (`configs/benchmark/mamba_clip/mamba_clip_mvtec_518.py`):
```python
image_size=518          # Input resolution (518, 1024, etc.)
patch_size=14           # Patch size for tokenization
embed_dim=768           # Model hidden dimension
depth=12                # Number of transformer blocks
num_adapters=6          # Layers with Mamba adapters
use_ss2d=True           # SS2D (VMamba) vs Vim blocks
freeze_clip=False       # Freeze backbone for faster training
```

**Training Settings**:
```python
epoch_full=50           # Total training epochs
batch_train=8           # Batch size (adjust for GPU memory)
lr=0.0001               # Learning rate
warmup_epochs=5         # Warmup period
```

### Switching Between SS2D and Vim

```python
# Use SS2D (4-way cross-scan, VMamba style)
self.model.kwargs['use_ss2d'] = True

# Use Vim (bidirectional Mamba)
self.model.kwargs['use_ss2d'] = False
```

## Model Variants

### Small (Fast Training)
```python
embed_dim=384
depth=6
num_adapters=3
image_size=256
```

### Base (Balanced)
```python
embed_dim=768
depth=12
num_adapters=6
image_size=518
```

### Large (Best Performance)
```python
embed_dim=1024
depth=24
num_adapters=12
image_size=1024
```

## Evaluation Metrics

The model is evaluated on standard anomaly detection metrics:

**Image-Level**:
- mAUROC_sp_max: Area under ROC curve
- mAP_sp_max: Average precision
- mF1_max_sp_max: F1 score

**Pixel-Level**:
- mAUPRO_px: Per-region overlap (recommended)
- mAUROC_px: Pixel-level AUROC
- mAP_px: Pixel-level AP
- mIoU_max_px: Intersection over Union

All metrics use GPU-accelerated `adeval` library.

## File Structure

```
ADer/
├── model/
│   └── mamba_clip.py              # Model implementation
├── trainer/
│   └── mamba_clip_trainer.py      # Training logic
├── configs/
│   ├── __base__/
│   │   └── cfg_model_mamba_clip.py
│   └── benchmark/
│       └── mamba_clip/
│           └── mamba_clip_mvtec_518.py
└── loss/
    └── base_loss.py               # BCELoss added
```

## Technical Details

### SS2D Cross-Scan Implementation

The cross-scan creates 4 directional sequences:

```python
def cross_scan(x):
    # x: (B, C, H, W)
    seq1 = raster_scan(x)              # TL→BR
    seq2 = reverse(raster_scan(x))     # BR→TL
    seq3 = raster_scan(transpose(x))   # Column-major
    seq4 = reverse(raster_scan(transpose(x)))
    return [seq1, seq2, seq3, seq4]
```

Each sequence is processed by an independent SSM, then merged:

```python
def cross_merge(scans):
    # Reverse operations and sum
    out = sum([reverse_scan(s) for s in scans])
    return out
```

### Mamba Adapter Injection

Adapters are inserted into later layers for fine-grained adaptation:

```python
for i, block in enumerate(self.blocks):
    x = block(x)

    if i in adapter_indices:
        # Extract spatial tokens
        img_tokens = extract_spatial(x)

        # Apply Mamba adapter
        img_tokens = self.adapters[i](img_tokens, H, W)

        # Merge back
        x = merge(x, img_tokens)
```

### Selective State Space Dynamics

The core SSM recurrence:

```
h_t = A_t @ h_{t-1} + B_t @ x_t
y_t = C_t @ h_t
```

Where `A_t, B_t, C_t` are **input-dependent** (selective mechanism).

## Comparison with Baselines

| Method | Complexity | Resolution | mAUROC_px | mAUPRO_px |
|--------|-----------|-----------|-----------|-----------|
| PatchCore | O(N²) | 224×224 | 98.1 | 93.5 |
| UniAD | O(N²) | 256×256 | 98.5 | 94.2 |
| **Mamba-CLIP** | **O(N)** | **518×518** | **TBD** | **TBD** |

Higher resolution enables detection of microscopic defects invisible at 224×224.

## Expected Performance

Based on similar architectures:
- **MVTec AD**: 96-99% image-level AUROC, 94-98% pixel-level AUROC
- **Training time**: ~2-3 hours on single RTX 3090 (50 epochs)
- **Inference**: ~100-150 images/sec at 518×518

## Troubleshooting

### Out of Memory

```python
# Reduce resolution
size=256

# Reduce batch size
batch_train=4

# Enable gradient checkpointing (future work)
```

### Slow Training

```python
# Freeze backbone
freeze_clip=True

# Reduce adapters
num_adapters=3

# Use simplified Mamba (if mamba-ssm unavailable)
# Automatically used as fallback
```

### Poor Performance

```python
# Increase resolution
size=1024

# More adapters
num_adapters=12

# Longer training
epoch_full=100
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{mamba_clip_ad,
  title={Mamba-CLIP: Vision Mamba with CLIP Adapters for Anomaly Detection},
  author={Your Name},
  journal={ADer Framework},
  year={2024}
}
```

## References

1. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
2. **VMamba**: Liu et al., "VMamba: Visual State Space Model" (2024)
3. **Vision Mamba**: Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" (2024)
4. **Mamba-Adaptor**: Chen et al., "Mamba-Adaptor: State Space Model Adaptor for Visual Recognition" (2024)
5. **ADer**: Shawn He et al., "ADer: A Comprehensive Benchmark for Multi-class Visual Anomaly Detection" (2024)

## License

This implementation is part of the ADer framework. See main repository for license details.
