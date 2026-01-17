# AA-CLIP Integration for ADer

This document describes how to run AA-CLIP within the ADer benchmark framework.

## Overview

AA-CLIP (Anomaly-Aware CLIP) has been integrated into ADer with a two-stage training approach:
- **Stage 1**: Train text adapter to learn anomaly-aware embeddings (5 epochs)
- **Stage 2**: Train image adapter to align with text embeddings (20 epochs)

## Requirements

1. **AA-CLIP Checkpoint**: The CLIP ViT-L-14-336 checkpoint must be available at:
   ```
   /mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt
   ```

2. **Dataset**: MVTec-AD dataset at:
   ```
   data/mvtec
   ```

## Files Created

### Model
- `/mnt/task_runtime/ADer/model/aaclip.py` - Model wrapper that loads AA-CLIP's AdaptedCLIP

### Trainer
- `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py` - Two-stage trainer implementation

### Configuration
- `/mnt/task_runtime/ADer/configs/__base__/cfg_model_aaclip.py` - Base model configuration
- `/mnt/task_runtime/ADer/configs/benchmark/aa_clip/aaclip_mvtec_518.py` - Benchmark config for MVTec-AD

## Running AA-CLIP

### Training

```bash
cd /mnt/task_runtime/ADer

# Train on MVTec-AD dataset with high resolution (518x518)
python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train
```

### Testing Only

```bash
# Test a trained model
python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m test
```

### Multi-GPU Training

```bash
# Use 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    -c configs.benchmark.aa_clip.aaclip_mvtec_518 \
    -m train \
    --dist
```

## Configuration Parameters

### Key Settings (in `aaclip_mvtec_518.py`)

```python
# Two-stage training
self.text_epochs = 5      # Stage 1 duration
self.image_epochs = 20    # Stage 2 duration

# Learning rates
self.text_lr = 0.00001    # LR for text adapter
self.image_lr = 0.0005    # LR for image adapter

# Batch sizes (adjust based on GPU memory)
self.batch_train = 2      # Small due to 518x518 resolution
self.batch_test_per = 4

# Image resolution
self.size = 518           # High-res for fine defects

# AA-CLIP specific
self.text_norm_weight = 0.1  # Orthogonal constraint weight
self.domain = 'Industrial'   # 'Industrial' or 'Medical'
```

### Adapter Configuration

```python
self.model.kwargs = dict(
    text_adapt_weight=0.1,      # Residual weight for text adapter
    image_adapt_weight=0.1,     # Residual weight for image adapter
    text_adapt_until=3,         # Adapt first 3 text layers
    image_adapt_until=6,        # Adapt first 6 visual layers
    levels=[6, 12, 18, 24],    # Multi-level feature extraction
)
```

## Training Process

### Stage 1: Text Adapter (5 epochs)
- Freezes image adapter
- Trains text adapter to generate anomaly-aware embeddings
- Uses segmentation loss (Dice + Focal) + orthogonal constraint
- Learns to distinguish "normal" vs "anomalous" descriptions

### Stage 2: Image Adapter (20 epochs)
- Freezes text adapter
- Trains image adapter to align with learned text embeddings
- Uses image-level classification loss + pixel-level segmentation loss
- Adapts visual features to match anomaly-aware text embeddings

## Output and Results

### Log Files
Results are saved in `runs/AAClipTrainer_*/`:
```
runs/AAClipTrainer_configs_benchmark_aa_clip_aaclip_mvtec_518_YYYYMMDD-HHMMSS/
├── log_train.txt         # Training logs
├── checkpoints/          # Model checkpoints
│   ├── epoch_5.pth       # After stage 1
│   ├── epoch_25.pth      # After stage 2
│   └── best.pth          # Best model
└── tensorboard/          # TensorBoard logs
```

### Metrics

The following metrics are evaluated:

**Image-Level (Detection)**:
- `mAUROC_sp_max` - Image AUROC (primary metric)
- `mAP_sp_max` - Image AP
- `mF1_max_sp_max` - Image F1

**Pixel-Level (Localization)**:
- `mAUPRO_px` - Pixel AUPRO (recommended)
- `mAUROC_px` - Pixel AUROC
- `mAP_px` - Pixel AP
- `mF1_max_px` - Pixel F1
- `mIoU_max_px` - Pixel IoU

### Comparing Results

After training, use the comparison scripts:

```bash
# Generate comparison CSV with all runs
python compare_runs.py

# Convert to paper-ready table
python csv_to_paper_table.py --format csv
python csv_to_paper_table.py --format latex
python csv_to_paper_table.py --format markdown
```

This will show AA-CLIP results alongside other methods (InvAD, MAMBAADTrainer, etc.).

## Memory Requirements

- **Training**: ~20GB GPU memory per GPU (batch_size=2, image_size=518)
- **Testing**: ~10GB GPU memory

To reduce memory usage:
1. Decrease batch size: `self.batch_train = 1`
2. Decrease image size: `self.size = 336` (but may hurt performance)
3. Use gradient checkpointing (requires model modification)

## Troubleshooting

### Issue: "No CLIP checkpoint found"
**Solution**: Ensure checkpoint exists at `/mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt`

### Issue: Out of memory
**Solution**:
- Reduce batch size in config
- Reduce image size to 336
- Use fewer GPUs with larger per-GPU batch size

### Issue: "ImportError: No module named 'model.adapter'"
**Solution**: The trainer adds AA-CLIP to Python path automatically via `sys.path.insert(0, '/mnt/task_runtime/AA-CLIP')`

### Issue: Text embeddings not found
**Solution**: The trainer automatically generates text embeddings after Stage 1. Ensure dataset name is correctly detected (MVTec, VisA, BTAD).

## Differences from Original AA-CLIP

1. **Framework Integration**: Uses ADer's trainer base class and data loaders
2. **Config System**: Uses ADer's configuration system instead of argparse
3. **Logging**: Integrated with ADer's logging and TensorBoard
4. **Evaluation**: Uses ADer's evaluator for consistent metrics across methods
5. **Multi-GPU**: Supports distributed training via ADer's DDP wrapper

## Expected Performance

Based on original AA-CLIP paper, expected performance on MVTec-AD:

| Metric | Expected Range |
|--------|----------------|
| Image-AUROC | 95-98% |
| Pixel-AUROC | 96-98% |
| Pixel-AUPRO | 92-95% |

Results may vary based on:
- Number of training epochs
- Batch size and learning rate
- Image resolution
- Hardware (GPU type)

## Citation

If you use AA-CLIP in your research, please cite:

```bibtex
@inproceedings{aaclip2024,
  title={AA-CLIP: Anomaly-Aware CLIP for Zero-shot and Few-shot Anomaly Detection},
  author={...},
  booktitle={...},
  year={2024}
}
```

## Next Steps

1. **Train AA-CLIP**: Run the training command above
2. **Compare Results**: Use `compare_runs.py` to compare with other methods
3. **Tune Hyperparameters**: Adjust learning rates, batch sizes, or epochs if needed
4. **Try Other Datasets**: Modify config for VisA, BTAD, or other datasets
