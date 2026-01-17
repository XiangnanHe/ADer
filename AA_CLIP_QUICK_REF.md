# AA-CLIP Quick Reference

## Quick Start

```bash
cd /mnt/task_runtime/ADer

# Train AA-CLIP on MVTec-AD
python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train

# Test only
python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m test

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train --dist
```

## Compare with Other Methods

```bash
# Generate comparison table
python compare_runs.py

# Convert to paper format
python csv_to_paper_table.py --format latex
python csv_to_paper_table.py --format markdown
```

## Key Features

- **Two-stage training**: Text adapter (5 epochs) → Image adapter (20 epochs)
- **High resolution**: 518x518 for fine-grained defect detection
- **Multi-level features**: Extracts from layers [6, 12, 18, 24]
- **Adapter-based**: Only trains small adapter modules, base CLIP frozen

## Configuration Files

- Model: `/mnt/task_runtime/ADer/model/aaclip.py`
- Trainer: `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py`
- Config: `/mnt/task_runtime/ADer/configs/benchmark/aa_clip/aaclip_mvtec_518.py`

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image size | 518 | High resolution |
| Batch size | 2 | Small due to high-res |
| Text epochs | 5 | Stage 1 duration |
| Image epochs | 20 | Stage 2 duration |
| Text LR | 0.00001 | Stage 1 learning rate |
| Image LR | 0.0005 | Stage 2 learning rate |

## Expected Performance (MVTec-AD)

- Image AUROC: ~95-98%
- Pixel AUROC: ~96-98%
- Pixel AUPRO: ~92-95%

## Memory Requirements

- Training: ~20GB GPU memory (batch_size=2)
- Testing: ~10GB GPU memory

## Output Location

Results saved in:
```
runs/AAClipTrainer_configs_benchmark_aa_clip_aaclip_mvtec_518_YYYYMMDD-HHMMSS/
```

## See Also

- Full documentation: `AA_CLIP_INTEGRATION.md`
- Test script: `test_aaclip_integration.py`
- Original AA-CLIP: `/mnt/task_runtime/AA-CLIP/`
