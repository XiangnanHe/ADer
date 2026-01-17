# AA-CLIP Fixed - Ready to Run

## Issue Fixed

The import error has been resolved. The problem was with how Python imports were handled when AA-CLIP modules were loaded from ADer.

**Fixed files:**
- `/mnt/task_runtime/ADer/model/aaclip.py` - Uses importlib for safe imports
- `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py` - Uses importlib for AA-CLIP utilities

## Correct Command Format

The correct command uses **dots** (not slashes) and **no .py extension**:

```bash
cd /mnt/task_runtime/ADer

# Single GPU training
CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train
```

**NOT** (incorrect):
```bash
# Wrong - uses slashes and .py extension
python run.py -c configs/benchmark/aa_clip/aaclip_mvtec_518.py -m train
```

## Quick Start Options

### Option 1: Use the training script
```bash
cd /mnt/task_runtime/ADer
bash train_aaclip.sh
```

### Option 2: Run directly
```bash
cd /mnt/task_runtime/ADer
CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train
```

### Option 3: Multi-GPU
```bash
cd /mnt/task_runtime/ADer
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py -c configs.benchmark.aa_clip.aaclip_mvtec_518 -m train --dist
```

## What Happens During Training

### Stage 1: Text Adapter (5 epochs)
- Trains text encoder to learn anomaly-aware embeddings
- Learns to distinguish "normal bottle" vs "damaged bottle" etc.
- Uses segmentation loss + orthogonal constraint
- Image adapter stays frozen

### Stage 2: Image Adapter (20 epochs)
- Text adapter frozen (uses learned embeddings)
- Trains image encoder to align with text embeddings
- Uses image classification + pixel segmentation losses
- Adapts visual features to match anomaly concepts

### Results Location
```
runs/AAClipTrainer_configs_benchmark_aa_clip_aaclip_mvtec_518_YYYYMMDD-HHMMSS/
├── log_train.txt       # Training logs with metrics
├── checkpoints/        # Model checkpoints
│   ├── epoch_5.pth     # After stage 1
│   ├── epoch_25.pth    # After stage 2
│   └── best.pth        # Best model
└── tensorboard/        # TensorBoard logs
```

## Expected Timeline

On a single GPU (e.g., A100):
- Stage 1: ~15-30 minutes (5 epochs, 15 classes)
- Stage 2: ~60-120 minutes (20 epochs, 15 classes)
- **Total: ~1.5-2.5 hours**

## Comparing Results

After training completes, compare with other methods:

```bash
cd /mnt/task_runtime/ADer

# Generate comparison CSV
python compare_runs.py

# View results
cat runs_comparison.csv | grep -E "(AAClip|InvAD|MAMBAAD)"

# Generate paper tables
python csv_to_paper_table.py --format csv
python csv_to_paper_table.py --format latex
```

Example expected output:
```
Method,Dataset,Img-AUROC,Px-AUROC,Px-AUPRO
InvADTrainer,mvtec,99.0,98.0,93.8
MAMBAADTrainer,mvtec,97.6,97.4,93.5
AAClipTrainer,mvtec,~96-98,~96-98,~92-95  # Expected range
```

## Troubleshooting

### "No module named 'model.adapter'"
**Fixed!** This error has been resolved with the import fix.

### "CUDA out of memory"
Reduce batch size in config:
```python
self.batch_train = 1  # Reduce from 2
```

### "Dataset not found"
Ensure MVTec-AD is at:
```
/mnt/task_runtime/ADer/data/mvtec
```

## Files Created

Integration complete with these files:
- ✓ Model: `model/aaclip.py`
- ✓ Trainer: `trainer/aaclip_trainer.py`
- ✓ Base config: `configs/__base__/cfg_model_aaclip.py`
- ✓ Benchmark config: `configs/benchmark/aa_clip/aaclip_mvtec_518.py`
- ✓ Documentation: `AA_CLIP_INTEGRATION.md`
- ✓ Quick reference: `AA_CLIP_QUICK_REF.md`
- ✓ Training script: `train_aaclip.sh`

All ready to go! 🚀
