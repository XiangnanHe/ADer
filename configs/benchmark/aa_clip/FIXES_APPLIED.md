# AA-CLIP Fixes Applied - 2026-01-15

## Critical Fix: Stage 2 Backward Error

**Error**: `RuntimeError: Trying to backward through the graph a second time`

**Root Cause**: Text features computed during Stage 1 retained gradient tracking. When used in Stage 2's backward pass, PyTorch attempted to backpropagate through an already-freed computation graph.

**Solution**: Modified `_compute_text_features()` in `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py` (line 278-287):

```python
def _compute_text_features(self):
    """Compute text features for all classes after stage 1"""
    with torch.no_grad():
        text_features_raw = get_adapted_text_embedding(
            self.net.model,
            self.dataset_name,
            self.imgs.device
        )
    # Detach all text features to prevent gradient tracking in Stage 2
    self.text_features = {k: v.detach() for k, v in text_features_raw.items()}
```

This ensures text features are completely detached from the computation graph before Stage 2 uses them.

## Diagnostic Addition: Zero Loss Warning

**Issue Observed**: During the previous training run, Stage 1 losses dropped to zero around iteration 7300 (epoch 4.0), resulting in poor test metrics (62.6% AUROC instead of expected ~95%).

**Addition**: Added a safety check in `_optimize_text_adapter()` (line 199-201):

```python
# Safety check: warn if loss is suspiciously low
if combined_loss < 0.01 and self.master:
    log_msg(self.logger, f"WARNING: Very low loss detected: {combined_loss.item():.6f}")
```

This will help diagnose if the zero loss issue occurs again.

## Testing Instructions

To test these fixes:

```bash
conda activate ader
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/benchmark/aa_clip/aaclip_mvtec_518 -m train
```

**Expected behavior**:
- Stage 1 losses should remain in range 0.5-2.0 throughout training
- Stage 1 test metrics (after epoch 5) should show >90% AUROC
- Stage 2 should start without backward errors
- Stage 2 should train successfully and improve metrics further

**Monitor for**:
- Zero loss warnings during Stage 1
- Stable learning rate: 0.0000100 (1e-5) for Stage 1, 0.0005000 (5e-4) for Stage 2
- Gradually decreasing losses in both stages

## Previous Issues Fixed

1. ✅ Module import conflicts (namespace isolation)
2. ✅ ipdb compatibility (mocking)
3. ✅ Config path format
4. ✅ Namespace import in config files
5. ✅ create_model argument order
6. ✅ Parameter freezing for positional_embedding
7. ✅ Checkpoint loading with clip_checkpoint_path
8. ✅ Positional embedding interpolation (336→518)
9. ✅ Patch dropout removal
10. ✅ Feature normalization and class token residual
11. ✅ FocalLoss → CrossEntropyLoss in Stage 2
12. ✅ Metric naming (metric_Avg)
13. ✅ Learning rate configuration (constant LR for Stage 1)
14. ✅ Target tensor shape for cross_entropy
15. ✅ **NEW**: Text feature gradient detachment for Stage 2

## Files Modified

- `/mnt/task_runtime/ADer/trainer/aaclip_trainer.py`:
  - Line 278-287: Text feature computation with gradient detachment
  - Line 199-201: Zero loss warning

## Pending Investigation

If zero loss issue persists, potential causes to investigate:
- Dataset sampling (are we getting only normal samples?)
- Gradient clipping effects
- Numerical stability in calculate_seg_loss
- Model parameter freezing state
