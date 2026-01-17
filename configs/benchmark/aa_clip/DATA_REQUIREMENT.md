# AA-CLIP Training Data Requirements

## **Critical Discovery**

AA-CLIP is **NOT a zero-shot anomaly detection method**. It requires **supervised training with anomaly samples and ground truth masks**.

## Data Comparison

### MVTec-AD Standard Split (Used by ADer)
```
Training set: 3,629 normal images (anomaly=0, empty masks)
Test set:     1,725 images
  ├─ Normal:    467 (27%)
  └─ Anomaly: 1,258 (73%) with defect masks
```

**Purpose**: Zero-shot/unsupervised anomaly detection
- Train only on normal samples
- Detect anomalies at test time without seeing them during training

### AA-CLIP Full-Shot Training (Original Implementation)
```
Training set: 1,725 images (ENTIRE TEST SET)
  ├─ Normal:    467 (27%)
  └─ Anomaly: 1,258 (73%) with defect masks
Test set:     Same 1,725 images
```

**Purpose**: Supervised/few-shot anomaly detection
- Train on both normal AND anomaly samples with masks
- Learn to segment defects using ground truth supervision

## Why Zero Loss Occurred

When using ADer's standard training split:

1. **Training data**: Only normal images with all-zero masks
2. **Model behavior**: Quickly learns to predict all zeros everywhere
3. **Loss values**:
   - Segmentation loss (Dice + Focal) on zero masks → near zero
   - Orthogonal loss → small constant (~0.001)
   - **Total loss → 0.000** after a few epochs
4. **Test metrics**: Poor (~62% AUROC) because model never learned to detect anomalies

## AA-CLIP Training Stages

### Stage 1: Text Adapter (5 epochs)
- **Input**: Images + ground truth masks + class names
- **Loss**: Segmentation (Dice + Focal) + Orthogonal constraint
- **Requires**: Anomaly samples with masks to compute segmentation loss
- **Learns**: Anomaly-aware text embeddings ("normal" vs "anomaly")

### Stage 2: Image Adapter (20 epochs)
- **Input**: Images + ground truth masks + class names
- **Loss**: Classification (is anomaly?) + Segmentation (where is anomaly?)
- **Requires**: Both normal and anomaly samples
- **Learns**: Visual features that align with text embeddings

## Solution Options

### Option 1: Use Test Set for Training (AA-CLIP Original)
Modify ADer dataset to include anomaly samples from test set:

**Pros:**
- Matches original AA-CLIP paper results
- High accuracy (~95%+ AUROC)
- Simple implementation

**Cons:**
- NOT zero-shot learning
- Uses test data for training (data leakage)
- Not comparable with zero-shot methods in ADer

**Use case:** Supervised anomaly segmentation benchmarking

### Option 2: Few-Shot Learning (AA-CLIP Paper)
Use only a small subset of anomaly samples (k-shot):

**Pros:**
- More realistic than full-shot
- Still supervised learning
- Can compare different k values (1, 2, 4, 8)

**Cons:**
- Still not zero-shot
- Requires careful data split management
- Results depend on which samples are selected

**Use case:** Few-shot anomaly detection research

### Option 3: Synthetic Anomalies (Not in Original Paper)
Generate synthetic defects during training:

**Pros:**
- True zero-shot learning
- No test data leakage
- Comparable with other ADer methods

**Cons:**
- Requires significant implementation changes
- Not validated in original paper
- May reduce accuracy
- Different method than published AA-CLIP

**Use case:** Zero-shot anomaly detection (would be a new variant)

### Option 4: Document Incompatibility (Current)
Document that AA-CLIP requires supervised data:

**Pros:**
- Accurate to original paper
- Clear about method requirements
- No misleading comparisons

**Cons:**
- Cannot directly compare with zero-shot methods
- Limited usefulness in ADer framework

**Use case:** Understanding method requirements

## Recommendation

**For Research/Benchmarking:**
Implement Option 1 or 2 with clear documentation that AA-CLIP is a supervised method. Create a separate dataset configuration for supervised learning.

**For Zero-Shot Comparison:**
AA-CLIP is fundamentally incompatible with zero-shot anomaly detection and should not be compared with methods like WinCLIP, CLIP-AD, etc. that don't see anomalies during training.

## Implementation for Supervised Training

To properly run AA-CLIP with supervised data, you would need to:

1. Create new dataset class that loads anomaly samples for training:
   ```python
   class SupervisedAD(DefaultAD):
       def __init__(self, cfg, train=True, ...):
           # Load both normal and anomaly samples for training
           if train:
               # Include samples from test split with anomalies
   ```

2. Update configuration:
   ```python
   self.data.type = 'SupervisedAD'
   self.data.train_anomaly_ratio = 0.73  # Match AA-CLIP
   ```

3. Modify meta.json or create new metadata file that includes anomalies in train split

## References

- Original AA-CLIP uses `dataset/metadata/MVTec/full-shot.jsonl`
- Contains ALL 1,725 test samples (467 normal + 1,258 anomaly)
- Training directly on test set to learn anomaly patterns
- Paper reports results as "full-shot" supervised learning
