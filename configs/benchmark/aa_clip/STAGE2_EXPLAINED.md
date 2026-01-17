# AA-CLIP Stage 2: What's Being Trained?

## Quick Answer

**Stage 2 trains**: Image adapter modules (10.2M parameters)
**Text adapters**: **FROZEN** - used to generate fixed text embeddings
**Training data**: **SAME** as Stage 1 (images + masks)

## Key Differences from Stage 1

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Trainable** | Text adapters (2.36M) | Image adapters (10.2M) |
| **Frozen** | Everything else | Text adapters + CLIP encoders |
| **Text features** | Computed fresh each batch | Cached once, reused |
| **Learning rate** | 1e-5 | 5e-4 (50x higher!) |
| **Epochs** | 5 | 20 |
| **Data** | Images + masks | Same images + masks |

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          STAGE 2 TRAINING                           │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Same as Stage 1 - Image + Mask + Class Name

┌────────────────────────┐         ┌────────────────────────┐
│    TEXT BRANCH         │         │    IMAGE BRANCH        │
│   (FULLY FROZEN)       │         │  (being trained)       │
└────────────────────────┘         └────────────────────────┘
          │                                  │
          │ Computed ONCE                    │ Every iteration
          │ after Stage 1                    │
          ▼                                  ▼
┌─────────────────────┐         ┌─────────────────────┐
│ Text Features       │         │ Image: grid.png     │
│ (CACHED)            │         │ Mask: defect mask   │
│                     │         └─────────────────────┘
│ For each class:     │                  │
│  "grid": [768, 2]   │                  ▼
│    ├─ normal        │         ┌─────────────────────┐
│    └─ anomaly       │         │ CLIP Image Encoder  │
│                     │         │ (ViT-L-14, FROZEN)  │
│ Stored in dict:     │         │                     │
│ self.text_features  │         │ Layer 0             │
│                     │         │ Layer 1             │
│ ✅ NO gradients     │         │ Layer 2             │
│ ✅ NO backprop      │         │ Layer 3             │
│ ✅ Pure lookup      │         │ Layer 4             │
└─────────────────────┘         │ Layer 5             │
          │                     │   ↓ IMG ADAPTER 0   │◄────┐
          │                     │ Layer 6             │     │
          │                     │   ↓ IMG ADAPTER 1   │◄────┤
          │                     │   ...               │     │
          │                     │   ↓ IMG ADAPTER 5   │◄────┤ TRAINABLE
          │                     │   ...               │     │ 10.2M params
          │                     │ Layer 23            │     │
          │                     │                     │     │
          │                     │ Extract at levels:  │     │
          │                     │  [6, 12, 18, 24]    │◄────┘
          │                     └─────────────────────┘
          │                              │
          │                              ▼
          │                   ┌─────────────────────┐
          │                   │ Seg Projections     │◄─── TRAINABLE
          │                   │ [4 x (1024→768)]    │     3.1M params
          │                   │                     │
          │                   │ Det Projection      │◄─── TRAINABLE
          │                   │ (1024→768)          │     0.8M params
          │                   └─────────────────────┘
          │                              │
          │                              ▼
          │                   ┌─────────────────────┐
          │                   │ seg_tokens: [B,L,768]│
          │                   │   (4 levels)        │
          │                   │                     │
          │                   │ det_token: [B, 768] │
          │                   │   (global feature)  │
          │                   └─────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ TASK 1: Image-level │
              │ Classification      │
              │                     │
              │ det_token @ text    │
              │ → [normal, anomaly] │
              │ CrossEntropyLoss    │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ TASK 2: Pixel-level │
              │ Segmentation        │
              │                     │
              │ seg_token @ text    │
              │ → anomaly map       │
              │ Dice + Focal Loss   │
              └─────────────────────┘
                         │
                         ▼
                    BACKPROP
              (only image adapters
               + projections)
```

## Detailed Stage 2 Process

### Before Training: Compute Text Features Once

After Stage 1 completes (line 308-311 in trainer):

```python
# This happens ONCE, between Stage 1 and Stage 2
with torch.no_grad():
    text_features = {}
    for class_name in ["bottle", "cable", "capsule", ...]:
        # Use trained text adapters to generate embeddings
        text_feat = encode_text_with_adapters(class_name)
        # text_feat shape: [768, 2] (normal + anomaly)
        text_features[class_name] = text_feat.detach()

# Cache for entire Stage 2
self.text_features = text_features
# Example:
# {
#   "grid": tensor([768, 2]),    # frozen
#   "bottle": tensor([768, 2]),  # frozen
#   ...
# }
```

**Key point**: These are **frozen lookup tables**. No gradients, no updates.

### During Stage 2 Training (each iteration)

```python
# Step 1: Forward through image adapters (TRAINABLE)
image = batch['img']  # Same training images as Stage 1

# Pass through CLIP + image adapters
seg_tokens, det_token = model(image)
# seg_tokens: [4 levels of [B, 1369, 768]] - for segmentation
# det_token: [B, 768] - for classification

# Step 2: Get text features (FROZEN lookup)
text_feat = self.text_features[class_name]  # Just retrieve from cache
# NO COMPUTATION - just dictionary lookup
# text_feat: [768, 2] (normal + anomaly)

# Step 3: Compute losses

# Loss 1: Image-level classification
det_similarity = det_token @ text_feat  # [B, 2]
# How similar is image to normal vs anomaly?
label = has_anomaly(mask)  # 0 or 1
img_loss = CrossEntropyLoss(det_similarity, label)

# Loss 2: Pixel-level segmentation
seg_similarity = seg_tokens[-1] @ text_feat  # [B, L, 2]
anomaly_map = compute_map(seg_similarity)  # [B, H, W]
seg_loss = DiceLoss(anomaly_map, mask) + FocalLoss(anomaly_map, mask)

total_loss = img_loss + seg_loss

# Step 4: Backprop
# Gradients flow through:
# ✅ Image adapters (6 layers)
# ✅ Seg projections (4 layers)
# ✅ Det projection (1 layer)
#
# Gradients BLOCKED at:
# ❌ Text features (cached, no graph)
# ❌ Text adapters (frozen)
# ❌ CLIP encoders (frozen)
```

## Why Use the Same Training Data?

**Stage 1 teaches text**: "What language describes anomalies?"
- Trains text adapters to generate meaningful "normal" vs "anomaly" embeddings

**Stage 2 teaches vision**: "What visual features indicate anomalies?"
- Trains image adapters to extract features that match text embeddings
- Uses those cached text embeddings as supervision targets

Same data, different learning objective:
- **Stage 1**: Learn text embeddings that segment anomalies when paired with frozen visual features
- **Stage 2**: Learn visual features that segment anomalies when paired with frozen text embeddings

## What Gets Updated in Stage 2?

### Image Adapter Modules (6.3M parameters)
```python
# Inserted at transformer layers 0-5
for i in range(6):  # image_adapt_until = 6
    x = transformer_layer(x)           # FROZEN
    adapt_out = image_adapter[i](x)    # TRAINABLE
    adapt_out = normalize(adapt_out)
    x = 0.1 * adapt_out + 0.9 * x     # Weighted residual
```

### Segmentation Projections (3.1M parameters)
```python
# 4 projections (one per feature level)
seg_proj = [
    Linear(1024 → 768),  # Level 1 (layer 6)
    Linear(1024 → 768),  # Level 2 (layer 12)
    Linear(1024 → 768),  # Level 3 (layer 18)
    Linear(1024 → 768),  # Level 4 (layer 24)
]
# Project multi-level features to match text embedding dimension
```

### Detection Projection (0.8M parameters)
```python
det_proj = Linear(1024 → 768)
# Project global feature for image-level classification
```

**Total trainable**: 10.2M parameters

## Training Configuration Changes

```python
# After Stage 1 completes:

# 1. Freeze text adapters
for param in text_adapter.parameters():
    param.requires_grad = False

# 2. Unfreeze image adapters
for param in image_adapter.parameters():
    param.requires_grad = True

# 3. Increase learning rate (50x!)
optimizer.lr = 5e-4  # was 1e-5 in Stage 1

# 4. Reset epoch counter
self.epoch = 0

# 5. Set new epoch limit
self.epoch_full = 20  # was 5 in Stage 1

# 6. Continue with SAME dataloader
# Same images, same masks, same batches
```

## Summary

**YES** to your questions:
1. ✅ **Uses trained text adapter**: The text adapters trained in Stage 1 are used to generate text features
2. ✅ **Uses same training data**: Same images + masks as Stage 1
3. ✅ **Text features frozen**: Computed once, cached, no gradients

**Key insight**:
- Stage 1: Images supervise text learning
- Stage 2: Text supervises image learning
- Both use the same supervised data (anomaly images + masks)

The two-stage approach creates a **curriculum**:
1. First learn what to say about anomalies (text)
2. Then learn what to see in anomalies (vision)
3. Both anchored to the same ground truth masks
