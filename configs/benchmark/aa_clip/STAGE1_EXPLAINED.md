# AA-CLIP Stage 1: What's Being Trained?

## Quick Answer

**Stage 1 trains**: Text adapter modules (small learned layers inserted into CLIP's text encoder)
**Uses**: Both text prompts AND image pairs with ground truth masks
**Frozen**: Base CLIP text encoder and entire CLIP image encoder

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          STAGE 1 TRAINING                           │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Image + Ground Truth Mask + Class Name (e.g., "grid")

┌────────────────────────┐         ┌────────────────────────┐
│    TEXT BRANCH         │         │    IMAGE BRANCH        │
│  (being trained)       │         │    (frozen)            │
└────────────────────────┘         └────────────────────────┘
          │                                  │
          │                                  │
          ▼                                  ▼
┌─────────────────────┐         ┌─────────────────────┐
│ Text Prompts:       │         │ Image: grid.png     │
│                     │         │ Mask: defect mask   │
│ Normal:             │         └─────────────────────┘
│  - "a grid."        │                  │
│  - "a photo of      │                  ▼
│    a grid."         │         ┌─────────────────────┐
│                     │         │ CLIP Image Encoder  │
│ Anomaly:            │         │ (ViT-L-14, FROZEN)  │
│  - "a damaged grid."│         └─────────────────────┘
│  - "a broken grid." │                  │
│  - "a grid with     │                  │
│    defect."         │         Extract layers [6,12,18,24]
└─────────────────────┘                  │
          │                              ▼
          │                   ┌─────────────────────┐
          ▼                   │ Patch Features      │
┌─────────────────────┐       │ [B, L, 768] x 4     │
│ CLIP Text Encoder   │       │ L=1369 patches      │
│ (12 layers, FROZEN) │       │ Normalized + CLS    │
│                     │       └─────────────────────┘
│ Layer 0             │                  │
│ Layer 1             │                  │
│ Layer 2             │                  │
│   ↓ TEXT ADAPTER 0  │◄─────────┐       │
│ Layer 3             │ TRAINABLE│       │
│   ↓ TEXT ADAPTER 1  │◄─────────┤       │
│   ...               │ 2.36M    │       │
│   ↓ TEXT ADAPTER 2  │ params   │       │
│   ...               │          │       │
│ Layer 11            │          │       │
│   ↓ TEXT ADAPTER 3  │◄─────────┘       │
│                     │                  │
│ Final LayerNorm     │                  │
└─────────────────────┘                  │
          │                              │
          ▼                              │
┌─────────────────────┐                  │
│ Text Embeddings     │                  │
│ [768, 2]            │                  │
│  ├─ Normal: [768]   │                  │
│  └─ Anomaly: [768]  │                  │
└─────────────────────┘                  │
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Similarity Map      │
              │ patch_feat @ text   │
              │ [B, H, W]           │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ LOSSES              │
              │ 1. Seg Loss (Dice   │
              │    + Focal) vs mask │
              │ 2. Orthogonal loss  │
              │    (normal ⊥ anom)  │
              └─────────────────────┘
                         │
                         ▼
                    BACKPROP
                (only text adapters)
```

## Text Adapter Details

### Structure
```python
text_adapt_until = 3  # Apply adapters at layers 0, 1, 2

text_adapter = nn.ModuleList([
    SimpleAdapter(768, 768),  # Layer 0
    SimpleAdapter(768, 768),  # Layer 1
    SimpleAdapter(768, 768),  # Layer 2
    SimpleProj(768, 768)      # Final projection (layer 3)
])

# Total: 2.36M trainable parameters
# vs 85M frozen CLIP text encoder parameters
```

### How Adapters Work
```python
# At each layer i < 3 in CLIP text transformer:
x = transformer_layer(x)           # FROZEN: base CLIP forward
adapt_out = text_adapter[i](x)     # TRAINABLE: adapter forward
adapt_out = normalize(adapt_out)   # Keep same magnitude
x = 0.1 * adapt_out + 0.9 * x     # Weighted residual (weight=0.1)

# After all layers:
x = text_adapter[3](x)             # TRAINABLE: final projection
```

## Concrete Example: Training on "grid" class

### Step 1: Generate Text Embeddings (trainable path)

**Input text prompts:**
- Normal: `["a grid.", "a photo of a grid.", "the grid."]`
- Anomaly: `["a damaged grid.", "a broken grid.", "a grid with defect.", ...]`

**Process:**
1. Tokenize prompts → [6 prompts x 77 tokens]
2. Pass through CLIP text encoder with adapters:
   - Token embedding (FROZEN)
   - Position embedding (FROZEN)
   - 12 transformer layers (FROZEN)
     - Adapters inserted at layers 0, 1, 2 (TRAINABLE)
   - Final adapter projection (TRAINABLE)
3. Get embeddings for each prompt
4. Average within each group (normal/anomaly)
5. Result: `text_features = [768, 2]` where:
   - `text_features[:, 0]` = normal embedding
   - `text_features[:, 1]` = anomaly embedding

### Step 2: Extract Image Features (frozen path)

**Input:** `grid/test/broken/001.png` + `ground_truth/broken/001_mask.png`

**Process:**
1. Pass image through CLIP ViT-L-14 (FROZEN)
2. Extract patch features from layers [6, 12, 18, 24]
3. Each layer: `[B, 1369, 1024]` → project to `[B, 1369, 768]`
4. Normalize: `patch_feat / ||patch_feat||`
5. Add class token residual: `patch_feat + cls_token`
6. Result: 4 levels of normalized patch features

### Step 3: Compute Similarity & Loss

For each feature level:
```python
# Compute similarity between patches and text
similarity = patch_features @ text_features  # [B, 1369, 2]
# Reshape to spatial map
sim_map = reshape(similarity, [B, 37, 37, 2])  # 37x37 = 1369 patches

# Anomaly score per pixel = max(anomaly_sim, 1 - normal_sim)
anomaly_map = calculate_similarity_map(sim_map)  # [B, H, W]

# Segmentation loss: compare with ground truth mask
seg_loss = dice_loss(anomaly_map, gt_mask) + focal_loss(anomaly_map, gt_mask)
```

**Orthogonal loss:**
```python
# Encourage normal and anomaly embeddings to be perpendicular
normal_emb = text_features[:, 0]    # [768]
anomaly_emb = text_features[:, 1]   # [768]
ortho_loss = |cos_similarity(normal_emb, anomaly_emb)|²
```

**Total loss:**
```python
loss = seg_loss + 0.1 * ortho_loss
```

### Step 4: Backpropagation

Gradients flow back through:
- ✅ text_adapter[3] (final projection)
- ✅ text_adapter[2] (layer 2 adapter)
- ✅ text_adapter[1] (layer 1 adapter)
- ✅ text_adapter[0] (layer 0 adapter)

Gradients blocked at:
- ❌ CLIP text transformer layers (frozen)
- ❌ CLIP text embeddings (frozen)
- ❌ CLIP image encoder (frozen)

## Why Both Text AND Images?

**The key insight**: Text adapters learn to generate anomaly-aware embeddings by:

1. **Seeing what anomalies look like** (via image features)
2. **Learning to make text distinguish them** (via segmentation loss)

Without images + masks, the text adapters have no supervision signal:
- How should "damaged grid" differ from "grid"?
- The adapters need to see actual damage patterns to learn meaningful distinctions

## Summary

| Component | Status | Parameters |
|-----------|--------|------------|
| CLIP Text Encoder | Frozen | 85M |
| Text Adapters | **Trainable** | **2.36M** |
| CLIP Image Encoder | Frozen | 305M |
| Image Adapters | Frozen (Stage 1) | 10.2M |

**Key point**: Stage 1 is NOT "text-only" training. It uses:
- ✅ Text prompts (to generate embeddings)
- ✅ Images (to extract visual features)
- ✅ Ground truth masks (for supervision)

The text adapters learn to create embeddings that, when compared with image features, produce accurate anomaly segmentations.
