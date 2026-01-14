# Key Differences: Mamba-CLIP vs MambaAD

## Executive Summary

**MambaAD** and **Mamba-CLIP** are fundamentally different approaches to anomaly detection using Mamba/SSM architectures:

- **MambaAD**: Teacher-Student knowledge distillation with reconstruction
- **Mamba-CLIP**: Adapter-based direct prediction with dual heads

---

## 🏗️ Architecture Comparison

| Aspect | **MambaAD** | **Mamba-CLIP** |
|--------|-------------|----------------|
| **Paradigm** | Knowledge Distillation | Adapter-based Learning |
| **Structure** | Encoder-Decoder | Encoder + Dual Heads |
| **Components** | Teacher + Student + MFF-OCE | Encoder + Adapters + Heads |
| **Frozen Parts** | Teacher network (WideResNet) | Optional (configurable) |
| **Trainable** | Student decoder + MFF | Adapters + Heads (or full) |

---

## 📐 Detailed Component Comparison

### 1. **SS2D (2D Selective Scan) - Core Difference**

#### **MambaAD SS2D** (Lines 122-302)

**Advanced Scanning Strategies:**
```python
scan_type ∈ ['scan', 'sweep', 'zorder', 'zigzag', 'hilbert']
```

- **Sweep**: Linear indexing (0, 1, 2, ..., N)
- **Scan**: Snake/raster (alternating row directions)
- **Z-order**: Morton curve (space-filling curve)
- **Zigzag**: Diagonal scanning pattern
- **Hilbert**: Hilbert space-filling curve (preserves locality)

**Number of Directions:**
```python
num_direction ∈ [2, 4, 8]
```

- **2 directions**: Forward + Backward scans
- **4 directions**: + Transposed versions
- **8 directions**: + Rotations (90°, 180°, 270°)

**Implementation:**
```python
# MambaAD uses SCANS class for complex space-filling curves
self.scans = SCANS(size=size, scan_type=scan_type)

# Encodes image into scan order
xs.append(self.scans.encode(x.view(B, -1, L)))
xs.append(self.scans.encode(torch.transpose(x, 2, 3).view(B, -1, L)))
xs.append(self.scans.encode(torch.rot90(x, k=1, dims=(2,3)).view(B, -1, L)))
# ... up to 8 directions with rotations

# After SSM processing, decode back
ys.append(self.scans.decode(out_y[:, 0]))
ys.append(torch.rot90(self.scans.decode(...), k=3, dims=(2,3)))
```

**Key Features:**
- ✅ Space-filling curves (Hilbert, Z-order) preserve 2D locality
- ✅ Up to 8 directional scans
- ✅ Rotation augmentation (90°, 180°, 270°)
- ✅ More comprehensive spatial coverage

---

#### **Mamba-CLIP SS2D** (Lines 88-159)

**Simple 4-Way Cross-Scan:**
```python
scan_type = 'fixed 4-way'  # Not configurable
```

- **Direction 1**: Top-Left → Bottom-Right (raster)
- **Direction 2**: Bottom-Right → Top-Left (reverse raster)
- **Direction 3**: Top-Right → Bottom-Left (transpose raster)
- **Direction 4**: Bottom-Left → Top-Right (reverse transpose)

**Implementation:**
```python
# Mamba-CLIP uses simple flips and transposes
def cross_scan(x):
    seq1 = x.view(B, C, L).transpose(1, 2)  # TL→BR
    seq2 = torch.flip(x, dims=[2,3]).view(B, C, L).transpose(1, 2)  # BR→TL
    seq3 = x.transpose(2,3).view(B, C, L).transpose(1, 2)  # Column-major
    seq4 = torch.flip(x.transpose(2,3), dims=[2,3]).view(B, C, L).transpose(1, 2)
    return [seq1, seq2, seq3, seq4]

# After SSM, reverse operations
def cross_merge(scans, H, W):
    out1 = scans[0].view(B, C, H, W)
    out2 = torch.flip(scans[1].view(B, C, H, W), dims=[2,3])
    out3 = scans[2].view(B, C, W, H).transpose(2, 3)
    out4 = torch.flip(scans[3].view(B, C, W, H), dims=[2,3]).transpose(2,3)
    return out1 + out2 + out3 + out4
```

**Key Features:**
- ✅ Simpler implementation (no external dependencies)
- ✅ Fixed 4 directions (adequate for most cases)
- ✅ No space-filling curves (just raster + transpose)
- ✅ Lighter computational overhead

---

### 2. **Overall Architecture**

#### **MambaAD Architecture** (Lines 607-635)

```
Input Image (B, 3, H, W)
       ↓
[Teacher Network - WideResNet50] ← Frozen, pretrained
       ↓
Multi-scale Features [F1, F2, F3]
       ↓
[MFF-OCE: Multi-Feature Fusion + One-Class Embedding]
   ├─ Conv layers to fuse F1, F2, F3
   └─ Bottleneck layers for compression
       ↓
Fused Feature (B, 512, 8, 8)
       ↓
[Student Network - MambaUPNet Decoder]
   ├─ VSSLayer_up (×4 stages)
   │   ├─ ConvBNSSMBlock (SSM + Multi-scale Conv)
   │   │   ├─ VSSBlock (×2-3 depth)
   │   │   └─ 3×3, 5×5, 7×7 depthwise convs
   │   └─ PatchExpand2D (upsample)
   └─ Progressive upsampling: 8→16→32→64
       ↓
Reconstructed Features [F1', F2', F3']
       ↓
Loss: ||F_teacher - F_student||
```

**Key Characteristics:**
- ✅ **Teacher-Student paradigm**: Learn to reconstruct teacher features
- ✅ **Multi-scale fusion**: Combines features from multiple layers
- ✅ **Decoder-only training**: Teacher frozen, only student trained
- ✅ **Reconstruction loss**: Anomalies = reconstruction errors
- ✅ **4 upsampling stages**: Progressive spatial resolution recovery

---

#### **Mamba-CLIP Architecture** (Lines 308-515)

```
Input Image (B, 3, 518, 518)
       ↓
Patch Embedding (B, D, 37, 37)
       ↓
[Encoder Blocks ×12]
   ├─ If use_ss2d=True:
   │   └─ VSSBlock (SS2D + LayerNorm)
   └─ If use_ss2d=False:
       └─ VimBlock (Bidirectional Mamba)
       ↓
   [Mamba Adapters] ← Injected in layers 7-12
       ├─ Adaptor-T: Temporal Mamba
       └─ Adaptor-S: Multi-scale convs (3×3, 5×5, 7×7)
       ↓
Layer Norm
       ↓
   ┌───────┴────────┐
   ↓                ↓
[Image Head]    [Pixel Head]
(B, 1)          (B, 1, H, W)
   ↓                ↓
Image Score   Anomaly Map
```

**Key Characteristics:**
- ✅ **Direct prediction**: No reconstruction, direct anomaly scoring
- ✅ **Dual heads**: Image-level + Pixel-level predictions
- ✅ **Adapter injection**: Lightweight adaptation in selected layers
- ✅ **Flexible backbone**: Can freeze for fast fine-tuning
- ✅ **High resolution**: Supports 518×518, 1024×1024+

---

### 3. **ConvBNSSMBlock - Spatial Enhancement**

#### **MambaAD** (Lines 326-408)

```python
class ConvBNSSMBlock:
    def __init__(self, hidden_dim, depth=2-3):
        # Multiple VSSBlocks (Mamba)
        self.smm_blocks = nn.ModuleList([VSSBlock(...) for _ in range(depth)])

        # Multi-scale convolution branches (3 scales)
        self.conv77 = Conv2d(kernel_size=7, padding=3)  # Large receptive field
        self.conv55 = Conv2d(kernel_size=5, padding=2)  # Medium
        self.conv33 = Conv2d(kernel_size=3, padding=1)  # Small

        # 1×1 convs before/after each scale
        self.conv1b3, self.conv1a3 = Conv2d(1×1), Conv2d(1×1)
        self.conv1b5, self.conv1a5 = Conv2d(1×1), Conv2d(1×1)

        # Fusion
        self.finalconv11 = Conv2d(hidden_dim*3 → hidden_dim)

    def forward(self, x):
        # SSM processing
        out_ssm = x
        for blk in self.smm_blocks:
            out_ssm = blk(out_ssm)

        # Parallel conv branches (on original input)
        x_conv = x.permute(0,3,1,2)  # Channel-first for conv
        out_77 = self.conv1a3(self.conv77(self.conv1b3(x_conv)))
        out_55 = self.conv1a5(self.conv55(self.conv1b5(x_conv)))

        # Concatenate SSM + 5×5 + 7×7 features
        output = torch.cat([out_ssm.permute(0,3,1,2), out_55, out_77], dim=1)
        output = self.finalconv11(output)  # 3D → D

        return output + x  # Residual connection
```

**Design Philosophy:**
- ✅ **Explicit multi-scale**: Separate 3×3, 5×5, 7×7 conv branches
- ✅ **Feature concatenation**: Fuse SSM + conv features
- ✅ **Instance normalization**: Used instead of LayerNorm
- ✅ **Residual on original input**: Skip SSM entirely if needed

---

#### **Mamba-CLIP MambaAdapterLayer** (Lines 239-307)

```python
class MambaAdapterLayer:
    def __init__(self, d_model, conv_scales=[3,5,7]):
        # Adaptor-T: Temporal/token modeling
        self.mamba = Mamba(d_model=d_model)

        # Adaptor-S: Multi-scale spatial convs (depthwise)
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=k,
                     padding=k//2, groups=d_model)  # Depthwise
            for k in conv_scales
        ])

        # Learnable scale (initialized to 0)
        self.adapter_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, H, W):
        residual = x

        # Adaptor-T: Mamba on flattened sequence
        x_mamba = self.mamba(self.norm(x))  # (B, L, C)

        # Adaptor-S: Reshape to 2D for spatial conv
        x_2d = x_mamba.transpose(1,2).reshape(B, C, H, W)
        x_spatial = sum([conv(x_2d) for conv in self.spatial_convs]) / len(self.spatial_convs)
        x_out = x_spatial.reshape(B, C, L).transpose(1, 2)

        # Residual with learnable scale
        return residual + self.adapter_scale * x_out
```

**Design Philosophy:**
- ✅ **Sequential**: Mamba → Spatial convs (not parallel)
- ✅ **Depthwise convs**: More efficient (groups=d_model)
- ✅ **Average fusion**: Simple averaging instead of concatenation
- ✅ **Learnable scale**: Adapter contribution controlled by parameter
- ✅ **Adapter pattern**: Minimal modification to frozen backbone

---

### 4. **Training Paradigm**

#### **MambaAD**

```python
# Teacher-Student setup
teacher = WideResNet50(pretrained=True)  # Frozen
student = MambaUPNet()  # Trainable decoder

def forward(imgs):
    # Extract teacher features (frozen)
    feats_t = teacher(imgs)  # [F1, F2, F3]
    feats_t = [f.detach() for f in feats_t]

    # Fuse and decode with student
    fused = mff_oce(feats_t)
    feats_s = student(fused)  # Reconstruct features

    return feats_t, feats_s

# Loss: Reconstruction error
loss = ||feats_t - feats_s||  # Cosine similarity or L2
```

**Anomaly Detection:**
- Anomalies = High reconstruction error
- Normal samples reconstruct well
- One-class learning (only normal samples in training)

---

#### **Mamba-CLIP**

```python
# End-to-end encoder with optional freezing
encoder = MambaEncoder(freeze_backbone=True/False)
adapters = {layer_i: MambaAdapter() for layer_i in [7,8,9,10,11,12]}
head_img = Linear(embed_dim → 1)
head_pixel = Conv2d(embed_dim → 1)

def forward(imgs):
    # Encoder with adapters
    features = encoder(imgs)  # With adapter injection

    # Dual predictions
    img_score = head_img(features['cls_token'])  # (B, 1)
    pixel_map = head_pixel(features['spatial'])  # (B, 1, H, W)

    return img_score, pixel_map

# Loss: Direct supervision
loss_img = BCE(img_score, anomaly_label)  # 0 or 1
loss_pixel = BCE(pixel_map, pixel_mask)   # 0/1 per pixel
loss = loss_img + loss_pixel
```

**Anomaly Detection:**
- Anomalies = High predicted scores
- Direct supervision on both image and pixel levels
- Binary classification (not one-class)

---

## 📊 Feature Comparison Table

| Feature | **MambaAD** | **Mamba-CLIP** |
|---------|-------------|----------------|
| **Scanning** | 2/4/8 directions, 5 scan types | Fixed 4-way cross-scan |
| **Space-Filling Curves** | ✅ Hilbert, Z-order | ❌ Simple raster/transpose |
| **Rotations** | ✅ 90°, 180°, 270° | ❌ None |
| **Learning Paradigm** | Knowledge Distillation | Direct Prediction |
| **Teacher Network** | ✅ WideResNet50 | ❌ None (or optional) |
| **Decoder** | ✅ MambaUPNet (4 stages) | ❌ None (direct heads) |
| **Multi-Feature Fusion** | ✅ MFF-OCE module | ❌ Single-level features |
| **Adapter Mechanism** | ❌ None | ✅ Mamba Adapters |
| **Output** | Reconstructed features | Image score + Pixel map |
| **Loss** | Reconstruction error | BCE (supervised) |
| **One-Class** | ✅ Yes (unsupervised) | ❌ No (supervised) |
| **Trainable Params** | Student decoder only | Adapters + heads (or full) |
| **Multi-Scale Convs** | Parallel branches (concat) | Sequential (averaged) |
| **Normalization** | InstanceNorm2d | LayerNorm |
| **Resolution** | Standard (256×256) | High (518×518+) |

---

## 🎯 Use Case Recommendations

### **Choose MambaAD if:**
- ✅ You want **unsupervised/one-class** learning (only normal samples)
- ✅ You need **knowledge distillation** from a strong teacher
- ✅ You prefer **feature reconstruction** paradigm
- ✅ You want **space-filling curves** for better locality preservation
- ✅ You have a **strong pretrained encoder** (WideResNet, ViT, etc.)
- ✅ You need **multi-scale feature fusion** from different layers

### **Choose Mamba-CLIP if:**
- ✅ You want **direct anomaly prediction** (simpler training)
- ✅ You have **labeled data** (anomaly labels available)
- ✅ You need **high-resolution** processing (518×518, 1024×1024+)
- ✅ You want **lightweight adaptation** (adapter-based fine-tuning)
- ✅ You prefer **end-to-end training** without teacher-student setup
- ✅ You need **both image and pixel-level** predictions simultaneously
- ✅ You want **faster inference** (no decoder upsampling)

---

## 🔬 Technical Innovations

### **MambaAD Innovations:**
1. **Advanced Scanning**: Hilbert curves, Z-order preserve 2D locality
2. **8-Directional Coverage**: Most comprehensive spatial coverage
3. **MFF-OCE**: Multi-Feature Fusion + One-Class Embedding
4. **ConvBNSSMBlock**: Hybrid SSM + Multi-scale convs
5. **Progressive Upsampling**: 4-stage decoder with PatchExpand2D

### **Mamba-CLIP Innovations:**
1. **Mamba Adapters**: Adaptor-T + Adaptor-S design
2. **Learnable Adapter Scale**: Stable training from 0 initialization
3. **Dual-Head Architecture**: Unified image + pixel predictions
4. **High-Resolution Support**: Linear complexity enables 1024×1024+
5. **Flexible Backbone**: Switchable SS2D/Vim, optional freezing

---

## 📈 Performance Characteristics

| Metric | **MambaAD** | **Mamba-CLIP** |
|--------|-------------|----------------|
| **Complexity** | O(N) (SSM) + O(N²) (teacher) | O(N) (full) |
| **Memory** | Teacher + Student + Features | Encoder + Adapters |
| **Training Speed** | Slower (teacher inference + student training) | Faster (direct training) |
| **Inference Speed** | Slower (teacher + decoder) | Faster (encoder + heads) |
| **Resolution** | 256×256 (standard) | 518×518+ (high) |
| **Parameters** | ~50M (teacher) + ~20M (student) | ~51M (base), 3M trainable with freeze |

---

## 🧪 Code Complexity Comparison

```bash
# File sizes
MambaAD:     662 lines  (more complex)
Mamba-CLIP:  523 lines  (simpler)

# Dependencies
MambaAD:     hilbert, pyzorder (space-filling curves)
Mamba-CLIP:  Standard PyTorch only (optional mamba_ssm)

# External modules
MambaAD:     SCANS, MFF_OCE, MambaUPNet, ConvBNSSMBlock
Mamba-CLIP:  SS2D, VimBlock, MambaAdapterLayer
```

---

## 💡 Summary

**MambaAD** is a **sophisticated, research-oriented** implementation:
- Advanced scanning strategies (Hilbert curves, 8 directions)
- Teacher-student knowledge distillation
- Multi-scale feature fusion
- Unsupervised one-class learning
- More complex but potentially more powerful

**Mamba-CLIP** is a **practical, production-ready** implementation:
- Simple 4-way cross-scan (sufficient for most cases)
- Direct supervised learning
- Adapter-based fine-tuning
- High-resolution support
- Easier to understand and deploy

**Both are valid approaches** with different strengths. MambaAD excels at **unsupervised feature reconstruction**, while Mamba-CLIP excels at **supervised direct prediction** with high resolution support.
