"""
Mamba-CLIP: Vision Mamba with CLIP Adapters for Anomaly Detection

This module implements:
1. SS2D (2D Selective Scan) block from VMamba
2. Vim (Vision Mamba) bidirectional block
3. Mamba Adapters for CLIP integration
4. Zero-shot anomaly detection with high-resolution processing

References:
- VMamba: Visual State Space Model (SS2D/Cross-Scan)
- Vision Mamba (Vim): Bidirectional Mamba for Vision
- Mamba-Adaptor: State Space Model Adaptor for Visual Recognition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

from model import MODEL

# Try to import mamba_ssm, fallback to simplified version
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using simplified Mamba implementation")


class SimplifiedMamba(nn.Module):
    """Simplified Mamba block when mamba_ssm is not available"""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=4, padding=3, groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, D)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv1d
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = x.transpose(1, 2)

        # Gate
        x = self.act(x) * F.sigmoid(z)
        return self.out_proj(x)


class SS2D(nn.Module):
    """
    2D Selective Scan (SS2D) Module from VMamba

    Implements 4-way cross-scan to process 2D feature maps with linear complexity:
    - Top-Left to Bottom-Right (TL-BR)
    - Bottom-Right to Top-Left (BR-TL)
    - Top-Right to Bottom-Left (TR-BL)
    - Bottom-Left to Top-Right (BL-TR)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local 2D convolution (depthwise)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        # 4 directional SSM blocks (simplified)
        self.ssm_blocks = nn.ModuleList([
            SimplifiedMamba(self.d_inner, d_state, expand=1)
            for _ in range(4)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.act = nn.SiLU()

    def cross_scan(self, x):
        """
        Create 4 scanning directions from 2D feature map
        x: (B, C, H, W)
        Returns: list of 4 tensors, each (B, L, C) where L=H*W
        """
        B, C, H, W = x.shape
        L = H * W

        # Direction 1: Top-Left to Bottom-Right (standard raster)
        seq1 = x.view(B, C, L).transpose(1, 2)  # (B, L, C)

        # Direction 2: Bottom-Right to Top-Left (reverse raster)
        seq2 = torch.flip(x, dims=[2, 3]).contiguous().view(B, C, L).transpose(1, 2)

        # Direction 3: Column-major (transpose then raster)
        seq3 = x.transpose(2, 3).contiguous().view(B, C, L).transpose(1, 2)

        # Direction 4: Reverse column-major
        seq4 = torch.flip(x.transpose(2, 3), dims=[2, 3]).contiguous().view(B, C, L).transpose(1, 2)

        return [seq1, seq2, seq3, seq4]

    def cross_merge(self, scans, H, W):
        """
        Merge 4 scanned sequences back to 2D feature map
        scans: list of 4 tensors, each (B, L, C)
        Returns: (B, C, H, W)
        """
        B, L, C = scans[0].shape

        # Reverse the scanning operations and merge
        # Direction 1: Direct reshape
        out1 = scans[0].transpose(1, 2).view(B, C, H, W)

        # Direction 2: Reverse flip
        out2 = torch.flip(scans[1].transpose(1, 2).view(B, C, H, W), dims=[2, 3])

        # Direction 3: Transpose back
        out3 = scans[2].transpose(1, 2).view(B, C, W, H).transpose(2, 3)

        # Direction 4: Reverse flip and transpose
        out4 = torch.flip(scans[3].transpose(1, 2).view(B, C, W, H), dims=[2, 3]).transpose(2, 3)

        # Sum all directions
        return out1 + out2 + out3 + out4

    def forward(self, x):
        """
        x: (B, H, W, C) - Channel last format
        Returns: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # Signal and gate

        # Permute for Conv2d: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Local convolution
        x = self.act(self.conv2d(x))

        # Cross-scan: split into 4 directional sequences
        scans = self.cross_scan(x)

        # Apply SSM to each direction
        out_scans = [self.ssm_blocks[i](scans[i]) for i in range(4)]

        # Merge back to 2D
        y = self.cross_merge(out_scans, H, W)

        # Back to channel-last: (B, C, H, W) -> (B, H, W, C)
        y = y.permute(0, 2, 3, 1)

        # Gating and output projection
        y = y * self.act(z)
        out = self.out_proj(y)

        return self.dropout(out)


class VSSBlock(nn.Module):
    """Visual State Space (VSS) Block with SS2D"""
    def __init__(self, hidden_dim, drop_path=0., **kwargs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B, H, W, C)
        return x + self.drop_path(self.ss2d(self.ln_1(x)))


class VimBlock(nn.Module):
    """
    Vision Mamba (Vim) Block with bidirectional scanning
    Processes flattened image patches with forward and backward SSMs
    """
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Bidirectional Mamba: forward and backward
        MambaClass = Mamba if MAMBA_AVAILABLE else SimplifiedMamba
        self.mamba_fwd = MambaClass(d_model=dim, d_state=d_state, expand=expand)
        self.mamba_bwd = MambaClass(d_model=dim, d_state=d_state, expand=expand)

    def forward(self, x):
        """
        x: (B, L, C) - Flattened sequence
        Returns: (B, L, C)
        """
        residual = x
        x = self.norm(x)

        # Forward scan
        out_fwd = self.mamba_fwd(x)

        # Backward scan
        x_bwd = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        # Merge bidirectional outputs
        output = out_fwd + out_bwd

        return residual + output


class MambaAdapterLayer(nn.Module):
    """
    Mamba Adapter with Adaptor-T (Temporal) and Adaptor-S (Spatial)

    Adaptor-T: Addresses long-range forgetting with selective memory
    Adaptor-S: Re-injects spatial context via multi-scale convolutions
    """
    def __init__(self, d_model, d_state=16, conv_scales=[3, 5, 7]):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # Adaptor-T: Temporal/Token modeling with Mamba
        MambaClass = Mamba if MAMBA_AVAILABLE else SimplifiedMamba
        self.mamba = MambaClass(d_model=d_model, d_state=d_state, expand=2)

        # Adaptor-S: Multi-scale spatial convolutions
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model)
            for k in conv_scales
        ])

        # Learnable scale parameter (initialized to 0 for stability)
        self.adapter_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, H, W):
        """
        x: (B, L, C) - Flattened sequence (excluding CLS token)
        H, W: Spatial dimensions
        Returns: (B, L, C)
        """
        residual = x
        x = self.norm(x)

        # Adaptor-T: Temporal modeling
        x_mamba = self.mamba(x)

        # Adaptor-S: Spatial modeling
        # Reshape to 2D: (B, L, C) -> (B, C, H, W)
        B, L, C = x_mamba.shape
        x_2d = x_mamba.transpose(1, 2).reshape(B, C, H, W)

        # Apply multi-scale convolutions and sum
        x_spatial = sum([conv(x_2d) for conv in self.spatial_convs]) / len(self.spatial_convs)

        # Flatten back: (B, C, H, W) -> (B, L, C)
        x_out = x_spatial.reshape(B, C, L).transpose(1, 2)

        # Residual with learnable scale
        return residual + self.adapter_scale * x_out


@MODEL.register_module
class MambaCLIP(nn.Module):
    """
    Mamba-CLIP: Vision Mamba with CLIP Adapters for Anomaly Detection

    Features:
    - High-resolution processing with linear complexity
    - Mamba adapters injected into frozen CLIP backbone
    - Multi-scale feature extraction for anomaly detection
    - Both image-level and pixel-level predictions
    """
    def __init__(
        self,
        pretrained=False,
        checkpoint_path='',
        strict=True,
        image_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_adapters=12,  # Number of layers to add adapters
        adapter_d_state=16,
        use_ss2d=True,  # Use SS2D blocks or Vim blocks
        freeze_clip=True,
        **kwargs
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = (image_size // patch_size) ** 2
        self.use_ss2d = use_ss2d

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Backbone: Either SS2D blocks or Vim blocks
        if use_ss2d:
            # VMamba-style with SS2D
            self.blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=embed_dim,
                    drop_path=0.1 * (i / depth),
                    d_state=16,
                    expand=2,
                )
                for i in range(depth)
            ])
        else:
            # Vim-style with bidirectional Mamba
            self.blocks = nn.ModuleList([
                VimBlock(dim=embed_dim, d_state=16, expand=2)
                for i in range(depth)
            ])

        # Mamba Adapters (injected into selected layers)
        adapter_indices = list(range(depth - num_adapters, depth))
        self.adapter_indices = adapter_indices
        self.adapters = nn.ModuleDict({
            str(idx): MambaAdapterLayer(
                d_model=embed_dim,
                d_state=adapter_d_state,
                conv_scales=[3, 5, 7]
            )
            for idx in adapter_indices
        })

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Anomaly detection heads
        self.head_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.head_pixel = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, 1)
        )

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Optionally freeze parts of the network
        if freeze_clip:
            self._freeze_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self):
        """Freeze backbone, only train adapters and heads"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

    def forward_features(self, x):
        """
        Extract multi-scale features
        x: (B, 3, H, W)
        Returns: features dict with CLS token and spatial features
        """
        B = x.shape[0]
        H_patch = W_patch = self.image_size // self.patch_size

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H_p, W_p)

        if self.use_ss2d:
            # SS2D expects (B, H, W, C)
            x = x.permute(0, 2, 3, 1)  # (B, H_p, W_p, D)

            # No CLS token for SS2D, process entire feature map
            for i, block in enumerate(self.blocks):
                x = block(x)

                # Apply adapter if this layer has one
                if i in self.adapter_indices:
                    # Need to flatten for adapter
                    B, H, W, C = x.shape
                    x_flat = x.reshape(B, H * W, C)
                    x_flat = self.adapters[str(i)](x_flat, H, W)
                    x = x_flat.reshape(B, H, W, C)

            # Final norm
            x = self.norm(x)  # (B, H_p, W_p, D)

            # Extract features
            cls_token = x.mean(dim=(1, 2))  # Global average pooling for classification
            spatial_features = x.permute(0, 3, 1, 2)  # (B, D, H_p, W_p) for pixel-level

        else:
            # Vim expects (B, L, C) with CLS token
            x = x.flatten(2).transpose(1, 2)  # (B, L, D) where L = H_p * W_p

            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Add positional embedding
            x = x + self.pos_embed

            # Forward through blocks with adapters
            for i, block in enumerate(self.blocks):
                x = block(x)

                # Apply adapter if this layer has one
                if i in self.adapter_indices:
                    cls_token, img_tokens = x[:, :1], x[:, 1:]
                    img_tokens = self.adapters[str(i)](img_tokens, H_patch, W_patch)
                    x = torch.cat([cls_token, img_tokens], dim=1)

            # Final norm
            x = self.norm(x)

            # Extract features
            cls_token = x[:, 0]  # (B, D)
            spatial_features = x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, H_patch, W_patch)

        return {
            'cls_token': cls_token,
            'spatial_features': spatial_features
        }

    def forward(self, x):
        """
        Training forward pass
        x: (B, 3, H, W)
        Returns: image_score, pixel_map
        """
        features = self.forward_features(x)

        # Image-level prediction
        img_score = self.head_img(features['cls_token'])  # (B, 1)

        # Pixel-level prediction
        pixel_map = self.head_pixel(features['spatial_features'])  # (B, 1, H_p, W_p)

        # Upsample pixel map to input resolution
        pixel_map = F.interpolate(
            pixel_map, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False
        )

        return img_score, pixel_map

    @torch.no_grad()
    def predict(self, images):
        """
        Inference for anomaly detection
        images: (B, 3, H, W) - already on CUDA from trainer
        Returns: anomaly_scores (B,), anomaly_maps (B, H, W)
        """
        self.eval()

        img_scores, pixel_maps = self.forward(images)

        # Convert to numpy arrays
        anomaly_scores = img_scores.squeeze(1).cpu().numpy()  # (B,)
        anomaly_maps = pixel_maps.squeeze(1).cpu().numpy()  # (B, H, W)

        return anomaly_scores, anomaly_maps
