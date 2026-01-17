"""
Mamba-CLIP Self-Supervised: Reconstruction-Based Anomaly Detection

This module implements self-supervised anomaly detection using:
1. Mamba encoder for feature extraction
2. Mamba decoder for reconstruction
3. Reconstruction error for anomaly scoring
4. Training only on normal samples

Key Features:
- Self-supervised: Only normal samples needed for training
- One-class learning: Learn to reconstruct normal patterns
- High-resolution: 518×518+ image support
- Dual reconstruction: Image + feature reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

from model import MODEL

# Import the base Mamba encoder components
from model.mamba_clip import SS2D, VSSBlock, VimBlock, MambaAdapterLayer


class PatchExpand2D(nn.Module):
    """Patch expansion for decoder upsampling"""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim_scale * (dim // dim_scale), bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: (B, H, W, C)
        Returns: (B, H*2, W*2, C//2)
        """
        B, H, W, C = x.shape
        x = self.expand(x)  # (B, H, W, 4 * (C//2)) = (B, H, W, 2C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                     p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)
        return x


class MambaDecoderBlock(nn.Module):
    """Decoder block with SS2D and upsampling"""
    def __init__(self, dim, upsample=True, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=16, expand=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if upsample:
            self.upsample = PatchExpand2D(dim=dim, dim_scale=2)
        else:
            self.upsample = None

    def forward(self, x):
        """
        x: (B, H, W, C)
        Returns: (B, H', W', C')
        """
        # SS2D block with residual
        x = x + self.drop_path(self.ss2d(self.norm(x)))

        # Upsample if needed
        if self.upsample is not None:
            x = self.upsample(x)

        return x


@MODEL.register_module
class MambaCLIPSelfSupervised(nn.Module):
    """
    Self-Supervised Anomaly Detection with Mamba-CLIP

    Training: Only normal samples (reconstruction loss)
    Inference: Anomaly = high reconstruction error

    Architecture:
        Encoder (Mamba) → Bottleneck → Decoder (Mamba) → Reconstruction
    """
    def __init__(
        self,
        pretrained=False,
        checkpoint_path='',
        strict=True,
        image_size=518,
        patch_size=14,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=4,
        num_adapters=6,
        adapter_d_state=16,
        use_ss2d=True,
        **kwargs
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_patches = (image_size // patch_size) ** 2
        self.use_ss2d = use_ss2d

        # ========== ENCODER ==========
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Encoder blocks
        if use_ss2d:
            self.encoder_blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=embed_dim,
                    drop_path=0.1 * (i / encoder_depth),
                    d_state=16,
                    expand=2,
                )
                for i in range(encoder_depth)
            ])
        else:
            self.encoder_blocks = nn.ModuleList([
                VimBlock(dim=embed_dim, d_state=16, expand=2)
                for i in range(encoder_depth)
            ])

        # Encoder adapters
        encoder_adapter_indices = list(range(encoder_depth - num_adapters, encoder_depth))
        self.encoder_adapter_indices = encoder_adapter_indices
        self.encoder_adapters = nn.ModuleDict({
            str(idx): MambaAdapterLayer(
                d_model=embed_dim,
                d_state=adapter_d_state,
                conv_scales=[3, 5, 7]
            )
            for idx in encoder_adapter_indices
        })

        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ========== DECODER ==========
        # Decoder will upsample from bottleneck back to image resolution
        decoder_dims = [embed_dim // (2**i) for i in range(decoder_depth)]

        self.decoder_blocks = nn.ModuleList([
            MambaDecoderBlock(
                dim=decoder_dims[i],
                upsample=(i < decoder_depth - 1),
                drop_path=0.1 * (i / decoder_depth)
            )
            for i in range(decoder_depth)
        ])

        # Final reconstruction head
        final_dim = decoder_dims[-1]
        self.recon_head = nn.Sequential(
            nn.Conv2d(final_dim, final_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_dim // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

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

    def encode(self, x):
        """
        Encode images to latent features
        x: (B, 3, H, W)
        Returns: (B, H_latent, W_latent, D)
        """
        B = x.shape[0]
        H_patch = W_patch = self.image_size // self.patch_size

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H_p, W_p)

        if self.use_ss2d:
            # SS2D path
            x = x.permute(0, 2, 3, 1)  # (B, H_p, W_p, D)

            # No positional embedding for SS2D (spatial structure preserved)
            for i, block in enumerate(self.encoder_blocks):
                x = block(x)

                if i in self.encoder_adapter_indices:
                    B, H, W, C = x.shape
                    x_flat = x.reshape(B, H * W, C)
                    x_flat = self.encoder_adapters[str(i)](x_flat, H, W)
                    x = x_flat.reshape(B, H, W, C)

            x = self.encoder_norm(x)  # (B, H_p, W_p, D)

        else:
            # Vim path
            x = x.flatten(2).transpose(1, 2)  # (B, L, D)

            # Add positional embedding (no CLS token for reconstruction)
            x = x + self.pos_embed

            for i, block in enumerate(self.encoder_blocks):
                x = block(x)

                if i in self.encoder_adapter_indices:
                    x = self.encoder_adapters[str(i)](x, H_patch, W_patch)

            x = self.encoder_norm(x)

            # Reshape back to 2D
            x = x.reshape(B, H_patch, W_patch, self.embed_dim)

        return x

    def decode(self, latent):
        """
        Decode latent features to reconstructed image
        latent: (B, H_latent, W_latent, D)
        Returns: (B, 3, H, W)
        """
        x = latent

        # Decoder blocks (progressive upsampling)
        for block in self.decoder_blocks:
            x = block(x)

        # x is now (B, H', W', D')
        # Convert to channel-first for conv
        x = x.permute(0, 3, 1, 2)  # (B, D', H', W')

        # Upsample to original image size if needed
        if x.shape[2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=False)

        # Reconstruction head
        recon = self.recon_head(x)  # (B, 3, H, W)

        return recon

    def forward(self, images):
        """
        Training forward pass

        Args:
            images: (B, 3, H, W) - normalized to [0, 1]

        Returns:
            reconstructed: (B, 3, H, W) - reconstructed images
        """
        # Encode
        latent = self.encode(images)

        # Decode
        reconstructed = self.decode(latent)

        return reconstructed

    @torch.no_grad()
    def predict(self, images):
        """
        Self-supervised anomaly detection via reconstruction error

        Args:
            images: (B, 3, H, W) - already on CUDA from trainer

        Returns:
            anomaly_scores: (B,) - mean reconstruction error per image
            anomaly_maps: (B, H, W) - pixel-level reconstruction error
        """
        self.eval()

        # Reconstruct
        reconstructed = self.forward(images)

        # Reconstruction error (L2 distance per pixel)
        error = (images - reconstructed) ** 2  # (B, 3, H, W)

        # Pixel-level anomaly map (mean across channels)
        anomaly_maps = error.mean(dim=1)  # (B, H, W)

        # Image-level anomaly score (mean across spatial dimensions)
        anomaly_scores = anomaly_maps.mean(dim=(1, 2))  # (B,)

        # Convert to numpy
        anomaly_scores = anomaly_scores.cpu().numpy()
        anomaly_maps = anomaly_maps.cpu().numpy()

        # Normalize to [0, 1] for consistency
        for i in range(len(anomaly_maps)):
            amap = anomaly_maps[i]
            anomaly_maps[i] = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

        return anomaly_scores, anomaly_maps
