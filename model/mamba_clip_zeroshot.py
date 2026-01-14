"""
Mamba-CLIP Zero-Shot: Text-Guided Anomaly Detection Without Training

This module implements zero-shot anomaly detection using:
1. Mamba encoder for high-resolution image processing
2. CLIP text encoder for semantic understanding
3. Text-image similarity for anomaly scoring
4. No anomaly-specific training required

Key Features:
- Zero-shot: No training on anomaly detection task
- Text prompts: "a photo of a flawless X" vs "a photo of a defective X"
- High-resolution: 518×518+ image support
- Pixel-level: Spatial token-text similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from model import MODEL

# Import the base Mamba encoder components
from model.mamba_clip import SS2D, VSSBlock, VimBlock, MambaAdapterLayer


try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder for zero-shot prompts"""
    def __init__(self, clip_model_name='ViT-B/16'):
        super().__init__()
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required for zero-shot. Install with: pip install git+https://github.com/openai/CLIP.git")

        # Load CLIP
        self.clip_model, _ = clip.load(clip_model_name, device='cpu')
        self.clip_model.eval()

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_prompts):
        """
        Encode text prompts
        text_prompts: list of strings or tokenized text
        Returns: (num_prompts, embed_dim)
        """
        if isinstance(text_prompts[0], str):
            text_tokens = clip.tokenize(text_prompts)
        else:
            text_tokens = text_prompts

        text_tokens = text_tokens.to(next(self.clip_model.parameters()).device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


@MODEL.register_module
class MambaCLIPZeroShot(nn.Module):
    """
    Zero-Shot Anomaly Detection with Mamba-CLIP

    No training required - uses text-image similarity for detection

    Usage:
        model = MambaCLIPZeroShot()
        prompts = ["a photo of a flawless product", "a photo of a defective product"]
        scores = model(images, prompts)
        anomaly_score = scores[:, 1] - scores[:, 0]  # defective - flawless
    """
    def __init__(
        self,
        pretrained=False,
        checkpoint_path='',
        strict=True,
        image_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_adapters=6,
        adapter_d_state=16,
        use_ss2d=True,
        clip_model_name='ViT-B/16',
        freeze_encoder=True,
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

        # Mamba encoder (same as supervised version)
        if use_ss2d:
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
            self.blocks = nn.ModuleList([
                VimBlock(dim=embed_dim, d_state=16, expand=2)
                for i in range(depth)
            ])

        # Mamba Adapters
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

        # Projection head to match CLIP embedding dimension
        self.clip_embed_dim = 512 if 'B' in clip_model_name else 768
        self.projection = nn.Linear(embed_dim, self.clip_embed_dim)

        # CLIP text encoder (frozen)
        self.text_encoder = CLIPTextEncoder(clip_model_name)

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Optionally freeze encoder
        if freeze_encoder:
            self._freeze_encoder()

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

    def _freeze_encoder(self):
        """Freeze encoder, only train adapters and projection"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

    def encode_image(self, x):
        """
        Encode images to CLIP-compatible features
        x: (B, 3, H, W)
        Returns: dict with 'cls_token' and 'spatial_features'
        """
        B = x.shape[0]
        H_patch = W_patch = self.image_size // self.patch_size

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H_p, W_p)

        if self.use_ss2d:
            # SS2D path
            x = x.permute(0, 2, 3, 1)  # (B, H_p, W_p, D)

            for i, block in enumerate(self.blocks):
                x = block(x)

                if i in self.adapter_indices:
                    B, H, W, C = x.shape
                    x_flat = x.reshape(B, H * W, C)
                    x_flat = self.adapters[str(i)](x_flat, H, W)
                    x = x_flat.reshape(B, H, W, C)

            x = self.norm(x)  # (B, H_p, W_p, D)

            # Extract features
            cls_token = x.mean(dim=(1, 2))  # (B, D)
            spatial_features = x.reshape(B, H_patch * W_patch, self.embed_dim)  # (B, L, D)

        else:
            # Vim path
            x = x.flatten(2).transpose(1, 2)  # (B, L, D)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed

            for i, block in enumerate(self.blocks):
                x = block(x)

                if i in self.adapter_indices:
                    cls_token, img_tokens = x[:, :1], x[:, 1:]
                    img_tokens = self.adapters[str(i)](img_tokens, H_patch, W_patch)
                    x = torch.cat([cls_token, img_tokens], dim=1)

            x = self.norm(x)

            cls_token = x[:, 0]  # (B, D)
            spatial_features = x[:, 1:]  # (B, L, D)

        # Project to CLIP space
        cls_token = self.projection(cls_token)
        cls_token = F.normalize(cls_token, dim=-1)

        spatial_features = self.projection(spatial_features)
        spatial_features = F.normalize(spatial_features, dim=-1, p=2)

        return {
            'cls_token': cls_token,  # (B, clip_dim)
            'spatial_features': spatial_features  # (B, L, clip_dim)
        }

    def forward(self, images, text_prompts):
        """
        Zero-shot inference with text prompts

        Args:
            images: (B, 3, H, W)
            text_prompts: list of strings, e.g., ["flawless", "defective"]

        Returns:
            dict with:
                - 'image_similarity': (B, num_prompts) - similarity scores
                - 'pixel_maps': (B, num_prompts, H, W) - pixel-level similarity
        """
        # Encode images
        img_features = self.encode_image(images)
        cls_token = img_features['cls_token']  # (B, D)
        spatial_features = img_features['spatial_features']  # (B, L, D)

        # Encode text prompts
        text_features = self.text_encoder.encode_text(text_prompts)  # (num_prompts, D)
        text_features = text_features.to(cls_token.device)

        # Image-level similarity (cosine similarity)
        # cls_token: (B, D), text_features: (num_prompts, D)
        image_similarity = cls_token @ text_features.T  # (B, num_prompts)

        # Pixel-level similarity
        # spatial_features: (B, L, D), text_features: (num_prompts, D)
        pixel_similarity = spatial_features @ text_features.T  # (B, L, num_prompts)

        # Reshape to image
        B, L, num_prompts = pixel_similarity.shape
        H = W = int(L ** 0.5)
        pixel_maps = pixel_similarity.permute(0, 2, 1).reshape(B, num_prompts, H, W)

        # Upsample to original resolution
        pixel_maps = F.interpolate(
            pixel_maps, size=(images.shape[2], images.shape[3]),
            mode='bilinear', align_corners=False
        )

        return {
            'image_similarity': image_similarity,  # (B, num_prompts)
            'pixel_maps': pixel_maps  # (B, num_prompts, H, W)
        }

    @torch.no_grad()
    def predict(self, images, normal_prompt="a photo of a flawless object",
                anomaly_prompt="a photo of a defective object"):
        """
        Zero-shot anomaly detection

        Args:
            images: (B, 3, H, W) - already on CUDA from trainer
            normal_prompt: text description of normal samples
            anomaly_prompt: text description of anomalous samples

        Returns:
            anomaly_scores: (B,) - higher = more anomalous
            anomaly_maps: (B, H, W) - pixel-level anomaly scores
        """
        self.eval()

        # Get similarities for both prompts
        prompts = [normal_prompt, anomaly_prompt]
        outputs = self.forward(images, prompts)

        image_similarity = outputs['image_similarity']  # (B, 2)
        pixel_maps = outputs['pixel_maps']  # (B, 2, H, W)

        # Anomaly score = similarity to anomaly - similarity to normal
        anomaly_scores_img = image_similarity[:, 1] - image_similarity[:, 0]  # (B,)
        anomaly_maps_pixel = pixel_maps[:, 1] - pixel_maps[:, 0]  # (B, H, W)

        # Convert to numpy
        anomaly_scores = anomaly_scores_img.cpu().numpy()
        anomaly_maps = anomaly_maps_pixel.cpu().numpy()

        # Normalize to [0, 1] for consistency with other methods
        anomaly_maps = (anomaly_maps - anomaly_maps.min()) / (anomaly_maps.max() - anomaly_maps.min() + 1e-8)

        return anomaly_scores, anomaly_maps
