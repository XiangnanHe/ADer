"""
AA-CLIP Model for ADer
Adapted from /mnt/task_runtime/AA-CLIP/model/adapter.py
"""
import sys
import os
import importlib.util
import types

# Import torch first (from ADer environment)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock ipdb to avoid import errors (AA-CLIP has unused ipdb imports)
if 'ipdb' not in sys.modules:
    ipdb_mock = types.ModuleType('ipdb')
    ipdb_mock.set_trace = lambda: None
    sys.modules['ipdb'] = ipdb_mock

# Load AA-CLIP modules by absolute file path to avoid package name conflicts
aa_clip_base = '/mnt/task_runtime/AA-CLIP'

# Helper to create a package structure and load module from file
def _load_aaclip_module_with_package(module_path, module_name, package_name=None):
    """Load a module from absolute file path with proper package setup"""
    spec = importlib.util.spec_from_file_location(module_name, module_path, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)

    # Set the package if provided
    if package_name:
        module.__package__ = package_name

    # Add to sys.modules before exec so relative imports work
    sys.modules[module_name] = module

    # Temporarily add AA-CLIP to path for imports within the module
    sys.path.insert(0, aa_clip_base)
    try:
        spec.loader.exec_module(module)
    finally:
        # Remove from front of path
        if sys.path and sys.path[0] == aa_clip_base:
            sys.path.pop(0)

    return module

# Create a fake package for AA-CLIP model to support relative imports
aaclip_model_package = types.ModuleType('_aaclip_model')
aaclip_model_package.__path__ = [f'{aa_clip_base}/model']
aaclip_model_package.__package__ = '_aaclip_model'
sys.modules['_aaclip_model'] = aaclip_model_package

# Load adapter_modules first (required by adapter.py)
adapter_modules_mod = _load_aaclip_module_with_package(
    f'{aa_clip_base}/model/adapter_modules.py',
    '_aaclip_model.adapter_modules',
    '_aaclip_model'
)

# Load adapter.py with the package structure
adapter_mod = _load_aaclip_module_with_package(
    f'{aa_clip_base}/model/adapter.py',
    '_aaclip_model.adapter',
    '_aaclip_model'
)

# Load clip.py
clip_mod = _load_aaclip_module_with_package(
    f'{aa_clip_base}/model/clip.py',
    '_aaclip_model.clip',
    '_aaclip_model'
)

AdaptedCLIP = adapter_mod.AdaptedCLIP
create_model = clip_mod.create_model

# Keep AA-CLIP in path for runtime
if aa_clip_base not in sys.path:
    sys.path.append(aa_clip_base)

from . import MODEL


@MODEL.register_module
class aaclip(nn.Module):
    """
    AA-CLIP model wrapper for ADer framework

    Two-stage training approach:
    - Stage 1: Train text adapter to generate anomaly-aware embeddings
    - Stage 2: Train image adapter to align with text embeddings
    """

    def __init__(
        self,
        pretrained=False,
        clip_checkpoint_path=None,
        model_name='ViT-L-14-336',
        img_size=518,
        text_adapt_weight=0.1,
        image_adapt_weight=0.1,
        text_adapt_until=3,
        image_adapt_until=6,
        levels=[6, 12, 18, 24],
        relu=True,
        **kwargs
    ):
        super().__init__()

        self.img_size = img_size
        self.model_name = model_name

        # Load base CLIP model
        # Create model with correct image size first
        clip_model = create_model(
            model_name,
            img_size,
            pretrained=False,  # Don't load pretrained yet
            force_image_size=img_size,  # Force image size to match our target size
            force_patch_dropout=0.0  # Disable patch dropout for anomaly detection
        )

        # Now load the checkpoint manually with proper positional embedding interpolation
        checkpoint_to_load = clip_checkpoint_path if (clip_checkpoint_path and os.path.exists(clip_checkpoint_path)) else '/mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt'

        if not os.path.exists(checkpoint_to_load):
            raise FileNotFoundError(
                f"No CLIP checkpoint found at {checkpoint_to_load}"
            )

        print(f'Loading pretrained {model_name} weights ({checkpoint_to_load}).')
        state_dict = torch.load(checkpoint_to_load, map_location='cpu')

        # Load resize function from AA-CLIP using our module loader
        model_mod = _load_aaclip_module_with_package(
            f'{aa_clip_base}/model/model.py',
            '_aaclip_model.model',
            '_aaclip_model'
        )
        resize_pos_embed = model_mod.resize_pos_embed

        # Resize positional embeddings to match our image size
        resize_pos_embed(state_dict, clip_model)

        # Load state dict
        clip_model.load_state_dict(state_dict, strict=True)

        # Create adapted CLIP model with adapters
        self.model = AdaptedCLIP(
            clip_model=clip_model,
            text_adapt_weight=text_adapt_weight,
            image_adapt_weight=image_adapt_weight,
            text_adapt_until=text_adapt_until,
            image_adapt_until=image_adapt_until,
            levels=levels,
            relu=relu,
        )

        # Freeze base CLIP parameters - only train adapters
        self._freeze_base_clip()

    def _freeze_base_clip(self):
        """Freeze base CLIP parameters, only train adapter modules"""
        # Freeze visual encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        # Freeze text encoder (except adapters which will be trained in stage 1)
        for param in self.model.clipmodel.transformer.parameters():
            param.requires_grad = False
        for param in self.model.clipmodel.token_embedding.parameters():
            param.requires_grad = False
        # positional_embedding is a Parameter, not a module
        if hasattr(self.model.clipmodel, 'positional_embedding'):
            self.model.clipmodel.positional_embedding.requires_grad = False
        for param in self.model.clipmodel.ln_final.parameters():
            param.requires_grad = False

    def freeze_text_adapter(self):
        """Freeze text adapter after stage 1 training"""
        for param in self.model.text_adapter.parameters():
            param.requires_grad = False

    def unfreeze_text_adapter(self):
        """Unfreeze text adapter for stage 1 training"""
        for param in self.model.text_adapter.parameters():
            param.requires_grad = True

    def freeze_image_adapter(self):
        """Freeze image adapter"""
        for param in self.model.image_adapter.parameters():
            param.requires_grad = False

    def unfreeze_image_adapter(self):
        """Unfreeze image adapter for stage 2 training"""
        for param in self.model.image_adapter.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through adapted CLIP

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            seg_tokens: Multi-level segmentation tokens [list of (B, L, 768)]
            det_token: Detection token [B, 768]
        """
        seg_tokens, det_token = self.model(x)
        return seg_tokens, det_token

    def encode_text(self, text, adapt_text=True):
        """
        Encode text with optional adaptation

        Args:
            text: Tokenized text [B, seq_len]
            adapt_text: Whether to use text adapter

        Returns:
            Text embeddings [B, 768]
        """
        return self.model.encode_text(text, adapt_text=adapt_text)

    def forward_original(self, x, modality="visual"):
        """
        Forward through original CLIP (without adapters)
        Used for feature extraction during training

        Args:
            x: Input images [B, 3, H, W]
            modality: "visual" or "text"

        Returns:
            patch_features: Patch features from layer 24
            cls_features: Class token features
        """
        return self.model.forward_original(x, modality=modality)
