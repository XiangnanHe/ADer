from util.net import Namespace


class cfg_model_mamba_clip_zeroshot(Namespace):
    def __init__(self):
        self.model = Namespace()
        self.model.name = 'MambaCLIPZeroShot'
        self.model.pretrained = False
        self.model.kwargs = dict(
            # Image resolution
            image_size=518,
            patch_size=14,

            # Model architecture
            embed_dim=768,           # Base: 768, Small: 384, Large: 1024
            depth=12,                # Base: 12, Small: 6, Large: 24

            # Mamba adapters
            num_adapters=6,          # Number of layers with adapters (last N layers)
            adapter_d_state=16,      # State dimension for adapters

            # Backbone type
            use_ss2d=True,          # True: SS2D blocks, False: Vim blocks

            # CLIP configuration
            clip_model_name='ViT-B/16',  # CLIP model: ViT-B/16 or ViT-L/14
            freeze_encoder=True,         # Freeze Mamba encoder (only train adapters)

            # Model loading
            pretrained=False,
            checkpoint_path='',
            strict=True,
        )
