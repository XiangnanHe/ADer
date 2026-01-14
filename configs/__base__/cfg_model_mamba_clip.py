from argparse import Namespace


class cfg_model_mamba_clip(Namespace):

    def __init__(self):
        Namespace.__init__(self)

        self.model = Namespace()
        self.model.name = 'MambaCLIP'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=True,
            image_size=518,        # High resolution for anomaly detection
            patch_size=14,         # Standard ViT patch size
            embed_dim=768,         # Base model dimension
            depth=12,              # Number of transformer blocks
            num_adapters=6,        # Number of layers with Mamba adapters
            adapter_d_state=16,    # State dimension for Mamba adapters
            use_ss2d=True,         # Use SS2D (VMamba) or Vim blocks
            freeze_clip=False,     # Whether to freeze backbone (set True for faster training)
        )
