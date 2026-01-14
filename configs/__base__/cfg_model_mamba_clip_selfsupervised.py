from util.net import Namespace


class cfg_model_mamba_clip_selfsupervised(Namespace):
    def __init__(self):
        self.model = Namespace()
        self.model.name = 'MambaCLIPSelfSupervised'
        self.model.pretrained = False
        self.model.kwargs = dict(
            # Image resolution
            image_size=518,
            patch_size=14,

            # Encoder architecture
            embed_dim=768,           # Base: 768, Small: 384, Large: 1024
            encoder_depth=12,        # Base: 12, Small: 6, Large: 24

            # Decoder architecture
            decoder_depth=4,         # Number of decoder stages (progressive upsampling)

            # Mamba adapters
            num_adapters=6,          # Number of encoder layers with adapters
            adapter_d_state=16,      # State dimension for adapters

            # Backbone type
            use_ss2d=True,          # True: SS2D blocks, False: Vim blocks

            # Model loading
            pretrained=False,
            checkpoint_path='',
            strict=True,
        )
