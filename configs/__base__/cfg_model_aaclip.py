from argparse import Namespace


class cfg_model_aaclip(Namespace):

    def __init__(self):
        Namespace.__init__(self)

        self.model = Namespace()
        self.model.name = 'aaclip'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',  # Empty - we load CLIP checkpoint internally via clip_checkpoint_path
            strict=False,
            clip_checkpoint_path='/mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt',
            model_name='ViT-L-14-336',          # CLIP model architecture
            img_size=518,                        # High resolution for fine-grained anomaly detection
            text_adapt_weight=0.1,               # Weight for text adapter residual connection
            image_adapt_weight=0.1,              # Weight for image adapter residual connection
            text_adapt_until=3,                  # Adapt first 3 text encoder layers
            image_adapt_until=6,                 # Adapt first 6 visual encoder layers
            levels=[6, 12, 18, 24],             # Extract features from these layers
            relu=True,                           # Use ReLU in adapter projections
        )
