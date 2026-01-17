from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_aaclip):
    """
    AA-CLIP Configuration for MVTec-AD Benchmark

    Two-Stage Training:
    - Stage 1: Text adapter training (5 epochs)
    - Stage 2: Image adapter training (20 epochs)
    """

    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_aaclip.__init__(self)

        self.seed = 42

        # High-resolution settings for anomaly detection
        self.size = 518  # High resolution to capture fine-grained defects
        self.image_size = 518
        self.input_size = (3, self.image_size, self.image_size)

        # Training settings (two-stage)
        self.text_epochs = 5      # Stage 1: text adapter
        self.image_epochs = 20    # Stage 2: image adapter
        self.epoch_full = self.text_epochs + self.image_epochs  # Total epochs
        self.warmup_epochs = 0    # No warmup - use constant LR for Stage 1
        self.test_start_epoch = self.text_epochs  # Test after stage 1
        self.test_per_epoch = 5

        # Batch sizes (adjust based on GPU memory)
        self.batch_train = 2  # Very small due to high resolution 518x518
        self.batch_test_per = 4

        # Optimizer settings (different for each stage)
        self.text_lr = 0.00001    # Stage 1: text adapter LR
        self.image_lr = 0.0005    # Stage 2: image adapter LR
        self.lr = self.text_lr    # Start with text LR
        self.weight_decay = 0.001

        # AA-CLIP specific parameters
        self.text_norm_weight = 0.1  # Orthogonal constraint weight
        self.domain = 'Industrial'   # Industrial or Medical

        # Metrics to evaluate
        self.metrics = [
            'mAUROC_sp_max',      # Image-level AUROC (key metric)
            'mAP_sp_max',         # Image-level AP
            'mF1_max_sp_max',     # Image-level F1
            'mAUPRO_px',          # Pixel-level AUPRO (recommended)
            'mAUROC_px',          # Pixel-level AUROC
            'mAP_px',             # Pixel-level AP
            'mF1_max_px',         # Pixel-level F1
            'mIoU_max_px',        # Pixel-level IoU (recommended)
        ]
        self.use_adeval = True  # Use GPU-accelerated evaluation

        # Dataset configuration
        self.data.type = 'DefaultAD'
        self.data.root = 'data/mvtec'
        self.data.meta = 'meta.json'
        self.data.cls_names = []  # Empty = all classes

        # Data transforms (high resolution, minimal augmentation)
        self.data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size),
                 interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size),
                 interpolation=F.InterpolationMode.NEAREST),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        # Model configuration (override base settings)
        self.model.name = 'aaclip'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',  # Empty - we load CLIP checkpoint internally via clip_checkpoint_path
            strict=False,
            clip_checkpoint_path='/mnt/task_runtime/AA-CLIP/model/ViT-L-14-336px.pt',
            model_name='ViT-L-14-336',
            img_size=self.image_size,
            text_adapt_weight=0.1,
            image_adapt_weight=0.1,
            text_adapt_until=3,
            image_adapt_until=6,
            levels=[6, 12, 18, 24],
            relu=True,
        )

        # Evaluator
        self.evaluator.kwargs = dict(
            metrics=self.metrics,
            pooling_ks=None,
            max_step_aupro=100,
            use_adeval=self.use_adeval
        )

        # Optimizer (Adam for AA-CLIP)
        self.optim.lr = self.lr
        self.optim.kwargs = dict(
            name='adam',
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False
        )

        # Trainer
        self.trainer.name = 'AAClipTrainer'
        self.trainer.logdir_sub = ''
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full

        # AA-CLIP specific trainer parameters
        self.trainer.text_epochs = self.text_epochs
        self.trainer.image_epochs = self.image_epochs
        self.trainer.text_norm_weight = self.text_norm_weight
        self.trainer.domain = self.domain

        # Learning rate scheduler
        # Stage 1 (text adapter): constant LR (as in original AA-CLIP)
        # Stage 2 (image adapter): MultiStepLR with decay at 60% and 80% of iterations
        self.trainer.scheduler_kwargs = dict(
            name='step',
            lr_noise=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            lr_min=self.lr,  # Keep minimum at base LR (no decay for Stage 1)
            warmup_lr=self.lr,  # Start at full LR immediately
            warmup_iters=-1,
            cooldown_iters=0,
            warmup_epochs=0,  # No warmup
            cooldown_epochs=0,
            use_iters=False,
            patience_iters=0,
            patience_epochs=0,
            decay_iters=0,
            decay_epochs=1000,  # Very large value = no decay for Stage 1
            cycle_decay=0.1,
            decay_rate=1.0  # No decay (keep at 100%)
        )

        # Mixup/Cutmix (disabled for anomaly detection)
        self.trainer.mixup_kwargs = dict(
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            cutmix_minmax=None,
            prob=0.0,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.0
        )

        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        # Batch sizes
        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        # Loss configuration (custom losses handled in trainer)
        self.loss.loss_terms = []

        # Logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>9.7f'),  # More precision for small LR values
            dict(name='total', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='seg_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='ortho_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
        ]
