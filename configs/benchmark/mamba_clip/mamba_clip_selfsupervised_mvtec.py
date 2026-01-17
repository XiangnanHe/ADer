from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_mamba_clip_selfsupervised):

    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_mamba_clip_selfsupervised.__init__(self)

        self.seed = 42

        # High-resolution settings for anomaly detection
        self.size = 518
        self.image_size = 518
        self.input_size = (3, self.image_size, self.image_size)

        # Training settings (self-supervised on normal samples)
        self.epoch_full = 100  # More epochs for reconstruction learning
        self.warmup_epochs = 10
        self.test_start_epoch = 10
        self.test_per_epoch = 10

        # Batch sizes
        self.batch_train = 8
        self.batch_test_per = 16

        # Optimizer settings
        self.lr = 0.0002  # Slightly higher for reconstruction task
        self.weight_decay = 0.05

        # Metrics to evaluate
        self.metrics = [
            'mAUROC_sp_max',      # Image-level AUROC
            'mAP_sp_max',         # Image-level AP
            'mF1_max_sp_max',     # Image-level F1
            'mAUPRO_px',          # Pixel-level AUPRO
            'mAUROC_px',          # Pixel-level AUROC
            'mAP_px',             # Pixel-level AP
            'mF1_max_px',         # Pixel-level F1
            'mIoU_max_px',        # Pixel-level IoU
        ]
        self.use_adeval = True

        # Dataset configuration
        self.data.type = 'DefaultAD'
        self.data.root = 'data/mvtec'
        self.data.meta = 'meta.json'
        self.data.cls_names = []  # Empty = all classes

        # Data transforms
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

        # Model configuration
        self.model.name = 'MambaCLIPSelfSupervised'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=True,
            image_size=self.image_size,
            patch_size=14,
            embed_dim=768,
            encoder_depth=12,
            decoder_depth=4,
            num_adapters=6,
            adapter_d_state=16,
            use_ss2d=True,
        )

        # Evaluator
        self.evaluator.kwargs = dict(
            metrics=self.metrics,
            pooling_ks=None,
            max_step_aupro=100,
            use_adeval=self.use_adeval
        )

        # Optimizer
        self.optim.lr = self.lr
        self.optim.kwargs = dict(
            name='adamw',
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False
        )

        # Trainer
        self.trainer.name = 'MambaCLIPSelfSupervisedTrainer'
        self.trainer.logdir_sub = ''
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.recon_weight = 1.0  # Reconstruction loss weight

        # Learning rate scheduler
        self.trainer.scheduler_kwargs = dict(
            name='cosine',
            lr_noise=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            lr_min=self.lr / 100,
            warmup_lr=self.lr / 10,
            warmup_iters=-1,
            cooldown_iters=0,
            warmup_epochs=self.warmup_epochs,
            cooldown_epochs=0,
            use_iters=False,
            patience_iters=0,
            patience_epochs=0,
            decay_iters=0,
            decay_epochs=0,
            cycle_decay=0.1,
            decay_rate=0.1
        )

        # Mixup/Cutmix (disabled for reconstruction)
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

        # Loss configuration (MSE for reconstruction)
        self.loss.loss_terms = [
            dict(type='MSELoss', name='mse', lam=1.0),  # Reconstruction loss
        ]

        # Logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
        ]
