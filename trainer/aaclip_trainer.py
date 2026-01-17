"""
AA-CLIP Trainer for ADer Framework

Implements two-stage training:
- Stage 1: Train text adapter for anomaly-aware embeddings (5 epochs default)
- Stage 2: Train image adapter to align with text embeddings (20 epochs default)
"""

import sys
import os
import glob
import time
import torch
import torch.nn.functional as F
import numpy as np
import importlib.util
import types

# Mock ipdb to avoid import errors (AA-CLIP has unused ipdb imports)
if 'ipdb' not in sys.modules:
    ipdb_mock = types.ModuleType('ipdb')
    ipdb_mock.set_trace = lambda: None
    sys.modules['ipdb'] = ipdb_mock

# Add AA-CLIP to sys.path for imports
aa_clip_base = '/mnt/task_runtime/AA-CLIP'
if aa_clip_base not in sys.path:
    sys.path.insert(0, aa_clip_base)

# Load AA-CLIP modules with proper package setup
def _load_aaclip_module(module_path, module_name, package_name=None):
    """Load a module from absolute file path"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if package_name:
        module.__package__ = package_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Pre-load model.tokenizer so forward_utils can import it
tokenizer_mod = _load_aaclip_module(
    f'{aa_clip_base}/model/tokenizer.py',
    'model.tokenizer'
)

# Now import AA-CLIP utilities - model.tokenizer is already loaded
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
    BinaryDiceLoss,
    FocalLoss
)
from dataset.constants import CLASS_NAMES

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.util import log_msg, update_log_term
from util.net import reduce_tensor, get_timepc
from util.vis import vis_rgb_gt_amp


@TRAINER.register_module
class AAClipTrainer(BaseTrainer):
    """
    AA-CLIP Trainer with two-stage training

    Stage 1: Text adapter training - learns anomaly-aware text embeddings
    Stage 2: Image adapter training - aligns visual features with text embeddings
    """

    def __init__(self, cfg):
        super(AAClipTrainer, self).__init__(cfg)

        # Training stage configuration
        self.text_epochs = getattr(cfg.trainer, 'text_epochs', 5)
        self.image_epochs = getattr(cfg.trainer, 'image_epochs', 20)
        self.current_stage = 'text'  # 'text' or 'image'

        # Loss functions for AA-CLIP
        self.focal_loss = FocalLoss()
        self.dice_loss = BinaryDiceLoss()

        # Text embeddings (will be computed after stage 1)
        self.text_features = None

        # Dataset name for text prompt generation
        self.dataset_name = self._get_dataset_name()

        # Orthogonal loss weight
        self.text_norm_weight = getattr(cfg.trainer, 'text_norm_weight', 0.1)

        # Domain type (Industrial or Medical) - affects post-processing
        self.domain = getattr(cfg.trainer, 'domain', 'Industrial')

    def _get_dataset_name(self):
        """Extract dataset name from config"""
        data_root = self.cfg.data.root
        if 'mvtec' in data_root.lower():
            return 'MVTec'
        elif 'visa' in data_root.lower():
            return 'VisA'
        elif 'btad' in data_root.lower():
            return 'BTAD'
        else:
            # Default to MVTec
            return 'MVTec'

    def set_input(self, inputs):
        """Prepare input data"""
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def forward(self):
        """Forward pass through AA-CLIP model"""
        self.seg_tokens, self.det_token = self.net(self.imgs)

    def optimize_parameters(self):
        """Training step - routes to appropriate stage"""
        if self.current_stage == 'text':
            self._optimize_text_adapter()
        else:
            self._optimize_image_adapter()

    def _optimize_text_adapter(self):
        """Stage 1: Train text adapter"""
        with self.amp_autocast():
            # Get patch features from original CLIP (frozen, for supervision)
            # This returns features from multiple layers
            patch_features_list, _ = self.net.forward_original(self.imgs, modality="visual")

            # Get class token for residual connection (as in original AA-CLIP)
            with torch.no_grad():
                cls_token, _ = self.net.model.clipmodel.encode_image(self.imgs, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

            # Normalize patch features and add class token (critical for AA-CLIP)
            # This is done in original AA-CLIP line 78-85
            normalized_patch_features = []
            for patch_feat in patch_features_list:
                # Normalize
                patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
                # Add class token as residual
                patch_feat = patch_feat + cls_token.unsqueeze(1)
                normalized_patch_features.append(patch_feat)

            # Generate text embeddings for all classes in batch
            text_features_dict = {}
            for cls in set(self.cls_name):
                text_feat = get_adapted_single_class_text_embedding(
                    self.net.model, self.dataset_name, cls, self.imgs.device
                )
                text_features_dict[cls] = text_feat

            # Compute similarity maps and segmentation loss across all patch feature levels
            total_loss = 0
            ortho_loss = 0

            for i, cls in enumerate(self.cls_name):
                text_feat = text_features_dict[cls]  # [768, 2] (normal, anomaly)

                # Calculate similarity map using features from ALL levels (as in original AA-CLIP)
                for patch_feat in normalized_patch_features:
                    # Get features for this sample
                    sample_feat = patch_feat[i:i+1]  # Keep batch dim
                    sim_map = calculate_similarity_map(
                        sample_feat,
                        text_feat,
                        img_size=self.imgs_mask.shape[-2:],
                        test=False,
                        domain=self.domain
                    )

                    # Segmentation loss (dice + focal)
                    mask = self.imgs_mask[i:i+1]
                    seg_loss = calculate_seg_loss(sim_map, mask)
                    total_loss += seg_loss

                # Orthogonal constraint: encourage normal/anomaly embeddings to be orthogonal
                normal_emb = text_feat[:, 0]
                anomaly_emb = text_feat[:, 1]
                ortho = torch.abs(F.cosine_similarity(normal_emb, anomaly_emb, dim=0))
                ortho_loss += ortho

            # Average losses over batch and feature levels
            num_levels = len(normalized_patch_features)
            total_loss = total_loss / (self.bs * num_levels)
            ortho_loss = ortho_loss / self.bs

            # Combined loss
            combined_loss = total_loss + self.text_norm_weight * ortho_loss

            # Safety check: warn if loss is suspiciously low
            if combined_loss < 0.01 and self.master:
                log_msg(self.logger, f"WARNING: Very low loss detected: {combined_loss.item():.6f}")

        # Backward
        self.backward_term(combined_loss, self.optim)

        # Log losses
        update_log_term(
            self.log_terms.get('total'),
            reduce_tensor(combined_loss, self.world_size).item(),
            1, self.master
        )
        update_log_term(
            self.log_terms.get('seg_loss'),
            reduce_tensor(total_loss, self.world_size).item(),
            1, self.master
        )
        update_log_term(
            self.log_terms.get('ortho_loss'),
            reduce_tensor(ortho_loss, self.world_size).item(),
            1, self.master
        )

    def _optimize_image_adapter(self):
        """Stage 2: Train image adapter"""
        with self.amp_autocast():
            self.forward()

            # Get text embeddings (frozen from stage 1)
            if self.text_features is None:
                self._compute_text_features()

            # Compute losses
            total_loss = 0

            for i, cls in enumerate(self.cls_name):
                text_feat = self.text_features[cls]  # [768, 2]

                # Image-level classification loss
                # det_token: [B, 768], text_feat: [768, 2]
                det_sim = torch.matmul(
                    self.det_token[i:i+1],  # [1, 768]
                    text_feat  # [768, 2]
                )  # [1, 2] - similarity to normal/anomaly

                # Target: anomaly if any pixel is anomaly (scalar tensor)
                # Need to check if mask has any positive values
                has_anomaly = (self.imgs_mask[i].max() > 0).long()  # Scalar: 0 or 1
                target = has_anomaly.unsqueeze(0)  # [1] - single element for batch size 1

                # Cross-entropy loss for classification (as in original AA-CLIP)
                img_loss = F.cross_entropy(det_sim, target)

                # Pixel-level segmentation loss
                # Use last-level seg_tokens for segmentation
                seg_token = self.seg_tokens[-1][i:i+1]  # [1, L, 768]
                sim_map = calculate_similarity_map(
                    seg_token,
                    text_feat,
                    img_size=self.imgs_mask.shape[-2:],
                    test=False,
                    domain=self.domain
                )

                mask = self.imgs_mask[i:i+1]
                seg_loss = calculate_seg_loss(sim_map, mask)

                total_loss += img_loss + seg_loss

            # Average over batch
            total_loss = total_loss / self.bs

        # Backward
        self.backward_term(total_loss, self.optim)

        # Log loss
        update_log_term(
            self.log_terms.get('total'),
            reduce_tensor(total_loss, self.world_size).item(),
            1, self.master
        )

    def _compute_text_features(self):
        """Compute text features for all classes after stage 1"""
        with torch.no_grad():
            text_features_raw = get_adapted_text_embedding(
                self.net.model,
                self.dataset_name,
                self.imgs.device
            )
        # Detach all text features to prevent gradient tracking in Stage 2
        self.text_features = {k: v.detach() for k, v in text_features_raw.items()}

    def train(self):
        """Override train to handle two-stage training"""
        # Stage 1: Train text adapter
        if self.master:
            log_msg(self.logger, "=" * 80)
            log_msg(self.logger, "Stage 1: Training Text Adapter")
            log_msg(self.logger, "=" * 80)

        self.current_stage = 'text'
        self.net.unfreeze_text_adapter()
        self.net.freeze_image_adapter()
        self.epoch_full = self.text_epochs

        super().train()

        # Compute and cache text features
        if self.master:
            log_msg(self.logger, "\nComputing text features...")
        self._compute_text_features()

        # Stage 2: Train image adapter
        if self.master:
            log_msg(self.logger, "=" * 80)
            log_msg(self.logger, "Stage 2: Training Image Adapter")
            log_msg(self.logger, "=" * 80)

        self.current_stage = 'image'
        self.net.freeze_text_adapter()
        self.net.unfreeze_image_adapter()
        self.epoch = 0  # Reset epoch counter
        self.epoch_full = self.image_epochs

        # Update learning rate for Stage 2 (5e-4 for image adapter, as in original AA-CLIP)
        image_lr = getattr(self.cfg, 'image_lr', 0.0005)
        for param_group in self.optim.param_groups:
            param_group['lr'] = image_lr
        if self.master:
            log_msg(self.logger, f"Updated learning rate to {image_lr} for image adapter training")

        super().train()

    @torch.no_grad()
    def test(self):
        """Evaluation loop"""
        if self.master:
            if os.path.exists(self.tmp_dir):
                import shutil
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.reset(isTrain=False)

        # Ensure text features are computed
        if self.text_features is None:
            self._compute_text_features()

        # Collection lists
        imgs_masks = []
        anomaly_maps = []
        anomaly_scores = []
        cls_names = []
        anomalys = []

        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)

        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1

            # Get test data
            test_data = next(test_loader)
            self.set_input(test_data)

            # Forward pass
            self.forward()

            # Compute anomaly scores and maps
            batch_scores = []
            batch_maps = []

            for i, cls in enumerate(self.cls_name):
                text_feat = self.text_features[cls]  # [768, 2]

                # Image-level anomaly score
                det_sim = 100.0 * torch.matmul(
                    self.det_token[i:i+1],
                    text_feat
                )  # [1, 2]
                # Anomaly score = (anomaly_sim + (1 - normal_sim)) / 2
                score = (det_sim[0, 1] + (100.0 - det_sim[0, 0])) / 2
                batch_scores.append(score.item())

                # Pixel-level anomaly map
                seg_token = self.seg_tokens[-1][i:i+1]
                sim_map = calculate_similarity_map(
                    seg_token,
                    text_feat,
                    img_size=self.imgs_mask.shape[-2:],
                    test=True,
                    domain=self.domain
                )
                # sim_map is [1, H, W] after test=True processing
                batch_maps.append(sim_map.squeeze(0).squeeze(0).cpu().numpy())

            # Convert to arrays
            batch_scores = np.array(batch_scores)
            batch_maps = np.array(batch_maps)[:, np.newaxis, :, :]  # Add channel dim

            # Binarize masks
            self.imgs_mask[self.imgs_mask > 0.5] = 1
            self.imgs_mask[self.imgs_mask <= 0.5] = 0

            # Visualization if enabled
            if self.cfg.vis:
                root_out = self.cfg.vis_dir if self.cfg.vis_dir else self.writer.logdir
                vis_rgb_gt_amp(
                    self.img_path,
                    self.imgs,
                    self.imgs_mask.cpu().numpy().astype(int),
                    batch_maps,
                    self.cfg.model.name,
                    root_out,
                    self.cfg.data.root.split('/')[-1]
                )

            # Collect results
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(batch_maps)
            anomaly_scores.append(batch_scores)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)

            # Logging
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    from util.util import able
                    msg = able(
                        self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'),
                        self.master, None
                    )
                    log_msg(self.logger, msg)
                    print(f'\r{batch_idx}/{test_length}', end='')

        # Merge results from all GPUs if distributed
        if self.cfg.dist:
            results = dict(
                imgs_masks=imgs_masks,
                anomaly_maps=anomaly_maps,
                anomaly_scores=anomaly_scores,
                cls_names=cls_names,
                anomalys=anomalys
            )
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)

            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], anomaly_scores=[], cls_names=[], anomalys=[])
                valid_results = False

                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True

                imgs_masks = results['imgs_masks']
                anomaly_maps = results['anomaly_maps']
                anomaly_scores = results['anomaly_scores']
                cls_names = results['cls_names']
                anomalys = results['anomalys']

        # Concatenate all results
        imgs_masks = np.concatenate(imgs_masks, axis=0)
        anomaly_maps = np.concatenate(anomaly_maps, axis=0)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        cls_names = np.concatenate(cls_names, axis=0)
        anomalys = np.concatenate(anomalys, axis=0)

        # Evaluate metrics per class
        if self.master:
            # Process results
            results = dict(
                imgs_masks=imgs_masks,
                anomaly_maps=anomaly_maps,
                anomaly_scores=anomaly_scores,
                cls_names=cls_names,
                anomalys=anomalys
            )

            # Run evaluator
            for cls in np.unique(cls_names):
                # Filter results for this class
                cls_idx = cls_names == cls
                cls_results = {
                    k: v[cls_idx] if isinstance(v, np.ndarray) else v
                    for k, v in results.items()
                }

                # Run evaluation
                metric_results = self.evaluator.run(cls_results, cls, self.logger)

                # Record metrics
                for metric in self.metrics:
                    metric_result = metric_results.get(metric, 0)
                    self.metric_recorder[f'{metric}_{cls}'].append(metric_result)

            # Compute average metrics
            from tabulate import tabulate
            table_data = []
            for cls in np.unique(cls_names):
                row = [cls]
                for metric in self.metrics:
                    value = self.metric_recorder[f'{metric}_{cls}'][-1]
                    row.append(f'{value:.3f}')
                table_data.append(row)

            # Add average row
            avg_row = ['Avg']
            for metric in self.metrics:
                values = [self.metric_recorder[f'{metric}_{cls}'][-1]
                         for cls in np.unique(cls_names)]
                avg_value = np.mean(values)
                avg_row.append(f'{avg_value:.3f}')
                self.metric_recorder[f'{metric}_Avg'].append(avg_value)
            table_data.append(avg_row)

            # Print results table
            headers = ['Name'] + self.metrics
            table_str = tabulate(table_data, headers=headers, tablefmt='pipe')
            log_msg(self.logger, '\n' + table_str)
