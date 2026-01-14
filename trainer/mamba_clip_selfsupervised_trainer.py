"""
Trainer for Mamba-CLIP Self-Supervised

Self-supervised anomaly detection:
- Training: Only normal samples (reconstruction loss)
- Testing: Anomaly = high reconstruction error
- One-class learning paradigm
"""

import os
import glob
import time
import torch
import numpy as np

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.util import log_msg, update_log_term
from util.net import reduce_tensor, get_timepc
from util.vis import vis_rgb_gt_amp


@TRAINER.register_module
class MambaCLIPSelfSupervisedTrainer(BaseTrainer):
    """
    Trainer for self-supervised Mamba-CLIP

    Training on normal samples only with reconstruction loss.
    Anomalies detected by high reconstruction error.
    """
    def __init__(self, cfg):
        super(MambaCLIPSelfSupervisedTrainer, self).__init__(cfg)

        # Reconstruction loss weight
        self.recon_weight = cfg.trainer.get('recon_weight', 1.0)

    def set_input(self, inputs):
        """Prepare input data"""
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def forward(self):
        """Forward pass: reconstruct images"""
        # Forward through encoder-decoder
        self.reconstructed = self.net(self.imgs)

    def optimize_parameters(self):
        """
        Self-supervised training with reconstruction loss

        Only train on normal samples (filter out anomalies if present in batch)
        """
        # Forward with AMP
        with self.amp_autocast():
            self.forward()

            # Filter to normal samples only (optional - depends on dataset)
            # For MVTec, training set should already be all normal
            is_normal = self.imgs_mask.reshape(self.bs, -1).max(dim=1)[0] == 0  # No anomalies

            if is_normal.sum() > 0:
                # Reconstruction loss (MSE) on normal samples
                imgs_normal = self.imgs[is_normal]
                recon_normal = self.reconstructed[is_normal]

                loss = self.loss_terms['mse'](recon_normal, imgs_normal) * self.recon_weight
            else:
                # If no normal samples in batch (shouldn't happen in training)
                # Use all samples for loss computation
                loss = self.loss_terms['mse'](self.reconstructed, self.imgs) * self.recon_weight

        # Backward
        self.backward_term(loss, self.optim)

        # Log
        update_log_term(
            self.log_terms.get('loss'),
            reduce_tensor(loss, self.world_size).clone().detach().item(),
            1, self.master
        )

    @torch.no_grad()
    def test(self):
        """Self-supervised anomaly detection via reconstruction error"""
        if self.master:
            if os.path.exists(self.tmp_dir):
                import shutil
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.reset(isTrain=False)

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

            test_data = next(test_loader)
            self.set_input(test_data)

            # Reconstruction-based prediction
            scores, preds = self.net.predict(self.imgs)

            # Binarize masks
            self.imgs_mask[self.imgs_mask > 0.5] = 1
            self.imgs_mask[self.imgs_mask <= 0.5] = 0

            # Visualization
            if self.cfg.vis:
                root_out = self.cfg.vis_dir if self.cfg.vis_dir else self.writer.logdir
                vis_rgb_gt_amp(
                    self.img_path,
                    self.imgs,
                    self.imgs_mask.cpu().numpy().astype(int),
                    preds,
                    self.cfg.model.name,
                    root_out,
                    self.cfg.data.root.split('/')[-1]
                )

            # Collect results
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(preds)
            anomaly_scores.append(scores)
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

        # Merge results from all GPUs
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
        else:
            results = dict(
                imgs_masks=imgs_masks,
                anomaly_maps=anomaly_maps,
                anomaly_scores=anomaly_scores,
                cls_names=cls_names,
                anomalys=anomalys
            )

        if self.master:
            # Concatenate all results
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}

            # Evaluate metrics per class
            import tabulate
            msg = {}

            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)

                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)

                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None

                # Record metrics
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)

                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1

                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

            # Display results
            msg = tabulate.tabulate(
                msg, headers='keys', tablefmt="pipe",
                floatfmt='.3f', numalign="center", stralign="center"
            )
            log_msg(self.logger, f'\n{msg}')
