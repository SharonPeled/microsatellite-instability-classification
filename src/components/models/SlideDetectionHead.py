import torch
import torch.nn as nn
from src.components.models.SubtypeClassifier import SubtypeClassifier
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
import torch.nn.functional as F
import pandas as pd
from src.training_utils import calc_safe_auc
import numpy as np


class SlideDetectionHead(SubtypeClassifier):
    def __init__(self, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 **other_kwargs):
        super(SlideDetectionHead, self).__init__(tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                 class_to_weight, num_iters_warmup_wo_backbone,
                                                 cohort_to_ind, cohort_weight, nn_output_size,
                                                 **other_kwargs)
        self.num_features *= 2
        self.head_out_size = 1
        if self.other_kwargs.get('n_nn_head', None) is not None:
            num_layers = self.other_kwargs['n_nn_head']['num_layers']
            dropout_value = self.other_kwargs['n_nn_head']['dropout_value']
            self.head = nn.Sequential(
                *(PretrainedClassifier.head_layer_block(self.num_features, dropout_value=dropout_value)
                  for _ in range(num_layers - 1)),
                nn.Linear(self.num_features, self.head_out_size)
                )
            Logger.log(f"{num_layers} layered head of size {self.num_features} with output size {self.head_out_size} created.", log_importance=1)
        else:
            self.head = nn.Linear(self.num_features, self.head_out_size)
            Logger.log(f"1 layered head of size {self.num_features} with output size {self.head_out_size} created.", log_importance=1)
        Logger.log(f"""SlideDetectionHead created with encoder name: {tile_encoder_name}.""", log_importance=1)

    def forward(self, x, c, paired_tile_0, c_0, paired_tile_1):
        x_embed = self.backbone(x, c)
        pair_0_embed = self.backbone(paired_tile_0, c_0)
        pair_1_embed = self.backbone(paired_tile_1, c)
        cat_pairs_0 = torch.cat((x_embed, pair_0_embed), dim=1)
        cat_pairs_1 = torch.cat((x_embed, pair_1_embed), dim=1)
        pairs = torch.cat((cat_pairs_0, cat_pairs_1), dim=0)
        y_0 = torch.zeros(cat_pairs_0.shape[0])
        y_1 = torch.ones(cat_pairs_1.shape[0])
        y = torch.cat((y_0, y_1)).to(pairs.device)
        return self.head(pairs).squeeze(), y

    def general_loop(self, batch, batch_idx):
        try:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
            self.logger.experiment.log_metric(self.logger.run_id, f"lr", lr)
        except:
            pass
        x, c, _, slide_id, patient_id, tile_path, paired_tile_0, c_0, paired_tile_1 = batch
        scores, y = self.forward(x, c, paired_tile_0, c_0, paired_tile_1)
        loss = self.loss(scores, y, c, tile_path+tile_path)
        return loss, {'loss': loss.detach().cpu(), 'c': torch.cat((c.detach().cpu(), c.detach().cpu())),
                      'scores': scores.detach().cpu(), 'y': y, 'slide_id': slide_id+slide_id,
                      'patient_id': patient_id+patient_id,
                      'tile_path': tile_path+tile_path}



