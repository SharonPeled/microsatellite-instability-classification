import torch
import torch.nn as nn
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
import torch.nn.functional as F
from torch.nn.functional import softmax
import pandas as pd
from src.training_utils import calc_safe_auc
import numpy as np
from src.general_utils import MultiInputSequential


class SubtypeClassifier(PretrainedClassifier):
    def __init__(self, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 **other_kwargs):
        super(SubtypeClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                class_to_weight, num_iters_warmup_wo_backbone, nn_output_size,
                                                **other_kwargs)
        self.cohort_to_ind = cohort_to_ind
        self.ind_to_cohort = {value: key for key, value in cohort_to_ind.items()}
        self.ind_to_class = {value: key for key, value in class_to_ind.items()}
        self.cohort_weight = cohort_weight
        if self.other_kwargs.get('one_hot_cohort_head', None):
            self.head = nn.Linear(self.num_features * len(self.cohort_to_ind), self.head_out_size)
            self.model = MultiInputSequential(self.backbone, self.head)
        if self.other_kwargs.get('learnable_cohort_prior_type', None):
            self.learnable_priors = nn.Parameter(torch.full((len(self.cohort_to_ind),), 0.1)).float()  # random init val
        Logger.log(f"""SubtypeClassifier created with cohort weights: {self.cohort_weight}.""", log_importance=1)

    def init_cohort_weight(self, train_dataset):
        if not self.other_kwargs.get('sep_cohort_w_loss', None):
            return
        df_tiles = train_dataset.df_labels
        tiles_per_cohort_subtype = df_tiles.groupby(['y', 'cohort'], as_index=False).tile_path.count()
        tiles_per_cohort = tiles_per_cohort_subtype.groupby('cohort')['tile_path'].sum()
        tiles_per_cohort_subtype['prop'] = tiles_per_cohort_subtype['tile_path'] / tiles_per_cohort_subtype[
            'cohort'].map(tiles_per_cohort)
        tiles_per_cohort_subtype['weight'] = 1 - tiles_per_cohort_subtype['prop']
        self.cohort_weight = tiles_per_cohort_subtype.groupby('cohort').apply(lambda df_c:
                                                                              df_c.sort_values(by='y').weight.values).to_dict()
        Logger.log(f"""SubtypeClassifier update cohort weights: {self.cohort_weight}.""", log_importance=1)

    def forward(self, x, c=None):
        if self.other_kwargs.get('tile_encoder', None) == 'SSL_VIT_PRETRAINED_COHORT_AWARE':
            if self.other_kwargs.get('one_hot_cohort_head', None):
                x = self.backbone(x, c).squeeze()
                x = self.features_to_one_hot(x, c, num_cohorts=len(self.cohort_to_ind))
                x = self.head(x).squeeze()
            else:
                x = self.model(x, c).squeeze()
        else:
            if self.other_kwargs.get('one_hot_cohort_head', None):
                x = self.backbone(x).squeeze()
                x = self.features_to_one_hot(x, c, num_cohorts=len(self.cohort_to_ind))
                x = self.head(x)
            else:
                x = self.model(x).squeeze()
        if self.other_kwargs.get('learnable_cohort_prior_type', None):
            c_one_hot = torch.eye((len(self.cohort_to_ind)), dtype=x.dtype, device=x.device)[c]
            if self.other_kwargs.get('learnable_cohort_prior_type') == '+':
                Logger.log(f'LP: {self.learnable_priors}', log_importance=1)
                priors = c_one_hot.matmul(self.learnable_priors)
                x += priors
            elif self.other_kwargs.get('learnable_cohort_prior_type') == '*':
                cohort_priors = softmax(self.learnable_priors)
                priors = c_one_hot.matmul(cohort_priors)
                x *= priors
            else:
                raise NotImplementedError("learnable cohort prior type not implemented.")
        return x

    def general_loop(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        if len(batch) == 4:
            x, y, slide_id, patient_id = batch
            scores = self.forward(x)
            loss = self.loss(scores, y)
            return {'loss': loss, 'scores': scores, 'y': y, 'slide_id': slide_id, 'patient_id': patient_id}
        if len(batch) == 5:
            x, c, y, slide_id, patient_id = batch
            scores = self.forward(x, c)
            loss = self.loss(scores, y, c)
            return {'loss': loss, 'c': c, 'scores': scores, 'y': y, 'slide_id': slide_id, 'patient_id': patient_id}

    def configure_optimizers(self):
        if not self.other_kwargs.get('learnable_cohort_prior_type', None):
            return super().configure_optimizers()
        optimizer_dict = super().configure_optimizers()
        if isinstance(self.learning_rate, list):
            lr = self.learning_rate[-1]
        else:
            lr = self.learning_rate
        new_group = {"params": self.learnable_priors, "lr": lr}
        optimizer_dict['optimizer'].add_param_group(new_group)
        optimizer_dict['lr_scheduler'] = self.create_lr_scheduler(optimizer_dict['optimizer'])
        return optimizer_dict

    def loss(self, scores, y, c=None):
        if self.cohort_weight is None or c is None:
            return super().loss(scores, y)
        y = y.to(scores.dtype)
        loss_list = []
        for c_name, c_ind in self.cohort_to_ind.items():
            scores_c = scores[c == c_ind]
            y_c = y[c == c_ind]
            if len(self.cohort_weight[c_name]) == 2:
                pos_weight = self.cohort_weight[c_name][1] /\
                             self.cohort_weight[c_name][0]
            else:
                pos_weight = 1
            loss_c = F.binary_cross_entropy_with_logits(scores_c, y_c, reduction='mean',
                                                        pos_weight=torch.tensor(pos_weight))
            loss_list.append(loss_c * y_c.shape[0])
        return sum(loss_list) / y.shape[0]

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        if len(scores.shape) == 1:
            cin_scores = scores
        else:
            cin_scores = scores[:, 1]
        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        cohort = torch.concat([out["c"] for out in outputs]).numpy()
        slide_id = np.concatenate([out["slide_id"] for out in outputs])
        patient_id = np.concatenate([out["patient_id"] for out in outputs])
        df = pd.DataFrame({
            "y_true": y_true,
            "cohort": cohort,
            "slide_id": slide_id,
            "patient_id": patient_id,
            "CIN_score": cin_scores
        })

        tile_cin_auc = calc_safe_auc(df.y_true, df.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_tile_CIN_AUC",
                                          tile_cin_auc)

        df_slide = df.groupby(['patient_id', 'cohort'], as_index=False).agg({
            'y_true': 'max',
            'CIN_score': 'mean'
        })
        slide_cin_auc = calc_safe_auc(df_slide.y_true, df_slide.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_CIN_AUC",
                                          slide_cin_auc)

        df_slide_cohort = df_slide.groupby('cohort').apply(lambda df_group:
                                                           calc_safe_auc(df_group.y_true,
                                                                         df_group.CIN_score))
        for cohort, auc in df_slide_cohort.iteritems():
            self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_{cohort}_CIN_AUC",
                                              auc)


