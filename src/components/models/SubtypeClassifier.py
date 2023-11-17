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
from src.components.objects.SCELoss import BSCELoss


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
        df_tiles = train_dataset.df_labels
        w_per_y_per_cohort = 1 / (df_tiles.y.nunique() * df_tiles.cohort.nunique())
        tile_counts = df_tiles.groupby(['y', 'cohort', 'slide_uuid'], as_index=False).tile_path.count()
        slide_counts = tile_counts.groupby(['y', 'cohort'], as_index=False).slide_uuid.nunique().rename(
            columns={'slide_uuid': 'num_slides_per_y_per_cohort'})
        tile_counts = tile_counts.merge(slide_counts, on=['y', 'cohort'], how='inner')
        tile_counts['slide_w'] = w_per_y_per_cohort / tile_counts.num_slides_per_y_per_cohort
        tile_counts['tile_w'] = tile_counts.slide_w / tile_counts.tile_path
        self.tile_weight = df_tiles.merge(tile_counts, on='slide_uuid', how='inner', suffixes=('', '__y'))
        # self.tile_weight.set_index('tile_path', inplace=True)
        self.tile_weight.set_index(self.tile_weight.tile_path.values, inplace=True)
        Logger.log(f"""SubtypeClassifier update tile weights {len(self.tile_weight)}.""", log_importance=1)

    def forward(self, x, c=None):
        if self.other_kwargs.get('tile_encoder', None) == 'SSL_VIT_PRETRAINED_COHORT_AWARE' or \
                self.other_kwargs.get('tile_encoder', None) == 'VIT_PRETRAINED_DINO':
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
                Logger.log(f'LP: {self.learnable_priors.cpu().detach().numpy().round(2)}', log_importance=1)
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
        try:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
            self.logger.experiment.log_metric(self.logger.run_id, f"lr", lr)
        except:
            pass
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        if len(batch) == 4:
            x, y, slide_id, patient_id = batch
            scores = self.forward(x)
            loss = self.loss(scores, y)
            return loss, {'loss': loss.detach().cpu(), 'scores': scores.detach().cpu(), 'y': y.cpu(),
                          'slide_id': slide_id, 'patient_id': patient_id}
        if len(batch) >= 5:
            x, c, y, slide_id, patient_id, tile_path = batch
            scores = self.forward(x, c)
            loss = self.loss(scores, y, c, tile_path)
            return loss, {'loss': loss.detach().cpu(), 'c': c.detach().cpu(),
                          'scores': scores.detach().cpu(), 'y': y, 'slide_id': slide_id, 'patient_id': patient_id,
                          'tile_path': tile_path}

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

    def loss(self, scores, y, c=None, tile_path=None):
        y = y.to(scores.dtype)
        if scores.dim() == 0:
            scores = scores.unsqueeze(dim=0)
        if self.tile_weight.index.isin(tile_path).sum() != len(tile_path):
            return torch.tensor(-1)
        tile_w = torch.Tensor(self.tile_weight.loc(axis=0)[tile_path].tile_w.values).to(scores.device).to(scores.dtype)
        tile_w = tile_w / tile_w.sum()
        loss_unreduced = F.binary_cross_entropy_with_logits(scores, y, reduction='none')
        return torch.dot(loss_unreduced, tile_w)

    def _get_df_for_metric_logging(self, outputs):
        scores = torch.concat([out["scores"] for out in outputs])
        scores = torch.sigmoid(scores)
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
        return df

    def log_epoch_level_metrics(self, outputs, dataset_str):
        df = self._get_df_for_metric_logging(outputs)
        tile_cin_auc = calc_safe_auc(df.y_true, df.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_tile_CIN_AUC",
                                          tile_cin_auc)
        self.metrics[f"{dataset_str}_tile_CIN_AUC"] = tile_cin_auc

        df_slide = df.groupby(['patient_id', 'cohort'], as_index=False).agg({
            'y_true': 'max',
            'CIN_score': 'mean'
        })
        slide_cin_auc = calc_safe_auc(df_slide.y_true, df_slide.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_CIN_AUC",
                                          slide_cin_auc)
        self.metrics[f"{dataset_str}_patient_CIN_AUC"] = slide_cin_auc


        df_slide_cohort = df_slide.groupby('cohort').apply(lambda df_group:
                                                           calc_safe_auc(df_group.y_true,
                                                                         df_group.CIN_score))
        for cohort, auc in df_slide_cohort.iteritems():
            self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_{cohort}_CIN_AUC",
                                              auc)
            self.metrics[f"{dataset_str}_C{cohort}_AUC"] = auc



