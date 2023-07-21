import torch
import torch.nn as nn
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
from src.training_utils import load_headless_tile_encoder
import torch.nn.functional as F
from torch.nn.functional import softmax
import pandas as pd
from src.training_utils import calc_safe_auc
import numpy as np


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
            self.model = nn.Sequential(self.backbone, self.head)
        if self.other_kwargs.get('learnable_cohort_prior_init_val', None):
            self.learnable_priors = nn.Parameter(torch.full((len(self.cohort_to_ind),),
                                                            self.other_kwargs.get('learnable_cohort_prior_init_val'))).float()
        Logger.log(f"""TransferLearningClassifier created with cohort weights: {self.cohort_weight}.""", log_importance=1)

    def forward(self, x, c=None):
        if self.other_kwargs.get('one_hot_cohort_head', None):
            x = self.backbone(x)
            x = self.features_to_one_hot(x, c, num_cohorts=len(self.cohort_to_ind))
            x = self.head(x)
        else:
            x = self.model(x).squeeze()
        if self.other_kwargs.get('learnable_cohort_prior_init_val', None):
            c = torch.eye((len(self.cohort_to_ind)), dtype=x.dtype, device=x.device)[c]
            priors = c.matmul(self.learnable_priors)
            x *= priors
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
            if self.other_kwargs.get('one_hot_cohort_head', None) or \
                    self.other_kwargs.get('learnable_cohort_prior_init_val', None):
                scores = self.forward(x, c)
            else:
                scores = self.forward(x)
            loss = self.loss(scores, y, c)
            return {'loss': loss, 'c': c, 'scores': scores, 'y': y, 'slide_id': slide_id, 'patient_id': patient_id}

    def configure_optimizers(self):
        if not self.other_kwargs.get('learnable_cohort_prior_init_val', None):
            return super().configure_optimizers()
        optimizer_dict = super().configure_optimizers()
        if isinstance(self.learning_rate, list):
            lr = self.learning_rate[-1]
        else:
            lr = self.learning_rate
        new_group = {"params": self.learnable_priors, "lr": lr}
        optimizer_dict['optimizer'].add_param_group(new_group)
        return optimizer_dict

    def loss(self, scores, y, c=None):
        if self.cohort_weight is None or c is None:
            return super().loss(scores, y)
        loss_per_sample = F.cross_entropy(scores, y, reduction='none')
        sample_weight = []
        for sample_cohort_ind, sample_class_ind in zip(c,y):
            sample_weight.append(self.cohort_weight[(self.ind_to_cohort[sample_cohort_ind.item()],
                                                     self.ind_to_class[sample_class_ind.item()])])
        sum_w = float(sum(sample_weight))
        weight_tensor = torch.Tensor([w / sum_w for w in sample_weight]).to(y.device)
        return (loss_per_sample * weight_tensor).mean()

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


