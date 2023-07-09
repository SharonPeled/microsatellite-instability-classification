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
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None):
        super(SubtypeClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                class_to_weight, num_iters_warmup_wo_backbone)
        self.cohort_to_ind = cohort_to_ind
        self.ind_to_cohort = {value: key for key, value in cohort_to_ind.items()}
        self.ind_to_class = {value: key for key, value in class_to_ind.items()}
        self.cohort_weight = cohort_weight
        Logger.log(f"""TransferLearningClassifier created with cohort weights: {self.cohort_weight}.""", log_importance=1)

    def general_loop(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        if len(batch) == 3:
            x, y, slide_id = batch
            scores = self.forward(x)
            loss = self.loss(scores, y)
            return {'loss': loss, 'scores': scores, 'y': y, 'slide_id': slide_id}
        else:
            x, c, y, slide_id = batch
            scores = self.forward(x)
            loss = self.loss(scores, y, c)
            return {'loss': loss, 'c': c, 'scores': scores, 'y': y, 'slide_id': slide_id}

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
        logits = softmax(scores, dim=1)
        y_pred = torch.argmax(logits, dim=1).numpy()
        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        cohort = torch.concat([out["c"] for out in outputs]).numpy()
        slide_id = np.concatenate([out["slide_id"] for out in outputs])
        df = pd.DataFrame({
            "y_true": y_true,
            "cohort": cohort,
            "slide_id": slide_id,
            "CIN_score": logits[:, 1]
        })

        tile_cin_auc = calc_safe_auc(df.y_true, df.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_tile_CIN_AUC",
                                          tile_cin_auc)

        df_slide = df.groupby('slide_id').agg({
            'y_true': 'max',
            'cohort': 'max',
            'CIN_score': 'mean'
        })
        slide_cin_auc = calc_safe_auc(df_slide.y_true, df_slide.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_slide_CIN_AUC",
                                          slide_cin_auc)

        df_slide_cohort = df_slide.groupby('cohort').apply(lambda df_group:
                                                           calc_safe_auc(df_group.y_true,
                                                                         df_group.CIN_score))
        for cohort, auc in df_slide_cohort.iteritems():
            self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_slide_{cohort}_CIN_AUC",
                                              slide_cin_auc)

        super().log_metrics(y_true, y_pred, logits, dataset_str=dataset_str)



