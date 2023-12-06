import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.general_utils import generate_confusion_matrix_figure
from src.components.objects.Logger import Logger
from src.components.models.PretrainedClassifier import PretrainedClassifier
import numpy as np
import pandas as pd


class VariantClassifier(PretrainedClassifier):
    # output_shape: #classes, #variants
    def __init__(self, output_shape, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                 class_to_weight=None, num_iters_warmup_wo_backbone=None):
        self.output_shape = list(output_shape)
        self.nn_output_size = self.output_shape[0] * self.output_shape[1]
        super(VariantClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                class_to_weight, num_iters_warmup_wo_backbone, self.nn_output_size)
        Logger.log(f"""VariantClassifier created with nn_output_size: {self.nn_output_size}.""",
                   log_importance=1)

    def general_loop(self, batch, batch_idx, test=False):
        x, c, y, slide_id, patient_id, tile_path = batch
        scores = self.forward(x)
        if test:
            loss = torch.tensor(-1)
        else:
            loss = self.loss(scores, y)
        return loss, {'loss': loss.detach().cpu(), 'c': c.detach().cpu(),
                      'scores': scores.detach().cpu(), 'y': y, 'slide_id': slide_id, 'patient_id': patient_id,
                      'tile_path': tile_path}

    def loss(self, scores, y):
        scores = scores.reshape(scores.shape[0], self.output_shape[0], self.output_shape[1])
        return super().loss(scores, y)

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"].reshape(-1, self.output_shape[0], self.output_shape[1])
                               for out in outputs]).cpu()
        logits = softmax(scores, dim=1).permute(2, 0, 1).cpu().numpy()
        y_true = torch.concat([out["y"] for out in outputs], dim=0).transpose(1, 0).cpu().numpy()
        auc_per_snp = []
        for i in range(y_true.shape[0]):
            auc_per_snp.append(self.calc_safe_auc(y_true[i, :], logits[i, :], multi_class='ovr', average=None))
        df_auc = pd.DataFrame(np.stack(auc_per_snp), columns=['0', '1', '2']).astype(float)
        df_auc['mean_auc'] = df_auc[['0', '1', '2']].mean(axis=1)
        for metric, metric_val in  df_auc.mean_auc.describe().to_dict().items():
            metric = ''.join([c for c in metric if c != '%'])
            self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_auc_{metric}",
                                              metric_val)

    def calc_safe_auc(self, y_true_i, logits_i, **kwargs):
        available_classes = np.unique(y_true_i)
        if len(available_classes) == logits_i.shape[-1]:
            return roc_auc_score(y_true_i, logits_i, **kwargs)
        if len(available_classes) == 1:
            return [None for _ in range(self.output_shape[0])]
        res = []
        for c in range(self.output_shape[0]):
            if c in available_classes:
                res.append(roc_auc_score(y_true_i, logits_i[:, c], **kwargs))
            else:
                res.append(None)
        return np.array(res)

