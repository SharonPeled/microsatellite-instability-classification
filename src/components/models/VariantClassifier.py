import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import generate_confusion_matrix_figure
from src.components.objects.Logger import Logger
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
import numpy as np
import pandas as pd


class VariantClassifier(TransferLearningClassifier):
    # output_shape: #classes, #variants
    def __init__(self, output_shape, learning_rate, class_to_ind):
        self.output_shape = list(output_shape)
        self.output_size = self.output_shape[0] * self.output_shape[1]
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        # for layer in layers:
        #     layer.requires_grad_(False)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_filters, self.output_size))
        model = nn.Sequential(*layers)
        super().__init__(model=model, class_to_ind=class_to_ind, learning_rate=learning_rate)

    def loss(self, scores, y):
        scores = scores.reshape(scores.shape[0], self.output_shape[0], self.output_shape[1])
        return F.cross_entropy(scores, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer}

    # def test_epoch_end(self, outputs):
    #     super(VariantClassifier, self).test_epoch_end(outputs)
    #     self.reshape_outputs(self.test_outputs, out_shape=[-1] + self.output_shape,
    #                          num_snps=self.output_shape[1])

    # def validation_epoch_end(self, outputs):
    #     super(VariantClassifier, self).validation_epoch_end(outputs)
    #     self.reshape_outputs(self.valid_outputs[-1], out_shape=[-1] + self.output_shape,
    #                          num_snps=self.output_shape[1])

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


    # @staticmethod
    # def reshape_outputs(outputs, out_shape, num_snps):
    #     for i in range(len(outputs)):
    #         outputs[i]['scores'] = outputs[i]['scores'].reshape(out_shape)
    #         outputs[i]['batch_idx'] = outputs[i]['scores'].repeat_interleave(num_snps)

