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
        logits = softmax(scores, dim=1).permute(0, 2, 1).reshape(-1, self.output_shape[0]).cpu().numpy()
        y_pred = torch.argmax(scores, dim=1).flatten().cpu().numpy()
        y_true = torch.concat([out["y"] for out in outputs]).flatten().cpu().numpy()
        self.log_metrics(y_true, y_pred, logits, dataset_str=dataset_str)

    # @staticmethod
    # def reshape_outputs(outputs, out_shape, num_snps):
    #     for i in range(len(outputs)):
    #         outputs[i]['scores'] = outputs[i]['scores'].reshape(out_shape)
    #         outputs[i]['batch_idx'] = outputs[i]['scores'].repeat_interleave(num_snps)

