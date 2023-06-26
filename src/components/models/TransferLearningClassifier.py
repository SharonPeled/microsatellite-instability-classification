import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...utils import generate_confusion_matrix_figure
from src.components.objects.Logger import Logger
import numpy as np


class TransferLearningClassifier(pl.LightningModule):
    def __init__(self, class_to_ind=None, model=None, learning_rate=1e-4, class_to_weight=None):
        super().__init__()
        if model is None and class_to_ind is None:
            raise "Invalid parameters."
        self.class_to_ind = class_to_ind
        self.class_weights = self.init_class_weights(class_to_weight)
        self.learning_rate = learning_rate
        if model is None:
            backbone = resnet50(weights="IMAGENET1K_V2")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            layers.append(nn.Flatten())
            layers.append(nn.Linear(num_filters, len(self.class_to_ind)))
            self.model = nn.Sequential(*layers)
        else:
            self.model = model
        self.test_outputs = None
        self.valid_outputs = []
        Logger.log(f"""TransferLearningClassifier created with loss weights: {self.class_weights}.""", log_importance=1)

    def init_class_weights(self, class_to_weight):
        if class_to_weight is None:
            return None
        sum_w = float(sum(class_to_weight.values()))
        return torch.Tensor([w / sum_w for w in class_to_weight.values()])

    def forward(self, x):
        return self.model(x)

    def loss(self, scores, y):
        if self.class_weights is None:
            return F.cross_entropy(scores, y)
        return F.cross_entropy(scores, y, weight=self.class_weights.to(scores.device))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def general_loop(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        train_loss, scores, y = self.general_loop(batch, batch_idx)
        self.logger.experiment.log_metric(self.logger.run_id, "train_loss", train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss, scores, y = self.general_loop(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.logger.experiment.log_metric(self.logger.run_id, "val_loss", val_loss)
        return {"scores": scores, "y": y, "batch_idx": batch_idx}

    def test_step(self, batch, batch_idx):
        test_loss, scores, y = self.general_loop(batch, batch_idx)
        self.logger.experiment.log_metric(self.logger.run_id, "test_loss", test_loss)
        return {"scores": scores, "y": y, "batch_idx": batch_idx}

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        logits = softmax(scores, dim=1).numpy()
        y_pred = torch.argmax(scores, dim=1).numpy()
        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        self.log_metrics(y_true, y_pred, logits, dataset_str=dataset_str)

    def validation_epoch_end(self, outputs):
        outputs_cpu = [{"scores": outputs[i]['scores'].cpu(),
                        "y": outputs[i]['y'].cpu(),
                        "batch_idx": outputs[i]['batch_idx']}
                       for i in range(len(outputs))]
        self.log_epoch_level_metrics(outputs_cpu, dataset_str='valid')
        self.valid_outputs.append(outputs_cpu)
        del outputs  # free from CUDA
        self.logger.experiment.log_param(self.logger.run_id, f"lr_epoch_{self.current_epoch}",
                                         self.optimizers().optimizer.defaults['lr'])

    def test_epoch_end(self, outputs):
        outputs_cpu = [{"scores": outputs[i]['scores'].cpu(),
                        "y": outputs[i]['y'].cpu(),
                        "batch_idx": outputs[i]['batch_idx']}
                       for i in range(len(outputs))]
        self.log_epoch_level_metrics(outputs_cpu, dataset_str='test')
        self.test_outputs = outputs_cpu
        del outputs  # free from CUDA

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def log_metrics(self, y_true, y_pred, logits, dataset_str):
        if self.class_to_ind is None or len(np.unique(y_true))==1:
            return
        target_names = self.class_to_ind.keys()
        # precision, recall, f1 per class
        metrics = classification_report(y_true, y_pred, output_dict=True,
                                        target_names=target_names)
        for class_str, class_metrics in metrics.items():
            if class_str not in target_names:
                continue
            for metric_str, metric_val in class_metrics.items():
                self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_{class_str}_{metric_str}",
                                                  metric_val)
        if logits is not None:
            # auc
            if len(target_names) == len(np.unique(y_true)):
                # in order to use auc y_true has to include all labels
                # this condition may not be satisfied in the sanity check, where the sampling is not stratified
                if len(target_names) == 2:
                    # binary case
                    for ind, class_str in enumerate(target_names):
                        auc_score = roc_auc_score((y_true == ind).astype(int), logits[:, ind])
                        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_{class_str}_auc",
                                                          auc_score)
                else:
                    auc_scores = roc_auc_score(y_true, logits, multi_class='ovr', average=None)
                    for ind, class_str in enumerate(target_names):
                        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_{class_str}_auc",
                                                          auc_scores[ind])
        # confusion matrix
        fig = generate_confusion_matrix_figure(y_true, y_pred, target_names)
        self.logger.experiment.log_figure(self.logger.run_id, fig,
                                          f"confusion_matrix_{dataset_str}_{self.current_epoch}.png")
