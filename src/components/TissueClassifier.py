import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from ..utils import generate_confusion_matrix_figure
from ..components.Logger import Logger


class TissueClassifier(pl.LightningModule):
    def __init__(self, class_to_ind, learning_rate, class_to_weight=None):
        super().__init__()
        self.class_to_ind = class_to_ind
        self.class_weights = self.init_class_weights(class_to_weight)
        self.learning_rate = learning_rate
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        # for layer in layers:
        #     layer.requires_grad_(False)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_filters, len(self.class_to_ind)))
        self.model = nn.Sequential(*layers)
        Logger.log(f"""TissueClassifier created with loss weights: {self.class_weights}.""", log_importance=1)

    def init_class_weights(self, class_to_ind):
        if class_to_ind is None:
            return None
        sum_w = float(sum(class_to_ind.values()))
        return torch.Tensor([w/sum_w for w in class_to_ind.values()])

    def forward(self, x):
        return self.model(x)

    def loss(self, scores, targets):
        if self.class_weights is None:
            return F.cross_entropy(scores, targets)
        return F.cross_entropy(scores, targets, weight=self.class_weights.to(scores.device))

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
        return {"scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        test_loss, scores, y = self.general_loop(batch, batch_idx)
        self.logger.experiment.log_metric(self.logger.run_id, "test_loss", test_loss)
        return {"scores": scores, "y": y}

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        logits = softmax(scores, dim=1).cpu().numpy()
        y_pred = torch.argmax(scores, dim=1).cpu().numpy()
        y_true = torch.concat([out["y"] for out in outputs]).cpu().numpy()
        TissueClassifier.log_metrics(y_true, y_pred, logits, target_names=self.class_to_ind.keys(),
                                     logger=self.logger, dataset_str=dataset_str, epoch=self.current_epoch)

    def validation_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='valid')
        # self.logger.experiment.log_param(self.logger.run_id, f"lr_epoch_{self.current_epoch}",
        #                                  self.optimizers().optimizer.get_lr())

    def test_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    @staticmethod
    def log_metrics(y_true, y_pred, logits, target_names, logger, dataset_str, epoch=0):
        # precision, recall, f1 per class
        metrics = classification_report(y_true, y_pred, output_dict=True,
                                        target_names=target_names)
        for class_str, class_metrics in metrics.items():
            if class_str not in target_names:
                continue
            for metric_str, metric_val in class_metrics.items():
                logger.experiment.log_metric(logger.run_id, f"{dataset_str}_{class_str}_{metric_str}",
                                                  metric_val)
        if logits is not None:
            # auc
            if set(target_names) == set(y_true):
                # in order to use auc y_true has to include all labels
                # this condition may not be satisfied in the sanity check, where the sampling is not stratified
                auc_scores = roc_auc_score(y_true, logits, multi_class='ovr', average=None)
                for ind, class_str in enumerate(target_names):
                    logger.experiment.log_metric(logger.run_id, f"{dataset_str}_{class_str}_auc", auc_scores[ind])
        # confusion matrix
        fig = generate_confusion_matrix_figure(y_true, y_pred, target_names)
        logger.experiment.log_figure(logger.run_id, fig, f"confusion_matrix_{dataset_str}_{epoch}.png")
