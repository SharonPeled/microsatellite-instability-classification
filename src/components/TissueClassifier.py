import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class TissueClassifier(pl.LightningModule):
    def __init__(self, class_to_ind, learning_rate):
        super().__init__()
        self.class_to_ind = class_to_ind
        self.learning_rate = learning_rate
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_filters, len(self.class_to_ind)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def loss(self, scores, targets):
        return F.cross_entropy(scores, targets)

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
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss, scores, y = self.general_loop(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        test_loss, scores, y = self.general_loop(batch, batch_idx)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"scores": scores, "y": y}

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        logits = softmax(scores, dim=1).cpu().numpy()
        y_pred = torch.argmax(scores, dim=1).cpu().numpy()
        y_true = torch.concat([out["y"] for out in outputs]).cpu().numpy()
        # precision, recall, f1
        metrics = classification_report(y_true, y_pred, output_dict=True,
                                        target_names=list(self.class_to_ind.keys()))
        for class_str, class_metrics in metrics.items():
            if class_str not in self.class_to_ind.keys():
                continue
            for metric_str, metric_val in class_metrics.items():
                self.log(f"{dataset_str}_{class_str}_{metric_str}", metric_val, on_step=False,
                         on_epoch=True, sync_dist=True)
        # auc
        if set(self.class_to_ind.values()) == set(y_true):
            # in order to use auc y_true has to include all labels
            # this condition may not be satisfied in the sanity check, where the sampling is not stratified
            auc_scores = roc_auc_score(y_true, logits, multi_class='ovr', average=None)
            for class_str, ind in self.class_to_ind.items():
                self.log(f"{dataset_str}_{class_str}_auc", auc_scores[ind], on_step=False,
                         on_epoch=True, sync_dist=True)
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt="d",
                    xticklabels=list(self.class_to_ind.keys()), yticklabels=list(self.class_to_ind.keys()))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        self.logger.experiment.log_figure(self.logger.run_id, fig, "confusion_matrix.png")

    def validation_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='valid')

    def test_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)
