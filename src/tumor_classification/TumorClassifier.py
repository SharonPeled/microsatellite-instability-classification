import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TumorClassifier(pl.LightningModule):
    def __init__(self, num_classes, tumor_class_ind, learning_rate):
        super().__init__()
        assert tumor_class_ind is not None
        self.tumor_class_ind = tumor_class_ind
        self.learning_rate = learning_rate
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_filters, num_classes))
        self.model = nn.Sequential(*layers)
        # self.save_hyperparameters()  # not working with MLFlow

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
        logits = softmax(scores, dim=1)
        tumor_prob = logits[:, self.tumor_class_ind].cpu().numpy()
        y_pred = (torch.argmax(scores, dim=1) == self.tumor_class_ind).int().cpu().numpy()
        y = (torch.concat([out["y"] for out in outputs]) == self.tumor_class_ind).int().cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        auc = roc_auc_score(y, tumor_prob)
        self.log(f"{dataset_str}_precision", precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{dataset_str}_recall", recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{dataset_str}_f1", f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{dataset_str}_auc", auc, on_step=False, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='valid')

    def test_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)
