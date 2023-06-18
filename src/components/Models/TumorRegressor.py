import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.components.Objects.Logger import Logger
import matplotlib.pyplot as plt


class TumorRegressor(pl.LightningModule):
    def __init__(self, learning_rate, class_weight_dict, dropout_value):
        super().__init__()
        self.learning_rate = learning_rate
        self.class_weight_dict = class_weight_dict
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        # for layer in layers:
        #     layer.requires_grad_(False)
        layers.append(nn.Flatten())
        hidden_size = 1024
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(num_filters, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)
        Logger.log(f"""TumorRegressor created.""", log_importance=1)

    def forward(self, x):
        return self.model(x)

    def loss(self, scores, targets):
        if self.class_weight_dict is None:
            return F.mse_loss(scores, targets)
        scores = scores.flatten()
        targets = targets.flatten()
        weights = torch.Tensor(
            [self.class_weight_dict[round(float(val.item()), 1)] for val in targets]).to(targets.device)
        weights /= weights.sum()
        return torch.sum(weights * ((scores - targets) ** 2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def general_loop(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1, 1).float()
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
        y_pred = torch.concat([out["scores"] for out in outputs]).cpu().numpy()
        y_true = torch.concat([out["y"] for out in outputs]).cpu().numpy()
        TumorRegressor.log_metrics(y_true, y_pred, logger=self.logger, dataset_str=dataset_str,
                                   epoch=self.current_epoch)

    def validation_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='valid')

    def test_epoch_end(self, outputs):
        self.log_epoch_level_metrics(outputs, dataset_str='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    @staticmethod
    def log_metrics(y_true, y_pred, logger, dataset_str, epoch=0):
        logger.experiment.log_metric(logger.run_id, f"{dataset_str}_mse", mean_squared_error(y_true, y_pred))
        logger.experiment.log_metric(logger.run_id, f"{dataset_str}_mae", mean_absolute_error(y_true, y_pred))
        logger.experiment.log_metric(logger.run_id, f"{dataset_str}_r2", r2_score(y_true, y_pred))
        logger.experiment.log_metric(logger.run_id, f"{dataset_str}_mape", mean_absolute_percentage_error(y_true, y_pred))

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title('Regression model performance')
        logger.experiment.log_figure(logger.run_id, fig, f"regression_scatter_{epoch}.png")

        errors = y_true - y_pred
        fig = plt.figure()
        plt.hist(errors, bins=100)
        plt.xlabel('Errors')
        plt.ylabel('Frequency')
        plt.title('Error distribution')
        logger.experiment.log_figure(logger.run_id, fig, f"error_histogram_{epoch}.png")
