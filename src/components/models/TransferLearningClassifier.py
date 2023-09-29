import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ...general_utils import generate_confusion_matrix_figure
from src.components.objects.Logger import Logger
import numpy as np
from torch.optim.lr_scheduler import StepLR


class TransferLearningClassifier(pl.LightningModule):
    def __init__(self, class_to_ind=None, model=None, learning_rate=1e-4, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, **other_kwargs):
        super().__init__()
        if model is None and class_to_ind is None:
            raise "Invalid parameters."
        self.other_kwargs = other_kwargs
        self.class_to_ind = class_to_ind
        self.class_weights = self.init_weights(class_to_weight)
        self.learning_rate = learning_rate
        self.num_iters_warmup_wo_backbone = num_iters_warmup_wo_backbone  # for this to work the model has to be a sequential
        self.backbone_grad_status = None
        if model is None:
            backbone = resnet50(weights="IMAGENET1K_V2")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            layers.append(nn.Flatten())
            layers.append(nn.Linear(num_filters, len(self.class_to_ind)))
            self.model = nn.Sequential(*layers)
        elif model == 'ignore':
            self.model = None
        else:
            self.model = model
        self.test_outputs = None
        self.valid_outputs = None
        self.is_fit = False
        self.is_training = False
        self.metrics = {}
        self.outputs = []
        Logger.log(f"""TransferLearningClassifier created with loss weights: {self.class_weights}.""", log_importance=1)

    def init_weights(self, class_to_weight):
        if class_to_weight is None:
            return None
        sum_w = float(sum(class_to_weight.values()))
        weights = [0 for _ in range(len(self.class_to_ind))]
        for c_name, ind in self.class_to_ind.items():
            weights[ind] = class_to_weight[c_name] / sum_w
        return torch.Tensor([w / sum_w for w in class_to_weight.values()])

    def on_train_start(self):
        self.is_training = True
        if self.other_kwargs.get('config_filepath', None):
            self.logger.experiment.log_artifact(self.logger.run_id, self.other_kwargs['config_filepath'],
                                                artifact_path="configs")
            Logger.log(f"""TransferLearningClassifier: config file logged.""",
                       log_importance=1)

    def on_train_end(self):
        self.is_fit = True
        self.is_training = False

    def forward(self, x):
        return self.model(x)

    def loss(self, scores, y):
        if self.class_weights is None:
            return F.cross_entropy(scores, y)
        if len(scores.shape) == 1 or (scores.shape[-1] == 1 and len(self.class_weights) == 2):
            return F.binary_cross_entropy_with_logits(scores, y.to(scores.dtype),
                                                      pos_weight=self.class_weights[1].to(scores.device))
        return F.cross_entropy(scores, y, weight=self.class_weights.to(scores.device))

    def set_training_warmup(self):
        if self.num_iters_warmup_wo_backbone is not None:
            for param in self.model[:-1].parameters():
                param.requires_grad = False
            self.backbone_grad_status = False
            Logger.log(f"Backbone frozen for {self.num_iters_warmup_wo_backbone} steps.", log_importance=1)
        num_training_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        Logger.log(f"Number of training params: {num_training_params}.", log_importance=1)

    def configure_optimizers(self):
        self.set_training_warmup()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = self.create_lr_scheduler(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def create_lr_scheduler(self, optimizer):
        return StepLR(optimizer, step_size=1, gamma=0.1)

    def general_loop(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        x, y, slide_id, patient_id = batch
        scores = self.forward(x)
        loss = self.loss(scores, y)
        return loss, {'loss': loss.detach().cpu(), 'scores': scores.detach().cpu(),
                      'y': y.cpu(), 'slide_id': slide_id, 'patient_id': patient_id}

    def training_step(self, batch, batch_idx):
        if self.num_iters_warmup_wo_backbone is not None and not self.backbone_grad_status \
                and batch_idx > self.num_iters_warmup_wo_backbone:
            for param in self.model[:-1].parameters():
                param.requires_grad = True
            self.backbone_grad_status = True
            Logger.log(f"Backbone unfrozen, step {batch_idx}.", log_importance=1)
            num_training_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            Logger.log(f"Number of training params: {num_training_params}.", log_importance=1)
        loss, loop_dict = self.general_loop(batch, batch_idx)
        self.logger.experiment.log_metric(self.logger.run_id, "train_loss", loop_dict['loss'])
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, loop_dict = self.general_loop(batch, batch_idx)
        self.log("val_loss", loop_dict['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.logger.experiment.log_metric(self.logger.run_id, "val_loss", loop_dict['loss'])
        loop_dict['batch_idx'] = batch_idx
        return loop_dict

    def test_step(self, batch, batch_idx):
        loss, loop_dict = self.general_loop(batch, batch_idx)
        self.logger.experiment.log_metric(self.logger.run_id, "test_loss", loop_dict['loss'])
        loop_dict['batch_idx'] = batch_idx
        self.outputs.append(loop_dict)
        return loop_dict

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        logits = softmax(scores, dim=1).numpy()
        y_pred = torch.argmax(scores, dim=1).numpy()
        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        self.log_metrics(y_true, y_pred, logits, dataset_str=dataset_str)

    def on_validation_epoch_end(self, outputs):
        outputs_cpu = TransferLearningClassifier.outputs_to_cpu(outputs)
        self.log_epoch_level_metrics(outputs_cpu, dataset_str='valid')
        self.valid_outputs = outputs_cpu
        del outputs  # free from CUDA
        if not self.optimizers():
            return
        self.logger.experiment.log_param(self.logger.run_id, f"lr_epoch_{self.current_epoch}",
                                         self.optimizers().optimizer.defaults['lr'])

    def on_test_epoch_end(self):
        outputs_cpu = TransferLearningClassifier.outputs_to_cpu(self.outputs)
        self.log_epoch_level_metrics(outputs_cpu, dataset_str='test')
        self.test_outputs = outputs_cpu
        self.outputs = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def log_metrics(self, y_true, y_pred, logits, dataset_str):
        if self.class_to_ind is None or len(np.unique(y_true)) == 1:
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

    @staticmethod
    def outputs_to_cpu(outputs):
        outputs_cpu = []
        for out in outputs:
            d = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu()
                d[k] = v
            outputs_cpu.append(d)
        return outputs_cpu
