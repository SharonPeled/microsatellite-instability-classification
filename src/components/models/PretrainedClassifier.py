import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.Logger import Logger
from src.training_utils import load_headless_tile_encoder
import torch.nn.functional as F


class PretrainedClassifier(TransferLearningClassifier):
    def __init__(self, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None):
        super(PretrainedClassifier, self).__init__(model='ignore', class_to_ind=class_to_ind, learning_rate=learning_rate,
                                                   class_to_weight=class_to_weight,
                                                   num_iters_warmup_wo_backbone=num_iters_warmup_wo_backbone)
        self.model, num_features = load_headless_tile_encoder(tile_encoder_name)
        self.cohort_to_ind = cohort_to_ind
        self.ind_to_cohort = {value: key for key, value in cohort_to_ind.items()}
        self.ind_to_class = {value: key for key, value in class_to_ind.items()}
        self.cohort_weight = cohort_weight
        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.num_iters_warmup_wo_backbone = None
            if isinstance(self.learning_rate, list):
                self.learning_rate = self.learning_rate[-1]
            Logger.log(f"Backbone frozen.", log_importance=1)
        self.head = nn.Linear(num_features, len(self.class_to_ind))
        self.model = nn.Sequential(self.model, self.head)
        Logger.log(f"""TransferLearningClassifier created with cohort weights: {self.cohort_weight}.""", log_importance=1)
        Logger.log(f"""TransferLearningClassifier created with encoder name: {tile_encoder_name}.""", log_importance=1)

    def configure_optimizers(self):
        self.set_training_warmup()
        if not isinstance(self.learning_rate, list):
            return super().configure_optimizers()

        grouped_parameters = [
            {"params": [p for p in self.model[:-1].parameters()], 'lr': self.learning_rate[0]},
            {"params": [p for p in self.model[-1].parameters()], 'lr': self.learning_rate[1]},
        ]

        optimizer = torch.optim.Adam(grouped_parameters)

        return {'optimizer': optimizer}

    def general_loop(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        if len(batch) == 2:
            x, y = batch
            scores = self.forward(x)
            loss = self.loss(scores, y)
        else:
            x, c, y = batch
            scores = self.forward(x)
            loss = self.loss(scores, y, c)
        return loss, scores, y

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


