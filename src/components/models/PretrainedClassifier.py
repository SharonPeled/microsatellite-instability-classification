import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.Logger import Logger
from src.training_utils import load_headless_tile_encoder


class PretrainedClassifier(TransferLearningClassifier):
    def __init__(self, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, nn_output_size=None, **other_kwargs):
        super(PretrainedClassifier, self).__init__(model='ignore', class_to_ind=class_to_ind, learning_rate=learning_rate,
                                                   class_to_weight=class_to_weight,
                                                   num_iters_warmup_wo_backbone=num_iters_warmup_wo_backbone,
                                                   **other_kwargs)
        self.backbone, self.num_features = load_headless_tile_encoder(tile_encoder_name)
        if len(self.class_to_ind) == 2:
            self.head_out_size = 1
        else:
            self.head_out_size = len(self.class_to_ind)
        self.nn_output_size = nn_output_size
        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.num_iters_warmup_wo_backbone = None
            if isinstance(self.learning_rate, list):
                self.learning_rate = self.learning_rate[-1]
            Logger.log(f"Backbone frozen.", log_importance=1)
        if self.nn_output_size is None:
            self.head = nn.Linear(self.num_features, self.head_out_size)
        else:
            self.head = nn.Linear(self.num_features, self.nn_output_size)
        self.model = nn.Sequential(self.backbone, self.head)
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

    @staticmethod
    def features_to_one_hot(x, c, num_cohorts):
        # x shape: (n, dim)
        # c shape: (n, 1)
        batch_size = x.shape[0]
        hidden_dim = x.shape[1]
        base_indices = torch.arange(hidden_dim).repeat(batch_size, 1).to(x.device)
        shift_indices = (c * hidden_dim).unsqueeze(dim=1).repeat(1, hidden_dim)
        cohort_indices = base_indices + shift_indices
        out = torch.zeros((batch_size, hidden_dim * num_cohorts), dtype=x.dtype).to(x.device)
        return out.scatter_(1, cohort_indices, x)


