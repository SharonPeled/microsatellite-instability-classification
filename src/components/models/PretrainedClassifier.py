import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.Logger import Logger
from src.training_utils import load_headless_tile_encoder
from src.general_utils import MultiInputSequential


class PretrainedClassifier(TransferLearningClassifier):
    def __init__(self, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, nn_output_size=None,
                 backbone=None, num_features=None, **other_kwargs):
        super(PretrainedClassifier, self).__init__(model='ignore', class_to_ind=class_to_ind, learning_rate=learning_rate,
                                                   class_to_weight=class_to_weight,
                                                   num_iters_warmup_wo_backbone=num_iters_warmup_wo_backbone,
                                                   **other_kwargs)
        if backbone is not None:
            self.backbone = backbone
            self.num_features = num_features
        else:
            self.backbone, self.num_features = load_headless_tile_encoder(tile_encoder_name, **other_kwargs)
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
        if self.nn_output_size is not None:
            self.head_out_size = self.nn_output_size
        if self.other_kwargs.get('n_nn_head', None) is not None:
            num_layers = self.other_kwargs['n_nn_head']['num_layers']
            dropout_value = self.other_kwargs['n_nn_head']['dropout_value']
            self.head = nn.Sequential(*(PretrainedClassifier.head_layer_block(self.num_features, dropout_value=dropout_value)
                 for _ in range(num_layers-1)),
                nn.Linear(self.num_features, self.head_out_size)
            )
            Logger.log(f"{num_layers} layered head output size {self.head_out_size} created.", log_importance=1)
        else:
            self.head = nn.Linear(self.num_features, self.head_out_size)
            Logger.log(f"1 layered head with output size {self.head_out_size} created.", log_importance=1)
        self.model = MultiInputSequential(self.backbone, self.head)
        Logger.log(f"""PretrainedClassifier created with encoder name: {tile_encoder_name}.""", log_importance=1)

    def configure_optimizers(self):
        self.set_training_warmup()

        if not isinstance(self.learning_rate, list):
            return super().configure_optimizers()

        grouped_parameters = [
            {"params": [p for p in self.model[:-1].parameters()], 'lr': self.learning_rate[0]},
            {"params": [p for p in self.model[-1].parameters()], 'lr': self.learning_rate[1]},
        ]

        optimizer = torch.optim.Adam(grouped_parameters)

        return {'optimizer': optimizer, 'lr_scheduler': self.create_lr_scheduler(optimizer)}

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

    @staticmethod
    def head_layer_block(num_features, dropout_value):
        return nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )


