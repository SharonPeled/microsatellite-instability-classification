import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.Logger import Logger
from src.training_utils import SLL_vit_small


class SSL_VIT(TransferLearningClassifier):
    def __init__(self, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None):
        super(SSL_VIT, self).__init__(model='ignore', class_to_ind=class_to_ind, learning_rate=learning_rate,
                                      class_to_weight=class_to_weight,
                                      num_iters_warmup_wo_backbone=num_iters_warmup_wo_backbone)
        self.vit_model = SLL_vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        self.frozen_backbone = frozen_backbone
        if self. frozen_backbone:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            self.num_iters_warmup_wo_backbone = None
            Logger.log(f"Backbone frozen.", log_importance=1)
        self.head = nn.Linear(self.vit_model.num_features, len(self.class_to_ind))
        self.model = nn.Sequential(self.vit_model, self.head)

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


