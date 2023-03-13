from torch import nn
import pytorch_lightning as pl
from torchvision.models import resnet50


class ResnetEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V2")
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)
