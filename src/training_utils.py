import numpy as np
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from torchvision.models import resnet50
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import Bottleneck, ResNet
from torch import nn
import torch
from src.components.objects.Logger import Logger
from sklearn.metrics import roc_auc_score, classification_report


def calc_safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        print(e)
        return np.nan


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch"
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def SLL_vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        Logger.log(verbose, log_importance=1)
    return model


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = self.fc.in_features
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def SSL_resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


def load_headless_tile_encoder(tile_encoder_name, path=None):
    if tile_encoder_name == 'pretrained_resent_tile_based':
        tile_encoder = TransferLearningClassifier.load_from_checkpoint(path,
                                                                       class_to_ind=None,
                                                                       learning_rate=None).model
        layers = list(tile_encoder.children())
        if len(layers) == 1:
            layers = layers[0]
        num_filters = layers[-1].in_features
        tile_encoder = nn.Sequential(*layers[:-1])
        return tile_encoder, num_filters
    elif tile_encoder_name == 'pretrained_resent_imagenet':
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        tile_encoder = nn.Sequential(*layers)
        return tile_encoder, num_filters
    elif tile_encoder_name == 'SSL_VIT_PRETRAINED':
        model = SLL_vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        return model, model.num_features
    elif tile_encoder_name == 'SSL_RESNET_PRETRAINED':
        model = SSL_resnet50(pretrained=True, progress=False, key="MoCoV2")
        return model, model.num_features


