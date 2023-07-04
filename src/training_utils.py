from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from torchvision.models import resnet50
from timm.models.vision_transformer import VisionTransformer
from torch import nn
import torch
from src.components.objects.Logger import Logger


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
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


def load_tile_encoder(Configs):
    if Configs.SC_TILE_ENCODER == 'pretrained_resent_tile_based':
        tile_encoder = TransferLearningClassifier.load_from_checkpoint(Configs.SC_TILE_BASED_TRAINED_MODEL,
                                                                       class_to_ind=Configs.SC_CLASS_TO_IND,
                                                                       learning_rate=Configs.SC_INIT_LR)
        layers = list(tile_encoder.children())
        if len(layers) == 1:
            layers = layers[0]
        num_filters = layers[-1].in_features
        tile_encoder = nn.Sequential(*layers[:-1])
        return tile_encoder, num_filters
    elif Configs.SC_TILE_ENCODER == 'pretrained_resent_imagenet':
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        tile_encoder = nn.Sequential(*layers)
        return tile_encoder, num_filters
    elif Configs.SC_TILE_ENCODER == 'SSL_VIT_PRETRAINED':
        model = SLL_vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        return model, model.num_features