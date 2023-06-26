from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier


class MIL_VIT(TransferLearningClassifier):
    def __init__(self, class_to_ind, learning_rate, tile_encoder, vit_variant='vit_b_16', pretrained=True):
        super(MIL_VIT, self).__init__(model=None, class_to_ind=class_to_ind, learning_rate=learning_rate)
        self.model = None  # TODO: fix this
        self.class_to_ind = class_to_ind
        self.tile_encoder, self.tile_encoder_out_features = MIL_VIT.init_tile_encoder(tile_encoder)
        self.vit_model = MIL_VIT.load_vit_model(vit_variant, pretrained)
        self.vit_model.encoder = MIL_Encoder(self.vit_model.encoder)
        self.vit_model.heads = nn.Linear(in_features=self.vit_model.hidden_dim,
                                         out_features=len(self.class_to_ind), bias=True)
        self.adapter = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.tile_encoder_out_features, self.tile_encoder_out_features),
                                     nn.ReLU(),
                                     nn.Linear(self.tile_encoder_out_features, self.vit_model.hidden_dim))
        self.training_phase = 0

    def next_training_phase(self):
        if self.training_phase == 0:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=False)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=False)
            MIL_VIT.grad_config(self.vit_model.heads, requires_grad=False)
        if self.training_phase == 1:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=False)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=True)
        if self.training_phase == 2:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=True)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=True)
        self.training_phase += 1

    @staticmethod
    def grad_config(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def load_vit_model(vit_variant, pretrained):
        if vit_variant == 'vit_b_16':
            if pretrained:
                return vit_b_16(weights=ViT_B_16_Weights)
        else:
            return vit_b_16()
        raise NotImplementedError()

    @staticmethod
    def init_tile_encoder(tile_encoder):
        layers = list(tile_encoder.children())[0]
        num_filters = layers[-1].in_features
        model = nn.Sequential(*layers[:-1])
        return model, num_filters

    def _process_input(self, x):
        n, p, c, h, w = x.shape

        # (n, p, c, h, w) -> (n*p, c, h, w)
        x = x.reshape((-1, c, h, w))

        # (n*p, c, h, w) -> (n*p, tile_encoder_dim)
        x = self.tile_encoder(x)

        # (n*p, tile_encoder_dim) -> (n*p, mil_encoder_hidden_dim)
        x = self.adapter(x)

        # (n*p, mil_patch_size**2) -> (n, p, mil_encoder_hidden_dim)
        x = x.reshape((n, p, self.vit_model.hidden_dim))

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit_model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.vit_model.heads(x)

        return x


class MIL_Encoder(nn.Module):
    def __init__(self, encoder):
        super(MIL_Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, input: torch.Tensor):
        # (n, p, hidden_dim)
        return self.encoder.ln(self.encoder.layers(self.encoder.dropout(input)))