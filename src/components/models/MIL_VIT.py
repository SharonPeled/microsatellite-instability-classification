from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.Logger import Logger


class MIL_VIT(TransferLearningClassifier):
    def __init__(self, class_to_ind, learning_rate, tile_encoder, cohort_dict, dropout=(0, 0, 0),
                 vit_variant='vit_b_16', pretrained=True):
        super(MIL_VIT, self).__init__(model=None, class_to_ind=class_to_ind, learning_rate=learning_rate)
        self.model = None  # TODO: fix this
        self.class_to_ind = class_to_ind
        self.cohort_dict = cohort_dict
        self.tile_encoder, self.tile_encoder_out_features = MIL_VIT.init_tile_encoder(tile_encoder)
        self.vit_model = MIL_VIT.load_vit_model(vit_variant, pretrained)
        self.vit_model.encoder = MIL_Encoder(self.vit_model.encoder)
        adapter_in = self.tile_encoder_out_features
        if self.cohort_dict['place']['before_adapter']:
            adapter_in *= self.cohort_dict['num_cohorts']
        self.adapter = nn.Sequential(nn.Dropout(dropout[0]),
                                     nn.Linear(adapter_in, adapter_in),
                                     nn.ReLU(),
                                     nn.Dropout(dropout[0]),
                                     nn.Linear(adapter_in, self.tile_encoder_out_features//2),
                                     nn.ReLU(),
                                     nn.Dropout(dropout[1]),
                                     nn.Linear(self.tile_encoder_out_features//2, self.vit_model.hidden_dim))
        head_in = self.vit_model.hidden_dim
        if self.cohort_dict['place']['before_head']:
            head_in *= self.cohort_dict['num_cohorts']
        self.vit_model.heads = nn.Linear(in_features=head_in,
                                         out_features=len(self.class_to_ind), bias=True)
        self.training_phase = 0

    def next_training_phase(self):
        if self.training_phase == 0:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=False)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=False)
            MIL_VIT.grad_config(self.vit_model.heads, requires_grad=True)
        if self.training_phase == 1:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=False)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=True)
        if self.training_phase == 2:
            MIL_VIT.grad_config(self.tile_encoder, requires_grad=True)
            MIL_VIT.grad_config(self.adapter, requires_grad=True)
            MIL_VIT.grad_config(self.vit_model, requires_grad=True)
        self.training_phase += 1
        Logger.log(f"""MIL_VIT training phase {self.training_phase}.""", log_importance=1)

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
        layers = list(tile_encoder.children())
        if len(layers) == 1:
            layers = layers[0]
        num_filters = layers[-1].in_features
        model = nn.Sequential(*layers[:-1])
        return model, num_filters

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer}

    def general_loop(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]
        if len(batch) == 2:
            x, y = batch
            scores = self.forward(x)
        else:
            x, c, y = batch
            scores = self.forward(x, c)
        loss = self.loss(scores, y)
        return loss, scores, y

    def _process_input(self, x, cohort):
        n, p, c, h, w = x.shape

        # (n, p, c, h, w) -> (n*p, c, h, w)
        x = x.reshape((-1, c, h, w))

        # (n*p, c, h, w) -> (n*p, tile_encoder_dim)
        x = self.tile_encoder(x)

        # one hot to encoded tiles
        if self.cohort_dict['place']['before_adapter']:
            x = MIL_VIT.features_to_one_hot(x, cohort.repeat_interleave(p), self.cohort_dict['num_cohorts'])
            
        # (n*p, tile_encoder_dim) -> (n*p, mil_encoder_hidden_dim)
        x = self.adapter(x)

        # (n*p, mil_patch_size**2) -> (n, p, mil_encoder_hidden_dim)
        x = x.reshape((n, p, self.vit_model.hidden_dim))

        return x

    def forward(self, x, cohort=None):
        # Reshape and permute the input tensor
        x = self._process_input(x, cohort)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit_model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        # one hot to final features
        if self.cohort_dict['place']['before_head']:
            x = MIL_VIT.features_to_one_hot(x, cohort, self.cohort_dict['num_cohorts'])

        x = self.vit_model.heads(x)

        return x


class MIL_Encoder(nn.Module):
    def __init__(self, encoder):
        super(MIL_Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, input: torch.Tensor):
        # (n, p, hidden_dim)
        return self.encoder.ln(self.encoder.layers(self.encoder.dropout(input)))

