import torch
import torch.nn as nn
from src.components.models.CombinedLossSubtypeClassifier import CombinedLossSubtypeClassifier
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from src.training_utils import lr_scheduler_linspace_steps
from copy import deepcopy
from torch.nn.functional import softmax
import torch.nn.functional as F


class CT_MIL(CombinedLossSubtypeClassifier):
    def __init__(self, ct_mil_args, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 **other_kwargs):
        super(CT_MIL, self).__init__(ct_mil_args['combined_loss_args'],
                                     tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                            class_to_weight, num_iters_warmup_wo_backbone,
                                                            cohort_to_ind, cohort_weight, nn_output_size,
                                                            **other_kwargs)
        self.ct_mil_args = ct_mil_args
        self.tier1_attention = Attention_Gated(L=self.num_features, D=self.ct_mil_args['attn_dim'])
        self.tier1_head = deepcopy(self.head)
        self.tier2_attention = Attention_Gated(L=self.num_features, D=self.ct_mil_args['attn_dim'])
        self.tier2_head = deepcopy(self.head)
        Logger.log(f"""CT_MIL created with: {self.ct_mil_args}.""", log_importance=1)

    def init_tile_weight(self, a):
        pass

    def on_train_start(self):
        super(CT_MIL, self).on_train_start()

    def configure_optimizers(self):
        self.set_training_warmup()
        param_groups = []
        if self.ct_mil_args['vit_adapter_trainable_blocks'] is not None \
                and self.ct_mil_args['vit_adapter_trainable_blocks'] > 0:
            param_groups.append({"params": [param for name, param in self.backbone.named_parameters()
                                            if name.startswith('norm') or name.startswith(tuple([f'blocks.{11 - i}'
                                                                                                 for i in range(self.ct_mil_args['vit_adapter_trainable_blocks'])]))],
                                 'lr': self.learning_rate[0]})
        param_groups.append({"params": [p for p in self.tier1_attention.parameters()], 'lr': self.learning_rate[1]})
        param_groups.append({"params": [p for p in self.tier1_head.parameters()], 'lr': self.learning_rate[1]})
        optimizer1 = torch.optim.Adam(param_groups)

        optimizer2 = torch.optim.Adam([
            {"params": [p for p in self.tier2_attention.parameters()], 'lr': self.learning_rate[1]},
            {"params": [p for p in self.tier2_head.parameters()], 'lr': self.learning_rate[1]}
        ])
        self.optimizers_list.append(optimizer1)
        self.optimizers_list.append(optimizer2)

    # def get_tile_embeddings(self, x, c):
    #     x_embed_list = []
    #     batch_size = self.ct_mil_args['inner_batch_size']
    #     for i in range(0, x.shape[0], batch_size):
    #         x_batch = x[i:i + batch_size, :]
    #         c_batch = torch.full((x_batch.shape[0],), c.item()).to(x_batch.device)
    #         x_embed_list.append(self.backbone(x_batch, c_batch))
    #     return torch.cat(x_embed_list)

    def general_loop(self, batch, batch_idx, test=False):
        x_embed, c, y, s, p = batch
        x_embed = x_embed.squeeze()
        if test:
            scores = self.forward(x_embed, c)
            loss = torch.tensor(-1)
            return loss, {'loss': loss.detach().cpu(), 'c': c.detach().cpu(),
                          'scores': scores.detach().cpu(), 'y': y, 'slide_id': s, 'patient_id': p,
                          'tile_path': ''}

        opt1, opt2, opt_s = self.optimizers_list

        bags_embed, tier1_scores = self.forward_tier1(x_embed)
        tier1_y = torch.full((tier1_scores.shape[0],), y.item()).to(tier1_scores.device)
        tier1_loss = F.binary_cross_entropy_with_logits(tier1_scores, tier1_y.float(), reduction='mean')
        opt1.zero_grad()
        self.manual_backward(tier1_loss, retain_graph=True)
        opt1.step()

        slide_embed, tier2_score = self.forward_tier2(bags_embed.clone().detach())
        tier2_y = torch.full((tier2_score.shape[0],), y.item()).to(tier2_score.device)
        tier2_loss = F.binary_cross_entropy_with_logits(tier2_score, tier2_y.float(), reduction='mean')
        opt2.zero_grad()
        self.manual_backward(tier2_loss)
        opt2.step()

        self.logger.experiment.log_metric(self.logger.run_id, "tier1_loss", tier1_loss.detach().cpu())
        self.logger.experiment.log_metric(self.logger.run_id, "tier2_loss", tier2_loss.detach().cpu())
        return tier2_loss, {'loss': tier2_loss.detach().cpu(), 'c': c.detach().cpu(),
                            'scores': tier2_score.detach().cpu(), 'y': y, 'slide_id': s, 'patient_id': p,
                            'tile_path': ''}

    def forward(self, x_embed, c):
        bags_embed, tier1_scores = self.forward_tier1(x_embed)
        slide_embed, tier2_score = self.forward_tier2(bags_embed)
        return tier2_score

    def forward_tier2(self, bags_embed):
        attn_w2 = self.tier2_attention(bags_embed)
        slide_embed = (bags_embed * attn_w2).sum(dim=0)
        tier2_score = self.tier2_head(slide_embed)
        return slide_embed, tier2_score

    def forward_tier1(self, x_embed):
        bag_size = x_embed.shape[0] // self.ct_mil_args['num_bags']
        tier1_scores = []
        tier2_embeds = []
        for i in range(0, x_embed.shape[0], bag_size):
            x_embed_b = x_embed[i:i + bag_size, :]
            attn_w1 = self.tier1_attention(x_embed_b)
            bag_embed = (x_embed_b * attn_w1).sum(dim=0)
            tier1_scores.append(self.tier1_head(bag_embed))
            tier2_embeds.append(bag_embed)
        tier1_scores = torch.cat(tier1_scores)
        bags_embed = torch.stack(tier2_embeds)
        return bags_embed, tier1_scores

    # def loss(self, scores, aux_c_scores, aux_s_scores, y, c, s, tile_path):
    #     task_loss = super(CombinedLossSubtypeClassifier, self).loss(scores, y, c=None, tile_path=tile_path)
    #     aux_c_loss = super(CombinedLossSubtypeClassifier, self).loss(aux_c_scores, y=c, c=None, tile_path=tile_path)
    #     aux_s_loss = super(CombinedLossSubtypeClassifier, self).loss(aux_s_scores, y=s, c=None, tile_path=tile_path)
    #     return task_loss, aux_c_loss, aux_s_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.global_iter += 1


# copy paste from https://github.com/hrzhang1123/DTFD-MIL
class Attention_Gated(nn.Module):
    def __init__(self, L, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = softmax(A, dim=0)  # softmax over N
        return A

