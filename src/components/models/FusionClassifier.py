import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, Mlp, DropPath
from timm.models.vision_transformer import LayerScale, VisionTransformer
from copy import deepcopy
from src.general_utils import MultiInputSequential


class CohortAwareVisionTransformer(VisionTransformer):
    
    def __init__(self, cohort_aware_dict,
                 **vit_kwargs):
        vit_kwargs['block_fn'] = cohort_aware_block_fn(cohort_aware_dict)
        super(CohortAwareVisionTransformer, self).__init__(**vit_kwargs)
        self.blocks = MultiInputSequential(*[block for block in self.blocks])  # to allow to multi input through blocks sequential

    def forward_features(self, x, c):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x, c = self.blocks(x, c)
        x = self.norm(x)
        return x

    def forward(self, x, c):
        x = self.forward_features(x, c)
        x = self.forward_head(x)
        return x

    def load_pretrained_model(self, state_dict_to_load, strict=True, strategy=None):
        state_dict_to_load = deepcopy(state_dict_to_load)
        curr_state_dict = self.state_dict()
        if state_dict_to_load.keys() == curr_state_dict.keys():
            return self.load_state_dict(state_dict_to_load, strict=strict)
        # loading from regular VisionTransformer to CohortAwareVisionTransformer
        keys_to_adjust = [k for k in state_dict_to_load.keys() if k not in curr_state_dict.keys() and
                          k.startswith('blocks')]
        if len(keys_to_adjust) == 0:
            raise ValueError("state_dict_to_load is not in format of either VisionTransformer or CohortAwareVisionTransformer.")
        for key in keys_to_adjust:
            key_parts = key.split('.')
            new_key_format = '.'.join(key_parts[:-1]) + f'_{key_parts[-1][0]}'

            state_dict_to_load[new_key_format] = state_dict_to_load[key]
            # state_dict_to_load[new_key_format] = CohortAwareVisionTransformer.adjust_weight(state_dict_to_load[key],
            #                                                                                 curr_state_dict[new_key_format],
            #                                                                                 strategy=strategy)

            del state_dict_to_load[key]
        self.load_state_dict(state_dict_to_load)

    @staticmethod
    def adjust_weight(original_tensor, target_tensor, strategy):
        if strategy == 'copy_first':
            adj_tensor = torch.cat(
                [original_tensor, original_tensor[:target_tensor.shape[0] - original_tensor.shape[0], ]], dim=0)
            if len(original_tensor.shape) == 2:
                adj_tensor = torch.cat([adj_tensor, adj_tensor[:, :target_tensor.shape[1] - original_tensor.shape[1]]],
                                       dim=1)
            return adj_tensor


def cohort_aware_block_fn(cohort_aware_dict):
    def block_fn(**block_kwargs):
        return CohortAwareBlock(cohort_aware_dict, **block_kwargs)
    return block_fn


class CohortAwareBlock(nn.Module):

    def __init__(
            self,
            cohort_aware_dict,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CohortAwareAttention(
            cohort_aware_dict=cohort_aware_dict,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, c):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), c)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, c


class CohortAwareAttention(nn.Module):

    def __init__(
            self,
            cohort_aware_dict,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.cohort_aware_dict = cohort_aware_dict
        self.num_heads = num_heads
        self.exclude_cohorts = cohort_aware_dict['exclude_cohorts']
        self.num_cohorts = cohort_aware_dict['num_cohorts']
        self.num_heads_per_cohort = cohort_aware_dict['num_heads_per_cohort']
        self.include_cohorts = [c for c in range(self.num_cohorts) if c not in self.exclude_cohorts]
        self.num_shared_heads = num_heads - len(self.include_cohorts)*self.num_heads_per_cohort
        assert dim % self.num_heads == 0, 'dim should be divisible by total number of num_heads.'

        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.dim = dim
        self.qkv_w = nn.Parameter(torch.randn(self.dim * 3, self.dim))
        self.qkv_b = nn.Parameter(torch.randn(self.dim * 3))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias

    def forward(self, x, c):
        B, N, C = x.shape
        qkv = torch.matmul(x, self.qkv_w.t())
        if self.qkv_bias:
            qkv += self.qkv_b
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.cohort_aware_dict['awareness_strategy'] == 'separate_head':
            head_mask = torch.ones((B, self.num_shared_heads), device=c.device)
            c_one_hot = torch.zeros((B, self.num_cohorts), device=c.device)
            c_one_hot.scatter_(1, c.unsqueeze(-1), 1)
            c_one_hot = c_one_hot[:, self.include_cohorts]
            c_one_hot = c_one_hot.repeat(1, self.num_heads_per_cohort)
            head_mask = torch.cat((head_mask, c_one_hot),
                                  dim=1).unsqueeze(-1).unsqueeze(-1)
            q = q * head_mask
        elif self.cohort_aware_dict['awareness_strategy'] == 'separate_query':
            q = q.clone()
            for head_ind, c_ind in enumerate(self.include_cohorts):
                q[c != c_ind , self.num_shared_heads + head_ind, :, :] = q[c != c_ind , self.num_shared_heads + head_ind,
                                                                      :, :].detach()
        else:
            raise NotImplementedError

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
