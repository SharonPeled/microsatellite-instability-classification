import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, Mlp, DropPath
from timm.models.vision_transformer import LayerScale, VisionTransformer
from copy import deepcopy
from src.general_utils import MultiInputSequential
import pandas as pd


class CohortAwareVisionTransformer(VisionTransformer):
    
    def __init__(self, cohort_aware_dict, depth, **vit_kwargs):
        vit_kwargs['block_fn'] = cohort_aware_block_fn(cohort_aware_dict)
        self.cohort_aware_dict = cohort_aware_dict
        self.cohort_aware_dict['depth'] = depth
        self.num_heads = vit_kwargs['num_heads']
        self.num_cohorts = cohort_aware_dict['num_cohorts']
        self.num_heads_per_cohort = cohort_aware_dict['num_heads_per_cohort']
        self.head_dim = vit_kwargs['embed_dim'] // self.num_heads
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
            block_number = int(key_parts[1]) + 1
            if self.cohort_aware_dict['awareness_strategy'] in ['one_hot_head', 'shared_query_separate_training',
                                                                'learnable_bias_matrices'] or \
                    (self.cohort_aware_dict['awareness_strategy'] == 'separate_query_per_block' and
                     (block_number <= self.cohort_aware_dict['depth'] - self.cohort_aware_dict['num_blocks_per_cohort'])):
                new_key_format = '.'.join(key_parts[:-1]) + f'_{key_parts[-1][0]}'
                state_dict_to_load[new_key_format] = state_dict_to_load[key]
            elif self.cohort_aware_dict['awareness_strategy'] in ['separate_query', 'separate_noisy_query',
                                                                  'separate_query_per_block']:
                kv_new_key_format = '.'.join(key_parts[:3]) + f'.kv_{key_parts[-1][0]}'
                q_shift = state_dict_to_load[key].shape[0] // 3
                state_dict_to_load[kv_new_key_format] = state_dict_to_load[key][q_shift:]
                shared_q_new_key_format = '.'.join(key_parts[:3]) + f'.shared_q_{key_parts[-1][0]}'
                state_dict_to_load[shared_q_new_key_format] = state_dict_to_load[key][:q_shift][:curr_state_dict[shared_q_new_key_format].shape[0]]
                cohort_q_new_key_format = '.'.join(key_parts[:3]) + f'.cohort_q_{key_parts[-1][0]}'
                target_shape = curr_state_dict[cohort_q_new_key_format].shape
                rest_q = state_dict_to_load[key][:q_shift][-target_shape[1]:]
                state_dict_to_load[cohort_q_new_key_format] = rest_q.repeat(target_shape[0], 1).reshape(target_shape)
                if self.cohort_aware_dict['awareness_strategy'] == 'separate_noisy_query':
                    std = state_dict_to_load[cohort_q_new_key_format].std()*0.2
                    noise = torch.normal(mean=0.0, std=std, size=state_dict_to_load[cohort_q_new_key_format].size())
                    state_dict_to_load[cohort_q_new_key_format] += noise
            else:
                raise NotImplementedError
            del state_dict_to_load[key]
        self.load_state_dict(state_dict_to_load, strict=False)

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
    cohort_aware_dict['block_num'] = 0
    def block_fn(**block_kwargs):
        cohort_aware_dict['block_num'] += 1
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
        if cohort_aware_dict['block_num'] <= cohort_aware_dict['depth'] - cohort_aware_dict['num_blocks_per_cohort']:
            apply_awareness = False
        else:
            apply_awareness = True
        self.norm1 = norm_layer(dim)
        self.attn = CohortAwareAttention(
            cohort_aware_dict=cohort_aware_dict,
            apply_awareness=apply_awareness,
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
            apply_awareness,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.apply_awareness = apply_awareness
        self.cohort_aware_dict = cohort_aware_dict
        self.num_heads = num_heads
        self.exclude_cohorts = cohort_aware_dict['exclude_cohorts']
        self.num_cohorts = cohort_aware_dict['num_cohorts']
        self.num_heads_per_cohort = cohort_aware_dict['num_heads_per_cohort']
        self.include_cohorts = [c for c in range(self.num_cohorts) if c not in self.exclude_cohorts]
        assert dim % self.num_heads == 0, 'dim should be divisible by total number of num_heads.'

        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.dim = dim
        self.init_qkv()
        self.init_cb()

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias

    def init_cb(self):
        if self.cohort_aware_dict['awareness_strategy'] == 'learnable_bias_matrices':
            self.cb_w = nn.Parameter(torch.randn(self.dim, self.dim)*0.1)
            self.cb_b = nn.Parameter(torch.randn(self.dim)*0.1)

    def init_qkv(self):
        if self.cohort_aware_dict['awareness_strategy'] in ['one_hot_head', 'shared_query_separate_training',
                                                            'learnable_bias_matrices', None] or\
                not self.apply_awareness:
            self.num_shared_heads = self.num_heads - len(self.include_cohorts) * self.num_heads_per_cohort
            self.qkv_w = nn.Parameter(torch.randn(self.dim * 3, self.dim))
            self.qkv_b = nn.Parameter(torch.randn(self.dim * 3))
        elif self.cohort_aware_dict['awareness_strategy'] in ['separate_query', 'separate_noisy_query',
                                                                  'separate_query_per_block']:
            self.num_shared_heads = self.num_heads - self.num_heads_per_cohort
            self.kv_w = nn.Parameter(torch.randn(self.dim * 2, self.dim))
            self.kv_b = nn.Parameter(torch.randn(self.dim * 2))
            self.shared_q_w =  nn.Parameter(torch.randn(self.head_dim*self.num_shared_heads,
                                                        self.dim))
            self.shared_q_b = nn.Parameter(torch.randn(self.head_dim*self.num_shared_heads))
            self.cohort_q_w = nn.Parameter(torch.randn(len(self.include_cohorts), self.head_dim*self.num_heads_per_cohort,
                                                       self.dim))
            self.cohort_q_b = nn.Parameter(torch.randn(len(self.include_cohorts), self.head_dim*self.num_heads_per_cohort))
        else:
            raise NotImplementedError

    def get_qkv_matrices(self, x, c):
        B, N, C = x.shape
        if self.cohort_aware_dict['awareness_strategy'] in ['one_hot_head', 'shared_query_separate_training',
                                                            'learnable_bias_matrices', None] \
                or not self.apply_awareness:
            qkv = torch.matmul(x, self.qkv_w.t())
            if self.qkv_bias:
                qkv += self.qkv_b
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if self.cohort_aware_dict['awareness_strategy'] == 'one_hot_head' and self.apply_awareness:
                head_mask = torch.ones((B, self.num_shared_heads), device=c.device)
                c_one_hot = torch.zeros((B, self.num_cohorts), device=c.device)
                c_one_hot.scatter_(1, c.unsqueeze(-1), 1)
                c_one_hot = c_one_hot[:, self.include_cohorts]
                c_one_hot = c_one_hot.repeat(1, self.num_heads_per_cohort)
                head_mask = torch.cat((head_mask, c_one_hot),
                                      dim=1).unsqueeze(-1).unsqueeze(-1)
                q = q * head_mask

            elif self.cohort_aware_dict['awareness_strategy'] == 'shared_query_separate_training' and self.apply_awareness:
                q = q.clone()
                for head_ind, c_ind in enumerate(self.include_cohorts):
                    q[c != c_ind, self.num_shared_heads + head_ind, :, :] = q[c != c_ind, self.num_shared_heads + head_ind,
                                                                            :, :].detach()

        elif self.cohort_aware_dict['awareness_strategy'] in ['separate_query', 'separate_noisy_query',
                                                              'separate_query_per_block']:
            # key, value only
            kv = torch.matmul(x, self.kv_w.t())
            if self.qkv_bias:
                kv += self.kv_b
            kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            # shared q
            shared_q = torch.matmul(x, self.shared_q_w.t())
            if self.qkv_bias:
                shared_q += self.shared_q_b
            # separate q
            c_cpu = c.cpu()
            indices = torch.arange(B)
            sep_q_list = []
            sep_inds_list = []
            for head_ind, c_ind in enumerate(self.include_cohorts):
                x_c = x[c == c_ind]
                sep_q_list.append(torch.matmul(x_c, self.cohort_q_w[head_ind].t()))
                if self.qkv_bias:
                    sep_q_list[-1] += self.cohort_q_b[head_ind]
                sep_inds_list.append(indices[c_cpu == c_ind])
            for c_ind in self.exclude_cohorts:
                x_c = x[c == c_ind]
                sep_q_list.append(torch.zeros((x_c.shape[0], x_c.shape[1],
                                               self.head_dim*self.num_heads_per_cohort), device=x.device))
                sep_inds_list.append(indices[c_cpu == c_ind])

            sep_q = torch.cat(sep_q_list, dim=0)
            sep_inds = torch.tensor(pd.Series(index=torch.cat(sep_inds_list).numpy(), data=indices).loc[indices].values,
                                    device=sep_q.device)
            sep_q = torch.index_select(sep_q, 0, sep_inds)
            q = torch.cat([shared_q, sep_q], dim=-1)
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            raise NotImplementedError
        return q, k, v

    def get_cb_matrix(self, x, c):
        B, N, C = x.shape
        if self.cohort_aware_dict['bias_matrices'] in ['z_before_fc', ]:
            c_cpu = c.cpu()
            indices = torch.arange(B)
            cb_list = []
            cb_inds_list = []
            for head_ind, c_ind in enumerate(range(self.num_cohorts)):
                x_c = x[c == c_ind]
                cb_list.append(torch.matmul(x_c, self.cb_w.t()) + self.cb_b)
                cb_inds_list.append(indices[c_cpu == c_ind])

            cb = torch.cat(cb_list, dim=0)
            cb_inds = torch.tensor(
                pd.Series(index=torch.cat(cb_inds_list).numpy(), data=indices).loc[indices].values,
                device=cb.device)
            cb = torch.index_select(cb, 0, cb_inds)
            return cb
        else:
            cb = torch.zeros(B, self.num_heads, N, self.head_dim)
        return cb

    def forward(self, x, c):
        B, N, C = x.shape
        q, k, v = self.get_qkv_matrices(x, c)
        cb = self.get_cb_matrix(x, c)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if self.cohort_aware_dict['bias_matrices'] in ['z_before_fc', ]:
            x = x + cb

        x = self.proj_drop(x)
        return x
