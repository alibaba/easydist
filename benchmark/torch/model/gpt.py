import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCore(nn.Module):

    def __init__(self, attention_head_size: int, attention_dropout: float) -> None:
        super().__init__()
        self.attention_head_size = attention_head_size
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

        # (avoid the const tensor init when forward)
        # self.register_buffer('causal_mask', None)
        self.causal_mask = None
        self.where_const = -1e4

    def forward(self, q, k, v):
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)

        # (avoid the const tensor init when forward)
        if self.causal_mask is None:
            q_len, k_len = q.size(-2), k.size(-2)
            self.causal_mask = torch.tril(
                torch.ones((q_len, k_len), dtype=torch.uint8,
                           device=q.device)).view(1, 1, q_len, k_len).bool()
        x = torch.where(self.causal_mask, x, self.where_const)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (-1, )
        x = x.reshape(new_context_layer_shape)

        return x


class FeedForward(nn.Module):

    def __init__(self, hidden_size, ratio=4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = hidden_size * ratio

        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.dense_4h_to_h = nn.Linear(self.ffn_hidden_size, self.hidden_size)

        self.activation_func = F.gelu

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class SelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float = 0.,
                 dropout: float = 0.) -> None:
        super().__init__()

        self.attention_head_size = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.dense = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.core_attention = AttentionCore(self.attention_head_size, attention_dropout)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        all_head_size = q.shape[-1]
        num_attention_heads = all_head_size // self.attention_head_size

        local_attention_heads = num_attention_heads
        new_qkv_shape = q.shape[:-1] + \
            (local_attention_heads, self.attention_head_size)
        q = q.view(new_qkv_shape)
        k = k.view(new_qkv_shape)
        v = v.view(new_qkv_shape)

        # TODO @botbw: solve non-contiguous
        q = q.permute((0, 2, 1, 3)).contiguous()
        k = k.permute((0, 2, 1, 3)).contiguous()
        v = v.permute((0, 2, 1, 3)).contiguous()

        x = self.core_attention(q, k, v)

        x = self.dense(x)
        x = self.dropout(x)

        return x


class GPTLayer(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 dtype: torch.dtype = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=1e-6, dtype=dtype)
        self.attn = SelfAttention(dim=dim,
                                  num_heads=num_heads,
                                  attention_dropout=attention_dropout,
                                  dropout=dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=1e-6, dtype=dtype)
        self.mlp = FeedForward(hidden_size=dim, ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):

    def __init__(self,
                 depth: int,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 dtype: torch.dtype = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            GPTLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                dtype=dtype,
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class LowCommGPTLayer(GPTLayer):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 comm_dim: int = 1,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 dtype: torch.dtype = None):
        super().__init__(dim, num_heads, mlp_ratio, attention_dropout, dropout, dtype)
        self.decompress = nn.Linear(comm_dim, dim)
        self.compress = nn.Linear(dim, comm_dim)

    def forward(self, x):
        x = self.decompress(x)
        x = super().forward(x)
        x = self.compress(x)
        return x

class SequentialLowCommGPT(nn.Sequential):
    def __init__(self,
                 depth: int,
                 dim: int,
                 num_heads: int,
                 comm_dim: int = 1,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 dtype: torch.dtype = None):
        super().__init__(OrderedDict(
            [
                (f'block{i}', LowCommGPTLayer(
                    dim=dim,
                    num_heads=num_heads,
                    comm_dim=comm_dim,
                    mlp_ratio=mlp_ratio,
                    attention_dropout=attention_dropout,
                    dropout=dropout,
                    dtype=dtype,
                )) for i in range(depth)
            ]
        ))

    def forward(self, x):
        return super().forward(x)