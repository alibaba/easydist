# code modified from https://github.com/jenkspt/gpt-jax

from dataclasses import dataclass
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.0
    use_bias: bool = True
    dtype: Optional[str] = "float32"


class SelfAttention(nn.Module):

    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads
        # deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        qkv = nn.Dense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        # calculate attention matrix
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate, deterministic=deterministic)(attn)

        # return weighted sum over values for each query position
        x = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(B, T, C)
        x = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(x)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x


class FeedForward(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=deterministic)(x)
        return x


class GPTBlock(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5,
                                 dtype=self.config.dtype,
                                 use_bias=self.config.use_bias)
        self.attn = SelfAttention(self.config.num_heads,
                                  self.config.dtype,
                                  dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5,
                                 dtype=self.config.dtype,
                                 use_bias=self.config.use_bias)
        self.mlp = FeedForward(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        if mask == None:
            attn_mask = nn.make_causal_mask(jax.numpy.ones((x.shape[0], x.shape[1])), dtype=bool)
        x = x + self.attn(self.ln_1(x), attn_mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


class GPTSimple(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, mask=None, deterministic=None):

        for i in range(self.config.num_layers):
            x = GPTBlock(self.config, name=str(i))(x, None, deterministic=deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        wte = nn.Embed(self.config.vocab_size,
                       self.config.num_embeds,
                       dtype=self.config.dtype,
                       name='wte')
        wpe = nn.Embed(self.config.block_size,
                       self.config.num_embeds,
                       dtype=self.config.dtype,
                       name='wpe')

        token_embed = wte(idx)  # [B, T, num_embeds]
        pos_embed = wpe(pos)  # [1, T, num_embeds]
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for i in range(self.config.num_layers):
            x = GPTBlock(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias,
                         name='ln_f')(x)
        logits = wte.attend(x)
        return logits

    def init(self, rng):
        """
        by jitting init, traced values instead of concrete values are used
        which saves memory (since un-jitted model may not fit in memory)
        """
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2, ))(rng, tokens, True)
        return params
