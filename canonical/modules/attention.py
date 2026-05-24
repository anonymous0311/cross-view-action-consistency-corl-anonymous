"""Pre-LN self- and cross-attention blocks for the canonical transformer.

Convention: Pre-LN residual blocks with MLP expansion of 4x and GELU activation.
This matches the standard recipe used by ViT and openpi's gemma stack.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def _make_ffn(d: int, mlp_ratio: float, *, rngs: nnx.Rngs) -> tuple[nnx.Linear, nnx.Linear]:
    hidden = int(round(d * mlp_ratio))
    fc1 = nnx.Linear(d, hidden, rngs=rngs)
    fc2 = nnx.Linear(hidden, d, rngs=rngs)
    return fc1, fc2


class SelfAttnBlock(nnx.Module):
    """Pre-LN self-attention + FFN residual block."""

    def __init__(
        self,
        d: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        self.d = d
        self.num_heads = num_heads
        self.norm1 = nnx.LayerNorm(d, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d,
            qkv_features=d,
            out_features=d,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(d, rngs=rngs)
        self.fc1, self.fc2 = _make_ffn(d, mlp_ratio, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.norm1(x)
        x = x + self.attn(h)
        h = self.norm2(x)
        return x + self.fc2(nnx.gelu(self.fc1(h)))


class CrossAttnBlock(nnx.Module):
    """Pre-LN cross-attention + FFN residual block.

    Queries attend into a separate key/value stream. LayerNorms are applied
    independently to the query stream (norm_q) and the kv stream (norm_kv) so
    the query residual path stays stable even when the kv distribution shifts.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        self.d = d
        self.num_heads = num_heads
        self.norm_q = nnx.LayerNorm(d, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(d, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d,
            qkv_features=d,
            out_features=d,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(d, rngs=rngs)
        self.fc1, self.fc2 = _make_ffn(d, mlp_ratio, rngs=rngs)

    def __call__(self, q: jax.Array, kv: jax.Array) -> jax.Array:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        q = q + self.attn(q_n, kv_n)
        h = self.norm2(q)
        return q + self.fc2(nnx.gelu(self.fc1(h)))
