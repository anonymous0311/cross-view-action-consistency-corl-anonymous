"""Configs for M6 canonical action-anchor training."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0M6AnchorConfig(pi0_config.Pi0Config):
    """pi0.5 wrapper that trains only the canonical action-anchor branch."""

    m6_canonical_dim: int = 512
    m6_hidden_dim: int = 512
    m6_num_anchor_tokens: int = 16
    m6_num_heads: int = 8
    m6_bottleneck_layers: int = 2
    m6_head_layers: int = 1
    m6_feature_noise_std: float = 0.01
    m6_token_dropout: float = 0.05
    m6_slot_dropout: float = 0.05
    m6_noise_floor_std: float = 1e-4
    m6_rank_margin: float = 0.05
    m6_lambda_rank: float = 0.5
    m6_preprocess_images: bool = True

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.m6_anchor import Pi0M6AnchorPretrain

        return Pi0M6AnchorPretrain(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        obs, actions = super().inputs_spec(batch_size=batch_size)
        canonical_spec = jax.ShapeDtypeStruct([batch_size, 128, self.m6_canonical_dim], jnp.float32)
        with at.disable_typechecking():
            obs = _model.Observation(
                images=obs.images,
                image_masks=obs.image_masks,
                state=obs.state,
                tokenized_prompt=obs.tokenized_prompt,
                tokenized_prompt_mask=obs.tokenized_prompt_mask,
                token_ar_mask=obs.token_ar_mask,
                token_loss_mask=obs.token_loss_mask,
                canonical_tokens=canonical_spec,
                canonical_tokens_neg=canonical_spec,
                canonical_tokens_mean=canonical_spec,
                task_index=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                episode_index=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                sample_index=jax.ShapeDtypeStruct([batch_size], jnp.int32),
            )
        return obs, actions


@dataclasses.dataclass(frozen=True)
class Pi0M6GatedFusionConfig(Pi0M6AnchorConfig):
    """pi0.5 wrapper for M6-B gated fusion training."""

    m6_beta_anchor: float = 0.25
    m6_eta_budget: float = 0.05
    m6_gate_init_logit: float = -2.0

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.m6_anchor import Pi0M6AnchorGatedFusion

        return Pi0M6AnchorGatedFusion(self, rngs=nnx.Rngs(rng))
