"""M6 canonical action-anchor modules.

These components implement the M6 action-anchor path:

    z + state + noisy action + time -> bottleneck memory -> full flow anchor

The branch consumes frozen canonical tokens and action context only. It does not
inject canonical tokens into the pi0.5 backbone and it does not predict a
residual target. M6-A trains the anchor directly; M6-B learns a small gate that
interpolates from the frozen pi0.5 base flow toward the canonical anchor.
"""

from __future__ import annotations

import dataclasses
import os
from typing import NamedTuple

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from canonical.modules.attention import CrossAttnBlock
from canonical.modules.attention import SelfAttnBlock
import openpi.models.gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import Pi0


def posemb_sincos_1d(
    pos: jax.Array,
    embedding_dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> jax.Array:
    """Sine-cosine embedding for scalar flow-matching time values."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum("...,...d->...d", pos, 1.0 / period * 2 * jnp.pi)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class CanonicalTokenPerturbation:
    """Training-time perturbations for canonical tokens.

    Dropout paths replace dropped values with small Gaussian noise instead of
    exact zeros. This avoids creating a special all-zero token distribution.
    """

    feature_noise_std: float = 0.0
    token_dropout: float = 0.0
    slot_dropout: float = 0.0
    noise_floor_std: float = 1e-4

    def __call__(self, tokens: jax.Array, rng: jax.Array, *, train: bool = True) -> jax.Array:
        if not train:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"canonical tokens must be [B, N, D], got {tokens.shape}")

        out = tokens
        rng_feature, rng_token, rng_slot, rng_floor = jax.random.split(rng, 4)

        if self.feature_noise_std > 0.0:
            out = out + jax.random.normal(rng_feature, out.shape, dtype=out.dtype) * self.feature_noise_std

        if self.token_dropout > 0.0:
            keep = jax.random.uniform(rng_token, (out.shape[0], 1, 1)) >= self.token_dropout
            floor = jax.random.normal(rng_floor, out.shape, dtype=out.dtype) * self.noise_floor_std
            out = jnp.where(keep, out, floor)

        if self.slot_dropout > 0.0:
            keep = jax.random.uniform(rng_slot, (out.shape[0], out.shape[1], 1)) >= self.slot_dropout
            floor = jax.random.normal(jax.random.fold_in(rng_floor, 1), out.shape, dtype=out.dtype) * self.noise_floor_std
            out = jnp.where(keep, out, floor)

        return out


class HardNegativeBatch(NamedTuple):
    matched: np.ndarray
    negative: np.ndarray
    mean: np.ndarray
    negative_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class HardNegativeCanonicalSampler:
    """Select same-task, different-episode canonical-token negatives.

    This sampler operates on an already-loaded batch. It is intentionally small
    and deterministic under an explicit NumPy RNG so it can be used in static
    diagnostics and dataset-side transforms.
    """

    min_action_distance: float = 0.0
    min_state_distance: float = 0.0
    fallback_to_same_task: bool = True

    def sample(
        self,
        canonical_tokens: np.ndarray,
        *,
        task_ids: np.ndarray,
        episode_ids: np.ndarray,
        actions: np.ndarray | None = None,
        states: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> HardNegativeBatch:
        if canonical_tokens.ndim != 3:
            raise ValueError(f"canonical_tokens must be [B, N, D], got {canonical_tokens.shape}")
        batch_size = canonical_tokens.shape[0]
        task_ids = np.asarray(task_ids)
        episode_ids = np.asarray(episode_ids)
        if task_ids.shape[0] != batch_size or episode_ids.shape[0] != batch_size:
            raise ValueError("task_ids and episode_ids must have length B")

        rng = np.random.default_rng() if rng is None else rng
        neg_indices = np.empty(batch_size, dtype=np.int64)

        flat_actions = None if actions is None else np.asarray(actions).reshape(batch_size, -1)
        flat_states = None if states is None else np.asarray(states).reshape(batch_size, -1)

        for i in range(batch_size):
            base_candidates = np.where((task_ids == task_ids[i]) & (episode_ids != episode_ids[i]))[0]
            candidates = base_candidates
            if flat_actions is not None and self.min_action_distance > 0.0:
                dist = np.linalg.norm(flat_actions[candidates] - flat_actions[i], axis=-1)
                candidates = candidates[dist >= self.min_action_distance]
            if flat_states is not None and self.min_state_distance > 0.0:
                dist = np.linalg.norm(flat_states[candidates] - flat_states[i], axis=-1)
                candidates = candidates[dist >= self.min_state_distance]

            if candidates.size == 0:
                candidates = base_candidates
            if candidates.size == 0 and self.fallback_to_same_task:
                candidates = np.where((task_ids == task_ids[i]) & (np.arange(batch_size) != i))[0]
            if candidates.size == 0:
                candidates = np.where(np.arange(batch_size) != i)[0]
            if candidates.size == 0:
                raise ValueError("Cannot sample a negative from a batch of size 1")

            neg_indices[i] = rng.choice(candidates)

        mean = np.mean(canonical_tokens, axis=0, keepdims=True).repeat(batch_size, axis=0)
        return HardNegativeBatch(
            matched=canonical_tokens,
            negative=canonical_tokens[neg_indices],
            mean=mean.astype(canonical_tokens.dtype, copy=False),
            negative_indices=neg_indices,
        )


class ActionConditionedCanonicalBottleneck(nnx.Module):
    """Cross-attend action-conditioned queries into canonical tokens."""

    def __init__(
        self,
        *,
        canonical_dim: int = 512,
        state_dim: int = 32,
        action_dim: int = 32,
        action_horizon: int = 10,
        hidden_dim: int = 512,
        num_anchor_tokens: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        language_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_anchor_tokens = num_anchor_tokens
        self.action_horizon = action_horizon
        self.language_dim = language_dim

        self.canonical_proj = nnx.Linear(canonical_dim, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.action_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.time_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.language_proj = None if language_dim is None else nnx.Linear(language_dim, hidden_dim, rngs=rngs)
        self.context_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.kv_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        query_init = jax.random.normal(
            rngs.params(), (num_anchor_tokens, hidden_dim), dtype=jnp.float32
        ) * 0.02
        self.anchor_queries = nnx.Param(query_init)

        self.cross_blocks = [
            CrossAttnBlock(d=hidden_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.self_blocks = [
            SelfAttnBlock(d=hidden_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(
        self,
        canonical_tokens: jax.Array,
        state: jax.Array,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array | None = None,
    ) -> jax.Array:
        if canonical_tokens.ndim != 3:
            raise ValueError(f"canonical_tokens must be [B, N, D], got {canonical_tokens.shape}")
        if noisy_actions.ndim != 3:
            raise ValueError(f"noisy_actions must be [B, H, A], got {noisy_actions.shape}")

        batch_size = canonical_tokens.shape[0]
        action_summary = jnp.mean(noisy_actions, axis=1)
        time_emb = posemb_sincos_1d(time, self.hidden_dim)

        context = self.state_proj(state) + self.action_proj(action_summary) + self.time_proj(time_emb)
        if language_summary is not None:
            if self.language_proj is None:
                raise ValueError("language_summary was provided but language_dim=None at construction")
            context = context + self.language_proj(language_summary)
        context = self.context_norm(context)

        q = jnp.broadcast_to(self.anchor_queries.value, (batch_size, self.num_anchor_tokens, self.hidden_dim))
        q = q + context[:, None, :]
        kv = self.kv_norm(self.canonical_proj(canonical_tokens))
        for cross, self_block in zip(self.cross_blocks, self.self_blocks, strict=True):
            q = cross(q, kv)
            q = self_block(q)
        return self.out_norm(q)


class CanonicalAnchorFlowHead(nnx.Module):
    """Predict a full pi0.5 flow/action anchor from anchor memory."""

    def __init__(
        self,
        *,
        action_dim: int = 32,
        action_horizon: int = 10,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.action_in = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.time_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.action_queries = nnx.Param(
            jax.random.normal(rngs.params(), (action_horizon, hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.cross_blocks = [
            CrossAttnBlock(d=hidden_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, anchor_memory: jax.Array, noisy_actions: jax.Array, time: jax.Array) -> jax.Array:
        if anchor_memory.ndim != 3:
            raise ValueError(f"anchor_memory must be [B, N, D], got {anchor_memory.shape}")
        batch_size = anchor_memory.shape[0]
        q = jnp.broadcast_to(self.action_queries.value, (batch_size, self.action_horizon, self.hidden_dim))
        q = q + self.action_in(noisy_actions)
        q = q + self.time_proj(posemb_sincos_1d(time, self.hidden_dim))[:, None, :]
        for cross in self.cross_blocks:
            q = cross(q, anchor_memory)
        return self.out(self.out_norm(q))


class CanonicalAnchorPretrainHead(nnx.Module):
    """Convenience wrapper for M6-A: bottleneck followed by flow head."""

    def __init__(
        self,
        *,
        canonical_dim: int = 512,
        state_dim: int = 32,
        action_dim: int = 32,
        action_horizon: int = 10,
        hidden_dim: int = 512,
        num_anchor_tokens: int = 16,
        num_heads: int = 8,
        bottleneck_layers: int = 2,
        head_layers: int = 1,
        language_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.bottleneck = ActionConditionedCanonicalBottleneck(
            canonical_dim=canonical_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_anchor_tokens=num_anchor_tokens,
            num_heads=num_heads,
            num_layers=bottleneck_layers,
            language_dim=language_dim,
            rngs=rngs,
        )
        self.flow_head = CanonicalAnchorFlowHead(
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=head_layers,
            rngs=rngs,
        )

    def __call__(
        self,
        canonical_tokens: jax.Array,
        state: jax.Array,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array | None = None,
    ) -> jax.Array:
        memory = self.encode(canonical_tokens, state, noisy_actions, time, language_summary)
        return self.decode(memory, noisy_actions, time)

    def encode(
        self,
        canonical_tokens: jax.Array,
        state: jax.Array,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array | None = None,
    ) -> jax.Array:
        return self.bottleneck(canonical_tokens, state, noisy_actions, time, language_summary)

    def decode(self, anchor_memory: jax.Array, noisy_actions: jax.Array, time: jax.Array) -> jax.Array:
        return self.flow_head(anchor_memory, noisy_actions, time)


class AnchorConfidenceGate(nnx.Module):
    """Predicts per-action-channel trust in the canonical anchor."""

    def __init__(
        self,
        *,
        state_dim: int = 32,
        action_dim: int = 32,
        hidden_dim: int = 512,
        language_dim: int | None = None,
        init_logit: float = -2.0,
        action_horizon: int = 10,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.init_logit = init_logit
        self.action_horizon = action_horizon
        self.memory_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.action_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.time_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.language_proj = None if language_dim is None else nnx.Linear(language_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.logit_bias = nnx.Param(jnp.full((action_dim,), init_logit, dtype=jnp.float32))

    def __call__(
        self,
        anchor_memory: jax.Array,
        state: jax.Array,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array | None = None,
    ) -> jax.Array:
        memory_summary = jnp.mean(anchor_memory, axis=1)
        action_summary = jnp.mean(noisy_actions, axis=1)
        time_emb = posemb_sincos_1d(time, self.hidden_dim)
        x = (
            self.memory_proj(memory_summary)
            + self.state_proj(state)
            + self.action_proj(action_summary)
            + self.time_proj(time_emb)
        )
        if language_summary is not None:
            if self.language_proj is None:
                raise ValueError("language_summary was provided but language_dim=None at construction")
            x = x + self.language_proj(language_summary)
        x = nnx.swish(self.norm(x))
        logits = self.out(x).astype(jnp.float32) + self.logit_bias.value
        alpha = jax.nn.sigmoid(logits)
        return alpha[:, None, :]


class AnchorLosses(NamedTuple):
    loss_anchor_match: jax.Array
    loss_anchor_neg: jax.Array
    loss_anchor_mean: jax.Array
    loss_rank_neg: jax.Array
    loss_rank_mean: jax.Array
    match_neg_gap: jax.Array
    match_mean_gap: jax.Array


def anchor_mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Per-example MSE over action horizon and action dimension."""
    pred = pred.astype(jnp.float32)
    target = target.astype(jnp.float32)
    return jnp.mean(jnp.square(pred - target), axis=(-2, -1))


def compute_anchor_losses(
    *,
    v_anchor_match: jax.Array,
    v_anchor_neg: jax.Array,
    v_anchor_mean: jax.Array,
    target_flow: jax.Array,
    margin: float = 0.05,
) -> AnchorLosses:
    """M6-A anchor and ranking losses for matched/negative/mean controls."""
    loss_match = anchor_mse(v_anchor_match, target_flow)
    loss_neg = anchor_mse(v_anchor_neg, target_flow)
    loss_mean = anchor_mse(v_anchor_mean, target_flow)
    rank_neg = jnp.maximum(0.0, margin + loss_match - loss_neg)
    rank_mean = jnp.maximum(0.0, margin + loss_match - loss_mean)
    return AnchorLosses(
        loss_anchor_match=jnp.mean(loss_match),
        loss_anchor_neg=jnp.mean(loss_neg),
        loss_anchor_mean=jnp.mean(loss_mean),
        loss_rank_neg=jnp.mean(rank_neg),
        loss_rank_mean=jnp.mean(rank_mean),
        match_neg_gap=jnp.mean(loss_neg - loss_match),
        match_mean_gap=jnp.mean(loss_mean - loss_match),
    )


class Pi0M6AnchorPretrain(_model.BaseModel):
    """M6-A training wrapper around a frozen pi0.5 base and trainable anchor."""

    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        base_config = pi0_config.Pi0Config(
            dtype=config.dtype,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            pi05=config.pi05,
            canonical_dim=0,
            canonical_token_dropout=0.0,
            discrete_state_input=config.discrete_state_input,
            pytorch_compile_mode=config.pytorch_compile_mode,
        )
        self.base = Pi0(base_config, rngs=rngs)
        self.anchor = CanonicalAnchorPretrainHead(
            canonical_dim=config.m6_canonical_dim,
            state_dim=config.action_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            hidden_dim=config.m6_hidden_dim,
            num_anchor_tokens=config.m6_num_anchor_tokens,
            num_heads=config.m6_num_heads,
            bottleneck_layers=config.m6_bottleneck_layers,
            head_layers=config.m6_head_layers,
            language_dim=config.m6_hidden_dim,
            rngs=rngs,
        )
        self.language_proj = nnx.Linear(_gemma.get_config(config.paligemma_variant).width, config.m6_hidden_dim, rngs=rngs)
        self.perturb = CanonicalTokenPerturbation(
            feature_noise_std=config.m6_feature_noise_std,
            token_dropout=config.m6_token_dropout,
            slot_dropout=config.m6_slot_dropout,
            noise_floor_std=config.m6_noise_floor_std,
        )
        self.preprocess_images = config.m6_preprocess_images
        self.rank_margin = config.m6_rank_margin
        self.lambda_rank = config.m6_lambda_rank

    def _language_summary(self, observation: _model.Observation) -> jax.Array:
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            return jnp.zeros((observation.state.shape[0], self.language_proj.out_features), dtype=observation.state.dtype)
        prompt_tokens = self.base.PaliGemma.llm(observation.tokenized_prompt, method="embed")
        mask = observation.tokenized_prompt_mask[..., None].astype(prompt_tokens.dtype)
        denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
        pooled = jnp.sum(prompt_tokens * mask, axis=1) / denom
        return self.language_proj(pooled)

    def _predict_anchor(
        self,
        canonical_tokens: jax.Array,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array,
        rng: jax.Array,
        *,
        train: bool,
        perturb: bool = True,
    ) -> jax.Array:
        canonical_tokens = self.perturb(canonical_tokens, rng, train=train and perturb)
        return self.anchor(canonical_tokens, observation.state, noisy_actions, time, language_summary)

    def _predict_anchor_with_memory(
        self,
        canonical_tokens: jax.Array,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        time: jax.Array,
        language_summary: jax.Array,
        rng: jax.Array,
        *,
        train: bool,
        perturb: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        canonical_tokens = self.perturb(canonical_tokens, rng, train=train and perturb)
        memory = self.anchor.encode(canonical_tokens, observation.state, noisy_actions, time, language_summary)
        return self.anchor.decode(memory, noisy_actions, time), memory

    def _base_observation(self, observation: _model.Observation) -> _model.Observation:
        return dataclasses.replace(
            observation,
            canonical_tokens=None,
            canonical_tokens_neg=None,
            canonical_tokens_mean=None,
            task_index=None,
            episode_index=None,
            sample_index=None,
        )

    def _sample_flow_batch(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
    ) -> tuple[_model.Observation, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        preprocess_rng, noise_rng, time_rng, match_rng, neg_rng, mean_rng = jax.random.split(rng, 6)
        if self.preprocess_images:
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        if observation.canonical_tokens is None:
            raise ValueError("M6 anchor pretrain requires observation.canonical_tokens")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        return observation, x_t, time, u_t, match_rng, neg_rng, mean_rng

    def _control_tokens(self, observation: _model.Observation) -> tuple[jax.Array, jax.Array, jax.Array]:
        z_match = observation.canonical_tokens
        if z_match is None:
            raise ValueError("M6 anchor pretrain requires observation.canonical_tokens")

        fallback_neg = observation.canonical_tokens_neg
        if fallback_neg is None:
            fallback_neg = jnp.roll(z_match, shift=1, axis=0)
        if observation.task_index is not None and observation.episode_index is not None:
            batch_size = z_match.shape[0]
            task = observation.task_index
            episode = observation.episode_index
            same_task_diff_episode = (task[:, None] == task[None, :]) & (episode[:, None] != episode[None, :])
            has_candidate = jnp.any(same_task_diff_episode, axis=1)
            first_candidate = jnp.argmax(same_task_diff_episode.astype(jnp.int32), axis=1)
            batch_neg = jnp.take(z_match, first_candidate, axis=0)
            z_neg = jnp.where(has_candidate[:, None, None], batch_neg, fallback_neg)
        else:
            z_neg = fallback_neg
        z_mean = observation.canonical_tokens_mean
        if z_mean is None:
            z_mean = jnp.broadcast_to(jnp.mean(z_match, axis=0, keepdims=True), z_match.shape)
        return z_match, z_neg, z_mean

    def _predict_controls(
        self,
        observation: _model.Observation,
        x_t: jax.Array,
        time: jax.Array,
        match_rng: jax.Array,
        neg_rng: jax.Array,
        mean_rng: jax.Array,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        z_match, z_neg, z_mean = self._control_tokens(observation)
        language_summary = self._language_summary(observation)
        v_match = self._predict_anchor(z_match, observation, x_t, time, language_summary, match_rng, train=train)
        v_neg = self._predict_anchor(z_neg, observation, x_t, time, language_summary, neg_rng, train=train)
        # Keep the constant-token control constant. Perturbing this path would
        # turn it into a random-noise control and make L_rank_mean too easy.
        v_mean = self._predict_anchor(
            z_mean, observation, x_t, time, language_summary, mean_rng, train=train, perturb=False
        )
        return v_match, v_neg, v_mean

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: int | jax.Array = 0,
    ) -> jax.Array:
        del step
        observation, x_t, time, u_t, match_rng, neg_rng, mean_rng = self._sample_flow_batch(
            rng, observation, actions, train=train
        )
        v_match, v_neg, v_mean = self._predict_controls(
            observation, x_t, time, match_rng, neg_rng, mean_rng, train=train
        )
        match_mse = anchor_mse(v_match, u_t)
        neg_mse = anchor_mse(v_neg, u_t)
        mean_mse = anchor_mse(v_mean, u_t)
        per_example = (
            match_mse
            + self.lambda_rank
            * (
                jnp.maximum(0.0, self.rank_margin + match_mse - neg_mse)
                + jnp.maximum(0.0, self.rank_margin + match_mse - mean_mse)
            )
        )
        return jnp.broadcast_to(per_example[..., None], actions.shape[:-1])

    def compute_train_metrics(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        step: int | jax.Array = 0,
    ) -> dict[str, jax.Array]:
        del step
        observation, x_t, time, u_t, match_rng, neg_rng, mean_rng = self._sample_flow_batch(
            rng, observation, actions, train=False
        )
        v_match, v_neg, v_mean = self._predict_controls(
            observation, x_t, time, match_rng, neg_rng, mean_rng, train=False
        )
        losses = compute_anchor_losses(
            v_anchor_match=v_match,
            v_anchor_neg=v_neg,
            v_anchor_mean=v_mean,
            target_flow=u_t,
            margin=self.rank_margin,
        )
        return {
            "m6_anchor/loss_anchor_match": losses.loss_anchor_match,
            "m6_anchor/loss_anchor_neg": losses.loss_anchor_neg,
            "m6_anchor/loss_anchor_mean": losses.loss_anchor_mean,
            "m6_anchor/match_neg_gap": losses.match_neg_gap,
            "m6_anchor/match_mean_gap": losses.match_mean_gap,
            "m6_anchor/loss_rank_neg": losses.loss_rank_neg,
            "m6_anchor/loss_rank_mean": losses.loss_rank_mean,
            "m6_anchor/v_anchor_norm": jnp.mean(jnp.linalg.norm(v_match, axis=-1)),
        }

    def sample_actions(self, rng: jax.Array, observation: _model.Observation, **kwargs) -> _model.Actions:
        raise NotImplementedError("M6-A anchor pretrain is not an inference policy; use gated fusion for rollout.")


class Pi0M6AnchorGatedFusion(Pi0M6AnchorPretrain):
    """M6-B wrapper: frozen pi0.5 base plus trainable anchor confidence gate."""

    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.gate = AnchorConfidenceGate(
            state_dim=config.action_dim,
            action_dim=config.action_dim,
            hidden_dim=config.m6_hidden_dim,
            language_dim=config.m6_hidden_dim,
            init_logit=config.m6_gate_init_logit,
            action_horizon=config.action_horizon,
            rngs=rngs,
        )
        self.beta_anchor = config.m6_beta_anchor
        self.eta_budget = config.m6_eta_budget
        alpha_override = os.environ.get("OPENPI_M6_ALPHA_OVERRIDE", "").strip()
        self.alpha_override: float | None = None
        if alpha_override:
            self.alpha_override = float(alpha_override)
            if not 0.0 <= self.alpha_override <= 1.0:
                raise ValueError("OPENPI_M6_ALPHA_OVERRIDE must be in [0, 1].")

    def _predict_gated(
        self,
        observation: _model.Observation,
        x_t: jax.Array,
        time: jax.Array,
        match_rng: jax.Array,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        z_match, _, _ = self._control_tokens(observation)
        language_summary = self._language_summary(observation)
        v_anchor, anchor_memory = self._predict_anchor_with_memory(
            z_match, observation, x_t, time, language_summary, match_rng, train=train
        )
        v_base = self.base.predict_flow(self._base_observation(observation), x_t, time, train=False).astype(jnp.float32)
        v_anchor = v_anchor.astype(jnp.float32)
        alpha = self.gate(anchor_memory, observation.state, x_t, time, language_summary).astype(jnp.float32)
        if self.alpha_override is not None:
            alpha = jnp.full_like(alpha, self.alpha_override)
        v_final = v_base + alpha * (v_anchor - jax.lax.stop_gradient(v_base))
        return v_final, v_base, v_anchor, alpha, language_summary

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: int | jax.Array = 0,
    ) -> jax.Array:
        del step
        observation, x_t, time, u_t, match_rng, neg_rng, mean_rng = self._sample_flow_batch(
            rng, observation, actions, train=train
        )
        v_final, v_base, v_match, alpha, language_summary = self._predict_gated(
            observation, x_t, time, match_rng, train=train
        )
        _, z_neg, z_mean = self._control_tokens(observation)
        v_neg = self._predict_anchor(z_neg, observation, x_t, time, language_summary, neg_rng, train=train)
        v_mean = self._predict_anchor(
            z_mean, observation, x_t, time, language_summary, mean_rng, train=train, perturb=False
        )

        match_mse = anchor_mse(v_match, u_t)
        neg_mse = anchor_mse(v_neg, u_t)
        mean_mse = anchor_mse(v_mean, u_t)
        final_mse = anchor_mse(v_final, u_t)
        delta_to_base = v_match.astype(jnp.float32) - jax.lax.stop_gradient(v_base).astype(jnp.float32)
        budget = jnp.mean(alpha * jnp.square(delta_to_base), axis=(-2, -1))
        per_example = (
            final_mse
            + self.beta_anchor * match_mse
            + self.lambda_rank
            * (
                jnp.maximum(0.0, self.rank_margin + match_mse - neg_mse)
                + jnp.maximum(0.0, self.rank_margin + match_mse - mean_mse)
            )
            + self.eta_budget * budget
        )
        return jnp.broadcast_to(per_example[..., None], actions.shape[:-1])

    def compute_train_metrics(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        step: int | jax.Array = 0,
    ) -> dict[str, jax.Array]:
        del step
        observation, x_t, time, u_t, match_rng, neg_rng, mean_rng = self._sample_flow_batch(
            rng, observation, actions, train=False
        )
        v_final, v_base, v_match, alpha, language_summary = self._predict_gated(
            observation, x_t, time, match_rng, train=False
        )
        _, z_neg, z_mean = self._control_tokens(observation)
        v_neg = self._predict_anchor(z_neg, observation, x_t, time, language_summary, neg_rng, train=False)
        v_mean = self._predict_anchor(
            z_mean, observation, x_t, time, language_summary, mean_rng, train=False, perturb=False
        )
        losses = compute_anchor_losses(
            v_anchor_match=v_match,
            v_anchor_neg=v_neg,
            v_anchor_mean=v_mean,
            target_flow=u_t,
            margin=self.rank_margin,
        )
        final_mse = anchor_mse(v_final, u_t)
        anchor_to_base = anchor_mse(v_match, jax.lax.stop_gradient(v_base))
        delta_to_base = v_match.astype(jnp.float32) - jax.lax.stop_gradient(v_base).astype(jnp.float32)
        budget = jnp.mean(alpha * jnp.square(delta_to_base), axis=(-2, -1))
        return {
            "m6_anchor/loss_final": jnp.mean(final_mse),
            "m6_anchor/loss_budget": jnp.mean(budget),
            "m6_anchor/loss_anchor_match": losses.loss_anchor_match,
            "m6_anchor/loss_anchor_neg": losses.loss_anchor_neg,
            "m6_anchor/loss_anchor_mean": losses.loss_anchor_mean,
            "m6_anchor/match_neg_gap": losses.match_neg_gap,
            "m6_anchor/match_mean_gap": losses.match_mean_gap,
            "m6_anchor/loss_rank_neg": losses.loss_rank_neg,
            "m6_anchor/loss_rank_mean": losses.loss_rank_mean,
            "m6_anchor/v_base_norm": jnp.mean(jnp.linalg.norm(v_base, axis=-1)),
            "m6_anchor/v_anchor_norm": jnp.mean(jnp.linalg.norm(v_match, axis=-1)),
            "m6_anchor/anchor_to_base_norm": jnp.mean(jnp.linalg.norm(v_match - v_base, axis=-1)),
            "m6_anchor/alpha_mean": jnp.mean(alpha),
            "m6_anchor/alpha_std": jnp.std(alpha),
            "m6_anchor/alpha_min": jnp.min(alpha),
            "m6_anchor/alpha_max": jnp.max(alpha),
        }

    def sample_actions(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if observation.canonical_tokens is None:
            raise ValueError("M6-B gated fusion requires observation.canonical_tokens for inference")
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        language_summary = self._language_summary(observation)
        base_obs = self._base_observation(observation)

        # Pre-compute the base prefix KV cache once — same images/language for all denoising steps.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.base.embed_prefix(base_obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.base.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, scalar_time = carry
            time = jnp.broadcast_to(scalar_time, (batch_size,))

            # Base flow: suffix-only pass with cached prefix KVs.
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.base.embed_suffix(
                base_obs, x_t, time
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.base.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                canonical_tokens=None,
            )
            v_base = self.base.action_out_proj(suffix_out[:, -self.action_horizon :]).astype(jnp.float32)

            # Anchor flow + gate.
            v_anchor, anchor_memory = self._predict_anchor_with_memory(
                observation.canonical_tokens,
                observation,
                x_t,
                time,
                language_summary,
                rng,
                train=False,
                perturb=False,
            )
            v_anchor = v_anchor.astype(jnp.float32)
            alpha = self.gate(anchor_memory, observation.state, x_t, time, language_summary).astype(jnp.float32)
            if self.alpha_override is not None:
                alpha = jnp.full_like(alpha, self.alpha_override)
            v_final = v_base + alpha * (v_anchor - v_base)
            return x_t + dt * v_final, scalar_time + dt

        def cond(carry):
            _, scalar_time = carry
            return scalar_time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
