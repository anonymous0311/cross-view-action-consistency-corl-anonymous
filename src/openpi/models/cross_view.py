"""Paired flow matching and bilateral action-flow consistency.

The policy reuses the observation prefix across flow samples and inherits
single-view action sampling from Pi0.
"""

import dataclasses

import augmax
import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.shared import array_typing as at
from openpi.shared import image_tools


def _flatten_pair_axis(x: jax.Array) -> jax.Array:
    """Flatten [B, 2, ...] into [2B, ...]."""
    if x.ndim < 2:
        raise ValueError(f"Expected a pair axis in an array with ndim >= 2, got shape {x.shape}")
    if x.shape[1] != 2:
        raise ValueError(f"Expected pair axis size 2 at axis 1, got shape {x.shape}")
    return jnp.reshape(x, (x.shape[0] * 2, *x.shape[2:]))


def _repeat_for_pair_axis(x: jax.Array | None) -> jax.Array | None:
    """Repeat a [B, ...] shared field into [2B, ...]."""
    if x is None:
        return None
    return jnp.repeat(x, repeats=2, axis=0)


def _flatten_or_repeat_pair_axis(x: jax.Array | None) -> jax.Array | None:
    """Flatten [B, 2, ...] if present, otherwise repeat shared [B, ...]."""
    if x is None:
        return None
    if x.ndim >= 2 and x.shape[1] == 2:
        return _flatten_pair_axis(x)
    return _repeat_for_pair_axis(x)


def _flatten_pair_observation_tree(observation: _model.Observation) -> _model.Observation:
    """Convert pair-batched observations into a regular flat view batch."""
    return dataclasses.replace(
        observation,
        images={key: _flatten_pair_axis(value) for key, value in observation.images.items()},
        image_masks={key: _flatten_pair_axis(value) for key, value in observation.image_masks.items()},
        state=_flatten_pair_axis(observation.state),
        tokenized_prompt=_flatten_or_repeat_pair_axis(observation.tokenized_prompt),
        tokenized_prompt_mask=_flatten_or_repeat_pair_axis(observation.tokenized_prompt_mask),
        token_ar_mask=_flatten_or_repeat_pair_axis(observation.token_ar_mask),
        token_loss_mask=_flatten_or_repeat_pair_axis(observation.token_loss_mask),
        task_index=_flatten_or_repeat_pair_axis(observation.task_index),
        episode_index=_flatten_or_repeat_pair_axis(observation.episode_index),
        sample_index=_flatten_or_repeat_pair_axis(observation.sample_index),
        camera_bin_id=_flatten_or_repeat_pair_axis(observation.camera_bin_id),
        pair_mask=None,
    )


def _random_derangement_permutation(rng: at.KeyArrayLike, batch_size: int) -> jax.Array:
    """Return a random batch permutation with no fixed points."""
    if batch_size < 2:
        raise ValueError("Clean wrong-CV control requires batch_size >= 2")
    order = jax.random.permutation(rng, batch_size)
    shifted = jnp.roll(order, shift=1)
    return jnp.zeros_like(order).at[order].set(shifted)


def _active_pair_normalize_cv_loss(
    loss_cv: jax.Array, pair_mask: jax.Array | None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mask [B,H] CV residuals and preserve per-active-pair loss strength."""
    if loss_cv.ndim != 2:
        raise ValueError(f"Expected CV loss shape [B, H], got {loss_cv.shape}")
    if pair_mask is None:
        mask = jnp.ones((loss_cv.shape[0],), dtype=loss_cv.dtype)
    else:
        mask = jnp.asarray(pair_mask, dtype=loss_cv.dtype)
        if mask.ndim != 1 or mask.shape[0] != loss_cv.shape[0]:
            raise ValueError(f"Expected pair_mask shape [{loss_cv.shape[0]}], got {mask.shape}")
    active = jnp.sum(mask)
    scale = jnp.asarray(loss_cv.shape[0], dtype=loss_cv.dtype) / jnp.maximum(active, 1.0)
    normalized = scale * mask[:, None] * loss_cv
    return (normalized, active, active / jnp.asarray(loss_cv.shape[0], dtype=loss_cv.dtype))


def preprocess_pairs(rng, observation, *, train, photometric_mode):
    """Resize without spatial augmentation; share or separate color jitter."""
    images = {}
    for key, pair in observation.images.items():
        if pair.ndim != 5 or pair.shape[1] != 2:
            raise ValueError(f"Expected [B,2,H,W,C] images, got {pair.shape}")
        shape = pair.shape
        flat = image_tools.resize_with_pad(_flatten_pair_axis(pair), 224, 224)
        pair = flat.reshape(shape[0], 2, 224, 224, 3)
        if train and photometric_mode != "none":
            _, photo_rng = jax.random.split(rng)
            chain = augmax.Chain(augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5))
            pair = pair / 2 + 0.5
            if photometric_mode == "shared":
                keys = jax.random.split(photo_rng, shape[0])
                pair = jax.vmap(lambda k, row: jax.vmap(lambda img: chain(k, img))(row))(keys, pair)
            else:
                keys = jax.random.split(photo_rng, shape[0] * 2)
                pair = jax.vmap(chain)(keys, _flatten_pair_axis(pair)).reshape(pair.shape)
            pair = pair * 2 - 1
        images[key] = pair
    return _flatten_pair_observation_tree(dataclasses.replace(observation, images=images))


def flow_losses(predictions, targets, *, action_dim, anchor_indices=None, time=None):
    """Return [B,H] losses, averaging FM over padded and CV over commanded dimensions.

    Predictions/targets: [K,B,2,H,A]. Shuffling changes only the consistency
    partners; flow-matching targets and inputs remain attached to their own rows.
    """
    fm_views = jnp.mean(jnp.square(predictions - targets), axis=-1).mean(axis=0)
    anchor = predictions[:, :, 0, :, :action_dim]
    student = predictions[:, :, 1, :, :action_dim]
    if anchor_indices is not None:
        anchor = anchor[:, anchor_indices]
    residual = jnp.mean(jnp.square(student - anchor), axis=-1)
    if time is not None:
        residual = residual * time[..., None] ** 2
    return (fm_views.mean(axis=1), residual.mean(axis=0), fm_views)


class CrossViewPolicy(pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.cv_config = config

    def _lambda_cv(self, step):
        cfg = self.cv_config
        return cfg.lambda_cv * jnp.clip((jnp.asarray(step) - cfg.cv_warmup_start) / max(cfg.cv_warmup_steps, 1), 0, 1)

    def _predict_flow_samples_from_prefix_cache(
        self, observation: _model.Observation, noisy_action_samples: jax.Array, time_samples: jax.Array, *, train: bool
    ) -> jax.Array:
        """Predict K flow samples while reusing the image/language prefix cache.

        ``noisy_action_samples`` has shape [K, 2B, H, A] and ``time_samples``
        has shape [K, 2B]. The prefix is independent of K, so this avoids
        repeating the SigLIP/PaliGemma prefix trunk for every flow sample.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        outputs = []
        for sample_idx in range(noisy_action_samples.shape[0]):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, noisy_action_samples[sample_idx], time_samples[sample_idx]
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            outputs.append(self.action_out_proj(suffix_out[:, -self.action_horizon :]))
        return jnp.stack(outputs, axis=0)

    def loss_components(self, rng, observation, actions, *, train=False, step=0):
        if actions.ndim != 4 or actions.shape[1] != 2:
            raise ValueError(f"Expected paired actions [B,2,H,A], got {actions.shape}")
        cfg = self.cv_config
        if cfg.observation_mode != "paired":
            flat = preprocess_pairs(rng, observation, train=train, photometric_mode=cfg.photometric_mode)
            flat_actions = _flatten_pair_axis(actions)
            if cfg.observation_mode == "nominal":
                flat = jax.tree.map(lambda value: value[::2], flat)
                flat_actions = flat_actions[::2]
            loss = super().compute_loss(rng, flat, flat_actions, train=False, step=step)
            return {"total": loss, "flow_matching": loss}
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        anchors = None
        if cfg.shuffle_pairs:
            pairing_rng, noise_rng = jax.random.split(noise_rng)
            anchors = _random_derangement_permutation(pairing_rng, actions.shape[0])
        flat_obs = preprocess_pairs(preprocess_rng, observation, train=train, photometric_mode=cfg.photometric_mode)
        action = actions[:, 0]
        k, b = (cfg.num_flow_samples, action.shape[0])
        noise = jax.random.normal(noise_rng, (k, *action.shape))
        time = jax.random.beta(time_rng, cfg.time_beta_alpha, cfg.time_beta_beta, (k, b)) * 0.999 + 0.001
        x = time[..., None, None] * noise + (1 - time[..., None, None]) * action[None]
        target = noise - action[None]
        x_pair = jnp.repeat(x[:, :, None], 2, axis=2).reshape(k, 2 * b, *action.shape[-2:])
        time_pair = jnp.repeat(time[:, :, None], 2, axis=2).reshape(k, 2 * b)
        pred = self._predict_flow_samples_from_prefix_cache(flat_obs, x_pair, time_pair, train=train)
        pred = pred.reshape(k, b, 2, *action.shape[-2:])
        targets = jnp.repeat(target[:, :, None], 2, axis=2)
        fm, cv, fm_views = flow_losses(
            pred,
            targets,
            action_dim=cfg.cv_action_dim,
            anchor_indices=anchors,
            time=time if cfg.action_space_weighting else None,
        )
        cv, active, fraction = _active_pair_normalize_cv_loss(cv, observation.pair_mask)
        weight = self._lambda_cv(step)
        return {
            "total": fm + weight * cv,
            "flow_matching": fm,
            "consistency": cv,
            "nominal_fm": fm_views[:, 0],
            "perturbed_fm": fm_views[:, 1],
            "lambda_cv": weight,
            "active_pairs": active,
            "active_fraction": fraction,
        }

    def compute_loss(self, rng, observation, actions, *, train=False, step=0):
        return self.loss_components(rng, observation, actions, train=train, step=step)["total"]

    def compute_train_metrics(self, rng, observation, actions, *, step=0):
        return {
            f"cross_view/{key}": jnp.mean(value)
            for key, value in self.loss_components(rng, observation, actions, step=step).items()
        }
