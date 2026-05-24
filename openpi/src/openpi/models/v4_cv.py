import dataclasses

import augmax
import einops
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import v4_cv_config
from openpi.shared import array_typing as at
from openpi.shared import image_tools

CV_PAIR_MODE_MATCHED = "matched"
CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT = "clean_wrong_batch_derangement"
CV_LOSS_MODE_SYMMETRIC = "symmetric"
CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC = "multi_sample_asymmetric"
CV_VIEW_NOMINAL = "nominal"
CV_VIEW_PERTURBED = "perturbed"
CV_TIME_DISTRIBUTION_LEGACY = "legacy"
CV_TIME_DISTRIBUTION_UNIFORM = "uniform"
CV_TIME_DISTRIBUTION_BETA_1P5_1 = "beta_1p5_1"
CV_TIME_DISTRIBUTION_BETA_2P0_3P0 = "beta_2p0_3p0"
CV_TIME_DISTRIBUTION_BETA_1P0_1P5 = "beta_1p0_1p5"
PAIR_SPATIAL_AUG_MODE_CURRENT = "current"
PAIR_SPATIAL_AUG_MODE_NONE = "none"
PAIR_SPATIAL_AUG_MODE_SHARED = "shared"
PAIR_PHOTOMETRIC_AUG_MODE_CURRENT = "current"
PAIR_PHOTOMETRIC_AUG_MODE_NONE = "none"
PAIR_PHOTOMETRIC_AUG_MODE_SHARED = "shared"
PAIR_PHOTOMETRIC_AUG_MODE_INDEPENDENT = "independent"


def _linear_ramp_lambda(
    lambda_max: float,
    step: int | jax.Array,
    *,
    total_steps: int,
    start_fraction: float,
    end_fraction: float,
) -> jax.Array:
    """0 until start, linear ramp to lambda_max, then constant."""
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    progress = jnp.clip(step_f / total, 0.0, 1.0)
    start = jnp.asarray(start_fraction, dtype=jnp.float32)
    end = jnp.asarray(end_fraction, dtype=jnp.float32)
    denom = jnp.maximum(end - start, 1e-6)
    ramp = jnp.clip((progress - start) / denom, 0.0, 1.0)
    return jnp.asarray(lambda_max, dtype=jnp.float32) * ramp


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
        canonical_tokens=_flatten_pair_axis(observation.canonical_tokens)
        if observation.canonical_tokens is not None
        else None,
        canonical_tokens_neg=_flatten_pair_axis(observation.canonical_tokens_neg)
        if observation.canonical_tokens_neg is not None
        else None,
        canonical_tokens_mean=_flatten_pair_axis(observation.canonical_tokens_mean)
        if observation.canonical_tokens_mean is not None
        else None,
        task_index=_flatten_or_repeat_pair_axis(observation.task_index),
        episode_index=_flatten_or_repeat_pair_axis(observation.episode_index),
        sample_index=_flatten_or_repeat_pair_axis(observation.sample_index),
        camera_bin_id=_flatten_or_repeat_pair_axis(observation.camera_bin_id),
    )


def _replace_second_pair_slot(x: jax.Array | None, perm: jax.Array) -> jax.Array | None:
    """Keep slot 0 and replace slot 1 with slot 1 from permuted rows."""
    if x is None:
        return None
    if x.ndim < 2 or x.shape[1] != 2:
        return x
    return jnp.concatenate([x[:, :1], x[perm, 1:2]], axis=1)


def _random_derangement_permutation(rng: at.KeyArrayLike, batch_size: int) -> jax.Array:
    """Return a random batch permutation with no fixed points."""
    if batch_size < 2:
        raise ValueError("Clean wrong-CV control requires batch_size >= 2")
    order = jax.random.permutation(rng, batch_size)
    shifted = jnp.roll(order, shift=1)
    return jnp.zeros_like(order).at[order].set(shifted)


def _apply_clean_wrong_pairing(
    observation: _model.Observation,
    actions: _model.Actions,
    perm: jax.Array,
) -> tuple[_model.Observation, _model.Actions]:
    """Derange only the second pair slot while preserving each view's own labels."""
    return (
        dataclasses.replace(
            observation,
            images={key: _replace_second_pair_slot(value, perm) for key, value in observation.images.items()},
            image_masks={key: _replace_second_pair_slot(value, perm) for key, value in observation.image_masks.items()},
            state=_replace_second_pair_slot(observation.state, perm),
            tokenized_prompt=_replace_second_pair_slot(observation.tokenized_prompt, perm),
            tokenized_prompt_mask=_replace_second_pair_slot(observation.tokenized_prompt_mask, perm),
            token_ar_mask=_replace_second_pair_slot(observation.token_ar_mask, perm),
            token_loss_mask=_replace_second_pair_slot(observation.token_loss_mask, perm),
            canonical_tokens=_replace_second_pair_slot(observation.canonical_tokens, perm),
            canonical_tokens_neg=_replace_second_pair_slot(observation.canonical_tokens_neg, perm),
            canonical_tokens_mean=_replace_second_pair_slot(observation.canonical_tokens_mean, perm),
            task_index=_replace_second_pair_slot(observation.task_index, perm),
            episode_index=_replace_second_pair_slot(observation.episode_index, perm),
            sample_index=_replace_second_pair_slot(observation.sample_index, perm),
            camera_bin_id=_replace_second_pair_slot(observation.camera_bin_id, perm),
        ),
        _replace_second_pair_slot(actions, perm),
    )


def _validate_pair_aug_modes(pair_spatial_aug_mode: str, pair_photometric_aug_mode: str) -> None:
    spatial_modes = {
        PAIR_SPATIAL_AUG_MODE_CURRENT,
        PAIR_SPATIAL_AUG_MODE_NONE,
        PAIR_SPATIAL_AUG_MODE_SHARED,
    }
    photometric_modes = {
        PAIR_PHOTOMETRIC_AUG_MODE_CURRENT,
        PAIR_PHOTOMETRIC_AUG_MODE_NONE,
        PAIR_PHOTOMETRIC_AUG_MODE_SHARED,
        PAIR_PHOTOMETRIC_AUG_MODE_INDEPENDENT,
    }
    if pair_spatial_aug_mode not in spatial_modes:
        raise ValueError(f"Unknown pair_spatial_aug_mode: {pair_spatial_aug_mode}")
    if pair_photometric_aug_mode not in photometric_modes:
        raise ValueError(f"Unknown pair_photometric_aug_mode: {pair_photometric_aug_mode}")


def _validate_cv_loss_config(
    *,
    cv_loss_mode: str,
    cv_num_samples: int,
    cv_anchor_view: str,
    cv_student_view: str,
    cv_time_distribution: str,
) -> None:
    if cv_loss_mode not in {CV_LOSS_MODE_SYMMETRIC, CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC}:
        raise ValueError(f"Unknown cv_loss_mode: {cv_loss_mode}")
    if cv_num_samples < 1:
        raise ValueError(f"cv_num_samples must be >= 1, got {cv_num_samples}")
    if cv_anchor_view != CV_VIEW_NOMINAL:
        raise ValueError(f"Only nominal anchor view is supported, got {cv_anchor_view}")
    if cv_student_view != CV_VIEW_PERTURBED:
        raise ValueError(f"Only perturbed student view is supported, got {cv_student_view}")
    if cv_time_distribution not in {
        CV_TIME_DISTRIBUTION_LEGACY,
        CV_TIME_DISTRIBUTION_UNIFORM,
        CV_TIME_DISTRIBUTION_BETA_1P5_1,
        CV_TIME_DISTRIBUTION_BETA_2P0_3P0,
        CV_TIME_DISTRIBUTION_BETA_1P0_1P5,
    }:
        raise ValueError(f"Unknown cv_time_distribution: {cv_time_distribution}")


def _sample_cv_time(
    rng: at.KeyArrayLike,
    distribution: str,
    shape: tuple[int, ...],
) -> jax.Array:
    if distribution in (CV_TIME_DISTRIBUTION_LEGACY, CV_TIME_DISTRIBUTION_BETA_1P5_1):
        time = jax.random.beta(rng, 1.5, 1, shape)
    elif distribution == CV_TIME_DISTRIBUTION_BETA_2P0_3P0:
        time = jax.random.beta(rng, 2.0, 3.0, shape)
    elif distribution == CV_TIME_DISTRIBUTION_BETA_1P0_1P5:
        time = jax.random.beta(rng, 1.0, 1.5, shape)
    elif distribution == CV_TIME_DISTRIBUTION_UNIFORM:
        time = jax.random.uniform(rng, shape)
    else:
        raise ValueError(f"Unknown cv_time_distribution: {distribution}")
    return time * 0.999 + 0.001


def _flatten_sample_pair_axis(x: jax.Array) -> jax.Array:
    """Flatten [K, B, 2, ...] into [K, 2B, ...]."""
    if x.ndim < 3:
        raise ValueError(f"Expected sample and pair axes in an array with ndim >= 3, got shape {x.shape}")
    if x.shape[2] != 2:
        raise ValueError(f"Expected pair axis size 2 at axis 2, got shape {x.shape}")
    return jnp.reshape(x, (x.shape[0], x.shape[1] * 2, *x.shape[3:]))


def _aggregate_cv_samples(x: jax.Array, *, average_over_samples: bool) -> jax.Array:
    if average_over_samples:
        return jnp.mean(x, axis=0)
    return jnp.sum(x, axis=0)


def _multi_sample_asymmetric_cv_loss(
    v_t: jax.Array,
    *,
    cv_dim: int,
    anchor_indices: jax.Array | None,
    stopgrad_anchor: bool,
) -> jax.Array:
    """Return per-sample, per-row, per-horizon asymmetric CV loss [K, B, H]."""
    cv_dim = min(cv_dim, v_t.shape[-1])
    anchor = v_t[:, :, 0, :, :cv_dim]
    student = v_t[:, :, 1, :, :cv_dim]
    if anchor_indices is not None:
        anchor = anchor[:, anchor_indices, :, :]
    if stopgrad_anchor:
        anchor = jax.lax.stop_gradient(anchor)
    return jnp.mean(jnp.square(student - anchor), axis=-1)


def _resize_pair_image_if_needed(
    image: jax.Array,
    image_resolution: tuple[int, int],
) -> jax.Array:
    if image.ndim != 5 or image.shape[1] != 2:
        raise ValueError(f"Expected pair image shape [B, 2, H, W, C], got {image.shape}")
    if image.shape[2:4] == image_resolution:
        return image
    flat = _flatten_pair_axis(image)
    flat = image_tools.resize_with_pad(flat, *image_resolution)
    return jnp.reshape(flat, (image.shape[0], 2, *flat.shape[1:]))


def _apply_pair_aug_chain(
    rng: at.KeyArrayLike,
    image: jax.Array,
    transforms: list,
    *,
    mode: str,
) -> jax.Array:
    if not transforms:
        return image

    chain = augmax.Chain(*transforms)
    if mode in (PAIR_SPATIAL_AUG_MODE_SHARED, PAIR_PHOTOMETRIC_AUG_MODE_SHARED):
        row_rngs = jax.random.split(rng, image.shape[0])

        def augment_row(row_rng, row):
            return jax.vmap(lambda slot_image: chain(row_rng, slot_image))(row)

        return jax.vmap(augment_row)(row_rngs, image)

    flat = _flatten_pair_axis(image)
    sub_rngs = jax.random.split(rng, flat.shape[0])
    flat = jax.vmap(chain)(sub_rngs, flat)
    return jnp.reshape(flat, image.shape)


def preprocess_pair_observation(
    rng: at.KeyArrayLike | None,
    observation: _model.Observation,
    *,
    train: bool,
    image_keys: tuple[str, ...],
    pair_spatial_aug_mode: str,
    pair_photometric_aug_mode: str,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> _model.Observation:
    """Preprocess a pair-batched observation and return a flattened view batch.

    ``current/current`` intentionally delegates to the legacy OpenPI path:
    flatten first, then call ``preprocess_observation``. Other modes keep the
    pair axis through augmentation so spatial and photometric randomness can be
    disabled or shared across nominal/perturbed slots.
    """
    _validate_pair_aug_modes(pair_spatial_aug_mode, pair_photometric_aug_mode)

    if (
        pair_spatial_aug_mode == PAIR_SPATIAL_AUG_MODE_CURRENT
        and pair_photometric_aug_mode == PAIR_PHOTOMETRIC_AUG_MODE_CURRENT
    ):
        observation_flat = _flatten_pair_observation_tree(observation)
        return _model.preprocess_observation(
            rng,
            observation_flat,
            train=train,
            image_keys=image_keys,
            image_resolution=image_resolution,
        )

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    out_images = {}
    for key in image_keys:
        image = _resize_pair_image_if_needed(observation.images[key], image_resolution)

        if train:
            if rng is None:
                raise ValueError("rng is required when train=True")
            spatial_rng, photometric_rng = jax.random.split(rng)
            image = image / 2.0 + 0.5

            if "wrist" not in key and pair_spatial_aug_mode != PAIR_SPATIAL_AUG_MODE_NONE:
                height, width = image.shape[2:4]
                image = _apply_pair_aug_chain(
                    spatial_rng,
                    image,
                    [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ],
                    mode=pair_spatial_aug_mode,
                )

            if pair_photometric_aug_mode != PAIR_PHOTOMETRIC_AUG_MODE_NONE:
                photo_mode = (
                    PAIR_PHOTOMETRIC_AUG_MODE_INDEPENDENT
                    if pair_photometric_aug_mode == PAIR_PHOTOMETRIC_AUG_MODE_CURRENT
                    else pair_photometric_aug_mode
                )
                image = _apply_pair_aug_chain(
                    photometric_rng,
                    image,
                    [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)],
                    mode=photo_mode,
                )

            image = image * 2.0 - 1.0

        out_images[key] = image

    batch_shape = observation.state.shape[:-1]
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return _flatten_pair_observation_tree(
        dataclasses.replace(observation, images=out_images, image_masks=out_masks)
    )


class Pi0V4CV(pi0.Pi0):
    """Vanilla pi0 with Phase 0B cross-view action-flow consistency loss."""

    def __init__(self, config: v4_cv_config.Pi0V4CVConfig, rngs):
        super().__init__(config, rngs)
        self.lambda_cv = config.lambda_cv
        self.total_train_steps = config.total_train_steps
        self.cv_warmup_start_fraction = config.cv_warmup_start_fraction
        self.cv_warmup_end_fraction = config.cv_warmup_end_fraction
        self.cv_action_dim = config.cv_action_dim
        self.cv_pair_mode = config.cv_pair_mode
        self.cv_loss_mode = config.cv_loss_mode
        self.cv_num_samples = config.cv_num_samples
        self.cv_anchor_view = config.cv_anchor_view
        self.cv_student_view = config.cv_student_view
        self.cv_stopgrad_anchor = config.cv_stopgrad_anchor
        self.cv_time_distribution = config.cv_time_distribution
        self.cv_eps_shared_across_views = config.cv_eps_shared_across_views
        self.cv_average_over_samples = config.cv_average_over_samples
        self.pair_spatial_aug_mode = config.pair_spatial_aug_mode
        self.pair_photometric_aug_mode = config.pair_photometric_aug_mode
        _validate_pair_aug_modes(self.pair_spatial_aug_mode, self.pair_photometric_aug_mode)
        _validate_cv_loss_config(
            cv_loss_mode=self.cv_loss_mode,
            cv_num_samples=self.cv_num_samples,
            cv_anchor_view=self.cv_anchor_view,
            cv_student_view=self.cv_student_view,
            cv_time_distribution=self.cv_time_distribution,
        )

    def _lambda_cv(self, step: int | jax.Array) -> jax.Array:
        return _linear_ramp_lambda(
            self.lambda_cv,
            step,
            total_steps=self.total_train_steps,
            start_fraction=self.cv_warmup_start_fraction,
            end_fraction=self.cv_warmup_end_fraction,
        )

    def _flatten_pair_observation(self, observation: _model.Observation) -> _model.Observation:
        """Convert pair-batched observations into a regular flat view batch.

        Image/state/mask leaves carry a pair axis [B, 2, ...]. Prompt and
        metadata leaves are shared per pair [B, ...] and are repeated so both
        views receive identical language/state-token context.
        """
        return _flatten_pair_observation_tree(observation)

    def _predict_flow_samples_from_prefix_cache(
        self,
        observation: _model.Observation,
        noisy_action_samples: jax.Array,
        time_samples: jax.Array,
        *,
        train: bool,
        canonical_drop_rng: at.KeyArrayLike | None = None,
    ) -> jax.Array:
        """Predict K flow samples while reusing the image/language prefix cache.

        ``noisy_action_samples`` has shape [K, 2B, H, A] and ``time_samples``
        has shape [K, 2B]. The prefix is independent of K, so this avoids
        repeating the SigLIP/PaliGemma prefix trunk for every flow sample.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        canonical_tokens = observation.canonical_tokens
        if train and self.canonical_token_dropout > 0.0 and canonical_tokens is not None:
            if canonical_drop_rng is None:
                raise ValueError("canonical_drop_rng is required when canonical_token_dropout is enabled")
            batch_size = canonical_tokens.shape[0]
            keep = jax.random.uniform(canonical_drop_rng, (batch_size, 1, 1)) >= self.canonical_token_dropout
            canonical_tokens = canonical_tokens * keep

        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

        outputs = []
        for sample_idx in range(noisy_action_samples.shape[0]):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                noisy_action_samples[sample_idx],
                time_samples[sample_idx],
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
                canonical_tokens=canonical_tokens,
            )
            outputs.append(self.action_out_proj(suffix_out[:, -self.action_horizon :]))

        return jnp.stack(outputs, axis=0)

    def _pair_loss_components(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
        step: int | jax.Array,
    ) -> dict[str, jax.Array]:
        if actions.ndim != 4 or actions.shape[1] != 2:
            raise ValueError(
                "Pi0V4CV expects paired actions with shape [B, 2, action_horizon, action_dim], "
                f"got {actions.shape}"
            )

        cv_pair_mode = getattr(self, "cv_pair_mode", CV_PAIR_MODE_MATCHED)
        cv_loss_mode = getattr(self, "cv_loss_mode", CV_LOSS_MODE_SYMMETRIC)
        if cv_pair_mode not in {CV_PAIR_MODE_MATCHED, CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT}:
            raise ValueError(f"Unknown cv_pair_mode: {cv_pair_mode}")

        if cv_loss_mode == CV_LOSS_MODE_SYMMETRIC:
            if cv_pair_mode == CV_PAIR_MODE_MATCHED:
                preprocess_rng, noise_rng, time_rng, canonical_drop_rng = jax.random.split(rng, 4)
                observation_loss = observation
                actions_loss = actions
                anchor_indices = jnp.arange(actions.shape[0])
            else:
                preprocess_rng, noise_rng, time_rng, canonical_drop_rng, pairing_rng = jax.random.split(rng, 5)
                anchor_indices = _random_derangement_permutation(pairing_rng, actions.shape[0])
                observation_loss, actions_loss = _apply_clean_wrong_pairing(observation, actions, anchor_indices)

            observation_flat = preprocess_pair_observation(
                preprocess_rng,
                observation_loss,
                train=train,
                image_keys=tuple(observation_loss.images),
                pair_spatial_aug_mode=getattr(self, "pair_spatial_aug_mode", PAIR_SPATIAL_AUG_MODE_CURRENT),
                pair_photometric_aug_mode=getattr(
                    self, "pair_photometric_aug_mode", PAIR_PHOTOMETRIC_AUG_MODE_CURRENT
                ),
            )

            actions_ref = actions_loss[:, 0]
            batch_shape = actions_ref.shape[:-2]
            noise = jax.random.normal(noise_rng, actions_ref.shape)
            time = _sample_cv_time(
                time_rng,
                getattr(self, "cv_time_distribution", CV_TIME_DISTRIBUTION_LEGACY),
                batch_shape,
            )
            time_expanded = time[..., None, None, None]
            x_t_pair = time_expanded * noise[:, None, :, :] + (1 - time_expanded) * actions_loss
            u_t_pair = noise[:, None, :, :] - actions_loss

            time_pair = jnp.repeat(time[:, None], repeats=2, axis=1)
            x_t_flat = _flatten_pair_axis(x_t_pair)
            time_flat = _flatten_pair_axis(time_pair)

            v_t_flat = self.predict_flow(
                observation_flat,
                x_t_flat,
                time_flat,
                train=train,
                canonical_drop_rng=canonical_drop_rng,
            )
            v_t = jnp.reshape(v_t_flat, (actions_ref.shape[0], 2, *actions_ref.shape[-2:]))

            fm_per_view = jnp.mean(jnp.square(v_t - u_t_pair), axis=-1)
            loss_fm = jnp.mean(fm_per_view, axis=1)

            cv_dim = min(self.cv_action_dim, actions_ref.shape[-1])
            loss_cv = jnp.mean(jnp.square(v_t[:, 0, :, :cv_dim] - v_t[:, 1, :, :cv_dim]), axis=-1)
            action_norm = jnp.linalg.norm(v_t[:, :, :, :cv_dim], axis=-1).mean(axis=1)
            num_samples = 1
        elif cv_loss_mode == CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC:
            preprocess_rng, noise_rng, time_rng, canonical_drop_rng = jax.random.split(rng, 4)
            if cv_pair_mode == CV_PAIR_MODE_MATCHED:
                anchor_indices = jnp.arange(actions.shape[0])
            else:
                pairing_rng, noise_rng = jax.random.split(noise_rng)
                anchor_indices = _random_derangement_permutation(pairing_rng, actions.shape[0])

            observation_flat = preprocess_pair_observation(
                preprocess_rng,
                observation,
                train=train,
                image_keys=tuple(observation.images),
                pair_spatial_aug_mode=getattr(self, "pair_spatial_aug_mode", PAIR_SPATIAL_AUG_MODE_CURRENT),
                pair_photometric_aug_mode=getattr(
                    self, "pair_photometric_aug_mode", PAIR_PHOTOMETRIC_AUG_MODE_CURRENT
                ),
            )

            actions_ref = actions[:, 0]
            batch_size = actions_ref.shape[0]
            batch_shape = actions_ref.shape[:-2]
            num_samples = int(getattr(self, "cv_num_samples", 1))
            time_distribution = getattr(self, "cv_time_distribution", CV_TIME_DISTRIBUTION_LEGACY)
            eps_shared = getattr(self, "cv_eps_shared_across_views", True)

            if eps_shared:
                noise = jax.random.normal(noise_rng, (num_samples, *actions_ref.shape))
                time = _sample_cv_time(time_rng, time_distribution, (num_samples, *batch_shape))
                time_expanded = time[..., None, None]
                x_t = time_expanded * noise + (1 - time_expanded) * actions_ref[None, ...]
                u_t = noise - actions_ref[None, ...]
                x_t_pair = jnp.repeat(x_t[:, :, None, :, :], repeats=2, axis=2)
                u_t_pair = jnp.repeat(u_t[:, :, None, :, :], repeats=2, axis=2)
                time_pair = jnp.repeat(time[:, :, None], repeats=2, axis=2)
            else:
                noise = jax.random.normal(noise_rng, (num_samples, batch_size, 2, *actions_ref.shape[-2:]))
                time = _sample_cv_time(time_rng, time_distribution, (num_samples, batch_size, 2))
                actions_pair = jnp.repeat(actions_ref[None, :, None, :, :], repeats=2, axis=2)
                time_expanded = time[..., None, None]
                x_t_pair = time_expanded * noise + (1 - time_expanded) * actions_pair
                u_t_pair = noise - actions_pair
                time_pair = time

            x_t_flat_samples = _flatten_sample_pair_axis(x_t_pair)
            time_flat_samples = _flatten_sample_pair_axis(time_pair[..., None])[..., 0]

            predict_flow_samples = getattr(self, "_predict_flow_samples_from_prefix_cache", None)
            if predict_flow_samples is None:
                v_t_flat_samples = jnp.stack(
                    [
                        self.predict_flow(
                            observation_flat,
                            x_t_flat_samples[sample_idx],
                            time_flat_samples[sample_idx],
                            train=train,
                            canonical_drop_rng=canonical_drop_rng,
                        )
                        for sample_idx in range(num_samples)
                    ],
                    axis=0,
                )
            else:
                v_t_flat_samples = predict_flow_samples(
                    observation_flat,
                    x_t_flat_samples,
                    time_flat_samples,
                    train=train,
                    canonical_drop_rng=canonical_drop_rng,
                )

            v_t = jnp.reshape(v_t_flat_samples, (num_samples, batch_size, 2, *actions_ref.shape[-2:]))

            fm_per_view_samples = jnp.mean(jnp.square(v_t - u_t_pair), axis=-1)
            loss_fm_samples = jnp.mean(fm_per_view_samples, axis=2)
            average_over_samples = getattr(self, "cv_average_over_samples", True)
            loss_fm = _aggregate_cv_samples(loss_fm_samples, average_over_samples=average_over_samples)
            loss_fm_nominal = _aggregate_cv_samples(
                fm_per_view_samples[:, :, 0],
                average_over_samples=average_over_samples,
            )
            loss_fm_perturbed = _aggregate_cv_samples(
                fm_per_view_samples[:, :, 1],
                average_over_samples=average_over_samples,
            )

            cv_dim = min(self.cv_action_dim, actions_ref.shape[-1])
            loss_cv_samples = _multi_sample_asymmetric_cv_loss(
                v_t,
                cv_dim=cv_dim,
                anchor_indices=anchor_indices,
                stopgrad_anchor=getattr(self, "cv_stopgrad_anchor", True),
            )
            loss_cv = _aggregate_cv_samples(loss_cv_samples, average_over_samples=average_over_samples)
            action_norm = _aggregate_cv_samples(
                jnp.linalg.norm(v_t[:, :, :, :, :cv_dim], axis=-1).mean(axis=2),
                average_over_samples=average_over_samples,
            )
            fm_per_view = jnp.stack([loss_fm_nominal, loss_fm_perturbed], axis=1)
        else:
            raise ValueError(f"Unknown cv_loss_mode: {cv_loss_mode}")

        lambda_cv = self._lambda_cv(step)
        total = loss_fm + lambda_cv * loss_cv

        return {
            "total": total,
            "loss_fm": loss_fm,
            "loss_fm_nominal": fm_per_view[:, 0],
            "loss_fm_perturbed": fm_per_view[:, 1],
            "loss_cv": loss_cv,
            "lambda_cv": lambda_cv,
            "flow_disagreement": jnp.sqrt(loss_cv + 1e-8),
            "action_norm": action_norm,
            "clean_wrong_cv_active": jnp.asarray(
                cv_pair_mode == CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT,
                dtype=jnp.float32,
            ),
            "multi_sample_asymmetric_cv_active": jnp.asarray(
                cv_loss_mode == CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC,
                dtype=jnp.float32,
            ),
            "cv_num_samples": jnp.asarray(num_samples, dtype=jnp.float32),
            "cv_anchor_indices": anchor_indices,
        }

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: int | jax.Array = 0,
    ) -> at.Float[at.Array, "b ah"]:
        return self._pair_loss_components(rng, observation, actions, train=train, step=step)["total"]

    def compute_train_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        step: int | jax.Array = 0,
    ) -> dict[str, jax.Array]:
        components = self._pair_loss_components(rng, observation, actions, train=False, step=step)
        return {
            "v4/loss_fm": jnp.mean(components["loss_fm"]),
            "v4/loss_fm_nominal": jnp.mean(components["loss_fm_nominal"]),
            "v4/loss_fm_perturbed": jnp.mean(components["loss_fm_perturbed"]),
            "v4/loss_cv": jnp.mean(components["loss_cv"]),
            "v4/lambda_cv": components["lambda_cv"],
            "v4/flow_disagreement": jnp.mean(components["flow_disagreement"]),
            "v4/action_flow_norm": jnp.mean(components["action_norm"]),
            "v4/clean_wrong_cv_active": components["clean_wrong_cv_active"],
            "v4/multi_sample_asymmetric_cv_active": components["multi_sample_asymmetric_cv_active"],
            "v4/cv_num_samples": components["cv_num_samples"],
        }
