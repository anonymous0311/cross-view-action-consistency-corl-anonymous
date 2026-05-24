import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import cross_view_action_consistency


class _FakeCrossViewModel:
    cv_action_dim = 7
    cv_pair_mode = cross_view_action_consistency.CV_PAIR_MODE_MATCHED
    cv_loss_mode = cross_view_action_consistency.CV_LOSS_MODE_SYMMETRIC
    cv_num_samples = 1
    cv_stopgrad_anchor = True
    cv_time_distribution = cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_1P5_1
    cv_eps_shared_across_views = True
    cv_average_over_samples = True
    pair_spatial_aug_mode = cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_CURRENT
    pair_photometric_aug_mode = cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_CURRENT

    def _lambda_cv(self, step):
        del step
        return jnp.asarray(0.05, dtype=jnp.float32)

    _flatten_pair_observation = cross_view_action_consistency.Pi0CrossViewActionConsistency._flatten_pair_observation

    def predict_flow(self, observation, noisy_actions, time, *, train=False, canonical_drop_rng=None):
        del observation, train, canonical_drop_rng
        noisy_pair = np.asarray(noisy_actions).reshape(2, 2, 3, 8)
        time_pair = np.asarray(time).reshape(2, 2)
        np.testing.assert_allclose(noisy_pair[:, 0], noisy_pair[:, 1], rtol=0, atol=0)
        np.testing.assert_allclose(time_pair[:, 0], time_pair[:, 1], rtol=0, atol=0)
        return jnp.zeros_like(noisy_actions)


class _FakeCleanWrongCrossViewModel(_FakeCrossViewModel):
    cv_pair_mode = cross_view_action_consistency.CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT

    def predict_flow(self, observation, noisy_actions, time, *, train=False, canonical_drop_rng=None):
        del observation, train, canonical_drop_rng
        time_pair = np.asarray(time).reshape(4, 2)
        np.testing.assert_allclose(time_pair[:, 0], time_pair[:, 1], rtol=0, atol=0)
        return jnp.zeros_like(noisy_actions)


class _FakeMultiSampleAsymmetricCrossViewModel(_FakeCrossViewModel):
    cv_loss_mode = cross_view_action_consistency.CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC
    cv_num_samples = 2

    def __init__(self):
        self.calls = []

    def predict_flow(self, observation, noisy_actions, time, *, train=False, canonical_drop_rng=None):
        del observation, train, canonical_drop_rng
        batch_pairs = noisy_actions.shape[0] // 2
        noisy_pair = np.asarray(noisy_actions).reshape(batch_pairs, 2, *noisy_actions.shape[1:])
        time_pair = np.asarray(time).reshape(batch_pairs, 2)
        np.testing.assert_allclose(noisy_pair[:, 0], noisy_pair[:, 1], rtol=0, atol=0)
        np.testing.assert_allclose(time_pair[:, 0], time_pair[:, 1], rtol=0, atol=0)
        self.calls.append(time_pair.copy())
        flow = np.zeros_like(noisy_pair)
        flow[:, 0, :, :7] = 1.0
        flow[:, 1, :, :7] = 2.0
        return jnp.asarray(flow.reshape(noisy_actions.shape))


class _FakeCleanWrongMultiSampleAsymmetricCrossViewModel(_FakeMultiSampleAsymmetricCrossViewModel):
    cv_pair_mode = cross_view_action_consistency.CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT

    def predict_flow(self, observation, noisy_actions, time, *, train=False, canonical_drop_rng=None):
        state_pair = np.asarray(observation.state).reshape(4, 2, 8)
        np.testing.assert_array_equal(state_pair[:, 0, 0], np.arange(4) * 16)
        np.testing.assert_array_equal(state_pair[:, 1, 0], np.arange(4) * 16 + 8)
        return super().predict_flow(
            observation,
            noisy_actions,
            time,
            train=train,
            canonical_drop_rng=canonical_drop_rng,
        )


def _make_pair_observation(batch_size: int = 2, *, include_wrist: bool = True) -> _model.Observation:
    images = {
        "base_0_rgb": jnp.zeros((batch_size, 2, 224, 224, 3), dtype=jnp.float32),
    }
    image_masks = {
        "base_0_rgb": jnp.ones((batch_size, 2), dtype=jnp.bool_),
    }
    if include_wrist:
        images["left_wrist_0_rgb"] = jnp.zeros((batch_size, 2, 224, 224, 3), dtype=jnp.float32)
        images["right_wrist_0_rgb"] = jnp.zeros((batch_size, 2, 224, 224, 3), dtype=jnp.float32)
        image_masks["left_wrist_0_rgb"] = jnp.zeros((batch_size, 2), dtype=jnp.bool_)
        image_masks["right_wrist_0_rgb"] = jnp.zeros((batch_size, 2), dtype=jnp.bool_)
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((batch_size, 2, 8), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((batch_size, 2, 16), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 2, 16), dtype=jnp.bool_),
        task_index=jnp.repeat(jnp.arange(batch_size)[:, None], repeats=2, axis=1),
        episode_index=jnp.repeat((jnp.arange(batch_size) + 10)[:, None], repeats=2, axis=1),
        sample_index=jnp.repeat((jnp.arange(batch_size) + 20)[:, None], repeats=2, axis=1),
    )


def test_cross_view_lambda_cv_linear_ramp():
    np.testing.assert_allclose(
        cross_view_action_consistency._linear_ramp_lambda(0.1, 0, total_steps=100, start_fraction=0.1, end_fraction=0.3),
        0.0,
    )
    np.testing.assert_allclose(
        cross_view_action_consistency._linear_ramp_lambda(0.1, 20, total_steps=100, start_fraction=0.1, end_fraction=0.3),
        0.05,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        cross_view_action_consistency._linear_ramp_lambda(0.1, 30, total_steps=100, start_fraction=0.1, end_fraction=0.3),
        0.1,
        rtol=1e-6,
    )


def test_sample_cv_time_supports_beta_1p0_1p5_distribution():
    time = cross_view_action_consistency._sample_cv_time(
        jax.random.key(0),
        cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_1P0_1P5,
        (4096,),
    )

    assert time.shape == (4096,)
    assert jnp.all(time >= 0.001)
    assert jnp.all(time <= 1.0)
    np.testing.assert_allclose(jnp.mean(time), 0.4, atol=0.03)


def test_cross_view_pair_loss_shares_noise_and_time_within_pair():
    model = _FakeCrossViewModel()
    observation = _make_pair_observation(batch_size=2)
    actions = jnp.ones((2, 2, 3, 8), dtype=jnp.float32)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )

    assert components["total"].shape == (2, 3)
    assert components["loss_fm"].shape == (2, 3)
    assert components["loss_fm_nominal"].shape == (2, 3)
    assert components["loss_fm_perturbed"].shape == (2, 3)
    assert components["loss_cv"].shape == (2, 3)
    np.testing.assert_allclose(components["loss_cv"], np.zeros((2, 3)), atol=1e-6)


def test_cross_view_pair_loss_accepts_base_only_images():
    model = _FakeCrossViewModel()
    observation = _make_pair_observation(batch_size=2, include_wrist=False)
    actions = jnp.ones((2, 2, 3, 8), dtype=jnp.float32)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )

    assert components["total"].shape == (2, 3)
    np.testing.assert_allclose(components["loss_cv"], np.zeros((2, 3)), atol=1e-6)


def test_multi_sample_asymmetric_pair_loss_k2_uses_shared_noise_and_asymmetric_cv():
    model = _FakeMultiSampleAsymmetricCrossViewModel()
    observation = _make_pair_observation(batch_size=2, include_wrist=False)
    actions = jnp.ones((2, 2, 3, 8), dtype=jnp.float32)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )

    assert len(model.calls) == 2
    assert components["total"].shape == (2, 3)
    assert components["loss_fm"].shape == (2, 3)
    assert components["loss_fm_nominal"].shape == (2, 3)
    assert components["loss_fm_perturbed"].shape == (2, 3)
    assert components["loss_cv"].shape == (2, 3)
    np.testing.assert_allclose(components["loss_cv"], np.ones((2, 3)), atol=1e-6)
    np.testing.assert_allclose(components["multi_sample_asymmetric_cv_active"], 1.0)
    np.testing.assert_allclose(components["cv_num_samples"], 2.0)


def test_multi_sample_asymmetric_cv_stopgrad_blocks_nominal_cv_gradient():
    v_t = jnp.zeros((1, 2, 2, 3, 8), dtype=jnp.float32)
    v_t = v_t.at[:, :, 0, :, :7].set(1.0)
    v_t = v_t.at[:, :, 1, :, :7].set(3.0)

    def loss_with_stopgrad(v):
        return jnp.mean(
            cross_view_action_consistency._multi_sample_asymmetric_cv_loss(
                v,
                cv_dim=7,
                anchor_indices=None,
                stopgrad_anchor=True,
            )
        )

    grad = jax.grad(loss_with_stopgrad)(v_t)
    np.testing.assert_allclose(np.asarray(grad[:, :, 0]), 0.0, atol=1e-6)
    assert np.max(np.abs(np.asarray(grad[:, :, 1]))) > 0.0

    def loss_without_stopgrad(v):
        return jnp.mean(
            cross_view_action_consistency._multi_sample_asymmetric_cv_loss(
                v,
                cv_dim=7,
                anchor_indices=None,
                stopgrad_anchor=False,
            )
        )

    grad_no_stop = jax.grad(loss_without_stopgrad)(v_t)
    assert np.max(np.abs(np.asarray(grad_no_stop[:, :, 0]))) > 0.0


def test_pair_preprocess_none_spatial_none_photometric_preserves_pair_axis_then_flattens():
    image = jnp.linspace(-1.0, 1.0, 224 * 224 * 3, dtype=jnp.float32).reshape(224, 224, 3)
    images = jnp.stack([image, image], axis=0)[None]
    observation = dataclasses.replace(
        _make_pair_observation(batch_size=1, include_wrist=False),
        images={"base_0_rgb": images},
    )

    out = cross_view_action_consistency.preprocess_pair_observation(
        jax.random.key(0),
        observation,
        train=True,
        image_keys=("base_0_rgb",),
        pair_spatial_aug_mode=cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_NONE,
        pair_photometric_aug_mode=cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_NONE,
    )

    assert out.images["base_0_rgb"].shape == (2, 224, 224, 3)
    np.testing.assert_allclose(out.images["base_0_rgb"][0], image, atol=1e-6)
    np.testing.assert_allclose(out.images["base_0_rgb"][1], image, atol=1e-6)
    assert out.state.shape == (2, 8)
    assert out.image_masks["base_0_rgb"].shape == (2,)


def test_pair_preprocess_shared_spatial_uses_same_transform_for_both_slots():
    image = jnp.linspace(-1.0, 1.0, 224 * 224 * 3, dtype=jnp.float32).reshape(224, 224, 3)
    images = jnp.stack([image, image], axis=0)[None]
    observation = dataclasses.replace(
        _make_pair_observation(batch_size=1, include_wrist=False),
        images={"base_0_rgb": images},
    )

    out = cross_view_action_consistency.preprocess_pair_observation(
        jax.random.key(1),
        observation,
        train=True,
        image_keys=("base_0_rgb",),
        pair_spatial_aug_mode=cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_SHARED,
        pair_photometric_aug_mode=cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_NONE,
    )

    pair = out.images["base_0_rgb"].reshape(1, 2, 224, 224, 3)
    np.testing.assert_allclose(pair[:, 0], pair[:, 1], atol=1e-6)


def test_pair_preprocess_shared_photometric_uses_same_transform_for_both_slots():
    image = jnp.linspace(-1.0, 1.0, 224 * 224 * 3, dtype=jnp.float32).reshape(224, 224, 3)
    images = jnp.stack([image, image], axis=0)[None]
    observation = dataclasses.replace(
        _make_pair_observation(batch_size=1, include_wrist=False),
        images={"base_0_rgb": images},
    )

    out = cross_view_action_consistency.preprocess_pair_observation(
        jax.random.key(2),
        observation,
        train=True,
        image_keys=("base_0_rgb",),
        pair_spatial_aug_mode=cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_NONE,
        pair_photometric_aug_mode=cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_SHARED,
    )

    pair = out.images["base_0_rgb"].reshape(1, 2, 224, 224, 3)
    np.testing.assert_allclose(pair[:, 0], pair[:, 1], atol=1e-6)


def test_cross_view_pair_loss_fixedaug_path_runs():
    class _FakeFixedAugCrossViewModel(_FakeCrossViewModel):
        pair_spatial_aug_mode = cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_NONE
        pair_photometric_aug_mode = cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_INDEPENDENT

    model = _FakeFixedAugCrossViewModel()
    observation = _make_pair_observation(batch_size=2, include_wrist=False)
    actions = jnp.ones((2, 2, 3, 8), dtype=jnp.float32)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=True,
        step=50,
    )

    assert components["total"].shape == (2, 3)
    assert jnp.all(jnp.isfinite(components["total"]))


def test_clean_wrong_pairing_deranges_second_slot_with_own_labels():
    observation = _make_pair_observation(batch_size=4, include_wrist=False)
    observation = dataclasses.replace(
        observation,
        state=jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8),
        tokenized_prompt=jnp.arange(4 * 2 * 16, dtype=jnp.int32).reshape(4, 2, 16),
        task_index=jnp.arange(4 * 2, dtype=jnp.int32).reshape(4, 2),
    )
    actions = jnp.arange(4 * 2 * 3 * 8, dtype=jnp.float32).reshape(4, 2, 3, 8)
    perm = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)

    wrong_observation, wrong_actions = cross_view_action_consistency._apply_clean_wrong_pairing(observation, actions, perm)

    np.testing.assert_array_equal(wrong_observation.state[:, 0], observation.state[:, 0])
    np.testing.assert_array_equal(wrong_observation.state[:, 1], observation.state[perm, 1])
    np.testing.assert_array_equal(wrong_observation.tokenized_prompt[:, 0], observation.tokenized_prompt[:, 0])
    np.testing.assert_array_equal(wrong_observation.tokenized_prompt[:, 1], observation.tokenized_prompt[perm, 1])
    np.testing.assert_array_equal(wrong_observation.task_index[:, 0], observation.task_index[:, 0])
    np.testing.assert_array_equal(wrong_observation.task_index[:, 1], observation.task_index[perm, 1])
    np.testing.assert_array_equal(wrong_actions[:, 0], actions[:, 0])
    np.testing.assert_array_equal(wrong_actions[:, 1], actions[perm, 1])


def test_clean_wrong_derangement_has_no_fixed_points():
    batch_size = 16
    perm = np.asarray(cross_view_action_consistency._random_derangement_permutation(jax.random.key(0), batch_size))

    np.testing.assert_array_equal(np.sort(perm), np.arange(batch_size))
    assert np.all(perm != np.arange(batch_size))


def test_clean_wrong_pair_loss_path_runs():
    model = _FakeCleanWrongCrossViewModel()
    observation = _make_pair_observation(batch_size=4, include_wrist=False)
    row_actions = jnp.arange(4 * 3 * 8, dtype=jnp.float32).reshape(4, 3, 8)
    actions = jnp.stack([row_actions, row_actions], axis=1)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )

    assert components["total"].shape == (4, 3)
    np.testing.assert_allclose(components["clean_wrong_cv_active"], 1.0)


def test_clean_wrong_multi_sample_asymmetric_deranges_only_cv_anchor():
    model = _FakeCleanWrongMultiSampleAsymmetricCrossViewModel()
    observation = _make_pair_observation(batch_size=4, include_wrist=False)
    observation = dataclasses.replace(
        observation,
        state=jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8),
    )
    row_actions = jnp.arange(4 * 3 * 8, dtype=jnp.float32).reshape(4, 3, 8)
    actions = jnp.stack([row_actions, row_actions], axis=1)

    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )

    anchor_indices = np.asarray(components["cv_anchor_indices"])
    np.testing.assert_array_equal(np.sort(anchor_indices), np.arange(4))
    assert np.all(anchor_indices != np.arange(4))
    assert components["total"].shape == (4, 3)
    np.testing.assert_allclose(components["clean_wrong_cv_active"], 1.0)
    np.testing.assert_allclose(components["multi_sample_asymmetric_cv_active"], 1.0)
