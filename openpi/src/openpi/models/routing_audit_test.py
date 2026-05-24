import jax
import jax.numpy as jnp

import openpi.models.pi0_config as _pi0_config


def test_capture_prefix_intermediates_shapes_dummy_pi0():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    captures = model.capture_prefix_intermediates(obs, layer_indices=(0, 2, 3))

    assert captures["layer_prefix_hidden"].shape == (3, 2, 816, 64)
    assert captures["prefix_out"].shape == (2, 816, 64)
    assert captures["prefix_mask"].shape == (2, 816)
    assert captures["n_image_tokens"] == 768
    assert captures["n_language_tokens"] == 48


def test_capture_prefix_intermediates_zero_image_tokens_dummy_pi0():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    captures = model.capture_prefix_intermediates(obs, layer_indices=(0,), zero_image_tokens=True)

    assert captures["raw_prefix"][:, : captures["n_image_tokens"], :].max() == 0.0


def test_capture_prefix_intermediates_does_not_break_predict_flow_dummy_pi0():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    actions = config.fake_act(batch_size=1)
    time = jnp.ones((1,), dtype=jnp.float32) * 0.5

    flow = model.predict_flow(obs, actions, time, train=False)

    assert flow.shape == actions.shape


def test_capture_prefix_intermediates_does_not_break_sample_actions_dummy_pi0():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    actions = model.sample_actions(jax.random.key(1), obs, num_steps=1)

    assert actions.shape == (1, config.action_horizon, config.action_dim)
