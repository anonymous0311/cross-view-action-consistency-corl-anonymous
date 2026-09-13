"""Scientific invariants of the objective; no model weights or GPUs required."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.cross_view import _active_pair_normalize_cv_loss, _random_derangement_permutation, flow_losses
from openpi.models.cross_view_config import CrossViewConfig


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.v = jax.random.normal(jax.random.key(2), (2, 4, 2, 10, 32))
        target = jax.random.normal(jax.random.key(3), (2, 4, 1, 10, 32))
        self.target = jnp.repeat(target, 2, axis=2)

    def test_mean_disagreement_identity_and_padding(self):
        fm, cv, _ = flow_losses(self.v, self.target, action_dim=7)
        mean = self.v.mean(axis=2)
        delta = (self.v[:, :, 0] - self.v[:, :, 1]) / 2
        expected = ((mean - self.target[:, :, 0]) ** 2).mean(axis=-1).mean(axis=0)
        expected += ((delta) ** 2).mean(axis=-1).mean(axis=0)
        np.testing.assert_allclose(fm, expected, rtol=1e-6)
        np.testing.assert_allclose(cv, 4 * (delta[..., :7] ** 2).mean(axis=-1).mean(axis=0), rtol=1e-6)
        changed = self.v.at[..., 7:].add(10)
        _, changed_cv, _ = flow_losses(changed, self.target, action_dim=7)
        np.testing.assert_allclose(cv, changed_cv)

    def test_bilateral_gradient(self):
        grad = jax.grad(lambda v: flow_losses(v, self.target, action_dim=7)[1].sum())(self.v)
        self.assertGreater(float(jnp.linalg.norm(grad[:, :, 0])), 0)
        np.testing.assert_allclose(grad[:, :, 0], -grad[:, :, 1], atol=1e-7)
        np.testing.assert_allclose(grad[..., 7:], 0)

    def test_shuffle_preserves_supervised_targets(self):
        perm = _random_derangement_permutation(jax.random.key(4), 4)
        self.assertTrue(np.all(np.asarray(perm) != np.arange(4)))
        fm, cv, _ = flow_losses(self.v, self.target, action_dim=7)
        shuffled_fm, shuffled_cv, _ = flow_losses(self.v, self.target, action_dim=7, anchor_indices=perm)
        np.testing.assert_allclose(fm, shuffled_fm)
        self.assertFalse(np.allclose(cv, shuffled_cv))

    def test_action_weighting(self):
        _, cv, _ = flow_losses(self.v, self.target, action_dim=7)
        _, weighted, _ = flow_losses(self.v, self.target, action_dim=7, time=jnp.full((2, 4), 0.5))
        np.testing.assert_allclose(weighted, cv / 4)

    def test_partial_coverage_preserves_per_active_pair_strength(self):
        values = jnp.ones((4, 10)) * 3
        for mask in [jnp.array([1, 0, 0, 0]), jnp.array([1, 1, 0, 0]), jnp.ones(4)]:
            loss, _, _ = _active_pair_normalize_cv_loss(values, mask)
            self.assertAlmostEqual(float(loss.mean()), 3)
        loss, _, _ = _active_pair_normalize_cv_loss(values, jnp.zeros(4))
        self.assertTrue(np.isfinite(loss).all())
        self.assertEqual(float(loss.sum()), 0)

    def test_paper_defaults(self):
        cfg = CrossViewConfig()
        self.assertEqual((cfg.action_horizon, cfg.cv_action_dim, cfg.action_dim), (10, 7, 32))
        self.assertEqual((cfg.num_flow_samples, cfg.time_beta_alpha, cfg.time_beta_beta), (2, 2, 3))
        self.assertEqual((cfg.cv_warmup_start, cfg.cv_warmup_steps, cfg.lambda_cv), (0, 500, 0.1))


if __name__ == "__main__":
    unittest.main()
