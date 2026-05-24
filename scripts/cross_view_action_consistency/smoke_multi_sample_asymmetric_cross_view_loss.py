#!/usr/bin/env python3
"""CPU smoke for multi-sample cross-view loss variants."""

from __future__ import annotations

# ruff: noqa: I001, SLF001

import json
import os
from pathlib import Path

# This smoke is intentionally CPU-only so it cannot interfere with active GPU jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import cross_view_action_consistency


class _SmokeCrossViewModel:
    cv_action_dim = 7
    cv_pair_mode = cross_view_action_consistency.CV_PAIR_MODE_MATCHED
    cv_loss_mode = cross_view_action_consistency.CV_LOSS_MODE_SYMMETRIC
    cv_num_samples = 1
    cv_stopgrad_anchor = True
    cv_time_distribution = cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_1P5_1
    cv_eps_shared_across_views = True
    cv_average_over_samples = True
    pair_spatial_aug_mode = cross_view_action_consistency.PAIR_SPATIAL_AUG_MODE_NONE
    pair_photometric_aug_mode = cross_view_action_consistency.PAIR_PHOTOMETRIC_AUG_MODE_NONE
    expect_row_local_state = False

    def __init__(self):
        self.calls = 0

    def _lambda_cv(self, step):
        del step
        return jnp.asarray(0.10, dtype=jnp.float32)

    def predict_flow(self, observation, noisy_actions, time, *, train=False, canonical_drop_rng=None):
        del train, canonical_drop_rng
        batch_pairs = noisy_actions.shape[0] // 2
        noisy_pair = np.asarray(noisy_actions).reshape(batch_pairs, 2, *noisy_actions.shape[1:])
        time_pair = np.asarray(time).reshape(batch_pairs, 2)
        np.testing.assert_allclose(noisy_pair[:, 0], noisy_pair[:, 1], rtol=0, atol=0)
        np.testing.assert_allclose(time_pair[:, 0], time_pair[:, 1], rtol=0, atol=0)
        if self.expect_row_local_state:
            expected_state0 = np.arange(batch_pairs * 16, step=8, dtype=np.float32)
            np.testing.assert_array_equal(np.asarray(observation.state)[:, 0], expected_state0)
        self.calls += 1
        flow = np.zeros_like(noisy_pair)
        flow[:, 0, :, :7] = 1.0
        flow[:, 1, :, :7] = 2.0
        return jnp.asarray(flow.reshape(noisy_actions.shape))


class _MultiSampleModel(_SmokeCrossViewModel):
    cv_loss_mode = cross_view_action_consistency.CV_LOSS_MODE_MULTI_SAMPLE_ASYMMETRIC
    cv_num_samples = 2


class _BilateralMultiSampleModel(_MultiSampleModel):
    cv_stopgrad_anchor = False


class _ActionBiasedBilateralMultiSampleModel(_BilateralMultiSampleModel):
    cv_time_distribution = cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_2P0_3P0


class _MultiSampleK1Model(_MultiSampleModel):
    cv_num_samples = 1


class _CleanWrongMultiSampleModel(_MultiSampleModel):
    cv_pair_mode = cross_view_action_consistency.CV_PAIR_MODE_CLEAN_WRONG_BATCH_DERANGEMENT
    expect_row_local_state = True


def _make_observation(batch_size: int) -> _model.Observation:
    return _model.Observation(
        images={"base_0_rgb": jnp.zeros((batch_size, 2, 224, 224, 3), dtype=jnp.float32)},
        image_masks={"base_0_rgb": jnp.ones((batch_size, 2), dtype=jnp.bool_)},
        state=jnp.arange(batch_size * 2 * 8, dtype=jnp.float32).reshape(batch_size, 2, 8),
        tokenized_prompt=jnp.ones((batch_size, 2, 16), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 2, 16), dtype=jnp.bool_),
    )


def _run_pair_loss(model, *, batch_size: int = 4) -> dict:
    observation = _make_observation(batch_size)
    row_actions = jnp.arange(batch_size * 3 * 8, dtype=jnp.float32).reshape(batch_size, 3, 8)
    actions = jnp.stack([row_actions, row_actions], axis=1)
    components = cross_view_action_consistency.Pi0CrossViewActionConsistency._pair_loss_components(
        model,
        jax.random.key(0),
        observation,
        actions,
        train=False,
        step=50,
    )
    return {
        "total_shape": list(components["total"].shape),
        "loss_fm_finite": bool(jnp.all(jnp.isfinite(components["loss_fm"]))),
        "loss_cv_finite": bool(jnp.all(jnp.isfinite(components["loss_cv"]))),
        "loss_cv_mean": float(jnp.mean(components["loss_cv"])),
        "clean_wrong_active": float(components["clean_wrong_cv_active"]),
        "multi_sample_active": float(components["multi_sample_asymmetric_cv_active"]),
        "cv_num_samples": float(components["cv_num_samples"]),
        "anchor_indices": np.asarray(components["cv_anchor_indices"]).tolist(),
        "predict_flow_calls": model.calls,
    }


def _grad_checks() -> dict:
    v_t = jnp.zeros((1, 2, 2, 3, 8), dtype=jnp.float32)
    v_t = v_t.at[:, :, 0, :, :7].set(1.0)
    v_t = v_t.at[:, :, 1, :, :7].set(3.0)

    def loss_with_stopgrad(v, *, stopgrad_anchor: bool):
        return jnp.mean(
            cross_view_action_consistency._multi_sample_asymmetric_cv_loss(
                v,
                cv_dim=7,
                anchor_indices=None,
                stopgrad_anchor=stopgrad_anchor,
            )
        )

    stopgrad_grad = jax.grad(loss_with_stopgrad)(v_t, stopgrad_anchor=True)
    bilateral_grad = jax.grad(loss_with_stopgrad)(v_t, stopgrad_anchor=False)
    return {
        "nominal_cv_grad_zero_with_stopgrad": bool(np.allclose(np.asarray(stopgrad_grad[:, :, 0]), 0.0)),
        "perturbed_cv_grad_nonzero_with_stopgrad": bool(
            np.max(np.abs(np.asarray(stopgrad_grad[:, :, 1]))) > 0.0
        ),
        "nominal_cv_grad_nonzero_without_stopgrad": bool(
            np.max(np.abs(np.asarray(bilateral_grad[:, :, 0]))) > 0.0
        ),
        "perturbed_cv_grad_nonzero_without_stopgrad": bool(
            np.max(np.abs(np.asarray(bilateral_grad[:, :, 1]))) > 0.0
        ),
    }


def _time_distribution_checks() -> dict:
    sample_shape = (32_768,)
    beta_1p5_1 = cross_view_action_consistency._sample_cv_time(
        jax.random.key(10),
        cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_1P5_1,
        sample_shape,
    )
    action_biased = cross_view_action_consistency._sample_cv_time(
        jax.random.key(11),
        cross_view_action_consistency.CV_TIME_DISTRIBUTION_BETA_2P0_3P0,
        sample_shape,
    )
    beta_1p5_1_mean = float(jnp.mean(beta_1p5_1))
    action_biased_mean = float(jnp.mean(action_biased))
    return {
        "beta_1p5_1_mean": beta_1p5_1_mean,
        "action_biased_mean": action_biased_mean,
        "action_biased_mean_below_beta_1p5_1": action_biased_mean < beta_1p5_1_mean,
        "action_biased_mean_in_expected_band": 0.35 <= action_biased_mean <= 0.45,
    }


def main() -> None:
    out_dir = Path("results/cross_view_action_consistency")
    out_dir.mkdir(parents=True, exist_ok=True)

    single_sample = _run_pair_loss(_SmokeCrossViewModel(), batch_size=4)
    k1 = _run_pair_loss(_MultiSampleK1Model(), batch_size=4)
    k2 = _run_pair_loss(_MultiSampleModel(), batch_size=4)
    bilateral = _run_pair_loss(_BilateralMultiSampleModel(), batch_size=4)
    action_biased_bilateral = _run_pair_loss(_ActionBiasedBilateralMultiSampleModel(), batch_size=4)
    clean_wrong = _run_pair_loss(_CleanWrongMultiSampleModel(), batch_size=4)
    grad = _grad_checks()
    time_checks = _time_distribution_checks()

    clean_wrong_anchor = np.asarray(clean_wrong["anchor_indices"])
    clean_wrong_checks = {
        "derangement_has_no_fixed_points": bool(np.all(clean_wrong_anchor != np.arange(clean_wrong_anchor.shape[0]))),
        "derangement_is_permutation": bool(
            np.array_equal(np.sort(clean_wrong_anchor), np.arange(clean_wrong_anchor.shape[0]))
        ),
        "fm_inputs_row_local": True,
        "cv_target_indices": clean_wrong["anchor_indices"],
    }

    report = {
        "cpu_only": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_CUDA_VISIBLE_DEVICES": os.environ.get("JAX_CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "jax_default_backend": jax.default_backend(),
        },
        "single_sample_symmetric": single_sample,
        "multi_sample_asymmetric_k1": k1,
        "multi_sample_asymmetric_k2": k2,
        "multi_sample_bilateral_k2": bilateral,
        "multi_sample_bilateral_action_biased_k2": action_biased_bilateral,
        "clean_wrong_multi_sample_asymmetric": clean_wrong,
        "clean_wrong_checks": clean_wrong_checks,
        "gradient_checks": grad,
        "time_distribution_checks": time_checks,
        "wrist_image_used": False,
    }

    (out_dir / "multi_sample_asymmetric_cross_view_loss_smoke.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Multi-Sample Asymmetric Cross-View Loss Smoke",
        "",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- single-sample symmetric loss works: `{single_sample['loss_fm_finite'] and single_sample['loss_cv_finite']}`",
        f"- K=1 works: `{k1['loss_fm_finite'] and k1['loss_cv_finite']}`",
        f"- K=2 works: `{k2['loss_fm_finite'] and k2['loss_cv_finite']}`",
        f"- K=2 bilateral works: `{bilateral['loss_fm_finite'] and bilateral['loss_cv_finite']}`",
        (
            f"- K=2 bilateral action-biased works: "
            f"`{action_biased_bilateral['loss_fm_finite'] and action_biased_bilateral['loss_cv_finite']}`"
        ),
        f"- K=2 predict_flow calls in fake fallback: `{k2['predict_flow_calls']}`",
        f"- nominal CV grad zero with stopgrad: `{grad['nominal_cv_grad_zero_with_stopgrad']}`",
        f"- perturbed CV grad nonzero with stopgrad: `{grad['perturbed_cv_grad_nonzero_with_stopgrad']}`",
        f"- nominal CV grad nonzero without stopgrad: `{grad['nominal_cv_grad_nonzero_without_stopgrad']}`",
        f"- perturbed CV grad nonzero without stopgrad: `{grad['perturbed_cv_grad_nonzero_without_stopgrad']}`",
        f"- beta_1p5_1 time mean: `{time_checks['beta_1p5_1_mean']:.4f}`",
        f"- action-biased time mean: `{time_checks['action_biased_mean']:.4f}`",
        f"- action-biased time mean below beta_1p5_1: `{time_checks['action_biased_mean_below_beta_1p5_1']}`",
        f"- action-biased time mean in expected band: `{time_checks['action_biased_mean_in_expected_band']}`",
        f"- no wrist image used: `{not report['wrist_image_used']}`",
        f"- loss_cv K=2 mean: `{k2['loss_cv_mean']:.6f}`",
        f"- loss_cv K=2 bilateral mean: `{bilateral['loss_cv_mean']:.6f}`",
        f"- loss_cv K=2 bilateral action-biased mean: `{action_biased_bilateral['loss_cv_mean']:.6f}`",
    ]
    (out_dir / "multi_sample_asymmetric_cross_view_loss_smoke.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    clean_lines = [
        "# Clean Wrong Multi-Sample Asymmetric Cross-View Smoke",
        "",
        f"- derangement has no fixed points: `{clean_wrong_checks['derangement_has_no_fixed_points']}`",
        f"- derangement is permutation: `{clean_wrong_checks['derangement_is_permutation']}`",
        f"- FM inputs remain row-local: `{clean_wrong_checks['fm_inputs_row_local']}`",
        f"- loss values finite: `{clean_wrong['loss_fm_finite'] and clean_wrong['loss_cv_finite']}`",
        f"- CV target indices: `{clean_wrong_checks['cv_target_indices']}`",
    ]
    (out_dir / "clean_wrong_multi_sample_asymmetric_cross_view_smoke.md").write_text(
        "\n".join(clean_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
