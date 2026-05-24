"""Tests for Pi0CanonicalConfig and canonical cross-attention correctness.

Covers three bugs found in code review:
  Bug 1 (CRITICAL): preprocess_observation drops canonical_tokens → feature is dead.
  Bug 2 (HIGH):     canonical cross-attn fires after FFN instead of before.
  Bug 3 (CRITICAL): double zero-init (adaRMS gate + can_out kernel) → zero gradients forever.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_canonical_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation_with_canonical(
    config: pi0_canonical_config.Pi0CanonicalConfig,
    batch_size: int,
    *,
    canonical_fill: float,
) -> _model.Observation:
    """Create a minimal Observation with canonical_tokens filled to a constant."""
    B = batch_size
    H, W = _model.IMAGE_RESOLUTION
    images = {k: jnp.zeros((B, H, W, 3)) for k in _model.IMAGE_KEYS}
    masks = {k: jnp.ones((B,), dtype=jnp.bool_) for k in _model.IMAGE_KEYS}
    # canonical_dim is forced to 512 by Pi0CanonicalConfig; use 4 tokens for speed.
    N_canonical = 4
    canonical_tokens = jnp.full((B, N_canonical, config.canonical_dim), canonical_fill)
    return _model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((B, config.action_dim)),
        canonical_tokens=canonical_tokens,
    )


# ---------------------------------------------------------------------------
# Bug 1 — preprocess_observation must preserve canonical_tokens
# ---------------------------------------------------------------------------

def test_preprocess_observation_preserves_canonical_tokens():
    """Bug 1: preprocess_observation was silently dropping canonical_tokens.

    After the fix, canonical_tokens must pass through unchanged so that
    compute_loss / sample_actions can actually reach the action expert.
    """
    B, N, D = 2, 128, 512
    images = {k: jnp.zeros((B, 224, 224, 3)) for k in _model.IMAGE_KEYS}
    masks = {k: jnp.ones((B,), dtype=jnp.bool_) for k in _model.IMAGE_KEYS}
    canonical = jnp.arange(B * N * D, dtype=jnp.float32).reshape(B, N, D)

    obs = _model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((B, 32)),
        canonical_tokens=canonical,
    )

    out = _model.preprocess_observation(None, obs, train=False)

    assert out.canonical_tokens is not None, (
        "preprocess_observation dropped canonical_tokens — "
        "add canonical_tokens=observation.canonical_tokens to the return Observation()"
    )
    np.testing.assert_array_equal(
        out.canonical_tokens,
        canonical,
        err_msg="canonical_tokens must be returned unchanged from preprocess_observation",
    )


def test_preprocess_observation_preserves_none_canonical_tokens():
    """When canonical_tokens=None, preprocess_observation must also return None (not error)."""
    B = 1
    images = {k: jnp.zeros((B, 224, 224, 3)) for k in _model.IMAGE_KEYS}
    masks = {k: jnp.ones((B,), dtype=jnp.bool_) for k in _model.IMAGE_KEYS}

    obs = _model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((B, 32)),
        canonical_tokens=None,
    )

    out = _model.preprocess_observation(None, obs, train=False)
    assert out.canonical_tokens is None


# ---------------------------------------------------------------------------
# Bug 2 — canonical cross-attn must be BEFORE the FFN in Block.__call__
# ---------------------------------------------------------------------------

def test_block_canonical_cross_attn_is_positioned_before_ffn():
    """Bug 2: canonical cross-attn code must be ordered before the FFN block.

    The specification (execution_plan.md Step 7) requires:
        self_attn → cross_attn(VLM) → cross_attn_canonical → FFN

    We verify by checking that 'can_pre_norm' (canonical block marker) appears
    before 'pre_ffw_norm' (FFN block marker) in Block.__call__ source code.
    """
    from openpi.models.gemma import Block  # noqa: PLC0415

    source = inspect.getsource(Block.__call__)

    ffn_pos = source.find("pre_ffw_norm")
    canonical_pos = source.find("can_pre_norm")

    assert ffn_pos != -1, "Could not find 'pre_ffw_norm' (FFN marker) in Block.__call__"
    assert canonical_pos != -1, "Could not find 'can_pre_norm' (canonical marker) in Block.__call__"

    assert canonical_pos < ffn_pos, (
        f"BUG 2: canonical cross-attention ('can_pre_norm' at char {canonical_pos}) "
        f"is placed AFTER FFN ('pre_ffw_norm' at char {ffn_pos}). "
        "Move the canonical block to before the FFN block in gemma.py Block.__call__."
    )


# ---------------------------------------------------------------------------
# Bug 3 — can_pre_norm must NOT use adaRMS (gate=None) to avoid zero gradients
# ---------------------------------------------------------------------------

def test_canonical_cross_attn_uses_standard_prenorm_not_adarms():
    """Bug 3: adaRMS gate is zero-initialized; combined with zero-init can_out this
    makes ∂loss/∂can_params = 0 permanently (both multiplicands in gate*out_can are 0).

    The fix: can_pre_norm receives cond=None so gate=None and the residual is x2+out_can,
    meaning ∂loss/∂out_can = downstream_grad (non-zero) from the first step.

    We verify by inspecting Block.__call__ source: 'can_pre_norm")(x2, None)' must appear
    (not 'adarms_cond' as the second argument to can_pre_norm).
    """
    from openpi.models.gemma import Block  # noqa: PLC0415

    source = inspect.getsource(Block.__call__)

    # The exact call must be RMSNorm(name="can_pre_norm")(x2, None) — not adarms_cond[1].
    # Checking the literal token on the same call-line is sufficient and avoids false matches
    # from adarms_cond appearing elsewhere in Block.__call__.
    assert 'can_pre_norm")(x2, None)' in source, (
        "BUG 3: can_pre_norm is using adaRMS conditioning (cond=adarms_cond[1]). "
        "This zero-inits the gate, and combined with zero-init can_out creates "
        "∂loss/∂can_params = 0 for all steps. "
        "Change to RMSNorm(name='can_pre_norm')(x2, None) to use standard pre-LN."
    )

    assert 'can_pre_norm")(x2, adarms_cond' not in source, (
        "BUG 3: can_pre_norm must not receive adarms_cond — use None instead."
    )
