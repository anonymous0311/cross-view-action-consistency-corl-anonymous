"""Losses for Stage 1 canonical-token pretraining.

The Stage 1 v3 objective is anchored only on the canonical token matrix
``z [B, N, D]``. There is no pooled vector and no reconstruction decoder in the
paper-facing training path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def safe_l2_normalize(x: jax.Array, *, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    """L2-normalize with finite gradients at zero norm."""
    sq = jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axis, keepdims=True)
    return x.astype(jnp.float32) * jax.lax.rsqrt(sq + eps)


def _masked_cross_entropy(logits: jax.Array, negative_mask: jax.Array | None) -> jax.Array:
    """Mean diagonal cross-entropy for logits ``[N, B, B]``."""
    n_slots, batch, _ = logits.shape
    labels = jnp.arange(batch)
    if negative_mask is not None:
        if negative_mask.shape != (batch, batch):
            raise ValueError(
                f"negative_mask must have shape {(batch, batch)}, got {negative_mask.shape}"
            )
        valid = negative_mask.astype(bool)
        valid = valid.at[labels, labels].set(True)
        floor = jnp.asarray(-1e30, dtype=logits.dtype)
        logits = jnp.where(valid[None, :, :], logits, floor)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[:, labels, labels])


def slot_info_nce(
    z_a: jax.Array,
    z_b: jax.Array,
    *,
    temperature: float = 0.10,
    negative_mask: jax.Array | None = None,
) -> jax.Array:
    """Symmetric per-slot InfoNCE on canonical tokens.

    Args:
        z_a, z_b: canonical tokens with shape ``[B, N, D]``.
        temperature: contrastive softmax temperature.
        negative_mask: optional ``[B, B]`` denominator mask. ``True`` keeps an
            entry; ``False`` removes it. The positive diagonal is forced valid.

    Returns:
        Scalar symmetric loss averaged over slots and examples.
    """
    if z_a.ndim != 3 or z_b.ndim != 3:
        raise ValueError(f"slot_info_nce expects [B,N,D] tokens, got {z_a.shape} and {z_b.shape}")
    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a and z_b must have identical shape, got {z_a.shape} and {z_b.shape}")

    za = safe_l2_normalize(z_a, axis=-1)
    zb = safe_l2_normalize(z_b, axis=-1)
    logits = jnp.einsum("bnd,cnd->nbc", za, zb) / temperature
    loss_ab = _masked_cross_entropy(logits, negative_mask)
    mask_ba = None if negative_mask is None else negative_mask.T
    loss_ba = _masked_cross_entropy(jnp.swapaxes(logits, 1, 2), mask_ba)
    return 0.5 * (loss_ab + loss_ba)


def per_slot_positive_similarity(z_a: jax.Array, z_b: jax.Array) -> jax.Array:
    """Mean positive cosine per slot, shape ``[N]``."""
    za = safe_l2_normalize(z_a, axis=-1)
    zb = safe_l2_normalize(z_b, axis=-1)
    return jnp.mean(jnp.sum(za * zb, axis=-1), axis=0)


def set_similarity_matrix(z_a: jax.Array, z_b: jax.Array) -> jax.Array:
    """Token-set similarity matrix used for retrieval, shape ``[B, B]``."""
    za = safe_l2_normalize(z_a, axis=-1)
    zb = safe_l2_normalize(z_b, axis=-1)
    return jnp.mean(jnp.einsum("bnd,cnd->bnc", za, zb), axis=1)


def variance_loss(embeddings: jax.Array, gamma: float = 1.0, eps: float = 1e-4) -> jax.Array:
    """VICReg variance hinge for a ``[B, D]`` matrix."""
    std = jnp.sqrt(jnp.var(embeddings.astype(jnp.float32), axis=0) + eps)
    return jnp.mean(jnp.maximum(0.0, gamma - std))


def covariance_loss(embeddings: jax.Array) -> jax.Array:
    """VICReg covariance penalty for a ``[B, D]`` matrix."""
    x = embeddings.astype(jnp.float32)
    batch, dim = x.shape
    if batch <= 1:
        return jnp.asarray(0.0, dtype=jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    cov = (x.T @ x) / (batch - 1)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    return jnp.sum(jnp.square(off_diag)) / dim


def per_slot_vicreg_var_cov(
    tokens: jax.Array,
    *,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    """Per-slot VICReg var/cov terms for ``tokens [B, N, D]``.

    Each slot is regularized independently across the batch, then the 128 slot
    losses are averaged. This prevents healthy slots from hiding collapsed
    slots, which can happen with a flattened ``[B*N, D]`` loss.
    """
    if tokens.ndim != 3:
        raise ValueError(f"per_slot_vicreg_var_cov expects [B,N,D], got {tokens.shape}")
    x = tokens.astype(jnp.float32)
    batch, _, dim = x.shape
    std = jnp.sqrt(jnp.var(x, axis=0) + eps)  # [N, D]
    var = jnp.mean(jnp.maximum(0.0, gamma - std))
    if batch <= 1:
        return var, jnp.asarray(0.0, dtype=jnp.float32)
    centered = x - jnp.mean(x, axis=0, keepdims=True)
    cov = jnp.einsum("bni,bnj->nij", centered, centered) / (batch - 1)
    eye = jnp.eye(dim, dtype=bool)
    off_diag = jnp.where(eye[None, :, :], 0.0, cov)
    cov_loss = jnp.sum(jnp.square(off_diag), axis=(-2, -1)) / dim
    return var, jnp.mean(cov_loss)
