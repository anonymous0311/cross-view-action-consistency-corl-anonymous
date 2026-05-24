"""ACB-VLA: Action-Grounded Canonical Bottleneck VLA.

Architecture overview
---------------------
During training:

    RGB ──► SigLIP ──► image_tokens [B, S_img, D_pg]
                                │
                       BottleneckEncoder
                       (cross-attn learnable queries)
                                │
                    ┌─────────────────────┐
                    │  b_can [B,N_can,D_pg]│  ◄── L_z_teacher, L_var, L_cam_adv
                    │  b_det [B,N_det,D_pg]│  ◄── free (contact / view detail)
                    └─────────────────────┘
                                │
    Language tokens ────────────┤
                                ▼
          PaliGemma (prefix stream):  [image | b_can | b_det | lang]
          Action expert (suffix):     [b_can | b_det | lang]  ← image BLOCKED
                                │
                      action_out_proj
                                │
                            action chunk

At inference: RGB + language + proprio only.  No Stage-1 encoder or cache is required.

Multi-GPU notes
---------------
- Training uses pure DDP (`fsdp_devices=1`); each GPU holds a full replica of
  model + optimizer state.  FSDP sharding is NOT used in this project.
- BottleneckEncoder params (queries, cross-attn weights) are standard nnx.Param
  and are replicated across all DDP ranks like every other model parameter.
- VICReg variance is computed per-device-shard.  No cross-device AllGather is
  needed; per-shard statistics are sufficient for anti-collapse regularisation.
- L_z_teacher is per-example cosine loss, fully parallel across devices.
- Gradient reversal (camera adversarial) is per-sample, no special handling.
- The attn_mask modification (hard bottleneck) is a pure tensor operation and
  compiles cleanly under jit/pmap.
"""
from __future__ import annotations

import functools

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from canonical.modules.attention import CrossAttnBlock
from canonical.modules.attention import SelfAttnBlock
from openpi.models import model as _model
from openpi.models.acb_vla_config import Pi0ACBConfig
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import Pi0
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for camera adversarial loss; used when lambda_cam>0)
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _gradient_reversal(x: jax.Array, alpha: float) -> jax.Array:  # noqa: D401
    return x


def _gr_fwd(x: jax.Array, alpha: float):
    return x, None


def _gr_bwd(alpha: float, _res, g: jax.Array):
    return (-alpha * g,)


_gradient_reversal.defvjp(_gr_fwd, _gr_bwd)


# ---------------------------------------------------------------------------
# BottleneckEncoder
# ---------------------------------------------------------------------------

class BottleneckEncoder(nnx.Module):
    """Compress image tokens into (b_can, b_detail) via cross-attention.

    Two independent sets of learnable queries cross-attend into image_tokens.
    Each set is followed by a self-attention block for lateral refinement.
    Output tokens are in PaliGemma embedding space (D_pg) so they can be
    concatenated into the shared prefix stream without any extra projection.

    Args:
        image_dim  : D_pg (= paligemma_config.width, typically 2048)
        n_can      : number of b_can query/output tokens
        n_det      : number of b_detail query/output tokens
        num_heads  : attention heads for cross- and self-attention blocks
        num_layers : number of (CrossAttn + SelfAttn) layer pairs
    """

    def __init__(
        self,
        *,
        image_dim: int,
        n_can: int,
        n_det: int,
        num_heads: int,
        num_layers: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_can = n_can
        self.n_det = n_det
        self.image_dim = image_dim

        init_std = 0.02

        # Learnable queries — initialised from N(0, 0.02) matching openpi convention
        can_init = jax.random.normal(rngs.params(), (n_can, image_dim)) * init_std
        self.can_queries = nnx.Param(can_init)  # [n_can, D_pg]

        # Shared KV normalisation (applied to image_tokens before attention)
        self.kv_norm = nnx.LayerNorm(image_dim, rngs=rngs)

        # Independent cross-attn + self-attn stacks for b_can
        self.can_cross = [CrossAttnBlock(d=image_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        self.can_self = [SelfAttnBlock(d=image_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        self.can_out_norm = nnx.LayerNorm(image_dim, rngs=rngs)

        # b_det branch — only instantiated when n_det > 0 (avoids zero-sized nnx.Param)
        if n_det > 0:
            det_init = jax.random.normal(rngs.params(), (n_det, image_dim)) * init_std
            self.det_queries = nnx.Param(det_init)
            self.det_cross = [CrossAttnBlock(d=image_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
            self.det_self = [SelfAttnBlock(d=image_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
            self.det_out_norm = nnx.LayerNorm(image_dim, rngs=rngs)
        else:
            self.det_queries = None
            self.det_cross = []
            self.det_self = []
            self.det_out_norm = None

    def __call__(
        self,
        image_tokens: at.Float[at.Array, "b s_img d_pg"],
    ) -> tuple[at.Float[at.Array, "b n_can d_pg"], at.Float[at.Array, "b n_det d_pg"]]:
        """
        Args:
            image_tokens: [B, S_img, D_pg]  (from SigLIP → PaliGemma embedding space)

        Returns:
            b_can : [B, n_can, D_pg]
            b_det : [B, n_det, D_pg]
        """
        batch_size = image_tokens.shape[0]

        # Normalise KV input once, shared by both branches
        kv = self.kv_norm(image_tokens)  # [B, S_img, D_pg]

        # ---- b_can branch ----
        can_q = jnp.broadcast_to(
            self.can_queries.value, (batch_size, self.n_can, self.image_dim)
        )
        for cross, self_block in zip(self.can_cross, self.can_self, strict=True):
            can_q = cross(can_q, kv)
            can_q = self_block(can_q)
        b_can = self.can_out_norm(can_q)  # [B, n_can, D_pg]

        # ---- b_det branch ----
        if self.n_det > 0:
            det_q = jnp.broadcast_to(
                self.det_queries.value, (batch_size, self.n_det, self.image_dim)
            )
            for cross, self_block in zip(self.det_cross, self.det_self, strict=True):
                det_q = cross(det_q, kv)
                det_q = self_block(det_q)
            b_det = self.det_out_norm(det_q)  # [B, n_det, D_pg]
        else:
            b_det = jnp.zeros((batch_size, 0, self.image_dim), dtype=image_tokens.dtype)

        return b_can, b_det


# ---------------------------------------------------------------------------
# Non-trainable constant buffer variable (used for camera_prior)
# ---------------------------------------------------------------------------

class ConstantBuffer(nnx.Variable):
    """Non-trainable constant array stored in nnx state but excluded from
    the optimizer (trainable_filter uses nnx.Param; this is not nnx.Param).

    Wrapping a raw jax/numpy array in an nnx.Variable is required because
    nnx.state() does not support bare array leaves on nnx.Module attributes.
    """
    pass


# ---------------------------------------------------------------------------
# Camera adversarial head (disabled when lambda_cam=0)
# ---------------------------------------------------------------------------

class CameraAdversarialHead(nnx.Module):
    """Camera classifier head used for split-discriminator KL confusion (v3) or GRL+CE (v1).

    Two call paths for v3:
      - apply_for_head(z): used in L_head with stopgrad(b_can) passed externally.
        No GRL. Trains classifier weights on detached features.
      - apply_for_feat(z): used in L_feat. Classifier params are frozen via
        nnx.split/merge stopgrad. Gradient flows through z → b_can encoder.

    The legacy GRL path (__call__) is kept for backward compat with
    lambda_cam > 0 configs (sparse_acb_full v1 and other existing configs).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        alpha: float = 1.0,
        rngs: nnx.Rngs,
    ) -> None:
        self.alpha = alpha
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)
        self.norm = nnx.LayerNorm(in_dim, rngs=rngs)

    def _forward_no_grl(self, z: jax.Array) -> jax.Array:
        """Forward pass without gradient reversal. z: [B, D] → logits [B, K]."""
        return self.fc2(nnx.relu(self.fc1(self.norm(z))))

    def apply_for_head(self, z_detached: jax.Array) -> jax.Array:
        """L_head path. Caller must pass jax.lax.stop_gradient(z) externally.

        Classifier weights receive gradient; b_can encoder does NOT (z is detached).
        """
        return self._forward_no_grl(z_detached)

    def apply_for_feat(self, z: jax.Array) -> jax.Array:
        """L_feat path. Classifier params are frozen; gradient flows through z.

        Uses nnx.split/merge with stop_gradient on all parameter values so
        that ∂L_feat/∂fc1.weight = 0 and ∂L_feat/∂fc2.weight = 0 while
        ∂L_feat/∂z ≠ 0 (gradient propagates back to b_can encoder).
        """
        frozen_state = jax.tree.map(jax.lax.stop_gradient, nnx.state(self))
        cloned = nnx.merge(nnx.graphdef(self), frozen_state)
        return cloned._forward_no_grl(z)

    def __call__(self, b_can: jax.Array) -> jax.Array:
        """Legacy GRL path. Kept for backward compat with lambda_cam > 0 configs."""
        z = jnp.mean(b_can, axis=1)  # pool over token dim → [B, D]
        z = _gradient_reversal(self.norm(z), self.alpha)
        return self.fc2(nnx.relu(self.fc1(z)))


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Unit-normalise along last dimension (float32)."""
    x = x.astype(jnp.float32)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _loss_z_teacher(
    b_can: jax.Array,
    z_teacher: jax.Array,
    z_proj: nnx.Linear,
) -> jax.Array:
    """Per-slot cosine alignment loss: each b_can slot → pooled z_teacher.

    b_can     : [B, n_can, D_pg]
    z_teacher : [B, N_z, D_z]  (frozen Stage-1 z; stopgrad applied here)
    z_proj    : Linear(D_pg → D_z)  — out_features must equal z_teacher.shape[-1]

    Teacher is mean-pooled (preserves weak-prior character). Each b_can slot
    is projected via reshape (one Dense call, not vmap over stateful module)
    and aligned to the shared pooled reference. Loss is mean over slots × batch.
    """
    z_teacher = jax.lax.stop_gradient(z_teacher.astype(jnp.float32))
    z_ref = jnp.mean(z_teacher, axis=1)  # [B, D_z] — pooled teacher
    B, N, D = b_can.shape
    z_hat_flat = z_proj(b_can.reshape(B * N, D).astype(jnp.float32))  # [B*N, D_z]
    if z_hat_flat.shape[-1] != z_ref.shape[-1]:
        raise ValueError(
            f"z_proj output dim ({z_hat_flat.shape[-1]}) != z_teacher dim ({z_ref.shape[-1]}). "
            "Set config.z_teacher_dim to match your Stage-1 checkpoint."
        )
    z_hat = z_hat_flat.reshape(B, N, -1)                              # [B, n_can, D_z]
    cos_per_slot = jnp.sum(
        _l2_normalize(z_hat) * _l2_normalize(z_ref)[:, None, :], axis=-1
    )  # [B, n_can]
    return jnp.mean(1.0 - cos_per_slot)


def _make_episode_random_teacher(episode_id: jax.Array, z_dim: int) -> jax.Array:
    """Generate fixed random unit vectors per episode_id: [B] -> [B, z_dim].

    Same episode_id always maps to the same vector (deterministic via fold_in).
    Vectors are unit-normalised so the cosine loss is well-conditioned.
    This is the episode-random teacher control (G2b): any consistent per-episode
    target that carries NO geometry content from Stage-1.
    """
    base = jax.random.key(42)  # fixed seed — never change
    keys = jax.vmap(lambda eid: jax.random.fold_in(base, eid))(episode_id.astype(jnp.uint32))
    vecs = jax.vmap(lambda k: jax.random.normal(k, (z_dim,)))(keys)
    return _l2_normalize(vecs)  # [B, z_dim]


def _loss_variance(b_can: jax.Array, gamma: float = 1.0) -> jax.Array:
    """VICReg variance term on mean-pooled b_can.

    Computes std over the batch dimension; penalises dimensions that collapse
    below gamma.  Multi-GPU: computed per-device shard (no AllGather needed;
    per-shard variance is sufficient for anti-collapse).

    b_can : [B, n_can, D_pg]
    Returns scalar.
    """
    z = jnp.mean(b_can, axis=1).astype(jnp.float32)  # [B, D_pg]
    # std over batch dimension (axis=0)
    std = jnp.sqrt(jnp.var(z, axis=0) + 1e-6)  # [D_pg]
    return jnp.mean(jnp.maximum(0.0, gamma - std))


def _lambda_schedule(
    lambda_max: float,
    step: jax.Array | int,
    total_steps: int,
) -> jax.Array:
    """Sigmoid warmup schedule.  lambda(t) = lambda_max * (2/(1+exp(-10p))-1), p = step/total_steps.
    At step=0: output=0.  At step=total_steps: output≈lambda_max.
    """
    p = jnp.clip(jnp.asarray(step, dtype=jnp.float32) / float(total_steps), 0.0, 1.0)
    return lambda_max * (2.0 / (1.0 + jnp.exp(-10.0 * p)) - 1.0)


def _loss_covariance(
    b_can: jax.Array,
    proj_cov_can: nnx.Linear,
) -> jax.Array:
    """Per-slot projected VICReg covariance loss.
    For each slot q, project b_can[:,q,:] → [B, D_proj], compute unbiased
    sample covariance C = (1/(B-1)) * Zc^T Zc, penalise off-diagonal entries:
    L_q = (1/D) * sum_{i≠j} C[i,j]^2.  Return mean over slots.
    """
    n_can = b_can.shape[1]
    slot_losses = []
    for q in range(n_can):
        h = proj_cov_can(b_can[:, q, :].astype(jnp.float32))  # [B, D_proj]
        B, D = h.shape[0], h.shape[1]
        h_c = h - jnp.mean(h, axis=0, keepdims=True)
        cov = (h_c.T @ h_c) / (B - 1)  # [D, D] unbiased covariance
        # Off-diagonal: total squared - diagonal squared, scaled by 1/D
        off_diag = (jnp.sum(jnp.square(cov)) - jnp.sum(jnp.square(jnp.diag(cov)))) / D
        slot_losses.append(off_diag)
    return jnp.mean(jnp.stack(slot_losses))


def _loss_diff(
    b_can: jax.Array,
    b_det: jax.Array,
    proj_diff_can: nnx.Linear,
    proj_diff_det: nnx.Linear,
) -> jax.Array:
    """DSN difference loss: penalise linear dependence between b_can and b_det.
    Projects mean-pooled b_can and b_det to [B, D_proj], then returns squared
    Frobenius norm of their unbiased cross-covariance matrix.
    Skipped when b_det is empty (n_det=0).
    """
    if b_det.shape[1] == 0:
        return jnp.zeros((), dtype=jnp.float32)
    h_can = proj_diff_can(jnp.mean(b_can, axis=1).astype(jnp.float32))  # [B, D]
    h_det = proj_diff_det(jnp.mean(b_det, axis=1).astype(jnp.float32))  # [B, D]
    B = h_can.shape[0]
    h_can_c = h_can - jnp.mean(h_can, axis=0, keepdims=True)
    h_det_c = h_det - jnp.mean(h_det, axis=0, keepdims=True)
    cross_cov = (h_can_c.T @ h_det_c) / (B - 1)  # [D_can, D_det]
    return jnp.sum(jnp.square(cross_cov))


# ---------------------------------------------------------------------------
# v3 helpers: KL prior confusion, b_det variance safety net, gated diff loss
# ---------------------------------------------------------------------------

def _camera_kl_to_prior(
    logits: jax.Array,        # [B, K]
    camera_prior: jax.Array,  # [K], sums to 1, > 0
) -> jax.Array:
    """KL(q || prior) where q = softmax(logits).

    Bounded above by log(K) when q is one-hot. Minimum 0 when q == prior.
    Replaces GRL gradient direction "push toward wrong class" with the bounded
    direction "push toward marginal prior" (the true DANN equilibrium).
    """
    log_q = jax.nn.log_softmax(logits, axis=-1)
    q = jnp.exp(log_q)
    log_pi = jnp.log(camera_prior + 1e-12)  # [K]
    kl = jnp.sum(q * (log_q - log_pi[None, :]), axis=-1)  # [B]
    return jnp.mean(kl)


def _loss_variance_det(b_det: jax.Array, gamma: float = 0.3) -> jax.Array:
    """VICReg variance hinge on mean-pooled b_det.

    Symmetric to _loss_variance(b_can) but with smaller gamma since b_det has
    fewer queries (n_det=8 vs n_can=32) and naturally lower variance budget.
    Acts as anti-extinction safety net; does NOT specify b_det content.
    """
    if b_det.shape[1] == 0:  # n_det = 0 ablation
        return jnp.zeros((), dtype=jnp.float32)
    z = jnp.mean(b_det, axis=1).astype(jnp.float32)
    std = jnp.sqrt(jnp.var(z, axis=0) + 1e-6)
    return jnp.mean(jnp.maximum(0.0, gamma - std))


def _loss_diff_gated(
    b_can: jax.Array,
    b_det: jax.Array,
    proj_diff_can: nnx.Linear,
    proj_diff_det: nnx.Linear,
    gamma_det: float,
    gate_steepness: float = 10.0,
) -> tuple[jax.Array, jax.Array]:
    """L_diff with soft-gate on stop_gradient(b_det_std).

    Returns (gated_loss, gate_value) so trainer can log the gate.

    The stop_gradient on b_det_std is critical: without it, the optimizer
    would discover that lowering b_det_std reduces gate, reduces L_diff, and
    is rewarded — accelerating b_det collapse. With stopgrad, the gate
    behaves as a scheduling signal only, not a target.
    """
    if b_det.shape[1] == 0:
        zero = jnp.zeros((), dtype=jnp.float32)
        return zero, zero
    raw = _loss_diff(b_can, b_det, proj_diff_can, proj_diff_det)
    z_det = jnp.mean(b_det, axis=1).astype(jnp.float32)
    b_det_std = jnp.mean(jnp.sqrt(jnp.var(z_det, axis=0) + 1e-6))
    gate = jax.nn.sigmoid(gate_steepness * (jax.lax.stop_gradient(b_det_std) - gamma_det))
    return gate * raw, gate


# ---------------------------------------------------------------------------
# Pi0ACB model
# ---------------------------------------------------------------------------

class Pi0ACB(Pi0):
    """ACB-VLA: pi0.5 with a canonical bottleneck on the action causal path.

    Inference interface is identical to Pi0/pi0.5 (RGB + language + proprio).
    The BottleneckEncoder and L_z_teacher projection are train-time only in the
    sense that their gradients regularise the main policy; at inference the
    bottleneck tokens are computed from the input image and attend into the
    action expert just like any other prefix tokens.
    """

    def __init__(self, config: Pi0ACBConfig, rngs: nnx.Rngs) -> None:
        # Initialise the base pi0.5 model (PaliGemma trunk + action expert).
        # canonical_dim=0 → no M5-style canonical cross-attention in Gemma.
        super().__init__(config, rngs=rngs)

        self.n_can = config.n_can
        self.n_det = config.n_det
        self.shortcut_mode = config.shortcut_mode
        self.lambda_z = config.lambda_z
        self.lambda_var = config.lambda_var
        self.lambda_cam = config.lambda_cam

        # Phase B schedule/loss parameters
        self.lambda_cam_max = config.lambda_cam_max
        self.lambda_z_max = config.lambda_z_max
        self.lambda_cov = config.lambda_cov
        self.lambda_diff = config.lambda_diff
        self.total_train_steps = config.total_train_steps
        self.teacher_mode = config.teacher_mode
        self.z_teacher_dim = config.z_teacher_dim

        # Phase B v3 parameters
        self.lambda_head = config.lambda_head
        self.lambda_feat_max = config.lambda_feat_max
        self.lambda_feat_warmup = config.lambda_feat_warmup
        self.lambda_var_det = config.lambda_var_det
        self.gamma_det = config.gamma_det
        self.lambda_diff_max = config.lambda_diff_max
        self.gate_steepness = config.gate_steepness

        pg_width = _gemma.get_config(config.paligemma_variant).width  # 2048 for gemma_2b

        # Bottleneck encoder: image_tokens → (b_can, b_det)
        self.bottleneck = BottleneckEncoder(
            image_dim=pg_width,
            n_can=config.n_can,
            n_det=config.n_det,
            num_heads=config.bottleneck_num_heads,
            num_layers=config.bottleneck_num_layers,
            rngs=rngs,
        )

        # z_teacher projection: pool(b_can) → Stage-1 z space [D_pg → z_teacher_dim]
        # Used for L_z_teacher; ignored if lambda_z = 0.
        self.z_proj = nnx.Linear(pg_width, config.z_teacher_dim, rngs=rngs)

        # Camera adversarial head: instantiated for legacy GRL (lambda_cam/_max > 0)
        # OR for v3 split discriminator (lambda_feat_max > 0).
        if config.lambda_cam > 0.0 or config.lambda_cam_max > 0.0 or config.lambda_feat_max > 0.0:
            self.cam_adv_head: CameraAdversarialHead | None = CameraAdversarialHead(
                in_dim=pg_width,
                num_classes=config.camera_bin_k,
                hidden_dim=256,
                alpha=1.0,
                rngs=rngs,
            )
        else:
            self.cam_adv_head = None

        # Load empirical camera prior for v3 KL confusion (training-time only).
        # Stored as a plain numpy array so nnx.state() does not traverse it as
        # a JAX-array leaf (raw jax.Array on an nnx.Module raises ValueError in
        # nnx.state() / jax.eval_shape). JAX ops accept numpy arrays directly.
        if config.lambda_feat_max > 0.0 and config.camera_prior_path is not None:
            import numpy as _np
            _prior = jnp.array(_np.load(config.camera_prior_path).astype(_np.float32))
            self.camera_prior: ConstantBuffer | None = ConstantBuffer(_prior)
        else:
            self.camera_prior = None

        # Per-slot covariance projection (Phase B VICReg)
        if config.lambda_cov > 0.0:
            self.proj_cov_can: nnx.Linear | None = nnx.Linear(pg_width, config.cov_proj_dim, rngs=rngs)
        else:
            self.proj_cov_can = None

        # Diff cross-covariance projections (Phase B DSN).
        # For v3, also instantiated when lambda_diff_max > 0 (gated diff loss).
        if config.lambda_diff > 0.0 or config.lambda_diff_max > 0.0:
            self.proj_diff_can: nnx.Linear | None = nnx.Linear(pg_width, config.diff_proj_dim, rngs=rngs)
            self.proj_diff_det: nnx.Linear | None = nnx.Linear(pg_width, config.diff_proj_dim, rngs=rngs)
        else:
            self.proj_diff_can = None
            self.proj_diff_det = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_prefix_split(
        self,
        obs: _model.Observation,
    ) -> tuple[
        at.Float[at.Array, "b s_img d"],
        at.Bool[at.Array, "b s_img"],
        at.Bool[at.Array, " s_img"],
        at.Float[at.Array, "b s_lang d"] | None,
        at.Bool[at.Array, "b s_lang"] | None,
        at.Bool[at.Array, " s_lang"],
    ]:
        """Split embed_prefix into image and language components.

        Returns:
            image_tokens [B, S_img, D_pg]
            image_mask   [B, S_img]
            image_ar     [S_img]   bool (all False = bidirectional)
            lang_tokens  [B, S_lang, D_pg] | None
            lang_mask    [B, S_lang] | None
            lang_ar      [S_lang]  bool (all False)
        """
        img_tok_list = []
        img_mask_list = []
        img_ar: list[bool] = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            img_tok_list.append(image_tokens)  # [B, 256, D_pg]
            img_mask_list.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            img_ar += [False] * image_tokens.shape[1]

        image_tokens = jnp.concatenate(img_tok_list, axis=1)  # [B, S_img, D_pg]
        image_mask = jnp.concatenate(img_mask_list, axis=1)   # [B, S_img]
        image_ar_arr = jnp.array(img_ar, dtype=jnp.bool_)

        lang_tokens = None
        lang_mask = None
        lang_ar_arr = jnp.zeros(0, dtype=jnp.bool_)

        if obs.tokenized_prompt is not None:
            lang_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # [B, S_lang, D_pg]
            lang_mask = obs.tokenized_prompt_mask  # [B, S_lang]
            lang_ar_arr = jnp.zeros(lang_tokens.shape[1], dtype=jnp.bool_)

        return image_tokens, image_mask, image_ar_arr, lang_tokens, lang_mask, lang_ar_arr

    def _build_bottleneck_prefix(
        self,
        obs: _model.Observation,
    ) -> tuple[
        at.Float[at.Array, "b s_new d"],
        at.Bool[at.Array, "b s_new"],
        at.Bool[at.Array, " s_new"],
        int,
        at.Float[at.Array, "b n_can d"],
        at.Float[at.Array, "b n_det d"],
    ]:
        """Build the modified prefix with bottleneck tokens inserted.

        Prefix layout: [image_tokens | b_can | b_det | lang_tokens]

        Returns:
            prefix_tokens [B, S_new, D_pg]
            prefix_mask   [B, S_new]
            prefix_ar     [S_new]
            n_image_tokens : int  (used to locate image columns for mask patching)
            b_can          [B, n_can, D_pg]  (for canonical losses)
            b_det          [B, n_det, D_pg]  (for Phase B diff loss / diagnostics)
        """
        image_tokens, image_mask, image_ar, lang_tokens, lang_mask, lang_ar = (
            self._embed_prefix_split(obs)
        )
        n_image_tokens = image_tokens.shape[1]  # static integer (S_img = 3 × 256 = 768)
        batch_size = image_tokens.shape[0]

        # Run bottleneck encoder
        b_can, b_det = self.bottleneck(image_tokens)  # [B, n_can/n_det, D_pg]

        # All bottleneck tokens are valid (input_mask = True), bidirectional (ar = False)
        bn_mask = jnp.ones((batch_size, self.n_can + self.n_det), dtype=jnp.bool_)
        bn_ar = jnp.zeros(self.n_can + self.n_det, dtype=jnp.bool_)

        # Concatenate into new prefix
        bn_tokens = jnp.concatenate([b_can, b_det], axis=1)  # [B, n_can+n_det, D_pg]

        if lang_tokens is not None:
            prefix_tokens = jnp.concatenate([image_tokens, bn_tokens, lang_tokens], axis=1)
            prefix_mask = jnp.concatenate([image_mask, bn_mask, lang_mask], axis=1)
            prefix_ar = jnp.concatenate([image_ar, bn_ar, lang_ar])
        else:
            prefix_tokens = jnp.concatenate([image_tokens, bn_tokens], axis=1)
            prefix_mask = jnp.concatenate([image_mask, bn_mask], axis=1)
            prefix_ar = jnp.concatenate([image_ar, bn_ar])

        return prefix_tokens, prefix_mask, prefix_ar, n_image_tokens, b_can, b_det

    def _apply_hard_bottleneck(
        self,
        attn_mask: jax.Array,
        s_prefix_new: int,
        n_image_tokens: int,
    ) -> jax.Array:
        """Block suffix (action expert) rows from attending to image token columns.

        attn_mask     : [B, S_full_new, S_full_new]
        s_prefix_new  : number of prefix tokens (image + b_can + b_det + lang)
        n_image_tokens: number of image tokens at the start of the prefix

        The action expert (suffix) queries are at rows [s_prefix_new:].
        Image tokens are at columns [0:n_image_tokens].
        """
        return attn_mask.at[:, s_prefix_new:, :n_image_tokens].set(False)

    # ------------------------------------------------------------------
    # Overridden forward methods
    # ------------------------------------------------------------------

    @override
    def predict_flow(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        canonical_drop_rng: at.KeyArrayLike | None = None,
    ) -> _model.Actions:
        """Forward pass with bottleneck.  Returns predicted flow [B, ah, ad]."""
        # Build modified prefix (image + b_can + b_det + lang)
        # canonical_tokens from observation are the Stage-1 z teacher — NOT injected here.
        prefix_tokens, prefix_mask, prefix_ar, n_image_tokens, _b_can, _b_det = (
            self._build_bottleneck_prefix(observation)
        )

        # Action expert suffix (unchanged from base Pi0)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
            observation, noisy_actions, time
        )

        # Build joint attention mask
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar])
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Hard bottleneck: block action expert from attending to image tokens
        s_prefix_new = prefix_tokens.shape[1]
        if self.shortcut_mode == "hard":
            attn_mask = self._apply_hard_bottleneck(attn_mask, s_prefix_new, n_image_tokens)

        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Two-stream attention (PaliGemma head + action expert).
        # canonical_tokens=None: we never inject Stage-1 z via M5-style cross-attn.
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            canonical_tokens=None,
        )
        del prefix_out

        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        step: int | jax.Array = 0,
    ) -> at.Float[at.Array, "*b ah"]:
        """Total loss = L_flow + phase-A losses + phase-B losses (when lambda>0).
        Returns [B, ah] same interface as base Pi0.compute_loss.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Build modified prefix and extract b_can / b_det for canonical losses
        prefix_tokens, prefix_mask, prefix_ar, n_image_tokens, b_can, b_det = (
            self._build_bottleneck_prefix(observation)
        )

        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar])
        attn_mask = make_attn_mask(input_mask, ar_mask)

        s_prefix_new = prefix_tokens.shape[1]
        if self.shortcut_mode == "hard":
            attn_mask = self._apply_hard_bottleneck(attn_mask, s_prefix_new, n_image_tokens)

        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            canonical_tokens=None,
        )
        del prefix_out

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        # --- L_flow: primary action flow matching loss ---
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B, ah]

        # --- L_z_teacher: resolve teacher target, then apply loss ---
        # teacher_mode controls the source of the teacher target:
        #   "matched"/"shuffled"/"constant" : teacher is observation.canonical_tokens (data pipeline handles shuffled/constant)
        #   "episode_random"                : teacher is a fixed random vector per episode_index (no canonical cache needed)
        # lambda_z_max > 0 uses sigmoid warmup (Phase B); lambda_z alone is static (Phase A).
        z_teacher_scalar = jnp.zeros((), dtype=jnp.float32)

        # Resolve teacher tensor
        z_teacher_target = None
        if self.teacher_mode == "episode_random":
            if observation.episode_index is None and (self.lambda_z > 0.0 or self.lambda_z_max > 0.0):
                raise ValueError(
                    "teacher_mode='episode_random' requires observation.episode_index, "
                    "but it is None. Ensure the data pipeline provides episode_index in the batch. "
                    f"(lambda_z={self.lambda_z}, lambda_z_max={self.lambda_z_max})"
                )
            if observation.episode_index is not None:
                # [B, z_teacher_dim] → unsqueeze to [B, 1, z_teacher_dim] for _loss_z_teacher compatibility
                z_rand = _make_episode_random_teacher(observation.episode_index, self.z_teacher_dim)
                z_teacher_target = z_rand[:, None, :]  # [B, 1, D_z]
        else:
            # matched / shuffled / constant — canonical_tokens provided by data pipeline
            z_teacher_target = observation.canonical_tokens

        if z_teacher_target is not None:
            if self.lambda_z_max > 0.0:
                lam_z = _lambda_schedule(self.lambda_z_max, step, self.total_train_steps)
                z_teacher_scalar = lam_z * _loss_z_teacher(b_can, z_teacher_target, self.z_proj)
            elif self.lambda_z > 0.0:
                z_teacher_scalar = self.lambda_z * _loss_z_teacher(b_can, z_teacher_target, self.z_proj)

        # --- L_variance: VICReg variance term on b_can (anti-collapse) ---
        var_scalar = jnp.zeros((), dtype=jnp.float32)
        if self.lambda_var > 0.0:
            var_scalar = self.lambda_var * _loss_variance(b_can)

        # --- L_covariance (Phase B VICReg covariance, per-slot projected) ---
        cov_scalar = jnp.zeros((), dtype=jnp.float32)
        if self.lambda_cov > 0.0 and self.proj_cov_can is not None:
            cov_scalar = self.lambda_cov * _loss_covariance(b_can, self.proj_cov_can)

        # --- L_diff (Phase B v1 DSN cross-covariance orthogonality, flat lambda) ---
        # Preserved for backward compat. v3 uses lambda_diff_max path (below).
        diff_scalar = jnp.zeros((), dtype=jnp.float32)
        if self.lambda_diff > 0.0 and self.proj_diff_can is not None and self.n_det > 0:
            diff_scalar = self.lambda_diff * _loss_diff(
                b_can, b_det, self.proj_diff_can, self.proj_diff_det
            )

        # --- L_camera_adversarial (Phase B v1 GRL, sigmoid warmup or static) ---
        # Legacy path — disabled in v3 by setting lambda_cam_max = 0.
        cam_scalar = jnp.zeros((), dtype=jnp.float32)
        if (
            self.lambda_cam_max > 0.0
            and self.cam_adv_head is not None
            and observation.camera_bin_id is not None
        ):
            lam_cam = _lambda_schedule(self.lambda_cam_max, step, self.total_train_steps)
            logits = self.cam_adv_head(b_can)  # [B, K] (GRL applies inside)
            labels = observation.camera_bin_id.astype(jnp.int32)  # [B]
            cam_loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
            )
            cam_scalar = lam_cam * cam_loss
        elif (
            self.lambda_cam > 0.0
            and self.cam_adv_head is not None
            and observation.camera_bin_id is not None
        ):
            # Legacy static lambda_cam path
            logits = self.cam_adv_head(b_can)
            labels = observation.camera_bin_id.astype(jnp.int32)
            cam_loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
            )
            cam_scalar = self.lambda_cam * cam_loss

        # -----------------------------------------------------------------------
        # Phase B v3: Split discriminator + KL prior confusion (replaces GRL+CE)
        # Active only when lambda_feat_max > 0; legacy GRL path is lambda_cam_max.
        # -----------------------------------------------------------------------
        cam_head_scalar = jnp.zeros((), dtype=jnp.float32)
        cam_feat_scalar = jnp.zeros((), dtype=jnp.float32)
        head_acc = jnp.zeros((), dtype=jnp.float32)
        lam_feat = jnp.zeros((), dtype=jnp.float32)
        if (
            self.lambda_feat_max > 0.0
            and self.cam_adv_head is not None
            and observation.camera_bin_id is not None
            and self.camera_prior is not None
        ):
            z = jnp.mean(b_can, axis=1)  # [B, D]

            # Path 1: classifier head training (b_can detached from gradient)
            z_detached = jax.lax.stop_gradient(z)
            logits_head = self.cam_adv_head.apply_for_head(z_detached)
            labels = observation.camera_bin_id.astype(jnp.int32)
            head_ce = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits_head, labels=labels
                )
            )
            head_pred = jnp.argmax(logits_head, axis=-1)
            head_acc = jnp.mean((head_pred == labels).astype(jnp.float32))
            cam_head_scalar = self.lambda_head * head_ce

            # Path 2: feature confusion (classifier params frozen; gradient flows through z)
            logits_feat = self.cam_adv_head.apply_for_feat(z)
            feat_kl = _camera_kl_to_prior(logits_feat, self.camera_prior.value)
            if self.lambda_feat_warmup:
                lam_feat = _lambda_schedule(self.lambda_feat_max, step, self.total_train_steps)
            else:
                lam_feat = jnp.asarray(self.lambda_feat_max, dtype=jnp.float32)
            cam_feat_scalar = lam_feat * feat_kl

        # --- L_var on b_det (v3 b_det extinction safety net) ---
        var_det_scalar = jnp.zeros((), dtype=jnp.float32)
        if self.lambda_var_det > 0.0:
            var_det_scalar = self.lambda_var_det * _loss_variance_det(b_det, gamma=self.gamma_det)

        # --- L_diff with soft-gate (v3 upgrade; gated on stopgrad(b_det_std)) ---
        diff_gate_value = jnp.zeros((), dtype=jnp.float32)
        raw_diff_loss = jnp.zeros((), dtype=jnp.float32)
        lam_diff = jnp.zeros((), dtype=jnp.float32)
        if (
            self.lambda_diff_max > 0.0
            and self.proj_diff_can is not None
            and self.n_det > 0
        ):
            raw_diff_loss, diff_gate_value = _loss_diff_gated(
                b_can, b_det,
                self.proj_diff_can, self.proj_diff_det,
                gamma_det=self.gamma_det,
                gate_steepness=self.gate_steepness,
            )
            lam_diff = _lambda_schedule(self.lambda_diff_max, step, self.total_train_steps)
            diff_scalar = diff_scalar + lam_diff * raw_diff_loss

        # --- Combine: scalars broadcast correctly over [B, ah] ---
        total = (
            flow_loss.astype(jnp.float32)
            + z_teacher_scalar
            + var_scalar
            + var_det_scalar      # v3: b_det variance safety net
            + cov_scalar
            + diff_scalar         # v1 flat or v3 gated (additive)
            + cam_scalar          # v1 legacy GRL (0 in v3 configs)
            + cam_head_scalar     # v3: L_head classifier training
            + cam_feat_scalar     # v3: L_feat KL confusion
        )
        return total.astype(flow_loss.dtype)

    def compute_train_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        step: int | jax.Array = 0,
    ) -> dict[str, jax.Array]:
        """Decomposed loss metrics for wandb logging."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar, n_image_tokens, b_can, b_det = (
            self._build_bottleneck_prefix(observation)
        )
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar])
        attn_mask = make_attn_mask(input_mask, ar_mask)
        s_prefix_new = prefix_tokens.shape[1]
        if self.shortcut_mode == "hard":
            attn_mask = self._apply_hard_bottleneck(attn_mask, s_prefix_new, n_image_tokens)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask, positions=positions,
            adarms_cond=[None, adarms_cond], canonical_tokens=None,
        )
        del prefix_out
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        flow_loss = jnp.mean(jnp.square(v_t - u_t))

        metrics: dict[str, jax.Array] = {
            "acb/loss_flow": flow_loss,
        }

        # Resolve teacher target (same logic as compute_loss)
        z_teacher_target_metrics = None
        if self.teacher_mode == "episode_random":
            if observation.episode_index is None and (self.lambda_z > 0.0 or self.lambda_z_max > 0.0):
                raise ValueError(
                    "teacher_mode='episode_random' requires observation.episode_index, "
                    "but it is None. Ensure the data pipeline provides episode_index in the batch. "
                    f"(lambda_z={self.lambda_z}, lambda_z_max={self.lambda_z_max})"
                )
            if observation.episode_index is not None:
                z_rand = _make_episode_random_teacher(observation.episode_index, self.z_teacher_dim)
                z_teacher_target_metrics = z_rand[:, None, :]
        else:
            z_teacher_target_metrics = observation.canonical_tokens

        if self.lambda_z_max > 0.0:
            lam_z_eff = _lambda_schedule(self.lambda_z_max, step, self.total_train_steps)
        else:
            lam_z_eff = self.lambda_z
        metrics["acb/lambda_z"] = lam_z_eff
        if (self.lambda_z_max > 0.0 or self.lambda_z > 0.0) and z_teacher_target_metrics is not None:
            z_loss = _loss_z_teacher(b_can, z_teacher_target_metrics, self.z_proj)
            metrics["acb/loss_z_teacher"] = z_loss
            metrics["acb/loss_z_teacher_weighted"] = lam_z_eff * z_loss

        var_loss = _loss_variance(b_can)
        metrics["acb/loss_variance"] = var_loss
        metrics["acb/loss_variance_weighted"] = self.lambda_var * var_loss

        # Diagnostics for b_can and b_det
        b_can_mean_norm = jnp.mean(jnp.linalg.norm(jnp.mean(b_can, axis=1), axis=-1))
        b_can_std = jnp.mean(jnp.std(jnp.mean(b_can, axis=1), axis=0))
        b_can_per_slot_std = jnp.mean(jnp.std(b_can, axis=0))
        metrics["acb/b_can_mean_norm"] = b_can_mean_norm
        metrics["acb/b_can_feat_std"] = b_can_std
        metrics["acb/b_can_per_slot_std"] = b_can_per_slot_std

        # Phase B covariance / diff metrics (only when corresponding head is enabled)
        if self.lambda_cov > 0.0 and self.proj_cov_can is not None:
            cov_loss = _loss_covariance(b_can, self.proj_cov_can)
            metrics["acb/loss_cov"] = cov_loss
            metrics["acb/loss_cov_weighted"] = self.lambda_cov * cov_loss

        if self.lambda_diff > 0.0 and self.proj_diff_can is not None and self.n_det > 0:
            diff_loss = _loss_diff(b_can, b_det, self.proj_diff_can, self.proj_diff_det)
            metrics["acb/loss_diff"] = diff_loss
            metrics["acb/loss_diff_weighted"] = self.lambda_diff * diff_loss

        lam_cam_eff = jnp.asarray(self.lambda_cam, dtype=jnp.float32)
        if self.lambda_cam_max > 0.0:
            lam_cam_eff = _lambda_schedule(self.lambda_cam_max, step, self.total_train_steps)
        metrics["acb/lambda_cam"] = lam_cam_eff

        if self.cam_adv_head is not None and observation.camera_bin_id is not None:
            labels = observation.camera_bin_id.astype(jnp.int32)
            logits = self.cam_adv_head(b_can)
            cam_loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
            )
            cam_acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            metrics["acb/loss_cam"] = cam_loss
            metrics["acb/loss_cam_weighted"] = lam_cam_eff * cam_loss
            metrics["acb/camera_cls_acc"] = cam_acc.astype(jnp.float32)

        # b_det diagnostics
        if self.n_det > 0:
            b_det_mean_norm = jnp.mean(jnp.linalg.norm(jnp.mean(b_det, axis=1), axis=-1))
            b_det_std = jnp.mean(jnp.std(jnp.mean(b_det, axis=1), axis=0))
            metrics["acb/b_det_mean_norm"] = b_det_mean_norm
            metrics["acb/b_det_feat_std"] = b_det_std

        # Phase B v3 metrics: split discriminator + KL prior confusion
        if (
            self.lambda_feat_max > 0.0
            and self.cam_adv_head is not None
            and observation.camera_bin_id is not None
            and self.camera_prior is not None
        ):
            z_v3 = jnp.mean(b_can, axis=1)
            labels_v3 = observation.camera_bin_id.astype(jnp.int32)

            # Head metrics: classifier accuracy on detached features
            z_detached_v3 = jax.lax.stop_gradient(z_v3)
            logits_head_v3 = self.cam_adv_head.apply_for_head(z_detached_v3)
            head_ce_v3 = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits_head_v3, labels=labels_v3)
            )
            head_acc_v3 = jnp.mean(
                (jnp.argmax(logits_head_v3, axis=-1) == labels_v3).astype(jnp.float32)
            )
            cam_head_w_v3 = self.lambda_head * head_ce_v3

            # Feat metrics: KL confusion loss
            logits_feat_v3 = self.cam_adv_head.apply_for_feat(z_v3)
            feat_kl_v3 = _camera_kl_to_prior(logits_feat_v3, self.camera_prior.value)
            if self.lambda_feat_warmup:
                lam_feat_v3 = _lambda_schedule(self.lambda_feat_max, step, self.total_train_steps)
            else:
                lam_feat_v3 = jnp.asarray(self.lambda_feat_max, dtype=jnp.float32)
            cam_feat_w_v3 = lam_feat_v3 * feat_kl_v3

            metrics.update({
                "acb/loss_cam_head": head_ce_v3,
                "acb/loss_cam_head_weighted": cam_head_w_v3,
                "acb/cam_cls_acc_head": head_acc_v3,
                "acb/loss_cam_feat_kl": feat_kl_v3,
                "acb/loss_cam_feat_kl_weighted": cam_feat_w_v3,
                "acb/lambda_feat": lam_feat_v3,
            })

        # Phase B v3: b_det variance safety net metrics
        if self.lambda_var_det > 0.0:
            var_det_loss = _loss_variance_det(b_det, gamma=self.gamma_det)
            metrics["acb/loss_var_det"] = var_det_loss
            metrics["acb/loss_var_det_weighted"] = self.lambda_var_det * var_det_loss

        # Phase B v3: gated diff loss metrics
        if self.lambda_diff_max > 0.0 and self.proj_diff_can is not None and self.n_det > 0:
            raw_diff_v3, gate_v3 = _loss_diff_gated(
                b_can, b_det,
                self.proj_diff_can, self.proj_diff_det,
                gamma_det=self.gamma_det,
                gate_steepness=self.gate_steepness,
            )
            lam_diff_v3 = _lambda_schedule(self.lambda_diff_max, step, self.total_train_steps)
            metrics["acb/loss_diff_raw"] = raw_diff_v3
            metrics["acb/loss_diff_gate"] = gate_v3
            metrics["acb/lambda_diff"] = lam_diff_v3

        return metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Flow-matching denoising with KV-cached prefix and bottleneck mask.

        The KV cache is built from [image | b_can | b_det | lang] tokens.
        During denoising, the action expert queries use a cross-attn mask that
        blocks attention to image token columns (hard bottleneck at inference).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build modified prefix once (includes bottleneck tokens)
        prefix_tokens, prefix_mask, prefix_ar, n_image_tokens, _b_can, _b_det = (
            self._build_bottleneck_prefix(observation)
        )

        # Build prefix self-attention mask and KV cache
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, scalar_time = carry
            time = jnp.broadcast_to(scalar_time, (batch_size,))

            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)

            # Cross-attention mask: suffix queries → prefix KV cache
            # Shape: [B, S_suffix, S_prefix_new]
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])

            # Hard bottleneck: block suffix from attending to image token columns in cache
            if self.shortcut_mode == "hard":
                cross_mask = cross_mask.at[:, :, :n_image_tokens].set(False)

            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)

            # Suffix positions are offset by the number of valid prefix tokens
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                canonical_tokens=None,
            )

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, scalar_time + dt

        def cond(carry):
            _, scalar_time = carry
            return scalar_time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_from_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s_new d"],
        prefix_mask: at.Bool[at.Array, "b s_new"],
        prefix_ar: at.Bool[at.Array, " s_new"],
        n_image_tokens: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Denoising from a pre-built prefix.  Used for shuffle probes.

        Identical to sample_actions but skips _build_bottleneck_prefix;
        caller supplies prefix_tokens/mask/ar directly, enabling b_can/b_det
        shuffle experiments.  observation is only used by embed_suffix.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, scalar_time = carry
            time = jnp.broadcast_to(scalar_time, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            if self.shortcut_mode == "hard":
                cross_mask = cross_mask.at[:, :, :n_image_tokens].set(False)
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                canonical_tokens=None,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, scalar_time + dt

        def cond(carry):
            _, scalar_time = carry
            return scalar_time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
