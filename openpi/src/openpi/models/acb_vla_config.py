"""Config for ACB-VLA: Action-grounded Canonical Bottleneck VLA.

The bottleneck sits on the action causal path: the action expert attends ONLY
to (b_can, b_detail, language) tokens, never to raw image tokens.  This forces
the canonical channel to become action-relevant by training (via L_flow), not
by assumption.  Inference is plain RGB+language+proprio and does not require a Stage-1 cache.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.acb_vla import Pi0ACB


@dataclasses.dataclass(frozen=True)
class Pi0ACBConfig(pi0_config.Pi0Config):
    """ACB-VLA extends pi0.5 with a canonical bottleneck on the action path.

    Inherits all Pi0Config fields.  Key new fields:

    Bottleneck architecture
    -----------------------
    n_can       : number of view-stable b_can tokens
    n_det       : number of view-dependent b_detail tokens
    bottleneck_num_heads  : attention heads inside BottleneckEncoder
    bottleneck_num_layers : cross-attn + self-attn layer pairs in encoder

    Loss weights (λ=0 disables)
    ---------------------------
    lambda_z    : L_z_teacher – weak cosine alignment b_can → frozen Stage-1 z
    lambda_var  : L_variance  – VICReg variance term on b_can (anti-collapse)
    lambda_cam  : L_camera_adversarial – GRL camera classifier on b_can (Phase B, static)
    lambda_inv  : L_multiview_invariance – same-state cross-view align (Phase B)

    Phase B loss fields (sigmoid-warmup & additional regularisers)
    --------------------------------------------------------------
    lambda_cam_max : sigmoid warmup max for camera adversarial GRL;
                     lambda_cam(t) = lambda_cam_max * (2/(1+exp(-10*p))-1),
                     p = step/total_train_steps.  Overrides static lambda_cam.
    lambda_z_max   : sigmoid warmup max for z-teacher cosine loss (same schedule).
                     Overrides static lambda_z when > 0.
    lambda_cov     : per-slot VICReg covariance loss weight (constant; hinge-style).
    lambda_diff    : DSN difference / cross-covariance orthogonality loss weight (const).
    cov_proj_dim   : projection dim for per-slot covariance head (D_pg → cov_proj_dim).
    diff_proj_dim  : projection dim for diff cross-covariance heads.
    camera_bin_k   : K for camera K-means bins (num_classes for CameraAdversarialHead).
    total_train_steps : total training steps used for sigmoid warmup schedule.

    Shortcut prevention
    -------------------
    shortcut_mode : 'hard' blocks suffix→image attention via attn_mask;
                    'none' disables (useful for ablation acb_no_shortcut).
    """

    # --- Bottleneck architecture ---
    n_can: int = 32
    n_det: int = 32
    bottleneck_num_heads: int = 8
    bottleneck_num_layers: int = 2

    # --- Loss weights ---
    lambda_z: float = 0.0     # L_z_teacher (skip if 0 or no canonical_tokens)
    lambda_var: float = 0.01  # L_variance  (anti-collapse; always on)
    lambda_cam: float = 0.0   # L_camera_adversarial (Phase B; needs camera labels)
    lambda_inv: float = 0.0   # L_multiview_invariance (Phase B; needs paired frames)

    # --- Stage-1 teacher projection dimension ---
    # Must match the z dimension of the Stage-1 canonical encoder (default: slot_info_nce z=512).
    # Used to build self.z_proj: Linear(pg_width → z_teacher_dim).
    z_teacher_dim: int = 512

    # --- Phase B loss fields ---
    # Sigmoid-warmup maxima (0.0 = disabled / use static lambda_* above)
    lambda_cam_max: float = 0.0    # Phase B sigmoid warmup max for camera adversarial GRL
    lambda_z_max: float = 0.0      # Phase B sigmoid warmup max for z-teacher cosine loss

    # Additional regularisation terms (constant weights)
    lambda_cov: float = 0.0        # Per-slot VICReg covariance loss (Phase B default: 1e-3)
    lambda_diff: float = 0.0       # DSN cross-covariance orthogonality loss (Phase B default: 1e-3)

    # Projection dimensions for Phase B heads
    cov_proj_dim: int = 128        # Projection dim for per-slot covariance head (D_pg → 128)
    diff_proj_dim: int = 128       # Projection dim for diff cross-covariance heads

    # Camera adversarial head configuration
    camera_bin_k: int = 8          # K for camera K-means bins (num_classes in CameraAdversarialHead)

    # Warmup schedule denominator
    total_train_steps: int = 30_000  # Total training steps for sigmoid warmup: p = step/total_train_steps

    # --- v3 Split Discriminator + KL Prior Confusion (Phase B v3) ---
    # All defaults 0 so existing configs are completely unaffected.
    # Set lambda_feat_max > 0 to activate the new v3 path; set lambda_cam_max = 0 simultaneously.

    # L_head: classifier training on stopgrad(b_can); constant weight (no warmup)
    lambda_head: float = 0.0           # 1.0 in v3 configs

    # L_feat: KL(q || prior) feature confusion; sigmoid warmup (bounded by log(K))
    lambda_feat_max: float = 0.0       # 0.03 in v3 configs
    # If True (default), apply sigmoid warmup on lambda_feat (v3 v1.0 behavior).
    # If False, use lambda_feat = lambda_feat_max as a constant from step 0 (v3.1+):
    # the bootstrap is naturally provided by KL itself growing from ~0 as the
    # classifier learns; the additional warmup unnecessarily delays the mechanism.
    lambda_feat_warmup: bool = True
    # Path to .npy of shape [K] summing to 1; required when lambda_feat_max > 0
    camera_prior_path: str | None = None

    # L_var(b_det) safety net: prevents b_det extinction
    lambda_var_det: float = 0.0        # 0.01 in v3 configs
    gamma_det: float = 0.3             # b_det std hinge threshold

    # L_diff upgrade: sigmoid warmup + soft-gate on stopgrad(b_det_std)
    lambda_diff_max: float = 0.0       # 5e-3 in v3 configs; replaces flat lambda_diff
    gate_steepness: float = 10.0       # steepness of the sigmoid gate on b_det_std

    # --- Teacher mode ---
    # Controls what target the z_teacher loss aligns b_can toward:
    #   "matched"       : canonical_tokens from the data pipeline (Stage-1 z of the current episode frame)
    #   "shuffled"      : handled in the data pipeline (canonical_tokens are shuffled across batch); model unchanged
    #   "constant"      : handled in the data pipeline (canonical_tokens replaced by batch mean); model unchanged
    #   "episode_random": a FIXED random unit vector per episode_id (generated from episode_index, no canonical cache needed)
    #                     This is the strongest consistency-only control (G2b): if matched ≈ episode_random,
    #                     Stage-1 geometry contributes nothing beyond regularisation consistency.
    teacher_mode: Literal["matched", "shuffled", "constant", "episode_random"] = "matched"

    # --- Shortcut prevention mode ---
    shortcut_mode: Literal["hard", "none"] = "hard"

    # canonical_dim stays 0 (no Gemma cross-attn injection; that is M5's approach)
    # canonical_token_dropout stays 0.0 (we manage canonical tokens separately)

    @override
    @property
    def model_type(self) -> _model.ModelType:
        # ACB-VLA is a pi0.5-compatible model (adaRMS, discrete state, flow matching)
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ACB":
        from openpi.models.acb_vla import Pi0ACB
        return Pi0ACB(self, rngs=nnx.Rngs(rng))
