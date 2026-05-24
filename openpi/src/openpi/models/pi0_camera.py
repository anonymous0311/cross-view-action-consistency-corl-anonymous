"""Pi0CameraConditioned: pi0.5 with a learnable camera-bin embedding token in the prefix.

This is a camera-aware baseline (G7 gate) for ACB-VLA.  The camera K-means bin ID
is embedded into a single learnable token that is appended to the prefix, giving the
action expert direct access to camera information.

Inference contract:
    camera_bin_id MUST be provided at both training AND inference.
    This is NOT an RGB-only policy — it is a camera-aware reference that documents
    how much explicit camera information is worth.

    ACB-VLA comparison: ACB-VLA uses camera_bin_id only during training (GRL supervision);
    at inference it is RGB+lang+proprio only.  Pi0CameraConditioned requires camera_bin_id
    at inference too.  Document this inference-contract difference explicitly in the paper.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0CameraConfig(pi0_config.Pi0Config):
    """Pi0.5 + camera-bin embedding token in the prefix.

    camera_bin_k : number of K-means camera bins (must match the K used at data time;
                   K=8 bins from assets/camera_bins_k8.parquet)

    Inference: requires observation.camera_bin_id at inference time.
    This is intentionally different from ACB-VLA (which is RGB-only at inference).
    """

    camera_bin_k: int = 8

    @override
    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0CameraConditioned":
        return Pi0CameraConditioned(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Pi0CameraConditioned(Pi0):
    """Pi0.5 with a camera-bin embedding token appended to the prefix.

    The camera token is appended at the END of the prefix (after language tokens),
    so the action expert can attend to it without any mask restriction.
    """

    def __init__(self, config: Pi0CameraConfig, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs=rngs)
        # Camera bin embedding: integer bin → pg_width embedding vector
        # Using nnx.Embed for a learnable lookup table
        import openpi.models.gemma as _gemma
        pg_width = _gemma.get_config(config.paligemma_variant).width
        self.camera_embed = nnx.Embed(config.camera_bin_k, pg_width, rngs=rngs)
        self.camera_bin_k = config.camera_bin_k

    @override
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Override to inject camera-bin embedding token at the end of the prefix."""
        tokens, input_mask, ar_mask = super().embed_prefix(obs)

        if obs.camera_bin_id is not None:
            B = tokens.shape[0]
            cam_ids = obs.camera_bin_id.astype(jnp.int32)  # [B]
            cam_tok = self.camera_embed(cam_ids[:, None])   # [B, 1, D]

            cam_mask = jnp.ones((B, 1), dtype=jnp.bool_)
            cam_ar = jnp.array([False])  # prefix token, no causal mask

            tokens = jnp.concatenate([tokens, cam_tok], axis=1)
            input_mask = jnp.concatenate([input_mask, cam_mask], axis=1)
            ar_mask = jnp.concatenate([ar_mask, cam_ar])

        return tokens, input_mask, ar_mask
