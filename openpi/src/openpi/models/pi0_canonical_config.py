"""Config for π0.5 with canonical spatial cross-attention (Stage 2 M5 model)."""
import dataclasses

from openpi.models import pi0_config


@dataclasses.dataclass(frozen=True)
class Pi0CanonicalConfig(pi0_config.Pi0Config):
    """π0.5 + canonical cross-attention in every action expert block.

    canonical_dim=512 is forced (matches Stage 1 CanonicalTransformer output dim).
    Canonical tokens arrive pre-computed from the data transform — the model itself
    does not contain the canonical transformer.
    """

    # Path to the Stage 1 checkpoint used to pre-compute canonical token cache.
    # Not used by the model at training time — only by the caching script.
    stage1_checkpoint: str = "checkpoints/stage1_v3_slotnce"

    # Override Pi0Config default (0.0) → 10% dropout for canonical configs by default.
    # During training, zeros out canonical_tokens for this fraction of batches, creating a
    # gradient contrast that incentivises the model to actively exploit canonical features.
    # Set to 0.0 in Phase 1 (pretrained frozen already forces canonical — no dropout needed).
    canonical_token_dropout: float = 0.1

    def __post_init__(self):
        # Force canonical_dim=512 (matches CanonicalTransformer output dim)
        object.__setattr__(self, "canonical_dim", 512)
        super().__post_init__()
