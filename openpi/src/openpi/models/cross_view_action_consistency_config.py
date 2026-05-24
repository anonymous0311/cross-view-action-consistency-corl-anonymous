import dataclasses
from typing import Literal

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0CrossViewActionConsistencyConfig(pi0_config.Pi0Config):
    """pi0.5 with training-time cross-view action-flow consistency.

    This config intentionally keeps the vanilla pi0 inference interface. The
    paired view axis is consumed only by ``compute_loss`` during cross-view
    training.
    """

    lambda_cv: float = 0.0
    total_train_steps: int = 10_000
    cv_warmup_start_fraction: float = 0.10
    cv_warmup_end_fraction: float = 0.30
    cv_action_dim: int = 7
    cv_pair_mode: str = "matched"
    cv_loss_mode: Literal["symmetric", "multi_sample_asymmetric"] = "symmetric"
    cv_num_samples: int = 1
    cv_anchor_view: Literal["nominal"] = "nominal"
    cv_student_view: Literal["perturbed"] = "perturbed"
    cv_stopgrad_anchor: bool = True
    cv_time_distribution: Literal["uniform", "beta_1p5_1", "beta_2p0_3p0", "beta_1p0_1p5"] = "beta_1p5_1"
    cv_eps_shared_across_views: bool = True
    cv_average_over_samples: bool = True
    pair_spatial_aug_mode: Literal["current", "none", "shared"] = "current"
    pair_photometric_aug_mode: Literal["current", "none", "shared", "independent"] = "current"

    @override
    def create(self, rng: at.KeyArrayLike) -> _model.BaseModel:
        from openpi.models.cross_view_action_consistency import Pi0CrossViewActionConsistency

        return Pi0CrossViewActionConsistency(self, rngs=nnx.Rngs(rng))
