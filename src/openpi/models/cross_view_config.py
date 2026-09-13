"""Configuration of the paper's training-only cross-view objective."""

import dataclasses
from typing import Literal

from flax import nnx

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class CrossViewConfig(Pi0Config):
    pi05: bool = True
    action_horizon: int = 10
    discrete_state_input: bool = False
    observation_mode: Literal["paired", "nominal", "mixed"] = "paired"
    lambda_cv: float = 0.10
    cv_action_dim: int = 7
    num_flow_samples: int = 2
    time_beta_alpha: float = 2.0
    time_beta_beta: float = 3.0
    cv_warmup_start: int = 0
    cv_warmup_steps: int = 500
    shuffle_pairs: bool = False
    action_space_weighting: bool = False
    photometric_mode: Literal["independent", "shared", "none"] = "independent"

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.cv_action_dim <= self.action_dim:
            raise ValueError("cv_action_dim must be within the padded action width")
        if self.num_flow_samples < 1 or self.lambda_cv < 0 or self.cv_warmup_steps < 0:
            raise ValueError("Invalid consistency configuration")

    def create(self, rng):
        from openpi.models.cross_view import CrossViewPolicy

        return CrossViewPolicy(self, nnx.Rngs(rng))
