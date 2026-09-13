"""Construct a JAX policy with checkpoint-local normalization."""

from pathlib import Path

import jax
import jax.numpy as jnp

from openpi import transforms
from openpi.models import model
from openpi.policies.policy import Policy
from openpi.shared import download
from openpi.training import checkpoints


def create_trained_policy(
    train_config, checkpoint_dir, *, repack_transforms=None, sample_kwargs=None, default_prompt=None, norm_stats=None
):
    checkpoint = Path(download.maybe_download(str(checkpoint_dir)))
    network = train_config.model.load(
        model.restore_params(checkpoint / "params", restore_type=jax.Array, dtype=jnp.bfloat16)
    )
    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data.asset_id)
    repack = repack_transforms or transforms.Group()
    return Policy(
        network,
        transforms=[
            *repack.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
            *repack.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
