"""Short compiled training, checkpoint/resume and single-view inference check.

Uses a small language/action expert with the regular vision backbone and synthetic
inputs. This is an execution test, not a reproduction of benchmark success rates.
"""

import argparse
import dataclasses
import gc
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as base_model
from openpi.shared import nnx_utils
from openpi.training import config, weight_loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    parser.add_argument("--dataset", type=Path, help="Optional exported paired dataset for a real-data check")
    parser.add_argument("--recipe", default="paired_cv", choices=["paired_cv", "nominal_only", "mixed_camera"])
    args = parser.parse_args()
    print("Devices:", jax.devices(), flush=True)
    spec = importlib.util.spec_from_file_location("afcv_train", Path(__file__).with_name("train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = config.get_config(args.recipe)
    cfg = dataclasses.replace(
        cfg,
        name="smoke",
        exp_name="execution",
        checkpoint_base_dir=str(args.output_dir),
        model=dataclasses.replace(cfg.model, paligemma_variant="dummy", action_expert_variant="dummy", max_token_len=8),
        data=config.FakeDataConfig()
        if args.dataset is None
        else dataclasses.replace(cfg.data, repo_id=str(args.dataset.resolve())),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        batch_size=2,
        num_workers=0,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        freeze_filter=nnx_utils.PathRegex("PaliGemma/img/.*"),
        wandb_enabled=False,
    )
    train.main(cfg)
    train.main(dataclasses.replace(cfg, resume=True, num_train_steps=3))
    gc.collect()
    checkpoint = cfg.checkpoint_dir / "3" / "params"
    policy = cfg.model.load(base_model.restore_params(checkpoint, restore_type=jax.Array, dtype=jnp.bfloat16))
    obs = cfg.model.fake_obs(batch_size=1)
    obs = dataclasses.replace(
        obs,
        image_masks={k: jnp.array([k == "base_0_rgb"]) for k in obs.images},
        tokenized_prompt_mask=jnp.ones((1, 8), dtype=jnp.bool_),
    )
    output = nnx_utils.module_jit(policy.sample_actions)(jax.random.key(7), obs, num_steps=2)
    output = np.asarray(output)
    assert output.shape == (1, 10, 32) and np.isfinite(output).all()
    print("PASS: compiled training, checkpoint resume, and single-view inference", flush=True)


if __name__ == "__main__":
    main()
