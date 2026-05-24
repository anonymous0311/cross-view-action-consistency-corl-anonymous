import dataclasses
import functools
import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _visible_nvidia_gpu_count() -> int | None:
    """Returns number of visible NVIDIA GPUs, or None if unknown."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    result = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))


def _to_wandb_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.size == 0:
        return image.astype(np.uint8)
    if np.nanmin(image) < 0.0:
        image = (image + 1.0) * 127.5
    elif np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _make_camera_log_image(images: dict[str, Any], sample_index: int) -> np.ndarray:
    """Build a wandb-compatible image grid for regular or paired image batches."""
    camera_images = []
    for image in images.values():
        sample = _to_wandb_uint8_image(np.asarray(image[sample_index]))
        if sample.ndim == 3:
            sample = sample[None, ...]
        elif sample.ndim > 4:
            sample = sample.reshape((-1, *sample.shape[-3:]))
        if sample.ndim != 4:
            raise ValueError(f"Expected image sample with shape [H,W,C] or [V,H,W,C], got {sample.shape}")
        camera_images.append(sample)

    num_views = max(sample.shape[0] for sample in camera_images)
    rows = []
    for view_index in range(num_views):
        row = [sample[min(view_index, sample.shape[0] - 1)] for sample in camera_images]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _log_and_validate_backend() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    logging.info(f"JAX backend: {backend}, devices: {devices}")

    if backend != "cpu":
        return

    allow_cpu = os.environ.get("OPENPI_ALLOW_CPU", "0") == "1"
    gpu_count = _visible_nvidia_gpu_count()

    if gpu_count is not None and gpu_count > 0 and not allow_cpu:
        raise RuntimeError(
            "JAX is running on CPU even though NVIDIA GPUs are visible. "
            "This commonly means CUDA plugin initialization failed (for example: cuDNN too old for current CUDA/cublasLt). "
            "Fix your CUDA/cuDNN stack so JAX can initialize GPUs, then retry. "
            "If you intentionally want CPU training, set OPENPI_ALLOW_CPU=1."
        )

    if not allow_cpu:
        logging.warning(
            "JAX backend is CPU. Training may be extremely slow and first-step compilation can take a long time. "
            "Set OPENPI_ALLOW_CPU=1 to suppress this warning."
        )


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True, step=state.step)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    # Canonical cross-attention diagnostics — logged only when canonical tokens are present.
    # canonical/token_norm:  mean L2 norm of the input canonical tokens (should be ~non-zero from step 0)
    # canonical/can_out_norm: Frobenius norm of can_out/kernel (zero-init; grows as canonical cross-attn activates)
    # canonical/param_norm:  global norm of all can_* kernels (tracks overall learning in canonical cross-attn)
    if observation.canonical_tokens is not None:
        info["canonical/token_norm"] = jnp.mean(
            jnp.linalg.norm(observation.canonical_tokens, axis=-1)
        )
        if observation.canonical_tokens_neg is not None:
            info["canonical/token_neg_norm"] = jnp.mean(
                jnp.linalg.norm(observation.canonical_tokens_neg, axis=-1)
            )
        if observation.canonical_tokens_mean is not None:
            info["canonical/token_mean_norm"] = jnp.mean(
                jnp.linalg.norm(observation.canonical_tokens_mean, axis=-1)
            )
        can_all = nnx.state(model, nnx.All(nnx.Param, nnx_utils.PathRegex(".*/can_.*/kernel")))
        can_out = nnx.state(model, nnx.All(nnx.Param, nnx_utils.PathRegex(".*/can_out/kernel")))
        if jax.tree_util.tree_leaves(can_all):
            info["canonical/param_norm"] = optax.global_norm(can_all)
        if jax.tree_util.tree_leaves(can_out):
            info["canonical/can_out_norm"] = optax.global_norm(can_out)
        anchor_params = nnx.state(model, nnx.All(nnx.Param, nnx_utils.PathRegex("anchor/.*")))
        if jax.tree_util.tree_leaves(anchor_params):
            info["m6_anchor/param_norm"] = optax.global_norm(anchor_params)
        gate_params = nnx.state(model, nnx.All(nnx.Param, nnx_utils.PathRegex("gate/.*")))
        if jax.tree_util.tree_leaves(gate_params):
            info["m6_anchor/gate_param_norm"] = optax.global_norm(gate_params)

    return new_state, info


@at.typecheck
def metrics_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    if not hasattr(model, "compute_train_metrics"):
        return {}
    observation, actions = batch
    metrics_rng = jax.random.fold_in(rng, state.step + 17_171)
    return model.compute_train_metrics(metrics_rng, observation, actions, step=state.step)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    _log_and_validate_backend()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    logging.info("Fetching first batch from data loader...")
    first_batch_start = time.time()
    batch = next(data_iter)
    logging.info("Fetched first batch in %.2f seconds.", time.time() - first_batch_start)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(_make_camera_log_image(batch[0].images, i))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pmetrics_step = jax.jit(
        metrics_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        step_start = time.time()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if step == start_step:
            logging.info("First train step (includes JIT compile) took %.2f seconds.", time.time() - step_start)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            with sharding.set_mesh(mesh):
                metrics_info = pmetrics_step(train_rng, train_state, batch)
            if metrics_info:
                reduced_info.update(jax.device_get(metrics_info))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
