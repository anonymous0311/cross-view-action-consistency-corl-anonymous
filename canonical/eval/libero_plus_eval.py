"""Reusable LIBERO-plus camera evaluation for scene-only and wrist-enabled pi0.5 policies."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
import csv
import dataclasses
import hashlib
import h5py
import io
import json
import math
import os
import pathlib
import re
import sys
import time
import types
from typing import Any

from flax import traverse_util
import jax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pandas as pd
from PIL import Image
from scipy.stats import norm

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import download as _download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as _transforms
from openpi_client import image_tools


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBERO_PLUS_ROOT = REPO_ROOT / "LIBERO-plus"
LIBERO_PLUS_PACKAGE_ROOT = LIBERO_PLUS_ROOT / "libero"
LIBERO_PLUS_BENCHMARK_ROOT = LIBERO_PLUS_ROOT / "libero" / "libero"
DEFAULT_LIBERO_PLUS_CONFIG_DIR = REPO_ROOT / ".libero-plus"
LIBERO_PLUS_TASK_CLASSIFICATION_PATH = LIBERO_PLUS_BENCHMARK_ROOT / "benchmark" / "task_classification.json"
LIBERO_ASSET_FALLBACK_ROOT = REPO_ROOT / "openpi" / "third_party" / "libero" / "libero" / "libero" / "assets"
LIBERO_PLUS_ASSET_LINK = LIBERO_PLUS_BENCHMARK_ROOT / "assets"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
CAMERA_TASK_RE = re.compile(r"^(?P<base>.+)_view_(?P<hv>\d+)_(?P<vv>\d+)_(?P<scale>\d+)_(?P<rotz>\d+)_(?P<roty>\d+)_initstate_(?P<init>\d+)$")
NOMINAL_BASELINE_TASK_SPECS = (
    "libero_spatial:0",
    "libero_spatial:1",
    "libero_object:0",
    "libero_goal:0",
    "libero_10:0",
)
RUNTIME_FLAGS = {"torch_load_patched": False}


def _ensure_repo_imports() -> None:
    for rel_path in ("openpi/src", "openpi/packages/openpi-client/src", "LIBERO-plus"):
        abs_path = REPO_ROOT / rel_path
        if abs_path.exists() and str(abs_path) not in sys.path:
            sys.path.insert(0, str(abs_path))


def _libero_config_dir() -> pathlib.Path:
    raw = os.environ.get("LIBERO_CONFIG_PATH")
    if raw:
        return pathlib.Path(raw).expanduser().resolve()
    return DEFAULT_LIBERO_PLUS_CONFIG_DIR


def ensure_libero_plus_runtime() -> pathlib.Path:
    """Bootstraps LIBERO-plus in-place without interactive prompts."""
    _ensure_repo_imports()
    ensure_wand_compat()
    config_dir = _libero_config_dir()
    config_path = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    if LIBERO_ASSET_FALLBACK_ROOT.exists() and not LIBERO_PLUS_ASSET_LINK.exists():
        LIBERO_PLUS_ASSET_LINK.symlink_to(LIBERO_ASSET_FALLBACK_ROOT)

    defaults = {
        "benchmark_root": str(LIBERO_PLUS_BENCHMARK_ROOT.resolve()),
        "bddl_files": str((LIBERO_PLUS_BENCHMARK_ROOT / "bddl_files").resolve()),
        "init_states": str((LIBERO_PLUS_BENCHMARK_ROOT / "init_files").resolve()),
        "datasets": str((LIBERO_PLUS_PACKAGE_ROOT / "datasets").resolve()),
        "assets": str(LIBERO_PLUS_ASSET_LINK.resolve()),
    }
    config_text = "\n".join(f"{key}: {value}" for key, value in defaults.items()) + "\n"
    current_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    if current_text != config_text:
        tmp_path = config_dir / f".config.{os.getpid()}.tmp"
        tmp_path.write_text(config_text, encoding="utf-8")
        tmp_path.replace(config_path)
    ensure_libero_torch_compat()
    return config_path


def ensure_wand_compat() -> None:
    try:
        from wand.api import library as _wand_library  # noqa: F401, PLC0415
        from wand.image import Image as _wand_image  # noqa: F401, PLC0415
        return
    except (ImportError, OSError):
        pass

    class _MissingMagickWandFunction:
        argtypes = ()

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("motion_blur requires Wand/ImageMagick, but MagickWand is not installed.")

    class _MissingLibrary:
        MagickMotionBlurImage = _MissingMagickWandFunction()

    class _MissingWandImage:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("motion_blur requires Wand/ImageMagick, but MagickWand is not installed.")

    api_module = types.ModuleType("wand.api")
    api_module.library = _MissingLibrary()
    image_module = types.ModuleType("wand.image")
    image_module.Image = _MissingWandImage
    sys.modules["wand.api"] = api_module
    sys.modules["wand.image"] = image_module


def ensure_libero_torch_compat() -> None:
    import torch  # noqa: PLC0415

    if RUNTIME_FLAGS["torch_load_patched"]:
        return

    original_torch_load = torch.load

    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    RUNTIME_FLAGS["torch_load_patched"] = True


def restore_params_portable(
    params_path: str | pathlib.Path,
    *,
    restore_type: type = np.ndarray,
    dtype: jnp.dtype | None = jnp.bfloat16,
) -> dict[str, Any]:
    """Restores a params-only Orbax checkpoint without depending on saved device sharding."""

    def _restore_nnx_sequence_keys(tree: Any) -> Any:
        if isinstance(tree, dict):
            return {
                int(key) if isinstance(key, str) and key.isdecimal() else key: _restore_nnx_sequence_keys(value)
                for key, value in tree.items()
            }
        return tree

    params_path = pathlib.Path(params_path).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = getattr(metadata, "tree", None)
        if item is None and hasattr(metadata, "item_metadata"):
            item = metadata.item_metadata.tree
        if item is None:
            raise AttributeError("Unable to locate checkpoint tree metadata for Orbax restore.")
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=restore_type, dtype=dtype),
                    item,
                ),
            ),
        )
        params = restored["params"]

    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return _restore_nnx_sequence_keys(traverse_util.unflatten_dict(flat_params))


def policy_infer_batched(policy: _policy.Policy, obs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batched equivalent of policy.infer(): runs N observations through the model in one GPU call.

    Per-element semantics are equivalent to N independent policy.infer() calls, except for
    the RNG schedule (one rng split per batched call vs. N sequential splits in the
    unbatched path). For LIBERO eval this means individual trial outcomes may not be
    bit-identical between sequential and batched runs, but aggregate success rates match
    in expectation. Result correctness is preserved (eval is still mathematically valid).

    Reaches into openpi Policy internals (_input_transform, _output_transform,
    _sample_actions, _rng, _sample_kwargs). These are stable across the openpi snapshot
    used here; if openpi changes, this function will fail loudly.
    """
    n = len(obs_list)
    if n == 0:
        return []

    transformed: list[dict[str, Any]] = []
    for obs in obs_list:
        inp = jax.tree.map(lambda x: x, obs)
        inp = policy._input_transform(inp)  # noqa: SLF001
        transformed.append(inp)

    batched = jax.tree.map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *transformed,
    )

    policy._rng, sample_rng = jax.random.split(policy._rng)  # noqa: SLF001

    observation = _model.Observation.from_dict(batched)
    raw_outputs = {
        "state": batched["state"],
        "actions": policy._sample_actions(sample_rng, observation, **policy._sample_kwargs),  # noqa: SLF001
    }

    results: list[dict[str, Any]] = []
    for i in range(n):
        single = jax.tree.map(lambda x, idx=i: np.asarray(x[idx, ...]), raw_outputs)
        single = policy._output_transform(single)  # noqa: SLF001
        results.append(single)
    return results


def warmup_policy(policy: _policy.Policy, *, batch_size: int = 1, resolution: int = 256) -> None:
    """Trigger JAX JIT compilation before evaluation to avoid first-task overhead.

    Without this, the first policy.infer() call in each worker process triggers XLA
    compilation, which can take 30-60 s and appears as a GPU utilization spike followed
    by an artificial performance gap on the first task.

    Pass ``batch_size`` equal to the batch dim that will be used in the hot loop so the
    JIT cache is primed for the right shape (different batch sizes recompile).
    """
    dummy = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "pick up the object and place it on the plate",
    }
    if batch_size <= 1:
        policy.infer(dummy)
    else:
        policy_infer_batched(policy, [dict(dummy) for _ in range(batch_size)])
    print(f"[info] JAX JIT warmup complete (batch_size={batch_size}).", flush=True)


def load_pi05_checkpoint(
    checkpoint_dir: str,
    *,
    config_name: str = "pi05_libero_scene",
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Loads a trained pi0.5 checkpoint with portable params-only restore."""
    train_config = _config.get_config(config_name)
    checkpoint_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir)))

    model = train_config.model.load(
        restore_params_portable(
            checkpoint_dir / "params",
            restore_type=np.ndarray,
            dtype=jnp.bfloat16,
        ),
        remove_extra_params=False,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    del pytorch_device
    return _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
    )


class CanonicalInferenceExtractor:
    """Compatibility stub for removed on-the-fly canonical-token extraction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "On-the-fly legacy Stage-1 canonical-token extraction is not included in this public release. "
            "Use precomputed canonical-token datasets or the current cross-view action-consistency configs."
        )


def _get_benchmark_module() -> Any:
    ensure_libero_plus_runtime()
    from libero.libero import benchmark  # noqa: PLC0415

    return benchmark


def _get_libero_path(query_key: str) -> str:
    ensure_libero_plus_runtime()
    from libero.libero import get_libero_path  # noqa: PLC0415

    return str(get_libero_path(query_key))


def default_max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name not in SUITE_MAX_STEPS:
        known = ", ".join(sorted(SUITE_MAX_STEPS))
        raise ValueError(f"Unknown task suite '{task_suite_name}'. Known: {known}")
    return SUITE_MAX_STEPS[task_suite_name]


def parse_task_spec(task_spec: str) -> tuple[str, int]:
    if ":" not in task_spec:
        raise ValueError(f"Task spec '{task_spec}' must be '<suite_name>:<task_id>'.")
    suite_name, task_id_str = task_spec.split(":", maxsplit=1)
    return suite_name, int(task_id_str)


def quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = float(np.sqrt(max(0.0, 1.0 - quat[3] * quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _orient_libero_image(image: np.ndarray, mode: str) -> np.ndarray:
    image = np.asarray(image)
    if mode == "both":
        return np.ascontiguousarray(image[::-1, ::-1])
    if mode == "flipud":
        return np.ascontiguousarray(image[::-1])
    if mode == "fliplr":
        return np.ascontiguousarray(image[:, ::-1])
    if mode == "none":
        return np.ascontiguousarray(image)
    raise ValueError("image_flip_mode must be one of: both, flipud, fliplr, none")


def build_policy_observation(
    obs: Mapping[str, Any],
    task_description: str,
    *,
    resize_size: int = 224,
    use_wrist_image: bool,
    mask_scene: bool = False,
    canonical_extractor: "CanonicalInferenceExtractor | None" = None,
    image_flip_mode: str = "both",
) -> dict[str, Any]:
    # OpenPI's official LIBERO eval uses "both". Local Phase0A rerendered
    # images were exported with only flipud, so keep eval orientation explicit.
    raw_img = _orient_libero_image(obs["agentview_image"], image_flip_mode)
    base_img = np.zeros_like(raw_img) if mask_scene else raw_img
    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, resize_size, resize_size))

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    policy_obs = {
        "observation/image": base_img,
        "observation/state": state,
        "prompt": str(task_description),
    }
    if use_wrist_image and "robot0_eye_in_hand_image" in obs:
        wrist_img = _orient_libero_image(obs["robot0_eye_in_hand_image"], image_flip_mode)
        policy_obs["observation/wrist_image"] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
        )
    # Inject canonical tokens when extractor is available.
    # When mask_scene=True we omit canonical tokens too so the masking check
    # tests full scene-camera dependency (both visual and spatial pathways zeroed).
    if canonical_extractor is not None and not mask_scene:
        policy_obs["canonical_tokens"] = canonical_extractor.extract(raw_img)
    return policy_obs


def wilson_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    if n_total <= 0:
        raise ValueError("n_total must be > 0 for Wilson CI.")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p_hat = n_success / n_total
    denom = 1 + (z * z) / n_total
    center = (p_hat + (z * z) / (2 * n_total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) / n_total) + ((z * z) / (4 * n_total * n_total))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    max_steps: int
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224


@dataclasses.dataclass(frozen=True)
class CameraTask:
    suite_name: str
    task_index: int
    json_id: int
    task_name: str
    base_task_name: str
    condition: str
    level: int
    horizon_view: int
    vertical_view: int
    scale_factor_percent: int
    endpoint_rot_deg: int
    endpoint_vertical_deg: int

    @property
    def task_spec(self) -> str:
        return f"{self.suite_name}:{self.task_index}"


@dataclasses.dataclass(frozen=True)
class TrialResult:
    condition: str
    suite_name: str
    task_index: int
    json_id: int
    task_name: str
    level: int
    trial_index: int
    rollout_seed: int
    success: bool
    done: bool
    steps_executed: int
    masked_scene: bool


def _stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _classify_camera_condition(match: re.Match[str]) -> str:
    hv = int(match.group("hv"))
    vv = int(match.group("vv"))
    scale = int(match.group("scale"))
    rotz = int(match.group("rotz"))
    roty = int(match.group("roty"))
    if scale != 100:
        return "C1"
    if rotz != 0 or roty != 0:
        return "C3"
    if hv != 0 or vv != 0:
        return "C2"
    raise ValueError(f"Could not classify camera task '{match.string}'.")


def enumerate_camera_tasks() -> list[CameraTask]:
    raw = json.loads(LIBERO_PLUS_TASK_CLASSIFICATION_PATH.read_text(encoding="utf-8"))
    tasks: list[CameraTask] = []
    for suite_name, entries in raw.items():
        for task_index, entry in enumerate(entries):
            if entry.get("category") != "Camera Viewpoints":
                continue
            match = CAMERA_TASK_RE.match(entry["name"])
            if match is None:
                continue
            tasks.append(
                CameraTask(
                    suite_name=suite_name,
                    task_index=task_index,
                    json_id=int(entry["id"]),
                    task_name=str(entry["name"]),
                    base_task_name=str(match.group("base")),
                    condition=_classify_camera_condition(match),
                    level=int(entry["difficulty_level"]),
                    horizon_view=int(match.group("hv")),
                    vertical_view=int(match.group("vv")),
                    scale_factor_percent=int(match.group("scale")),
                    endpoint_rot_deg=int(match.group("rotz")),
                    endpoint_vertical_deg=int(match.group("roty")),
                )
            )
    return tasks


def prefetch_libero_plus_envs(task: CameraTask, *, resolution: int = 256, seed: int = 7, n: int = 3) -> list[Any]:
    """Create N independent LIBERO-plus envs for the given camera task (one per trial).

    All envs are bound to the same BDDL/task but maintain independent MuJoCo state
    so they can be advanced in lockstep with batched policy inference. Returns the
    list of envs (caller closes them).
    """
    if n <= 0:
        return []
    suite = create_task_suite(task.suite_name)
    suite_task = suite.get_task(task.task_index)
    envs: list[Any] = []
    for _ in range(n):
        env, _ = create_libero_plus_env(suite_task, resolution=resolution, seed=seed)
        envs.append(env)
    return envs


def create_task_suite(task_suite_name: str) -> Any:
    benchmark = _get_benchmark_module()
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        known = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"Unknown benchmark '{task_suite_name}'. Known: {known}")
    return benchmark_dict[task_suite_name]()


def create_libero_plus_env(task: Any, *, resolution: int = 256, seed: int = 7) -> tuple[Any, str]:
    ensure_libero_plus_runtime()
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    task_description = task.language
    task_bddl_file = pathlib.Path(_get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def rollout_episode(
    policy: _policy.Policy,
    env: Any,
    task_description: str,
    rollout_config: RolloutConfig,
    *,
    initial_state: np.ndarray,
    use_wrist_image: bool,
    mask_scene: bool = False,
    canonical_extractor: "CanonicalInferenceExtractor | None" = None,
    image_flip_mode: str = "both",
) -> dict[str, Any]:
    reset_obs = env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else reset_obs
    action_plan: list[np.ndarray] = []
    done = False
    info: Mapping[str, Any] | dict[str, Any] = {}
    step_idx = -1

    for step_idx in range(rollout_config.max_steps + rollout_config.num_steps_wait):
        if step_idx < rollout_config.num_steps_wait:
            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue

        if not action_plan:
            policy_obs = build_policy_observation(
                obs,
                task_description,
                resize_size=rollout_config.resize_size,
                use_wrist_image=use_wrist_image,
                mask_scene=mask_scene,
                canonical_extractor=canonical_extractor,
                image_flip_mode=image_flip_mode,
            )
            action_chunk = np.asarray(policy.infer(policy_obs)["actions"])
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError(f"Unexpected action shape {action_chunk.shape}; expected (H, 7)")
            if action_chunk.shape[0] < rollout_config.replan_steps:
                raise ValueError(
                    f"Need at least {rollout_config.replan_steps} planned actions, got {action_chunk.shape[0]}"
                )
            action_plan = [np.asarray(x, dtype=np.float32) for x in action_chunk[: rollout_config.replan_steps]]

        action = action_plan.pop(0)
        obs, _, done, info = env.step(action.tolist())
        if done:
            break

    success = bool(done)
    if isinstance(info, Mapping) and "success" in info:
        success = bool(info["success"])
    if hasattr(env, "check_success"):
        success = success or bool(env.check_success())
    return {
        "success": success,
        "done": bool(done),
        "steps_executed": int(step_idx + 1),
        "info": dict(info) if isinstance(info, Mapping) else {},
    }


def rollout_episode_batched(
    policy: _policy.Policy,
    envs: list[Any],
    task_description: str,
    rollout_config: RolloutConfig,
    *,
    initial_states: list[Any],
    use_wrist_image: bool,
    mask_scene: bool = False,
    canonical_extractor: "CanonicalInferenceExtractor | None" = None,
    image_flip_mode: str = "both",
    env_executor: Any = None,
) -> list[dict[str, Any]]:
    """Run N envs in lockstep with one batched policy.infer() per replan boundary.

    All envs share the same task_description and rollout_config (they are different
    trials of the same task). Each env independently resets to its own initial_state.

    Image orientation (image_flip_mode), wrist image inclusion, mask_scene zeroing,
    and canonical token injection are applied per env via build_policy_observation,
    exactly as in the sequential rollout; the batching happens at the model call
    only, not at the observation builder.

    Done envs are skipped from further env.step() calls but kept in the inference
    batch (their slot is ignored when collecting actions) so the JIT-compiled
    inference path always sees batch_size = len(envs). Changing batch size mid-run
    would trigger XLA recompilation, defeating the warmup.

    Keep env_executor=None for LIBERO/robosuite. MuJoCo/EGL env.step/render is not
    thread-safe across these env instances and can silently destroy task success.
    """
    n = len(envs)
    if len(initial_states) != n:
        raise ValueError(f"envs has {n} entries but initial_states has {len(initial_states)}")
    if n == 0:
        return []

    obs_list: list[Any] = []
    for env_i, init_state in zip(envs, initial_states):
        env_i.reset()
        if init_state is not None:
            obs_list.append(env_i.set_init_state(init_state))
        else:
            obs_list.append(env_i.reset())

    action_plans: list[list[np.ndarray]] = [[] for _ in range(n)]
    done_flags: list[bool] = [False] * n
    info_list: list[Any] = [{} for _ in range(n)]
    step_count: list[int] = [0] * n

    def _step_one(env_idx: int, action: list[float]) -> None:
        if done_flags[env_idx]:
            return
        obs_new, _, done, info = envs[env_idx].step(action)
        obs_list[env_idx] = obs_new
        info_list[env_idx] = info
        step_count[env_idx] += 1
        if bool(done):
            done_flags[env_idx] = True

    def _step_all(actions: list[list[float]]) -> None:
        if env_executor is None:
            for i in range(n):
                _step_one(i, actions[i])
        else:
            futs = [env_executor.submit(_step_one, i, actions[i]) for i in range(n)]
            for fut in futs:
                fut.result()

    # Wait phase: all envs step a dummy action together.
    for _ in range(rollout_config.num_steps_wait):
        if all(done_flags):
            break
        _step_all([list(LIBERO_DUMMY_ACTION)] * n)

    for _ in range(rollout_config.max_steps):
        if all(done_flags):
            break

        needs_replan = any((not action_plans[i]) and (not done_flags[i]) for i in range(n))
        if needs_replan:
            batched_obs: list[dict[str, Any]] = []
            for i in range(n):
                # Build the observation for every env (including done ones) to keep
                # batch_size constant; outputs for done envs are discarded below.
                batched_obs.append(
                    build_policy_observation(
                        obs_list[i],
                        task_description,
                        resize_size=rollout_config.resize_size,
                        use_wrist_image=use_wrist_image,
                        mask_scene=mask_scene,
                        canonical_extractor=canonical_extractor,
                        image_flip_mode=image_flip_mode,
                    )
                )
            infer_results = policy_infer_batched(policy, batched_obs)
            for i in range(n):
                if done_flags[i] or action_plans[i]:
                    continue
                chunk = np.asarray(infer_results[i]["actions"])
                if chunk.ndim != 2 or chunk.shape[1] != 7:
                    raise ValueError(f"Unexpected action shape {chunk.shape}; expected (H, 7)")
                if chunk.shape[0] < rollout_config.replan_steps:
                    raise ValueError(
                        f"Need at least {rollout_config.replan_steps} planned actions, got {chunk.shape[0]}"
                    )
                action_plans[i] = [np.asarray(x, dtype=np.float32) for x in chunk[: rollout_config.replan_steps]]

        step_actions: list[list[float]] = []
        for i in range(n):
            if done_flags[i] or not action_plans[i]:
                step_actions.append(list(LIBERO_DUMMY_ACTION))  # ignored by _step_one for done envs
            else:
                step_actions.append(action_plans[i].pop(0).tolist())
        _step_all(step_actions)

    results: list[dict[str, Any]] = []
    for i in range(n):
        info = info_list[i]
        success = bool(done_flags[i])
        if isinstance(info, Mapping) and "success" in info:
            success = bool(info["success"])
        if hasattr(envs[i], "check_success"):
            success = success or bool(envs[i].check_success())
        results.append(
            {
                "success": success,
                "done": bool(done_flags[i]),
                "steps_executed": int(step_count[i]),
                "info": dict(info) if isinstance(info, Mapping) else {},
            }
        )
    return results


def _aggregate_trials(
    trials: Iterable[TrialResult],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    rows = list(trials)
    n_total = len(rows)
    n_success = sum(int(row.success) for row in rows)
    ci_low, ci_high = wilson_ci(n_success, n_total, confidence=confidence)
    exemplar = rows[0]
    return {
        "condition": exemplar.condition,
        "suite_name": exemplar.suite_name,
        "task_id": exemplar.task_index,
        "json_id": exemplar.json_id,
        "task_name": exemplar.task_name,
        "level": exemplar.level,
        "n_success": n_success,
        "n_total": n_total,
        "success_rate": n_success / n_total,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


class LIBEROPlusCameraEvaluator:
    """Evaluate a policy on the LIBERO-plus camera perturbation subset."""

    def __init__(
        self,
        *,
        policy: _policy.Policy,
        n_trials_per_task: int = 20,
        results_dir: str | pathlib.Path = "results/main_table/m0",
        rollout_config: RolloutConfig | None = None,
        resolution: int = 256,
        base_seed: int = 7,
        use_wrist_image: bool = False,
        progress_every: int = 25,
        canonical_extractor: "CanonicalInferenceExtractor | None" = None,
        image_flip_mode: str = "both",
    ) -> None:
        ensure_libero_plus_runtime()
        self.policy = policy
        self.n_trials_per_task = int(n_trials_per_task)
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = int(resolution)
        self.base_seed = int(base_seed)
        self.use_wrist_image = bool(use_wrist_image)
        self.progress_every = int(progress_every)
        self.camera_tasks = enumerate_camera_tasks()
        self.rollout_config = rollout_config or RolloutConfig(max_steps=0)
        self.raw_rows: list[TrialResult] = []
        self.evaluation_log_path = self.results_dir / "evaluation_log.txt"
        self.canonical_extractor = canonical_extractor
        if image_flip_mode not in {"both", "flipud", "fliplr", "none"}:
            raise ValueError("image_flip_mode must be one of: both, flipud, fliplr, none")
        self.image_flip_mode = image_flip_mode

    def _log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.evaluation_log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(line + "\n")

    def _rollout_config_for_suite(self, suite_name: str) -> RolloutConfig:
        max_steps = self.rollout_config.max_steps or default_max_steps_for_suite(suite_name)
        return dataclasses.replace(self.rollout_config, max_steps=max_steps)

    def _task_records(self, condition: str) -> list[CameraTask]:
        return [task for task in self.camera_tasks if task.condition == condition]

    def run_single_task(
        self,
        task: CameraTask,
        n_trials: int | None = None,
        *,
        mask_scene: bool = False,
        prebuilt_envs: list[Any] | None = None,
    ) -> list[TrialResult]:
        """Run all trials for one camera task using batched lockstep rollout.

        Each trial gets its own env so we can advance them in parallel and submit
        observations to the policy as a single batched call (N=n_trials).
        Provide `prebuilt_envs` (length n_trials) to skip in-thread env creation
        (typical use is a background prefetch from the caller's task loop).
        """
        n_trials = self.n_trials_per_task if n_trials is None else int(n_trials)
        suite = create_task_suite(task.suite_name)
        suite_task = suite.get_task(task.task_index)
        initial_states = suite.get_task_init_states(task.task_index)
        if n_trials > len(initial_states):
            raise ValueError(
                f"Requested {n_trials} trials for {task.task_spec}, but only {len(initial_states)} initial states exist."
            )

        task_description = str(suite_task.language)
        if prebuilt_envs is not None:
            if len(prebuilt_envs) != n_trials:
                raise ValueError(
                    f"prebuilt_envs has {len(prebuilt_envs)} entries but n_trials={n_trials}"
                )
            envs = list(prebuilt_envs)
        else:
            envs = []
            for _ in range(n_trials):
                env_i, _ = create_libero_plus_env(
                    suite_task,
                    resolution=self.resolution,
                    seed=self.base_seed,
                )
                envs.append(env_i)

        rollout_cfg = self._rollout_config_for_suite(task.suite_name)

        rollout_seeds: list[int] = []
        for trial_index in range(n_trials):
            rs = _stable_seed(self.base_seed, task.task_spec, trial_index, int(mask_scene))
            envs[trial_index].seed(rs)
            rollout_seeds.append(rs)

        initial_states_used = [initial_states[t] for t in range(n_trials)]

        task_rows: list[TrialResult] = []
        try:
            results = rollout_episode_batched(
                self.policy,
                envs,
                task_description,
                rollout_cfg,
                initial_states=initial_states_used,
                use_wrist_image=self.use_wrist_image,
                mask_scene=mask_scene,
                canonical_extractor=self.canonical_extractor,
                image_flip_mode=self.image_flip_mode,
                env_executor=None,
            )
            for trial_index, result in enumerate(results):
                row = TrialResult(
                    condition=task.condition,
                    suite_name=task.suite_name,
                    task_index=task.task_index,
                    json_id=task.json_id,
                    task_name=task.task_name,
                    level=task.level,
                    trial_index=trial_index,
                    rollout_seed=rollout_seeds[trial_index],
                    success=bool(result["success"]),
                    done=bool(result["done"]),
                    steps_executed=int(result["steps_executed"]),
                    masked_scene=bool(mask_scene),
                )
                task_rows.append(row)
        finally:
            for env_i in envs:
                try:
                    env_i.close()
                except Exception:
                    pass
        return task_rows

    def run_condition(self, condition: str, *, mask_scene: bool = False) -> list[dict[str, Any]]:
        if condition not in {"C1", "C2", "C3"}:
            raise ValueError("condition must be one of {'C1', 'C2', 'C3'}.")

        condition_rows: list[TrialResult] = []
        tasks = self._task_records(condition)
        total_rollouts = len(tasks) * self.n_trials_per_task
        self._log(f"Starting condition {condition} with {len(tasks)} tasks and {total_rollouts} rollouts.")
        completed = 0
        for task in tasks:
            task_rows = self.run_single_task(task, mask_scene=mask_scene)
            condition_rows.extend(task_rows)
            completed += len(task_rows)
            if completed % max(1, self.progress_every) == 0 or completed == total_rollouts:
                successes = sum(int(row.success) for row in condition_rows)
                self._log(
                    f"{condition}: completed {completed}/{total_rollouts} rollouts "
                    f"(success_rate={successes / max(1, completed):.3f})"
                )
        self.raw_rows.extend(condition_rows)
        return self.compute_aggregate_with_ci(condition_rows)

    def run_full_evaluation(self) -> dict[str, Any]:
        started_at = time.time()
        self.raw_rows = []
        per_task_rows: list[dict[str, Any]] = []
        for condition in ("C1", "C2", "C3"):
            per_task_rows.extend(self.run_condition(condition))

        aggregate_rows = self.aggregate_by_condition_and_level(self.raw_rows)
        self.write_results_csv(per_task_rows, self.results_dir / "per_task_results.csv")
        self.write_markdown_summary(aggregate_rows, self.results_dir / "aggregate_summary.md")
        return {
            "per_task_rows": per_task_rows,
            "aggregate_rows": aggregate_rows,
            "raw_rows": [dataclasses.asdict(row) for row in self.raw_rows],
            "wall_time_sec": time.time() - started_at,
        }

    def compute_aggregate_with_ci(self, results: Iterable[TrialResult]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, int], list[TrialResult]] = {}
        for row in results:
            key = (row.condition, row.suite_name, row.task_index)
            grouped.setdefault(key, []).append(row)
        return [_aggregate_trials(group) for group in grouped.values()]

    def aggregate_by_condition_and_level(self, results: Iterable[TrialResult]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int], list[TrialResult]] = {}
        condition_grouped: dict[str, list[TrialResult]] = {}
        for row in results:
            grouped.setdefault((row.condition, row.level), []).append(row)
            condition_grouped.setdefault(row.condition, []).append(row)

        rows: list[dict[str, Any]] = []
        for (condition, level), group_rows in sorted(grouped.items()):
            n_total = len(group_rows)
            n_success = sum(int(row.success) for row in group_rows)
            ci_low, ci_high = wilson_ci(n_success, n_total)
            rows.append(
                {
                    "condition": condition,
                    "level": level,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        for condition, group_rows in sorted(condition_grouped.items()):
            n_total = len(group_rows)
            n_success = sum(int(row.success) for row in group_rows)
            ci_low, ci_high = wilson_ci(n_success, n_total)
            rows.append(
                {
                    "condition": condition,
                    "level": "Aggregate",
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        return rows

    def write_results_csv(self, results: Iterable[Mapping[str, Any]], path: str | pathlib.Path) -> None:
        rows = list(results)
        if not rows:
            return
        path = pathlib.Path(path)
        fieldnames = [
            "condition",
            "suite_name",
            "task_id",
            "json_id",
            "task_name",
            "level",
            "n_success",
            "n_total",
            "success_rate",
            "ci_low",
            "ci_high",
        ]
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def write_markdown_summary(self, results: Iterable[Mapping[str, Any]], path: str | pathlib.Path) -> None:
        rows = list(results)
        path = pathlib.Path(path)
        lines = [
            "# LIBERO-plus Camera Evaluation",
            "",
            "| Condition | Level | Success Rate (95% Wilson CI) | n_success / n_total |",
            "|---|---:|---:|---:|",
        ]
        def _level_key(value: Any) -> tuple[int, str]:
            if isinstance(value, int):
                return (0, f"{value:02d}")
            return (1, str(value))

        for row in sorted(rows, key=lambda item: (str(item["condition"]), _level_key(item["level"]))):
            lines.append(
                "| "
                f"{row['condition']} | {row['level']} | "
                f"{100 * float(row['success_rate']):.1f}% "
                f"([{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]) | "
                f"{row['n_success']} / {row['n_total']} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_with_scene_camera_masked(
    policy: _policy.Policy,
    evaluator: LIBEROPlusCameraEvaluator,
    *,
    n_trials: int = 20,
    condition: str = "nominal",
) -> dict[str, Any]:
    # Note: mask_scene=True suppresses canonical_extractor too (see build_policy_observation)
    """Sanity check: zero the scene camera and evaluate."""
    if condition != "nominal":
        raise ValueError("Only condition='nominal' is supported for the masking sanity check.")

    raw_rows: list[TrialResult] = []
    for task_spec in NOMINAL_BASELINE_TASK_SPECS:
        suite_name, task_id = parse_task_spec(task_spec)
        suite = create_task_suite(suite_name)
        suite_task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = create_libero_plus_env(
            suite_task,
            resolution=evaluator.resolution,
            seed=evaluator.base_seed,
        )
        rollout_cfg = evaluator._rollout_config_for_suite(suite_name)
        try:
            for trial_index in range(min(n_trials, len(initial_states))):
                rollout_seed = _stable_seed(evaluator.base_seed, task_spec, trial_index, "mask")
                env.seed(rollout_seed)
                result = rollout_episode(
                    policy,
                    env,
                    task_description,
                    rollout_cfg,
                    initial_state=initial_states[trial_index],
                    use_wrist_image=evaluator.use_wrist_image,
                    mask_scene=True,
                    canonical_extractor=evaluator.canonical_extractor,
                    image_flip_mode=evaluator.image_flip_mode,
                )
                raw_rows.append(
                    TrialResult(
                        condition="masked_nominal",
                        suite_name=suite_name,
                        task_index=task_id,
                        json_id=-1,
                        task_name=str(suite_task.name),
                        level=0,
                        trial_index=trial_index,
                        rollout_seed=rollout_seed,
                        success=bool(result["success"]),
                        done=bool(result["done"]),
                        steps_executed=int(result["steps_executed"]),
                        masked_scene=True,
                    )
                )
        finally:
            env.close()

    n_total = len(raw_rows)
    n_success = sum(int(row.success) for row in raw_rows)
    ci_low, ci_high = wilson_ci(n_success, n_total)
    return {
        "condition": condition,
        "n_success": n_success,
        "n_total": n_total,
        "success_rate": n_success / max(1, n_total),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "raw_rows": [dataclasses.asdict(row) for row in raw_rows],
    }
