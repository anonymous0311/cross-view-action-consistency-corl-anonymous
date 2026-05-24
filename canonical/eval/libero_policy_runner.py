"""Shared LIBERO policy rollout utilities for camera-robustness evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import dataclasses
import math
import os
import pathlib
from typing import Any

import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import tqdm

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_LIBERO_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"

SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
RUNTIME_FLAGS = {"torch_load_patched": False}


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    max_steps: int
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224


def warmup_policy(policy: _policy.Policy, *, batch_size: int = 1, resolution: int = 256) -> None:
    """Trigger JAX JIT compilation before evaluation to avoid first-task overhead.

    Without this, the first policy.infer() call in each worker process triggers XLA
    compilation, which can take 30-60 s and creates a misleading GPU utilization spike.
    The LIBERO ID policy uses both scene and wrist images, so both are included.

    Pass ``batch_size`` matching the batch dim used in the hot loop (the batched and
    unbatched code paths trigger different XLA cache entries).
    """
    dummy = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "pick up the object and place it on the plate",
    }
    if batch_size <= 1:
        policy.infer(dummy)
    else:
        # Local import to avoid a hard dependency at module import time.
        from canonical.eval.libero_plus_eval import policy_infer_batched  # noqa: PLC0415
        policy_infer_batched(policy, [dict(dummy) for _ in range(batch_size)])
    print(f"[info] JAX JIT warmup complete (batch_size={batch_size}).", flush=True)


def ensure_libero_config(libero_config_path: str | None = None) -> pathlib.Path:
    """Ensures LIBERO can be imported without interactive prompts."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_cfg_dir = repo_root / ".libero"
    cfg_dir_raw = libero_config_path if libero_config_path is not None else os.environ.get("LIBERO_CONFIG_PATH")
    cfg_dir = pathlib.Path(cfg_dir_raw) if cfg_dir_raw is not None else default_cfg_dir
    cfg_dir = cfg_dir.expanduser().resolve()

    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"

    if not cfg_path.exists():
        benchmark_root = (repo_root / "openpi" / "third_party" / "libero" / "libero" / "libero").resolve()
        defaults = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str((benchmark_root / "bddl_files").resolve()),
            "init_states": str((benchmark_root / "init_files").resolve()),
            "datasets": str((benchmark_root.parent / "datasets").resolve()),
            "assets": str((benchmark_root / "assets").resolve()),
        }
        cfg_text = "\n".join(f"{key}: {value}" for key, value in defaults.items()) + "\n"
        cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def ensure_libero_torch_compat() -> None:
    """Ensures LIBERO init-state loading works with torch>=2.6 defaults."""
    import torch  # noqa: PLC0415

    if RUNTIME_FLAGS["torch_load_patched"]:
        return

    original_torch_load = torch.load

    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    RUNTIME_FLAGS["torch_load_patched"] = True


def default_max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name not in SUITE_MAX_STEPS:
        known = ", ".join(sorted(SUITE_MAX_STEPS))
        raise ValueError(f"Unknown task suite '{task_suite_name}'. Known: {known}")
    return SUITE_MAX_STEPS[task_suite_name]


def create_libero_policy(
    *,
    checkpoint_dir: str = DEFAULT_LIBERO_CHECKPOINT,
    config_name: str = "pi05_libero",
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    train_config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )


def get_task_suite(task_suite_name: str) -> Any:
    ensure_libero_config()
    ensure_libero_torch_compat()
    try:
        from libero.libero import benchmark  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO import failed. Ensure PYTHONPATH includes "
            "`openpi/third_party/libero` and required dependencies are installed "
            f"(missing module: {exc.name})."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        known = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"Unknown LIBERO task suite '{task_suite_name}'. Known: {known}")
    return benchmark_dict[task_suite_name]()


def create_libero_env(task: Any, *, resolution: int = 256, seed: int = 7) -> tuple[Any, str]:
    ensure_libero_config()
    ensure_libero_torch_compat()
    try:
        from libero.libero import get_libero_path  # noqa: PLC0415
        from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO environment dependencies are incomplete. "
            "Install LIBERO/robosuite dependencies in the active isolated env and set PYTHONPATH per "
            "the LIBERO environment setup notes "
            f"(missing module: {exc.name})."
        ) from exc

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Converts quaternion `[x, y, z, w]` into axis-angle (3D)."""
    quat = quat_xyzw.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

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
    image_flip_mode: str = "both",
) -> dict[str, Any]:
    from openpi_client import image_tools  # noqa: PLC0415

    # OpenPI's official LIBERO data uses "both". Locally re-rendered LIBERO pair-data
    # images were saved with only flipud, so keep this configurable.
    base_img = _orient_libero_image(obs["agentview_image"], image_flip_mode)
    wrist_img = _orient_libero_image(obs["robot0_eye_in_hand_image"], image_flip_mode)

    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _override_scene_image(
    obs: Mapping[str, Any],
    *,
    mode: str,
    gray_value: int,
    rng: np.random.Generator | None,
) -> Mapping[str, Any]:
    if mode == "none":
        return obs
    if "agentview_image" not in obs:
        raise KeyError("Observation is missing 'agentview_image'.")

    image = np.asarray(obs["agentview_image"])
    if mode == "black":
        replaced = np.zeros_like(image)
    elif mode == "gray":
        fill = np.uint8(np.clip(gray_value, 0, 255))
        replaced = np.full_like(image, fill)
    elif mode == "noise":
        if rng is None:
            rng = np.random.default_rng(0)
        replaced = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    else:
        valid = "none, black, gray, noise"
        raise ValueError(f"Unknown scene image mode '{mode}'. Valid: {valid}")

    obs_mut = dict(obs)
    obs_mut["agentview_image"] = replaced
    return obs_mut


def _override_wrist_image(
    obs: Mapping[str, Any],
    *,
    mode: str,
    gray_value: int,
    rng: np.random.Generator | None,
) -> Mapping[str, Any]:
    if mode == "none":
        return obs
    if "robot0_eye_in_hand_image" not in obs:
        raise KeyError("Observation is missing 'robot0_eye_in_hand_image'.")

    image = np.asarray(obs["robot0_eye_in_hand_image"])
    if mode == "black":
        replaced = np.zeros_like(image)
    elif mode == "gray":
        fill = np.uint8(np.clip(gray_value, 0, 255))
        replaced = np.full_like(image, fill)
    elif mode == "noise":
        if rng is None:
            rng = np.random.default_rng(0)
        replaced = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    else:
        valid = "none, black, gray, noise"
        raise ValueError(f"Unknown wrist image mode '{mode}'. Valid: {valid}")

    obs_mut = dict(obs)
    obs_mut["robot0_eye_in_hand_image"] = replaced
    return obs_mut


def rollout_episode(
    policy: _policy.Policy,
    env: Any,
    task_description: str,
    rollout_config: RolloutConfig,
    *,
    initial_state: np.ndarray | None = None,
    progress: bool = False,
    scene_image_mode: str = "none",
    scene_gray_value: int = 127,
    noise_seed: int | None = None,
    wrist_image_mode: str = "none",
    wrist_gray_value: int = 127,
    wrist_noise_seed: int | None = None,
    image_flip_mode: str = "both",
) -> dict[str, Any]:
    """Runs a single rollout and returns success metadata."""
    reset_obs = env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else reset_obs

    action_plan: deque[np.ndarray] = deque()
    done = False
    info: dict[str, Any] = {}
    rng = np.random.default_rng(noise_seed) if scene_image_mode == "noise" else None
    wrist_rng = np.random.default_rng(wrist_noise_seed) if wrist_image_mode == "noise" else None

    step_iter: range | tqdm.tqdm = range(rollout_config.max_steps + rollout_config.num_steps_wait)
    if progress:
        step_iter = tqdm.tqdm(
            step_iter,
            dynamic_ncols=True,
            leave=False,
            unit="step",
        )

    for step_idx in step_iter:
        if step_idx < rollout_config.num_steps_wait:
            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue

        if not action_plan:
            policy_input_obs = _override_scene_image(
                obs,
                mode=scene_image_mode,
                gray_value=scene_gray_value,
                rng=rng,
            )
            policy_input_obs = _override_wrist_image(
                policy_input_obs,
                mode=wrist_image_mode,
                gray_value=wrist_gray_value,
                rng=wrist_rng,
            )
            policy_obs = build_policy_observation(
                policy_input_obs,
                task_description,
                resize_size=rollout_config.resize_size,
                image_flip_mode=image_flip_mode,
            )
            action_chunk = np.asarray(policy.infer(policy_obs)["actions"])
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError(f"Unexpected action shape {action_chunk.shape}; expected (H, 7)")
            if action_chunk.shape[0] < rollout_config.replan_steps:
                raise ValueError(
                    f"Need >= {rollout_config.replan_steps} planned actions, "
                    f"but policy returned {action_chunk.shape[0]}"
                )
            action_plan.extend(action_chunk[: rollout_config.replan_steps])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, info = env.step(action.tolist())
        if done:
            break

    success = bool(done)
    if isinstance(info, Mapping) and "success" in info:
        success = bool(info["success"])
    return {
        "success": success,
        "done": bool(done),
        "steps_executed": int(step_idx + 1),
        "scene_image_mode": scene_image_mode,
        "wrist_image_mode": wrist_image_mode,
        "info": dict(info) if isinstance(info, Mapping) else {},
    }


def rollout_episode_batched(
    policy: _policy.Policy,
    envs: list[Any],
    task_description: str,
    rollout_config: RolloutConfig,
    *,
    initial_states: list[Any],
    image_flip_mode: str = "both",
    env_executor: Any = None,
) -> list[dict[str, Any]]:
    """Lockstep batched version of rollout_episode for the LIBERO ID eval.

    Image orientation (image_flip_mode) is applied per-env inside build_policy_observation,
    exactly as in the sequential rollout; the batching only affects the policy.infer()
    call, not the observation builder. The image flip fix (which was a critical bug fix)
    is therefore preserved verbatim.

    Done envs are skipped from further env.step() calls but kept in the inference batch
    (their slot is ignored when collecting actions) so the JIT-compiled inference path
    always sees batch_size = len(envs). Keep env_executor=None for LIBERO/robosuite:
    MuJoCo/EGL env.step/render is not thread-safe across env instances.

    Note: scene_image_mode / wrist_image_mode overrides are NOT supported here. The ID
    eval main loop does not use them (only the standalone masking script does), so
    keeping this batched function simple. The standalone callers should keep using the
    sequential rollout_episode.
    """
    from canonical.eval.libero_plus_eval import policy_infer_batched  # noqa: PLC0415

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

    for _ in range(rollout_config.num_steps_wait):
        if all(done_flags):
            break
        _step_all([list(LIBERO_DUMMY_ACTION)] * n)

    for _ in range(rollout_config.max_steps):
        if all(done_flags):
            break

        if any((not action_plans[i]) and (not done_flags[i]) for i in range(n)):
            batched_obs: list[dict[str, Any]] = []
            for i in range(n):
                batched_obs.append(
                    build_policy_observation(
                        obs_list[i],
                        task_description,
                        resize_size=rollout_config.resize_size,
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
                        f"Need >= {rollout_config.replan_steps} planned actions, "
                        f"but policy returned {chunk.shape[0]}"
                    )
                action_plans[i] = [np.asarray(x, dtype=np.float32) for x in chunk[: rollout_config.replan_steps]]

        step_actions: list[list[float]] = []
        for i in range(n):
            if done_flags[i] or not action_plans[i]:
                step_actions.append(list(LIBERO_DUMMY_ACTION))
            else:
                step_actions.append(action_plans[i].pop(0).tolist())
        _step_all(step_actions)

    results: list[dict[str, Any]] = []
    for i in range(n):
        info = info_list[i]
        success = bool(done_flags[i])
        if isinstance(info, Mapping) and "success" in info:
            success = bool(info["success"])
        results.append(
            {
                "success": success,
                "done": bool(done_flags[i]),
                "steps_executed": int(step_count[i]),
                "scene_image_mode": "none",
                "wrist_image_mode": "none",
                "info": dict(info) if isinstance(info, Mapping) else {},
            }
        )
    return results
