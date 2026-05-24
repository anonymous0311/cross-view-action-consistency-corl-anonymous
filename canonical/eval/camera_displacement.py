"""Camera displacement utilities for LIBERO Phase-0 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

DISPLACEMENT_CONFIGS: dict[str, dict[str, tuple[float, float, float]]] = {
    # Translation units are meters in world-frame XYZ by default.
    # Rotation units are degrees in Euler XYZ (roll, pitch, yaw).
    # Scene-view protocol for Step 3:
    # 1) look_from_above
    # 2) look_from_left
    # 3) look_from_left_front_45
    # Positive rotation is counter-clockwise about world XYZ axes.
    # x means going further, y means going left and z means going up.
    "nominal": {"translation": (0.0, 0.0, 0.0), "rotation_deg": (0.0, 0.0, 0.0)},
    "look_from_above": {"translation": (-0.40, 0.00, 0.45), "rotation_deg": (0.0, -35.0, 0.0)},
    "look_from_left": {"translation": (-0.80, 0.85, 0.0), "rotation_deg": (0.0, 0.0, 90.0)},
    "look_from_left_front_45": {"translation": (-0.35, 0.55, 0.20), "rotation_deg": (0.0, -10.0, 45.0)},
}


def _quat_normalize_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def quat_multiply_wxyz(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion multiplication with MuJoCo-compatible `[w, x, y, z]` ordering."""
    lw, lx, ly, lz = lhs_wxyz
    rw, rx, ry, rz = rhs_wxyz
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quat_from_euler_xyz_degrees_wxyz(rotation_deg_xyz: Sequence[float]) -> np.ndarray:
    """Builds a quaternion from intrinsic XYZ Euler angles in degrees.

    For intrinsic XYZ (roll/pitch/yaw applied on the rotating camera axes),
    the equivalent quaternion product is `qz * qy * qx`.
    """
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg_xyz, dtype=np.float64))

    cx, sx = math.cos(rx / 2.0), math.sin(rx / 2.0)
    cy, sy = math.cos(ry / 2.0), math.sin(ry / 2.0)
    cz, sz = math.cos(rz / 2.0), math.sin(rz / 2.0)

    qx = np.array([cx, sx, 0.0, 0.0], dtype=np.float64)
    qy = np.array([cy, 0.0, sy, 0.0], dtype=np.float64)
    qz = np.array([cz, 0.0, 0.0, sz], dtype=np.float64)
    return _quat_normalize_wxyz(quat_multiply_wxyz(quat_multiply_wxyz(qz, qy), qx))


def quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Converts MuJoCo-order quaternion `[w, x, y, z]` to 3x3 rotation matrix."""
    w, x, y, z = _quat_normalize_wxyz(quat_wxyz.astype(np.float64))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def verify_wrist_view_unchanged(
    nominal_obs: dict[str, Any],
    displaced_obs: dict[str, Any],
    wrist_key: str = "robot0_eye_in_hand_image",
) -> dict[str, float]:
    """Computes simple wrist-view difference stats for sanity checks."""
    if wrist_key not in nominal_obs or wrist_key not in displaced_obs:
        return {"mean_abs_diff": float("nan"), "max_abs_diff": float("nan")}
    a = np.asarray(nominal_obs[wrist_key], dtype=np.float32)
    b = np.asarray(displaced_obs[wrist_key], dtype=np.float32)
    diff = np.abs(a - b)
    return {"mean_abs_diff": float(diff.mean()), "max_abs_diff": float(diff.max())}


class CameraDisplacementWrapper:
    """Applies scene-camera displacement to a LIBERO environment."""

    def __init__(
        self,
        env: Any,
        config_name: str = "nominal",
        *,
        scene_camera_name: str = "agentview",
        fallback_camera_names: Sequence[str] = ("canonical_agentview", "frontview"),
        translation_frame: str = "world",
        rotation_frame: str = "world",
    ) -> None:
        if config_name not in DISPLACEMENT_CONFIGS:
            valid = ", ".join(sorted(DISPLACEMENT_CONFIGS))
            raise ValueError(f"Unknown config '{config_name}'. Valid configs: {valid}")
        if translation_frame not in {"world", "camera"}:
            raise ValueError("translation_frame must be one of: {'world', 'camera'}")
        if rotation_frame not in {"world", "camera"}:
            raise ValueError("rotation_frame must be one of: {'world', 'camera'}")

        self.env = env
        self.config_name = config_name
        self.scene_camera_name = scene_camera_name
        self.fallback_camera_names = tuple(fallback_camera_names)
        self.translation_frame = translation_frame
        self.rotation_frame = rotation_frame

        self._scene_camera_id: int | None = None
        self._resolved_scene_camera_name: str | None = None
        self._nominal_cam_pos: np.ndarray | None = None
        self._nominal_cam_quat_wxyz: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def resolved_scene_camera_name(self) -> str | None:
        return self._resolved_scene_camera_name

    @property
    def scene_camera_id(self) -> int | None:
        return self._scene_camera_id

    def set_config(self, config_name: str) -> None:
        if config_name not in DISPLACEMENT_CONFIGS:
            valid = ", ".join(sorted(DISPLACEMENT_CONFIGS))
            raise ValueError(f"Unknown config '{config_name}'. Valid configs: {valid}")
        self.config_name = config_name

    def reset(self) -> dict[str, Any]:
        _ = self.env.reset()
        self._ensure_nominal_pose_captured()
        self._apply_displacement()
        return self._regenerate_observation()

    def set_init_state(self, init_state: np.ndarray) -> dict[str, Any]:
        _ = self.env.set_init_state(init_state)
        self._ensure_nominal_pose_captured()
        self._apply_displacement()
        return self._regenerate_observation()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self.env.step(action)

    def current_camera_pose(self) -> dict[str, list[float]]:
        self._ensure_nominal_pose_captured()
        assert self._scene_camera_id is not None
        model = self.env.sim.model
        return {
            "camera_name": self._resolved_scene_camera_name or "",
            "pos": model.cam_pos[self._scene_camera_id].astype(float).tolist(),
            "quat_wxyz": model.cam_quat[self._scene_camera_id].astype(float).tolist(),
        }

    def nominal_camera_pose(self) -> dict[str, list[float]]:
        self._ensure_nominal_pose_captured()
        assert self._nominal_cam_pos is not None
        assert self._nominal_cam_quat_wxyz is not None
        return {
            "camera_name": self._resolved_scene_camera_name or "",
            "pos": self._nominal_cam_pos.astype(float).tolist(),
            "quat_wxyz": self._nominal_cam_quat_wxyz.astype(float).tolist(),
        }

    def _ensure_nominal_pose_captured(self) -> None:
        if self._scene_camera_id is None:
            self._scene_camera_id, self._resolved_scene_camera_name = self._resolve_scene_camera_id()
        if self._nominal_cam_pos is None or self._nominal_cam_quat_wxyz is None:
            assert self._scene_camera_id is not None
            model = self.env.sim.model
            self._nominal_cam_pos = np.asarray(model.cam_pos[self._scene_camera_id], dtype=np.float64).copy()
            self._nominal_cam_quat_wxyz = np.asarray(model.cam_quat[self._scene_camera_id], dtype=np.float64).copy()

    def _resolve_scene_camera_id(self) -> tuple[int, str]:
        model = self.env.sim.model
        camera_names = self._list_camera_names()
        candidates = [self.scene_camera_name, *self.fallback_camera_names]

        for name in candidates:
            try:
                if hasattr(model, "camera_name2id"):
                    cam_id = int(model.camera_name2id(name))
                    return cam_id, name
            except Exception:
                continue

            if name in camera_names:
                return camera_names.index(name), name

        raise ValueError(
            "Failed to resolve scene camera id. "
            f"Tried {candidates}, available cameras: {camera_names}"
        )

    def _list_camera_names(self) -> list[str]:
        model = self.env.sim.model
        names: list[str] = []

        if hasattr(model, "camera_id2name") and hasattr(model, "ncam"):
            for cam_id in range(int(model.ncam)):
                try:
                    maybe_name = model.camera_id2name(cam_id)
                except Exception:
                    maybe_name = None
                if maybe_name is not None:
                    names.append(str(maybe_name))
            return names

        if hasattr(model, "camera_names"):
            for item in model.camera_names:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                names.append(decoded)
            return names

        return names

    def _regenerate_observation(self) -> dict[str, Any]:
        if hasattr(self.env, "regenerate_obs_from_state") and hasattr(self.env, "get_sim_state"):
            return self.env.regenerate_obs_from_state(self.env.get_sim_state())

        # Fallback path.
        self.env.sim.forward()
        if hasattr(self.env, "_get_observations"):
            return self.env._get_observations()  # noqa: SLF001
        if hasattr(self.env, "env") and hasattr(self.env.env, "_get_observations"):
            return self.env.env._get_observations()  # noqa: SLF001
        raise RuntimeError("Cannot regenerate observations after camera displacement.")

    def _apply_displacement(self) -> None:
        self._ensure_nominal_pose_captured()
        assert self._scene_camera_id is not None
        assert self._nominal_cam_pos is not None
        assert self._nominal_cam_quat_wxyz is not None

        cfg = DISPLACEMENT_CONFIGS[self.config_name]
        translation = np.asarray(cfg["translation"], dtype=np.float64)
        rotation_deg = np.asarray(cfg["rotation_deg"], dtype=np.float64)

        if self.translation_frame == "camera":
            rotation_mat = quat_to_rotmat_wxyz(self._nominal_cam_quat_wxyz)
            translation = rotation_mat @ translation

        delta_quat = quat_from_euler_xyz_degrees_wxyz(rotation_deg)
        if self.rotation_frame == "camera":
            # Rotate about camera-local axes (intrinsic): q_nominal * q_delta.
            displaced_quat = quat_multiply_wxyz(self._nominal_cam_quat_wxyz, delta_quat)
        else:
            # Rotate about world axes (extrinsic): q_delta * q_nominal.
            displaced_quat = quat_multiply_wxyz(delta_quat, self._nominal_cam_quat_wxyz)
        displaced_quat = _quat_normalize_wxyz(displaced_quat)

        displaced_pos = self._nominal_cam_pos + translation

        model = self.env.sim.model
        model.cam_pos[self._scene_camera_id] = displaced_pos.astype(model.cam_pos.dtype)
        model.cam_quat[self._scene_camera_id] = displaced_quat.astype(model.cam_quat.dtype)
        self.env.sim.forward()
