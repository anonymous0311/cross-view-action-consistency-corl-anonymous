"""Render two observations of each stored state, with one fixed camera per episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from afcv.evaluation.libero_policy_runner import ensure_libero_config

_SCALE_PIVOT = np.array([0.0, 0.0, 0.8])
C2_HORIZON_POOL = list(range(1, 76)) + list(range(285, 360))
C2_VERTICAL_POOL = [0, 15]
C3_ROT_POOL = [2, 4, 6, 8, 10, 350, 352, 354, 356, 358]
CATEGORY_PROBS = {"C1": 0.196, "C2": 0.620, "C3": 0.184}


def _rot_z(
    quat: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rotate pos and/or quat (WXYZ) around the world Z-axis."""
    result: dict[str, np.ndarray] = {}
    R = Rotation.from_euler("z", deg, degrees=True)
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # WXYZ → XYZW
        xyzw = (R * orig).as_quat()
        result["new_quat"] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if pos is not None:
        result["new_pos"] = R.apply(np.asarray(pos, dtype=float))
    return result


def _rot_y_pivot(
    quat: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rotate pos and/or quat (WXYZ) around the Y-axis through _SCALE_PIVOT."""
    result: dict[str, np.ndarray] = {}
    axis = np.array([0.0, 1.0, 0.0])
    R = Rotation.from_rotvec(np.radians(-deg) * axis)
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        xyzw = (R * orig).as_quat()
        result["new_quat"] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if pos is not None:
        p = np.asarray(pos, dtype=float)
        result["new_pos"] = R.apply(p - _SCALE_PIVOT) + _SCALE_PIVOT
    return result


def _scale_dist(
    quat: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    factor: float = 1.0,
) -> dict[str, np.ndarray]:
    """Scale distance from _SCALE_PIVOT; quat is unchanged."""
    result: dict[str, np.ndarray] = {}
    if quat is not None:
        result["new_quat"] = np.asarray(quat, dtype=float).copy()
    if pos is not None:
        p = np.asarray(pos, dtype=float)
        result["new_pos"] = _SCALE_PIVOT + (p - _SCALE_PIVOT) * factor
    return result


def compute_camera(
    nom_pos: np.ndarray,
    nom_quat: np.ndarray,
    *,
    horizon: int,
    vertical: int,
    scale_pct: int,
    end_rot: int,
    end_vert: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply LIBERO-Plus camera transformation to a given nominal pos/quat.

    Parameters use parquet integer conventions:
      scale_pct: 100 = nominal, 150 = 1.5x farther
      horizon, vertical, end_rot, end_vert: integer degrees

    Returns (new_pos, new_quat) as float64 numpy arrays, quat in WXYZ.
    """
    pos = nom_pos.copy().astype(float)
    quat = nom_quat.copy().astype(float)

    # C2: vertical elevation shift, then azimuth rotation (both change pos+quat)
    if vertical != 0:
        r = _rot_y_pivot(quat=quat, pos=pos, deg=float(vertical))
        pos, quat = r["new_pos"], r["new_quat"]
    if horizon != 0:
        r = _rot_z(quat=quat, pos=pos, deg=float(horizon))
        pos, quat = r["new_pos"], r["new_quat"]

    # C1: scale distance from pivot (moves pos along radial direction; quat unchanged)
    factor = scale_pct / 100.0
    if factor != 1.0:
        r = _scale_dist(quat=quat, pos=pos, factor=factor)
        pos, quat = r["new_pos"], r["new_quat"]

    # C3: roll/pitch in camera frame (quat only — pos stays fixed)
    if end_rot != 0:
        r = _rot_z(quat=quat, deg=float(end_rot))
        quat = r["new_quat"]
    if end_vert != 0:
        r = _rot_y_pivot(quat=quat, deg=float(end_vert))
        quat = r["new_quat"]

    return pos, quat


def sample_camera(rng: np.random.Generator, category: str | None = None) -> dict[str, Any]:
    """Sample C1/C2/C3 camera parameters.

    If category is None, draw proportionally per CATEGORY_PROBS.
    Returns dict with keys: horizon, vertical, scale, end_rot, end_vert, category.
    """
    if category is None:
        cats = list(CATEGORY_PROBS.keys())
        probs = [CATEGORY_PROBS[c] for c in cats]
        category = rng.choice(cats, p=probs)

    if category == "C1":
        return {
            "horizon": 0,
            "vertical": 0,
            "scale": int(rng.integers(115, 201)),  # uniform in [115, 200]
            "end_rot": 0,
            "end_vert": 0,
            "category": "C1",
        }
    if category == "C2":
        return {
            "horizon": int(rng.choice(C2_HORIZON_POOL)),
            "vertical": int(rng.choice(C2_VERTICAL_POOL)),
            "scale": 100,
            "end_rot": 0,
            "end_vert": 0,
            "category": "C2",
        }
    # C3
    return {
        "horizon": 0,
        "vertical": 0,
        "scale": 100,
        "end_rot": int(rng.choice(C3_ROT_POOL)),
        "end_vert": int(rng.choice(C3_ROT_POOL)),
        "category": "C3",
    }


def hdf5_to_bddl(hdf5_path: Path, bddl_root: Path) -> Path:
    """Derive local BDDL path from HDF5 filename.

    HDF5:  {suite}/{task_name}_demo.hdf5
    BDDL:  {bddl_root}/{suite}/{task_name}.bddl
    """
    suite = hdf5_path.parent.name
    task_stem = hdf5_path.stem.removesuffix("_demo")
    return bddl_root / suite / f"{task_stem}.bddl"


def build_env(bddl_file: Path, img_size: int = 256, gpu_device_id: int = -1):
    """Create OffScreenRenderEnv for original LIBERO (no LIBERO-Plus camera args)."""
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=img_size,
        camera_widths=img_size,
        camera_names=["agentview"],
        render_gpu_device_id=gpu_device_id,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        ignore_done=True,
        hard_reset=True,
    )


def get_cam(env, cam_name: str = "agentview") -> tuple[int, np.ndarray, np.ndarray]:
    """Return (cam_id, nom_pos, nom_quat[WXYZ]) from the initialized env model."""
    cam_id = env.sim.model.camera_name2id(cam_name)
    nom_pos = env.sim.model.cam_pos[cam_id].copy()
    nom_quat = env.sim.model.cam_quat[cam_id].copy()  # MuJoCo stores WXYZ
    return cam_id, nom_pos, nom_quat


def _render(env, cam_id: int, pos: np.ndarray, quat: np.ndarray, img_size: int) -> np.ndarray:
    """Set agentview camera and render.  Returns raw uint8 (H,W,3), upside-down.

    sim.render() reads data.cam_xpos (world-space, computed by forward()).
    We must call forward() after modifying model.cam_pos so the change
    propagates from model → data before rendering.
    """
    env.sim.model.cam_pos[cam_id][:] = pos
    env.sim.model.cam_quat[cam_id][:] = quat
    env.sim.forward()  # propagate model.cam_pos → data.cam_xpos
    return env.sim.render(width=img_size, height=img_size, camera_name="agentview")


def render_views(
    env,
    cam_id: int,
    nom_pos: np.ndarray,
    nom_quat: np.ndarray,
    state_vec: np.ndarray,
    perturb_params: list[dict[str, Any]],
    img_size: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Reset to state_vec and render nominal + one view per entry in perturb_params.

    Restores nominal camera after rendering to leave env clean.
    Returns:
      nom_img  — (H,W,3) uint8, robosuite upside-down convention
      perturbed — list of (img, pos, quat) per perturbation
    """
    env.sim.set_state_from_flattened(state_vec)
    env.sim.forward()

    nom_img = _render(env, cam_id, nom_pos, nom_quat, img_size)

    perturbed = []
    for p in perturb_params:
        p_pos, p_quat = compute_camera(
            nom_pos,
            nom_quat,
            horizon=p["horizon"],
            vertical=p["vertical"],
            scale_pct=p["scale"],
            end_rot=p["end_rot"],
            end_vert=p["end_vert"],
        )
        p_img = _render(env, cam_id, p_pos, p_quat, img_size)
        perturbed.append((p_img, p_pos, p_quat))

    # Restore nominal camera (avoids drift across calls)
    env.sim.model.cam_pos[cam_id][:] = nom_pos
    env.sim.model.cam_quat[cam_id][:] = nom_quat

    return nom_img, perturbed


def _save_png(path: Path, img_raw: np.ndarray) -> None:
    """Flip from robosuite upside-down convention and save as PNG."""
    import imageio.v2 as imageio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.flipud(img_raw).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True, help="Root of the four original HDF5 suites")
    parser.add_argument("--output-dir", type=Path, default=Path("data/rendered_pairs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int, help="For a short pipeline check only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--support", choices=["full", "restricted"], default="full")
    args = parser.parse_args()
    os.environ.setdefault("LIBERO_ROOT", str(Path(__file__).resolve().parents[2] / "third_party/libero"))
    ensure_libero_config()
    from libero.libero import get_libero_path

    bddl_root = Path(get_libero_path("bddl_files"))
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    files = sorted(p for suite in suites for p in (args.libero_root / suite).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError("No original LIBERO HDF5 files found")
    files = files[: args.max_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "source": str(args.libero_root.resolve()),
        "seed": args.seed,
        "max_tasks": args.max_tasks,
        "max_episodes": args.max_episodes,
        "max_frames": args.max_frames,
        "camera_per_episode": True,
    }
    spec_path = args.output_dir / "render_config.json"
    if spec_path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume or a new directory")
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("Resume configuration differs")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    manifests = []
    for task_index, source in enumerate(tqdm(files, desc="Tasks", dynamic_ncols=True)):
        suite = source.parent.name
        task = source.stem.removesuffix("_demo")
        env = build_env(hdf5_to_bddl(source, bddl_root))
        env.reset()
        camera_id, pos, quat = get_cam(env)
        try:
            with h5py.File(source) as f:
                demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))[: args.max_episodes]
                raw = f["data"].attrs.get("problem_info", "{}")
                language = json.loads(raw).get("language_instruction", task.replace("_", " "))
                for demo_key in tqdm(demos, desc="Episodes", leave=False, dynamic_ncols=True):
                    demo_id = int(demo_key.split("_")[-1])
                    out = args.output_dir / suite / task / demo_key
                    out.mkdir(parents=True, exist_ok=True)
                    manifest = out / "pairs.jsonl"
                    manifests.append(manifest)
                    if manifest.exists() and args.resume:
                        continue
                    demo = f["data"][demo_key]
                    camera = sample_camera(
                        np.random.default_rng(np.random.SeedSequence([args.seed, task_index, demo_id]))
                    )
                    rows = []
                    for t in tqdm(
                        range(min(len(demo["states"]), args.max_frames or len(demo["states"]))),
                        desc="Frames",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        nominal, perturbed = render_views(env, camera_id, pos, quat, demo["states"][t], [camera], 256)
                        a = out / f"{t:06d}_nominal.png"
                        b = out / f"{t:06d}_perturbed.png"
                        _save_png(a, nominal)
                        _save_png(b, perturbed[0][0])
                        rows.append(
                            {
                                "suite": suite,
                                "task_name": task,
                                "hdf5_path": str(source.resolve()),
                                "demo_key": demo_key,
                                "timestep": t,
                                "language": language,
                                "img_a_path": str(a.resolve()),
                                "img_b_path": str(b.resolve()),
                                "camera_params_b": camera,
                            }
                        )
                    temporary = manifest.with_suffix(".tmp")
                    temporary.write_text("".join(json.dumps(r) + "\n" for r in rows))
                    temporary.replace(manifest)
        finally:
            env.close()
    with (args.output_dir / "pairs.jsonl").open("w") as dst:
        for manifest in manifests:
            dst.write(manifest.read_text())
    print(f"Manifest: {args.output_dir / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
