"""LIBERO pair-data.3 — Same-state multiview rerender for LIBERO HDF5 demonstrations.

For each sampled timestep, renders:
  - one nominal agentview (same camera as stored obs/agentview_rgb)
  - views_per_state C1/C2/C3 perturbed views (same MuJoCo state, different camera)

The camera perturbation math replicates LIBERO-Plus semantics so that
training-time camera diversity matches LIBERO-Plus C1/C2/C3 eval categories.

Outputs:
  data/rendered_libero_multiview/{suite}/{task}/demo_{N}/t{t:06d}_{label}.png
  results/libero_pair_audit/libero_pair_manifest_train.jsonl
  results/libero_pair_audit/libero_pair_manifest_val.jsonl
  results/libero_pair_audit/libero_wrong_pair_manifest_{train,val}.jsonl
  results/libero_pair_audit/rerender_reconstruction_report.md
  results/libero_pair_audit/libero_pair_audit_report.md
  debug/rerender_sanity/smoke_{task}.png

Interpreter: venv-libero-plus/bin/python  (has robosuite/mujoco/scipy)
Original LIBERO is added to sys.path automatically.

Smoke test (verify state reset works before batch render):
  venv-libero-plus/bin/python scripts/libero_pair_data/render_libero_multiview_states.py \\
    --smoke-test --suite libero_spatial --max-tasks 1 \\
    --libero-root data/libero_hdf5_original

Batch render (screening, ~25k pairs per suite):
  venv-libero-plus/bin/python scripts/libero_pair_data/render_libero_multiview_states.py \\
    --libero-root data/libero_hdf5_original \\
    --suite libero_spatial libero_object libero_goal libero_10 libero_90 \\
    --max-pairs-per-suite 25000 \\
    --output-dir data/rendered_libero_multiview \\
    --results-dir results/libero_pair_audit
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ── Repo-relative paths ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
OPENPI_LIBERO = REPO_ROOT / "openpi" / "third_party" / "libero"

# Original LIBERO must be first in sys.path so it shadows any LIBERO-Plus import.
sys.path.insert(0, str(OPENPI_LIBERO))
sys.path.insert(0, str(_HERE))  # libero_pair_data_common

from libero_pair_data_common import (
    DEFAULT_RESULTS_DIR,
    ensure_dir,
    hash_array,
    now_iso,
    rel_to_repo,
    stable_id,
    write_json,
    write_jsonl,
)

# ── Default paths ───────────────────────────────────────────────────────────────
DEFAULT_LIBERO_ROOT = Path("data/libero_hdf5_original")
DEFAULT_OUTPUT_DIR  = REPO_ROOT / "data"    / "rendered_libero_multiview"
DEFAULT_RESULTS_DIR_= REPO_ROOT / "results" / "libero_pair_audit"
DEFAULT_DEBUG_DIR   = REPO_ROOT / "debug"   / "rerender_sanity"
BDDL_ROOT_DEFAULT   = OPENPI_LIBERO / "libero" / "libero" / "bddl_files"
SUITE_DIRS = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

# ── Pivot for scale / vertical rotation (LIBERO-Plus convention) ────────────────
_SCALE_PIVOT = np.array([0.0, 0.0, 0.8])

# ── Camera sampling pools (from configs/camera_perturbation_spec.yaml) ───────
C2_HORIZON_POOL = list(range(1, 76)) + list(range(285, 360))   # {1..75}∪{285..359}
C2_VERTICAL_POOL = [0, 15]
C3_ROT_POOL = [2, 4, 6, 8, 10, 350, 352, 354, 356, 358]
CATEGORY_PROBS = {"C1": 0.196, "C2": 0.620, "C3": 0.184}

# ── Smoke-test threshold (pixel RMS error below which nominal reconstruction passes) ──
SMOKE_L2_PASS = 20.0   # per-channel RMS; images are uint8


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Camera math — replicates LIBERO-Plus _setup_camera geometry               ║
# ║  Source: LIBERO-plus/libero/libero/envs/problems/libero_tabletop_manipulation.py
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _rot_z(
    quat: np.ndarray | None = None,
    pos:  np.ndarray | None = None,
    deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rotate pos and/or quat (WXYZ) around the world Z-axis."""
    result: dict[str, np.ndarray] = {}
    R = Rotation.from_euler("z", deg, degrees=True)
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])   # WXYZ → XYZW
        xyzw = (R * orig).as_quat()
        result["new_quat"] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if pos is not None:
        result["new_pos"] = R.apply(np.asarray(pos, dtype=float))
    return result


def _rot_y_pivot(
    quat: np.ndarray | None = None,
    pos:  np.ndarray | None = None,
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
    pos:  np.ndarray | None = None,
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
    nom_pos:  np.ndarray,
    nom_quat: np.ndarray,
    *,
    horizon:   int,
    vertical:  int,
    scale_pct: int,
    end_rot:   int,
    end_vert:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply LIBERO-Plus camera transformation to a given nominal pos/quat.

    Parameters use parquet integer conventions:
      scale_pct: 100 = nominal, 150 = 1.5x farther
      horizon, vertical, end_rot, end_vert: integer degrees

    Returns (new_pos, new_quat) as float64 numpy arrays, quat in WXYZ.
    """
    pos  = nom_pos.copy().astype(float)
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Camera sampling                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sample_camera(rng: np.random.Generator, category: str | None = None) -> dict[str, Any]:
    """Sample C1/C2/C3 camera parameters.

    If category is None, draw proportionally per CATEGORY_PROBS.
    Returns dict with keys: horizon, vertical, scale, end_rot, end_vert, category.
    """
    if category is None:
        cats  = list(CATEGORY_PROBS.keys())
        probs = [CATEGORY_PROBS[c] for c in cats]
        category = rng.choice(cats, p=probs)

    if category == "C1":
        return {
            "horizon": 0, "vertical": 0,
            "scale": int(rng.integers(115, 201)),   # uniform in [115, 200]
            "end_rot": 0, "end_vert": 0,
            "category": "C1",
        }
    if category == "C2":
        return {
            "horizon":  int(rng.choice(C2_HORIZON_POOL)),
            "vertical": int(rng.choice(C2_VERTICAL_POOL)),
            "scale": 100, "end_rot": 0, "end_vert": 0,
            "category": "C2",
        }
    # C3
    return {
        "horizon": 0, "vertical": 0, "scale": 100,
        "end_rot":  int(rng.choice(C3_ROT_POOL)),
        "end_vert": int(rng.choice(C3_ROT_POOL)),
        "category": "C3",
    }


def _cam_label(p: dict[str, Any]) -> str:
    """Short filename-safe label for camera params."""
    cat = p["category"]
    if cat == "C1":
        return f"C1_s{p['scale']}"
    if cat == "C2":
        return f"C2_h{p['horizon']}_v{p['vertical']}"
    return f"C3_er{p['end_rot']}_ev{p['end_vert']}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BDDL path mapping                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def hdf5_to_bddl(hdf5_path: Path, bddl_root: Path) -> Path:
    """Derive local BDDL path from HDF5 filename.

    HDF5:  {suite}/{task_name}_demo.hdf5
    BDDL:  {bddl_root}/{suite}/{task_name}.bddl
    """
    suite     = hdf5_path.parent.name
    task_stem = hdf5_path.stem.removesuffix("_demo")
    return bddl_root / suite / f"{task_stem}.bddl"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Environment management                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

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
    cam_id   = env.sim.model.camera_name2id(cam_name)
    nom_pos  = env.sim.model.cam_pos[cam_id].copy()
    nom_quat = env.sim.model.cam_quat[cam_id].copy()   # MuJoCo stores WXYZ
    return cam_id, nom_pos, nom_quat


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Low-level rendering                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _render(env, cam_id: int, pos: np.ndarray, quat: np.ndarray, img_size: int) -> np.ndarray:
    """Set agentview camera and render.  Returns raw uint8 (H,W,3), upside-down.

    sim.render() reads data.cam_xpos (world-space, computed by forward()).
    We must call forward() after modifying model.cam_pos so the change
    propagates from model → data before rendering.
    """
    env.sim.model.cam_pos[cam_id][:] = pos
    env.sim.model.cam_quat[cam_id][:] = quat
    env.sim.forward()   # propagate model.cam_pos → data.cam_xpos
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
            nom_pos, nom_quat,
            horizon=p["horizon"], vertical=p["vertical"],
            scale_pct=p["scale"], end_rot=p["end_rot"], end_vert=p["end_vert"],
        )
        p_img = _render(env, cam_id, p_pos, p_quat, img_size)
        perturbed.append((p_img, p_pos, p_quat))

    # Restore nominal camera (avoids drift across calls)
    env.sim.model.cam_pos[cam_id][:] = nom_pos
    env.sim.model.cam_quat[cam_id][:] = nom_quat

    return nom_img, perturbed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Image I/O                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _save_png(path: Path, img_raw: np.ndarray) -> None:
    """Flip from robosuite upside-down convention and save as PNG."""
    import imageio.v2 as imageio   # noqa: PLC0415
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.flipud(img_raw).astype(np.uint8))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Smoke test                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_smoke_test(
    hdf5_path: Path,
    bddl_root: Path,
    debug_dir: Path,
    img_size: int = 256,
    gpu_device_id: int = -1,
    n_frames: int = 10,
) -> dict[str, Any]:
    """Rerender nominal agentview for n_frames of demo_0; compare to stored frame.

    Saves a side-by-side comparison grid (stored | rerendered) to debug_dir.
    Returns report dict.
    """
    bddl = hdf5_to_bddl(hdf5_path, bddl_root)
    if not bddl.exists():
        return {"error": f"BDDL not found: {bddl}", "status": "ERROR"}

    task_name = hdf5_path.stem.removesuffix("_demo")
    print(f"  env init ...", flush=True)
    t0 = time.time()
    env = build_env(bddl, img_size=img_size, gpu_device_id=gpu_device_id)
    env.reset()
    init_time = time.time() - t0
    print(f"  env ready in {init_time:.1f}s", flush=True)

    cam_id, nom_pos, nom_quat = get_cam(env)
    print(f"  agentview pos  = {nom_pos.tolist()}")
    print(f"  agentview quat = {nom_quat.tolist()}")

    with h5py.File(hdf5_path, "r") as f:
        demo  = f["data/demo_0"]
        states     = demo["states"][:]                   # (T, state_dim)
        stored_rgb = demo["obs/agentview_rgb"][:]        # (T, h, w, 3) upside-down

    n = min(n_frames, len(states))
    l2_list: list[float] = []
    grids: list[np.ndarray] = []

    for t_idx in tqdm(range(n), desc="  smoke frames", leave=False, dynamic_ncols=True):
        env.sim.set_state_from_flattened(states[t_idx])
        env.sim.forward()
        rendered = _render(env, cam_id, nom_pos, nom_quat, img_size)  # (H,W,3) upside-down

        stored = stored_rgb[t_idx]   # (h, w, 3) upside-down (same convention)
        # Resize stored if demos were recorded at a different resolution
        if stored.shape[:2] != (img_size, img_size):
            from PIL import Image  # noqa: PLC0415
            # PIL expects right-side-up: flipud before resize, then flipud back
            pil = Image.fromarray(np.flipud(stored)).resize(
                (img_size, img_size), Image.BILINEAR
            )
            stored = np.flipud(np.array(pil))

        l2 = float(np.sqrt(np.mean((rendered.astype(float) - stored.astype(float)) ** 2)))
        l2_list.append(l2)

        # Build display row (flip both to right-side-up for the comparison image)
        s_up = np.flipud(stored)
        r_up = np.flipud(rendered)
        sep  = np.full((img_size, 6, 3), 180, dtype=np.uint8)
        grids.append(np.concatenate([s_up, sep, r_up], axis=1))

    env.close()
    gc.collect()

    # Save comparison grid
    ensure_dir(debug_dir)
    try:
        import imageio.v2 as imageio   # noqa: PLC0415
        grid_img = np.concatenate(grids, axis=0)
        out_path = debug_dir / f"smoke_{task_name}.png"
        imageio.imwrite(str(out_path), grid_img)
        grid_str = str(out_path)
    except Exception as exc:
        grid_str = f"(save failed: {exc})"

    l2_mean = float(np.mean(l2_list))
    status = "PASS" if l2_mean < SMOKE_L2_PASS else "WARN"

    return {
        "task": task_name,
        "hdf5": str(hdf5_path),
        "bddl": str(bddl),
        "n_frames_tested": n,
        "pixel_rms_mean":  round(l2_mean, 3),
        "pixel_rms_max":   round(float(np.max(l2_list)), 3),
        "pixel_rms_min":   round(float(np.min(l2_list)), 3),
        "pixel_rms_per_frame": [round(x, 3) for x in l2_list],
        "nominal_cam_pos":  nom_pos.tolist(),
        "nominal_cam_quat": nom_quat.tolist(),
        "comparison_grid":  grid_str,
        "env_init_sec":     round(init_time, 1),
        "status": status,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Per-file batch processing                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def process_hdf5(
    hdf5_path: Path,
    bddl_root: Path,
    output_dir: Path,
    rng: np.random.Generator,
    *,
    timestep_sample_rate: float = 0.10,
    views_per_state: int = 1,
    max_pairs: int | None = None,
    img_size: int = 256,
    gpu_device_id: int = -1,
    val_demo_fraction: float = 0.10,
) -> list[dict[str, Any]]:
    """Render all selected pairs from one HDF5 file.

    Returns list of pair manifest dicts (matched pairs only; wrong-state
    controls are built later from the merged list).
    """
    bddl = hdf5_to_bddl(hdf5_path, bddl_root)
    if not bddl.exists():
        print(f"    [SKIP] BDDL not found: {bddl}", flush=True)
        return []

    suite     = hdf5_path.parent.name
    task_name = hdf5_path.stem.removesuffix("_demo")
    task_out  = output_dir / suite / task_name

    # ── read demo list once ─────────────────────────────────────────────────
    with h5py.File(hdf5_path, "r") as f:
        data_grp  = f["data"]
        demo_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[1]),
        )
        n_demos = len(demo_keys)
        prob_info_str = data_grp.attrs.get("problem_info", "{}")
        try:
            language = json.loads(prob_info_str).get("language_instruction", "")
        except Exception:
            language = ""

    # Split: last val_demo_fraction of demos → val
    n_val = max(1, int(n_demos * val_demo_fraction))
    val_set = set(range(n_demos - n_val, n_demos))

    # ── build env once per file ─────────────────────────────────────────────
    env = build_env(bddl, img_size=img_size, gpu_device_id=gpu_device_id)
    env.reset()
    cam_id, nom_pos, nom_quat = get_cam(env)

    pairs: list[dict[str, Any]] = []
    pair_count = 0

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]

        for demo_idx in tqdm(
            range(n_demos),
            desc=f"    demos",
            leave=False,
            dynamic_ncols=True,
        ):
            if max_pairs is not None and pair_count >= max_pairs:
                break

            dk = f"demo_{demo_idx}"
            demo = data_grp[dk]

            states       = demo["states"][:]          # (T, state_dim)
            actions      = demo["actions"][:]          # (T, 7)
            robot_states = (
                demo["robot_states"][:] if "robot_states" in demo
                else np.zeros((len(states), 1), dtype=np.float32)
            )
            n_steps = len(states)
            split   = "val" if demo_idx in val_set else "train"

            # Sample timestep indices
            n_sample = max(1, int(n_steps * timestep_sample_rate))
            sampled_t = sorted(
                rng.choice(n_steps, size=min(n_sample, n_steps), replace=False).tolist()
            )

            for t in tqdm(sampled_t, desc="      frames", leave=False, dynamic_ncols=True):
                if max_pairs is not None and pair_count >= max_pairs:
                    break

                state_vec  = states[t]
                state_hash = hash_array(state_vec)
                action_hash= hash_array(actions[t])
                robot_hash = hash_array(robot_states[t])

                # Sample perturbation params for each view
                perturb_params = [sample_camera(rng) for _ in range(views_per_state)]

                # Render
                nom_img, pert_list = render_views(
                    env, cam_id, nom_pos, nom_quat, state_vec, perturb_params, img_size
                )

                # Save nominal image (skip if already exists — supports resume)
                nom_abs = task_out / f"demo_{demo_idx}" / f"t{t:06d}_nominal.png"
                if not nom_abs.exists():
                    _save_png(nom_abs, nom_img)

                # Save perturbed images + build pair rows
                for cam_params, (pert_img, pert_pos, pert_quat) in zip(perturb_params, pert_list):
                    label    = _cam_label(cam_params)
                    pert_abs = task_out / f"demo_{demo_idx}" / f"t{t:06d}_{label}.png"
                    if not pert_abs.exists():
                        _save_png(pert_abs, pert_img)

                    pair_id = stable_id(suite, task_name, demo_idx, t, label)
                    row: dict[str, Any] = {
                        "pair_id":           pair_id,
                        "source_type":       "libero_original_same_state_rerender",
                        "suite":             suite,
                        "task_name":         task_name,
                        "hdf5_path":         str(hdf5_path),
                        "demo_id":           demo_idx,
                        "timestep":          t,
                        "split":             split,
                        "language":          language,
                        "state_hash":        state_hash,
                        "action_chunk_hash": action_hash,
                        "robot_state_hash":  robot_hash,
                        "pair_confidence":   1.0,
                        "action_equivalence":"same HDF5 demo, same timestep, exact MuJoCo state reset",
                        "pair_type":         "matched",
                        "img_size":          img_size,
                        # Camera A = nominal
                        "img_a_path":    rel_to_repo(nom_abs),
                        "cam_pos_a":     nom_pos.tolist(),
                        "cam_quat_a":    nom_quat.tolist(),
                        "camera_params_a": {
                            "horizon": 0, "vertical": 0,
                            "scale":   100, "end_rot": 0, "end_vert": 0,
                        },
                        # Camera B = perturbed
                        "img_b_path":    rel_to_repo(pert_abs),
                        "cam_pos_b":     pert_pos.tolist(),
                        "cam_quat_b":    pert_quat.tolist(),
                        "camera_params_b": {
                            k: cam_params[k]
                            for k in ("horizon", "vertical", "scale", "end_rot", "end_vert")
                        },
                        "camera_category": cam_params["category"],
                    }
                    pairs.append(row)
                    pair_count += 1

    env.close()
    gc.collect()
    return pairs


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Wrong-state controls                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_wrong_state_controls(
    matched_pairs: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Shuffle img_b across matched pairs within same camera category.

    Produces wrong-state controls where camera distribution is preserved
    but state matching is broken (state_hash_a != state_hash_b).
    """
    by_cat: dict[str, list[dict]] = {}
    for p in matched_pairs:
        by_cat.setdefault(p["camera_category"], []).append(p)

    wrong: list[dict[str, Any]] = []
    for cat, rows in by_cat.items():
        idxs = list(range(len(rows)))
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        # Rotate by 1 to guarantee at least index mismatch
        shuffled = shuffled[1:] + shuffled[:1]

        for orig_i, shuf_i in zip(idxs, shuffled):
            a = rows[orig_i]
            b = rows[shuf_i]
            if a["state_hash"] == b["state_hash"]:
                continue   # accidentally same state — skip (rare)
            w = a.copy()
            w["pair_id"]        = "wrong_" + a["pair_id"]
            w["pair_type"]      = "wrong_state"
            w["pair_confidence"]= 0.0
            w["state_hash_b"]   = b["state_hash"]   # records that it differs
            w["img_b_path"]     = b["img_b_path"]
            w["cam_pos_b"]      = b["cam_pos_b"]
            w["cam_quat_b"]     = b["cam_quat_b"]
            w["camera_params_b"]= b["camera_params_b"]
            wrong.append(w)

    return wrong


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Report builders                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _rerender_report(
    smoke_reports: list[dict],
    all_pairs: list[dict],
    generated_at: str,
) -> str:
    lines = [
        "# LIBERO pair-data.3 — Rerender Reconstruction Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Smoke Test Results",
        "",
    ]
    if smoke_reports:
        lines += [
            "| Task | Frames | RMS mean | RMS max | Cam pos | Status |",
            "|---|---|---|---|---|---|",
        ]
        for r in smoke_reports:
            if "error" in r:
                lines.append(f"| — | — | — | — | — | ERROR: {r['error'][:60]} |")
            else:
                pos_str = "[" + ", ".join(f"{v:.3f}" for v in r["nominal_cam_pos"]) + "]"
                lines.append(
                    f"| {r['task'][:40]} "
                    f"| {r['n_frames_tested']} "
                    f"| {r['pixel_rms_mean']:.2f} "
                    f"| {r['pixel_rms_max']:.2f} "
                    f"| {pos_str} "
                    f"| {r['status']} |"
                )
    else:
        lines += ["No smoke tests run."]

    lines += ["", "## Pair Manifest Statistics", ""]
    if all_pairs:
        train_n = sum(1 for p in all_pairs if p["split"] == "train")
        val_n   = sum(1 for p in all_pairs if p["split"] == "val")
        by_cat: dict[str, int] = {}
        by_suite: dict[str, int] = {}
        for p in all_pairs:
            by_cat[p["camera_category"]] = by_cat.get(p["camera_category"], 0) + 1
            by_suite[p["suite"]] = by_suite.get(p["suite"], 0) + 1

        lines += [
            f"- Total matched pairs: {len(all_pairs)}",
            f"  - train: {train_n}",
            f"  - val:   {val_n}",
            "- By camera category:",
        ]
        for cat in ("C1", "C2", "C3"):
            n = by_cat.get(cat, 0)
            pct = 100 * n / len(all_pairs) if all_pairs else 0
            lines.append(f"  - {cat}: {n} ({pct:.1f}%)")
        lines += ["- By suite:"]
        for s, n in sorted(by_suite.items()):
            lines.append(f"  - {s}: {n}")
    else:
        lines += ["No pairs generated."]

    lines += [
        "",
        "## Checkpoint",
        "",
        "```",
        "PASS: nominal rerender RMS < 20 per channel (pixel values 0-255).",
        "Object poses and robot pose must be visually correct.",
        "See debug/rerender_sanity/ for comparison images (stored | rerendered).",
        "```",
    ]
    return "\n".join(lines)


def _pair_audit_report(all_pairs: list[dict], generated_at: str) -> str:
    lines = [
        "# LIBERO Pair Manifest Audit Report",
        "",
        f"Generated: {generated_at}",
        "",
    ]
    if not all_pairs:
        lines += ["No pairs generated."]
        return "\n".join(lines)

    by_suite: dict[str, dict] = {}
    for p in all_pairs:
        s = p["suite"]
        if s not in by_suite:
            by_suite[s] = {"total": 0, "train": 0, "val": 0, "C1": 0, "C2": 0, "C3": 0}
        by_suite[s]["total"] += 1
        by_suite[s][p["split"]] += 1
        by_suite[s][p["camera_category"]] += 1

    lines += [
        "| Suite | Total | Train | Val | C1 | C2 | C3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, d in sorted(by_suite.items()):
        lines.append(
            f"| {s} | {d['total']} | {d['train']} | {d['val']} "
            f"| {d['C1']} | {d['C2']} | {d['C3']} |"
        )

    total = len(all_pairs)
    lines += [
        "",
        f"**Total matched pairs: {total}**",
        "",
        "## Pair Schema",
        "",
        "Each row in `libero_pair_manifest_*.jsonl` contains:",
        "```",
        "pair_id, source_type, suite, task_name, hdf5_path,",
        "demo_id, timestep, split, language,",
        "state_hash, action_chunk_hash, robot_state_hash,",
        "pair_confidence=1.0, action_equivalence, pair_type='matched',",
        "img_size, img_a_path (nominal), img_b_path (perturbed),",
        "cam_pos_a/b, cam_quat_a/b, camera_params_a/b, camera_category",
        "```",
    ]
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO pair-data.3: same-state multiview LIBERO rerender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--libero-root",    type=Path, default=DEFAULT_LIBERO_ROOT,
                        help="Root of original LIBERO HDF5 suite directories.")
    parser.add_argument("--suite",          nargs="*", default=SUITE_DIRS, dest="suites",
                        help="Suite subdirectory names to process.")
    parser.add_argument("--output-dir",     type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Root for rendered PNG files.")
    parser.add_argument("--results-dir",    type=Path, default=DEFAULT_RESULTS_DIR_,
                        help="Root for JSONL manifests and .md reports.")
    parser.add_argument("--debug-dir",      type=Path, default=DEFAULT_DEBUG_DIR,
                        help="Root for smoke-test comparison images.")
    parser.add_argument("--bddl-root",      type=Path, default=BDDL_ROOT_DEFAULT,
                        help="Root of original LIBERO BDDL files.")
    parser.add_argument("--img-size",       type=int,  default=256)
    parser.add_argument("--gpu-device-id",  type=int,  default=0,
                        help="GPU device ID for offscreen rendering (-1 = CPU).")
    parser.add_argument("--timestep-sample-rate", type=float, default=0.10,
                        help="Fraction of timesteps to sample per demo.")
    parser.add_argument("--views-per-state",      type=int,   default=1,
                        help="Number of perturbed views to render per sampled timestep.")
    parser.add_argument("--max-pairs-per-suite",  type=int,   default=None,
                        help="Cap matched pairs per suite (for screening).")
    parser.add_argument("--max-tasks",      type=int,  default=None,
                        help="Cap HDF5 files processed per suite (for testing).")
    parser.add_argument("--val-demo-fraction", type=float, default=0.10,
                        help="Fraction of demos reserved for val split.")
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Run nominal reconstruction sanity check only (no batch render).")
    parser.add_argument("--smoke-frames",   type=int,  default=10,
                        help="Number of frames per task to test in smoke mode.")
    # ── Multi-GPU sharding ──────────────────────────────────────────────────
    parser.add_argument("--n-shards",    type=int, default=1,
                        help="Total number of parallel shards (= number of GPUs).")
    parser.add_argument("--shard-idx",   type=int, default=0,
                        help="Index of this shard (0-based). Each process runs one shard.")
    args = parser.parse_args()

    if args.shard_idx >= args.n_shards:
        raise ValueError(f"--shard-idx {args.shard_idx} must be < --n-shards {args.n_shards}")

    rng          = np.random.default_rng(args.seed)
    generated_at = now_iso()

    ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.debug_dir)

    # ── Smoke-test mode ─────────────────────────────────────────────────────
    if args.smoke_test:
        smoke_reports: list[dict] = []
        for suite_name in args.suites:
            suite_dir = args.libero_root / suite_name
            if not suite_dir.exists():
                print(f"[SKIP] {suite_dir} not found")
                continue
            hdf5_files = sorted(suite_dir.glob("*.hdf5"))
            if not hdf5_files:
                print(f"[SKIP] no HDF5 in {suite_dir}")
                continue
            n_tasks = min(len(hdf5_files), args.max_tasks if args.max_tasks else 1)
            for hdf5_path in hdf5_files[:n_tasks]:
                print(f"\n[smoke] {suite_name}/{hdf5_path.name}", flush=True)
                rep = run_smoke_test(
                    hdf5_path, args.bddl_root, args.debug_dir,
                    img_size=args.img_size,
                    gpu_device_id=args.gpu_device_id,
                    n_frames=args.smoke_frames,
                )
                smoke_reports.append(rep)
                if "error" in rep:
                    print(f"  ERROR: {rep['error']}")
                else:
                    print(
                        f"  RMS mean={rep['pixel_rms_mean']:.2f} "
                        f"max={rep['pixel_rms_max']:.2f}  status={rep['status']}"
                    )
                    print(f"  grid → {rep['comparison_grid']}")

        report_path = args.results_dir / "rerender_reconstruction_report.md"
        report_path.write_text(
            _rerender_report(smoke_reports, [], generated_at), encoding="utf-8"
        )
        write_json(
            args.results_dir / "smoke_test_results.json",
            {"generated_at": generated_at, "reports": smoke_reports},
        )
        print(f"\nSmoke-test report → {report_path}")
        return

    # ── Batch render mode ───────────────────────────────────────────────────

    # Build flat (suite, hdf5_path) work list across all requested suites
    all_work: list[tuple[str, Path]] = []
    for suite_name in args.suites:
        suite_dir = args.libero_root / suite_name
        if not suite_dir.exists():
            print(f"[SKIP] suite dir not found: {suite_dir}")
            continue
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"[SKIP] no HDF5 in {suite_dir}")
            continue
        if args.max_tasks:
            hdf5_files = hdf5_files[: args.max_tasks]
        for hdf5_path in hdf5_files:
            all_work.append((suite_name, hdf5_path))

    # Apply file-level sharding: this process owns files where global_idx % n_shards == shard_idx
    if args.n_shards > 1:
        all_work = [
            (s, h) for i, (s, h) in enumerate(all_work)
            if i % args.n_shards == args.shard_idx
        ]
        print(
            f"[shard {args.shard_idx}/{args.n_shards}] "
            f"This GPU owns {len(all_work)} files"
        )

    # Process files, tracking per-suite pair counts for optional cap
    all_pairs: list[dict[str, Any]] = []
    suite_pair_counts: dict[str, int] = {}

    for suite_name, hdf5_path in tqdm(
        all_work,
        desc=f"GPU{args.gpu_device_id} shard{args.shard_idx}",
        leave=True,
        dynamic_ncols=True,
    ):
        suite_count = suite_pair_counts.get(suite_name, 0)

        # Per-suite cap (shared across files within this shard)
        remaining = None
        if args.max_pairs_per_suite is not None:
            remaining = args.max_pairs_per_suite - suite_count
            if remaining <= 0:
                tqdm.write(f"  [{suite_name}] reached cap {args.max_pairs_per_suite}, skipping rest")
                continue

        task_pairs = process_hdf5(
            hdf5_path=hdf5_path,
            bddl_root=args.bddl_root,
            output_dir=args.output_dir,
            rng=rng,
            timestep_sample_rate=args.timestep_sample_rate,
            views_per_state=args.views_per_state,
            max_pairs=remaining,
            img_size=args.img_size,
            gpu_device_id=args.gpu_device_id,
            val_demo_fraction=args.val_demo_fraction,
        )
        all_pairs.extend(task_pairs)
        suite_pair_counts[suite_name] = suite_count + len(task_pairs)
        tqdm.write(
            f"  {suite_name}/{hdf5_path.name}: "
            f"+{len(task_pairs)} pairs  (suite total this shard: {suite_pair_counts[suite_name]})"
        )

    # ── Shard-specific manifest names ───────────────────────────────────────
    # When sharding, each GPU writes its own shard file.
    # After all GPUs finish, merge with merge_render_shards.sh.
    shard_tag = f"_shard{args.shard_idx:02d}" if args.n_shards > 1 else ""

    train_pairs = [p for p in all_pairs if p["split"] == "train"]
    val_pairs   = [p for p in all_pairs if p["split"] == "val"]
    write_jsonl(args.results_dir / f"libero_pair_manifest_train{shard_tag}.jsonl",  train_pairs)
    write_jsonl(args.results_dir / f"libero_pair_manifest_val{shard_tag}.jsonl",    val_pairs)
    print(f"[shard{args.shard_idx}] Matched pairs: train={len(train_pairs)}  val={len(val_pairs)}")

    wrong_train = build_wrong_state_controls(train_pairs, rng)
    wrong_val   = build_wrong_state_controls(val_pairs,   rng)
    write_jsonl(args.results_dir / f"libero_wrong_pair_manifest_train{shard_tag}.jsonl", wrong_train)
    write_jsonl(args.results_dir / f"libero_wrong_pair_manifest_val{shard_tag}.jsonl",   wrong_val)
    print(f"[shard{args.shard_idx}] Wrong-state controls: train={len(wrong_train)}  val={len(wrong_val)}")

    # Per-shard report (merged by merge script)
    (args.results_dir / f"libero_pair_audit_report{shard_tag}.md").write_text(
        _pair_audit_report(all_pairs, generated_at), encoding="utf-8"
    )
    print(f"\n[shard{args.shard_idx}] Done. Total matched pairs this shard: {len(all_pairs)}")


if __name__ == "__main__":
    main()
