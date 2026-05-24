"""LIBERO pair-data.3 visual camera sanity check.

Renders a comprehensive grid of C1/C2/C3 camera configurations for one
task + one timestep so you can visually verify camera coverage before
launching the full batch rerender.

Outputs (in --out-dir):
  nominal.png            — reference agentview frame
  C1_distance.png        — 6 scale variants (115→200)
  C2_horizontal.png      — azimuth sweep, vertical=0
  C2_elevated.png        — azimuth sweep, vertical=15
  C3_roll.png            — end_rot variants, end_vert=0
  C3_pitch.png           — end_vert variants, end_rot=0
  C3_combined.png        — end_rot+end_vert both non-zero
  ALL_overview.png       — one big labeled grid of everything

Usage:
  venv-libero-plus/bin/python scripts/libero_pair_data/visual_camera_check.py \\
    --libero-root data/libero_hdf5_original \\
    --suite libero_spatial \\
    --task-idx 0 \\
    --demo-idx 0 \\
    --timestep 30 \\
    --out-dir debug/camera_visual_check \\
    --gpu-device-id 0
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
OPENPI_LIBERO = REPO_ROOT / "openpi" / "third_party" / "libero"
sys.path.insert(0, str(OPENPI_LIBERO))
sys.path.insert(0, str(_HERE))

from render_libero_multiview_states import (
    build_env,
    compute_camera,
    get_cam,
    hdf5_to_bddl,
    _render,
    BDDL_ROOT_DEFAULT,
)

# ── Camera configurations to visualize ────────────────────────────────────────

NOMINAL = {"label": "NOMINAL\nh=0 v=0 s=100\ner=0 ev=0",
           "horizon": 0, "vertical": 0, "scale_pct": 100, "end_rot": 0, "end_vert": 0}

C1_CONFIGS = [
    {"label": f"C1 s={s}", "horizon": 0, "vertical": 0,
     "scale_pct": s, "end_rot": 0, "end_vert": 0}
    for s in [115, 130, 145, 160, 175, 200]
]

# C2 azimuth sweep at vertical=0 (positive = left, 285-359 = right i.e. -75..-1)
C2_H_ZERO = [
    {"label": f"C2 h={h}\nv=0",
     "horizon": h, "vertical": 0, "scale_pct": 100, "end_rot": 0, "end_vert": 0}
    for h in [15, 30, 45, 60, 75, 315, 330, 345]   # 315=-45, 330=-30, 345=-15
]

# C2 azimuth sweep at vertical=15 (camera raised)
C2_H_V15 = [
    {"label": f"C2 h={h}\nv=15",
     "horizon": h, "vertical": 15, "scale_pct": 100, "end_rot": 0, "end_vert": 0}
    for h in [0, 15, 30, 45, 315, 330, 345, 60]
]

# C3 roll only (end_rot, end_vert=0)
C3_ROLL = [
    {"label": f"C3 er={er}\nev=0",
     "horizon": 0, "vertical": 0, "scale_pct": 100, "end_rot": er, "end_vert": 0}
    for er in [2, 6, 10, 350, 354, 358]
]

# C3 pitch only (end_vert, end_rot=0)
C3_PITCH = [
    {"label": f"C3 er=0\nev={ev}",
     "horizon": 0, "vertical": 0, "scale_pct": 100, "end_rot": 0, "end_vert": ev}
    for ev in [2, 6, 10, 350, 354, 358]
]

# C3 both non-zero
C3_COMBINED = [
    {"label": f"C3 er={er}\nev={ev}",
     "horizon": 0, "vertical": 0, "scale_pct": 100, "end_rot": er, "end_vert": ev}
    for er, ev in [(6, 6), (10, 10), (6, 354), (354, 6), (10, 350), (350, 10)]
]


# ── Image helpers ──────────────────────────────────────────────────────────────

def _label_image(img_raw: np.ndarray, text: str, img_size: int) -> Image.Image:
    """Flip (robosuite→display), add black label bar at bottom."""
    arr = np.flipud(img_raw).astype(np.uint8)
    pil = Image.fromarray(arr)

    bar_h = 36
    canvas = Image.new("RGB", (img_size, img_size + bar_h), (30, 30, 30))
    canvas.paste(pil, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    y = img_size + 2
    for line in text.split("\n"):
        draw.text((4, y), line, fill=(220, 220, 80), font=font)
        y += 12

    return canvas


def _make_grid(images: list[Image.Image], ncols: int, title: str = "") -> Image.Image:
    """Arrange images in a grid with optional title bar."""
    if not images:
        return Image.new("RGB", (100, 100), (50, 50, 50))

    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    pad = 4
    title_h = 30 if title else 0

    grid_w = ncols * (w + pad) + pad
    grid_h = nrows * (h + pad) + pad + title_h
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    if title:
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((pad, 6), title, fill=(255, 255, 120), font=font)

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols
        x = pad + col * (w + pad)
        y = title_h + pad + row * (h + pad)
        grid.paste(img, (x, y))

    return grid


def render_config(env, cam_id, nom_pos, nom_quat, state_vec, cfg, img_size) -> Image.Image:
    """Render one camera configuration and return a labeled PIL image."""
    pos, quat = compute_camera(
        nom_pos, nom_quat,
        horizon=cfg["horizon"],
        vertical=cfg["vertical"],
        scale_pct=cfg["scale_pct"],
        end_rot=cfg["end_rot"],
        end_vert=cfg["end_vert"],
    )
    env.sim.model.cam_pos[cam_id][:] = pos
    env.sim.model.cam_quat[cam_id][:] = quat
    env.sim.forward()   # propagate model.cam_pos → data.cam_xpos before render
    img_raw = env.sim.render(width=img_size, height=img_size, camera_name="agentview")
    # restore nominal
    env.sim.model.cam_pos[cam_id][:] = nom_pos
    env.sim.model.cam_quat[cam_id][:] = nom_quat
    return _label_image(img_raw, cfg["label"], img_size)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--libero-root", type=Path,
                        default=Path("data/libero_hdf5_original"))
    parser.add_argument("--suite",       default="libero_spatial")
    parser.add_argument("--task-idx",    type=int, default=0,
                        help="Index of HDF5 file (0-based) within the suite dir.")
    parser.add_argument("--demo-idx",    type=int, default=0)
    parser.add_argument("--timestep",    type=int, default=30,
                        help="Timestep index within the demo to render.")
    parser.add_argument("--img-size",    type=int, default=256)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--bddl-root",   type=Path, default=BDDL_ROOT_DEFAULT)
    parser.add_argument("--out-dir",     type=Path,
                        default=Path("debug/camera_visual_check"))
    args = parser.parse_args()

    # ── locate HDF5 ────────────────────────────────────────────────────────────
    suite_dir  = args.libero_root / args.suite
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {suite_dir}")
    hdf5_path = hdf5_files[args.task_idx]
    bddl      = hdf5_to_bddl(hdf5_path, args.bddl_root)
    task_name = hdf5_path.stem.removesuffix("_demo")
    print(f"Task   : {task_name}")
    print(f"HDF5   : {hdf5_path}")
    print(f"BDDL   : {bddl}")

    # ── load state ─────────────────────────────────────────────────────────────
    with h5py.File(hdf5_path, "r") as f:
        demo   = f[f"data/demo_{args.demo_idx}"]
        states = demo["states"][:]
        n_steps = len(states)

    t = min(args.timestep, n_steps - 1)
    print(f"Demo   : demo_{args.demo_idx}  (total timesteps: {n_steps})")
    print(f"Timestep: {t}")

    # ── build env ──────────────────────────────────────────────────────────────
    print("Building env ...", flush=True)
    env = build_env(bddl, img_size=args.img_size, gpu_device_id=args.gpu_device_id)
    env.reset()
    cam_id, nom_pos, nom_quat = get_cam(env)
    print(f"Nominal cam pos  = {nom_pos.tolist()}")
    print(f"Nominal cam quat = {nom_quat.tolist()}")

    # ── set state ──────────────────────────────────────────────────────────────
    env.sim.set_state_from_flattened(states[t])
    env.sim.forward()

    # ── render all configs ─────────────────────────────────────────────────────
    print("Rendering ...", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    S = args.img_size

    def render(cfg: dict[str, Any]) -> Image.Image:
        return render_config(env, cam_id, nom_pos, nom_quat, states[t], cfg, S)

    # Nominal
    nom_img = render(NOMINAL)
    nom_img.save(str(args.out_dir / "nominal.png"))
    print("  nominal.png")

    # C1 — distance
    c1_imgs = [nom_img] + [render(c) for c in C1_CONFIGS]
    c1_labels = ["NOMINAL (ref)"] + [f"scale={c['scale_pct']}" for c in C1_CONFIGS]
    c1_grid = _make_grid(c1_imgs, ncols=7,
                         title=f"C1 — Distance  |  task: {task_name[:60]}  |  t={t}")
    c1_grid.save(str(args.out_dir / "C1_distance.png"))
    print("  C1_distance.png")

    # C2 — horizontal sweep (vertical=0)
    c2h_imgs = [nom_img] + [render(c) for c in C2_H_ZERO]
    c2h_grid = _make_grid(
        c2h_imgs, ncols=9,
        title=f"C2 — Azimuth sweep (vertical=0)  |  t={t}  |  positive h = left, 315=-45°, 330=-30°, 345=-15°",
    )
    c2h_grid.save(str(args.out_dir / "C2_horizontal.png"))
    print("  C2_horizontal.png")

    # C2 — elevated (vertical=15)
    c2v_imgs = [nom_img] + [render(c) for c in C2_H_V15]
    c2v_grid = _make_grid(
        c2v_imgs, ncols=9,
        title=f"C2 — Azimuth sweep (vertical=15, camera raised)  |  t={t}",
    )
    c2v_grid.save(str(args.out_dir / "C2_elevated.png"))
    print("  C2_elevated.png")

    # C3 — roll
    c3r_imgs = [nom_img] + [render(c) for c in C3_ROLL]
    c3r_grid = _make_grid(
        c3r_imgs, ncols=7,
        title=f"C3 — Roll (end_rot only, end_vert=0)  |  t={t}  |  350-358 = negative rotation",
    )
    c3r_grid.save(str(args.out_dir / "C3_roll.png"))
    print("  C3_roll.png")

    # C3 — pitch
    c3p_imgs = [nom_img] + [render(c) for c in C3_PITCH]
    c3p_grid = _make_grid(
        c3p_imgs, ncols=7,
        title=f"C3 — Pitch (end_vert only, end_rot=0)  |  t={t}  |  350-358 = negative rotation",
    )
    c3p_grid.save(str(args.out_dir / "C3_pitch.png"))
    print("  C3_pitch.png")

    # C3 — combined roll+pitch
    c3c_imgs = [nom_img] + [render(c) for c in C3_COMBINED]
    c3c_grid = _make_grid(
        c3c_imgs, ncols=7,
        title=f"C3 — Roll+Pitch combined  |  t={t}",
    )
    c3c_grid.save(str(args.out_dir / "C3_combined.png"))
    print("  C3_combined.png")

    # Big overview grid: nominal + best-of each category
    overview_configs = (
        [NOMINAL]
        + C1_CONFIGS[::2]                        # s=115,145,175,200 → every other
        + C2_H_ZERO[:4] + C2_H_ZERO[4:]          # full C2 horizontal
        + C2_H_V15[:4]                            # elevated subset
        + C3_ROLL[:3] + C3_PITCH[:3]             # small angle examples
        + C3_COMBINED[:3]
    )
    all_imgs = [render(c) for c in overview_configs]
    all_grid = _make_grid(
        all_imgs, ncols=8,
        title=f"ALL CAMERA TYPES OVERVIEW  |  task: {task_name[:55]}  |  demo={args.demo_idx}  t={t}",
    )
    all_grid.save(str(args.out_dir / "ALL_overview.png"))
    print("  ALL_overview.png")

    env.close()
    gc.collect()

    print(f"\nDone. Images saved to: {args.out_dir.resolve()}")
    print("\nFiles:")
    for f in sorted(args.out_dir.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:30s}  {size_kb:5d} KB")


if __name__ == "__main__":
    main()
