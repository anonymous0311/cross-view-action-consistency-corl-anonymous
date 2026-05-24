"""Audit original LIBERO HDF5 demonstration files for simulator-state rerender feasibility.

This script targets the original LIBERO HDF5 archives (from LIBERO-Project on HuggingFace),
NOT the OpenVLA RLDS/TFRecord exports. It inspects each suite, reports key structural fields,
and classifies per-suite rerender feasibility.

Feasibility classification:
  LIBERO_RERENDER_PRIMARY  — per-frame simulator states + task XML/env_args found
  LIBERO_REPLAY_APPROX_ONLY — only init_states found (init + replay chain, not per-frame reset)
  LIBERO_HDF5_UNAVAILABLE — no replayable original LIBERO HDF5 state found

Usage:
  python scripts/libero_pair_data/audit_libero_hdf5_original.py \
    --libero-root /path/to/original_libero_hdf5 \
    [--suite-dirs libero_spatial libero_object libero_goal libero_90 libero_10] \
    [--max-files-per-suite 3] \
    [--output-dir results/libero_pair_audit]

The script writes:
  results/libero_pair_audit/libero_pair_data_libero_hdf5_audit.md
  results/libero_pair_audit/libero_pair_data_suite_inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Allow importing libero_pair_data_common whether run from repo root or scripts/libero_pair_data
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from libero_pair_data_common import DEFAULT_RESULTS_DIR, ensure_dir, now_iso, rel_to_repo, write_json


# ── field classifiers ──────────────────────────────────────────────────────────

# Per-frame flattened MuJoCo simulator state fields (allow exact frame-level reset).
# Keep this strict: `obs/ee_states`, `obs/joint_states`, and `robot_states` are
# useful proprio fields, but they are not sufficient for same-state rerender.
PER_FRAME_STATE_NAMES = ("states", "sim_states", "mujoco_state")
# Initial-only simulator state (allows replay from start, not per-frame reset)
INIT_STATE_HINTS = ("init_states", "initial_state")
# Task asset / environment config fields (needed to reconstruct the MuJoCo scene)
ASSET_HINTS = ("model_file", "xml", "env_args", "bddl", "task_info", "problem_info", "env_name")
# Action fields
ACTION_HINTS = ("actions", "action")
# Observation image fields
IMAGE_HINTS = (
    "agentview_rgb",
    "agentview_image",
    "eye_in_hand_rgb",
    "wrist_image",
    "eye_in_hand",
    "frontview_image",
    "image",
)
# Robot state fields
ROBOT_STATE_HINTS = ("ee_pos", "ee_ori", "joint_states", "gripper_states", "robot0_eef")


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.dtype.kind in ("S", "U"):
            return value.tolist()
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _matches(name: str, hints: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(h in lower for h in hints)


def _is_per_frame_sim_state(name: str) -> bool:
    return name.lower().split("/")[-1] in PER_FRAME_STATE_NAMES


def _collect_datasets(f: h5py.File, max_items: int) -> tuple[list[dict], list[str]]:
    datasets: list[dict] = []
    groups: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        if len(datasets) + len(groups) >= max_items:
            return
        if isinstance(obj, h5py.Dataset):
            datasets.append({
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            })
        elif isinstance(obj, h5py.Group):
            groups.append(name)

    f.visititems(_visit)
    return datasets, groups


def inspect_hdf5(path: Path, max_items: int = 120) -> dict[str, Any]:
    """Inspect a single LIBERO HDF5 file and classify its state fields."""
    result: dict[str, Any] = {
        "file": str(path),
        "error": None,
        "top_keys": [],
        "file_attrs": {},
        "n_demos": 0,
        "demo_key_sample": None,
        "demo_keys": [],
        "demo_attrs": {},
        # Field presence flags
        "has_actions": False,
        "has_obs_images": False,
        "has_per_frame_states": False,
        "has_init_states": False,
        "has_model_xml_or_env_args": False,
        "has_robot_state": False,
        # Matched field names
        "per_frame_state_fields": [],
        "init_state_fields": [],
        "asset_fields": [],
        "action_fields": [],
        "image_fields": [],
        "robot_state_fields": [],
        # Shape summaries
        "action_shape": None,
        "image_shape": None,
        "state_shape": None,
        "init_state_shape": None,
        # Feasibility per file
        "exact_frame_reset_feasible": "unknown",
        "replay_feasible": "unknown",
        "rerender_feasible": "unknown",
    }

    try:
        with h5py.File(path, "r") as f:
            result["top_keys"] = list(f.keys())
            result["file_attrs"] = {k: _decode_attr(v) for k, v in f.attrs.items()}

            # Classify file-level attributes
            for attr_name, attr_val in f.attrs.items():
                lower = attr_name.lower()
                decoded = _decode_attr(attr_val)
                if _matches(attr_name, ASSET_HINTS):
                    result["asset_fields"].append(f"@attr:{attr_name}")
                    result["has_model_xml_or_env_args"] = True
                if lower == "env_args" and isinstance(decoded, str):
                    try:
                        env_args = json.loads(decoded)
                        result["env_args_keys"] = list(env_args.keys()) if isinstance(env_args, dict) else []
                    except Exception:
                        result["env_args_keys"] = ["<parse-error>"]

            # Find demo groups under data/
            if "data" not in f:
                result["error"] = "no 'data' group in file"
                return result

            data_grp = f["data"]
            demo_keys = sorted(
                [k for k in data_grp.keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
            )
            result["n_demos"] = len(demo_keys)

            if not demo_keys:
                result["error"] = "no demo_ groups found under data/"
                return result

            demo_key = demo_keys[0]
            result["demo_key_sample"] = demo_key
            demo = data_grp[demo_key]
            result["demo_keys"] = list(demo.keys())
            result["demo_attrs"] = {k: _decode_attr(v) for k, v in demo.attrs.items()}

            # Classify demo-level keys and sub-keys
            def _classify(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    shape = list(obj.shape)
                    lower_name = name.lower()
                    if _is_per_frame_sim_state(name):
                        result["per_frame_state_fields"].append(name)
                        result["has_per_frame_states"] = True
                        if result["state_shape"] is None:
                            result["state_shape"] = shape
                    if _matches(name, INIT_STATE_HINTS):
                        result["init_state_fields"].append(name)
                        result["has_init_states"] = True
                        if result["init_state_shape"] is None:
                            result["init_state_shape"] = shape
                    if _matches(name, ASSET_HINTS):
                        result["asset_fields"].append(name)
                        result["has_model_xml_or_env_args"] = True
                    if _matches(name, ACTION_HINTS):
                        result["action_fields"].append(name)
                        result["has_actions"] = True
                        if result["action_shape"] is None:
                            result["action_shape"] = shape
                    if _matches(name, IMAGE_HINTS):
                        result["image_fields"].append(name)
                        result["has_obs_images"] = True
                        if result["image_shape"] is None:
                            result["image_shape"] = shape
                    if _matches(name, ROBOT_STATE_HINTS):
                        result["robot_state_fields"].append(name)
                        result["has_robot_state"] = True
                elif isinstance(obj, h5py.Group):
                    # Also check group-level attributes for env_args/model_file
                    for attr_name, attr_val in obj.attrs.items():
                        if _matches(attr_name, ASSET_HINTS):
                            result["asset_fields"].append(f"@{name}.attr:{attr_name}")
                            result["has_model_xml_or_env_args"] = True

            demo.visititems(_classify)

            # Also check data-level group attributes
            for attr_name in data_grp.attrs:
                if _matches(attr_name, ASSET_HINTS):
                    result["asset_fields"].append(f"@data.attr:{attr_name}")
                    result["has_model_xml_or_env_args"] = True

    except Exception as exc:
        result["error"] = repr(exc)
        return result

    # Classify feasibility for this file
    has_state = result["has_per_frame_states"]
    has_init = result["has_init_states"]
    has_asset = result["has_model_xml_or_env_args"]

    if has_state and has_asset:
        result["exact_frame_reset_feasible"] = True
        result["replay_feasible"] = True
        result["rerender_feasible"] = True
    elif has_state and not has_asset:
        result["exact_frame_reset_feasible"] = "partial"
        result["replay_feasible"] = True
        result["rerender_feasible"] = "partial"
    elif has_init and has_asset:
        result["exact_frame_reset_feasible"] = False
        result["replay_feasible"] = True
        result["rerender_feasible"] = "partial"
    elif has_init:
        result["exact_frame_reset_feasible"] = False
        result["replay_feasible"] = True
        result["rerender_feasible"] = False
    else:
        result["exact_frame_reset_feasible"] = False
        result["replay_feasible"] = False
        result["rerender_feasible"] = False

    return result


def audit_suite(suite_name: str, suite_root: Path, max_files: int) -> dict[str, Any]:
    """Audit one LIBERO suite directory."""
    h5_paths = sorted(suite_root.rglob("*.hdf5")) + sorted(suite_root.rglob("*.h5"))
    # Exclude cache/venv artifacts
    h5_paths = [p for p in h5_paths if ".venv" not in str(p) and "site-packages" not in str(p)]
    n_total = len(h5_paths)
    sample = h5_paths[:max_files]

    suite_entry: dict[str, Any] = {
        "suite_name": suite_name,
        "suite_root": str(suite_root),
        "found": suite_root.exists(),
        "n_hdf5_files": n_total,
        "n_demos_sampled": 0,
        "has_actions": False,
        "has_obs_images": False,
        "has_states": False,
        "has_init_states": False,
        "has_model_xml_or_env_args": False,
        "exact_frame_reset_feasible": "unknown",
        "replay_feasible": "unknown",
        "rerender_feasible": "unknown",
        "notes": "",
        "file_inspections": [],
    }

    if not suite_root.exists():
        suite_entry["notes"] = "directory not found — download pending"
        suite_entry["exact_frame_reset_feasible"] = "unknown"
        suite_entry["replay_feasible"] = "unknown"
        suite_entry["rerender_feasible"] = "unknown"
        return suite_entry

    if n_total == 0:
        suite_entry["notes"] = "directory exists but no HDF5 files found"
        suite_entry["exact_frame_reset_feasible"] = False
        suite_entry["replay_feasible"] = False
        suite_entry["rerender_feasible"] = False
        return suite_entry

    inspections = []
    total_demos = 0
    for path in sample:
        info = inspect_hdf5(path)
        inspections.append(info)
        total_demos += info.get("n_demos", 0)
        if info.get("has_actions"):
            suite_entry["has_actions"] = True
        if info.get("has_obs_images"):
            suite_entry["has_obs_images"] = True
        if info.get("has_per_frame_states"):
            suite_entry["has_states"] = True
        if info.get("has_init_states"):
            suite_entry["has_init_states"] = True
        if info.get("has_model_xml_or_env_args"):
            suite_entry["has_model_xml_or_env_args"] = True

    suite_entry["n_demos_sampled"] = total_demos
    suite_entry["file_inspections"] = inspections

    # Aggregate feasibility
    feasibilities = [i["rerender_feasible"] for i in inspections if i["error"] is None]
    if any(f is True for f in feasibilities):
        suite_entry["rerender_feasible"] = True
    elif any(f == "partial" for f in feasibilities):
        suite_entry["rerender_feasible"] = "partial"
    elif all(f is False for f in feasibilities):
        suite_entry["rerender_feasible"] = False
    else:
        suite_entry["rerender_feasible"] = "unknown"

    frame_feas = [i["exact_frame_reset_feasible"] for i in inspections if i["error"] is None]
    if any(f is True for f in frame_feas):
        suite_entry["exact_frame_reset_feasible"] = True
    elif any(f == "partial" for f in frame_feas):
        suite_entry["exact_frame_reset_feasible"] = "partial"
    elif all(f is False for f in frame_feas):
        suite_entry["exact_frame_reset_feasible"] = False
    else:
        suite_entry["exact_frame_reset_feasible"] = "unknown"

    replay_feas = [i["replay_feasible"] for i in inspections if i["error"] is None]
    if any(f is True for f in replay_feas):
        suite_entry["replay_feasible"] = True
    elif all(f is False for f in replay_feas):
        suite_entry["replay_feasible"] = False
    else:
        suite_entry["replay_feasible"] = "unknown"

    # Build notes
    notes = []
    if suite_entry["has_states"]:
        notes.append("per-frame simulator states present")
    elif suite_entry["has_init_states"]:
        notes.append("init_states only — replay chain required, not per-frame reset")
    if suite_entry["has_model_xml_or_env_args"]:
        notes.append("task XML / env_args found — scene reconstruction possible")
    else:
        notes.append("no task XML / env_args — external asset lookup required")
    errors = [i["error"] for i in inspections if i["error"]]
    if errors:
        notes.append(f"errors on {len(errors)} file(s): {errors[0]}")
    suite_entry["notes"] = "; ".join(notes)

    return suite_entry


def classify_global(inventory: list[dict]) -> str:
    """Assign global feasibility decision from suite inventory."""
    found = [s for s in inventory if s["found"] and s["n_hdf5_files"] > 0]
    if not found:
        return "LIBERO_HDF5_UNAVAILABLE"
    rerenders = [s["rerender_feasible"] for s in found]
    if any(r is True for r in rerenders):
        return "LIBERO_RERENDER_PRIMARY"
    if any(r == "partial" for r in rerenders):
        replay_chains = [s["replay_feasible"] for s in found]
        if any(r is True for r in replay_chains):
            return "LIBERO_REPLAY_APPROX_ONLY"
    replays = [s["replay_feasible"] for s in found]
    if any(r is True for r in replays):
        return "LIBERO_REPLAY_APPROX_ONLY"
    return "LIBERO_HDF5_UNAVAILABLE"


def build_markdown(inventory: list[dict], decision: str, generated_at: str) -> str:
    lines = [
        "# LIBERO pair-data: Original LIBERO HDF5 Audit",
        "",
        f"Generated: {generated_at}",
        "",
        f"## Global Decision: `{decision}`",
        "",
    ]

    decision_text = {
        "LIBERO_RERENDER_PRIMARY": (
            "Per-frame simulator states and task assets found in original LIBERO HDF5 files. "
            "Same-state multi-camera rerender is feasible. This is the primary LIBERO pair-data pair source."
        ),
        "LIBERO_REPLAY_APPROX_ONLY": (
            "Only `init_states` (no per-frame states) found. Exact frame-level reset is not possible. "
            "Deterministic replay from init_state + action sequence can reconstruct specific timesteps "
            "with additional compute. Rerender quality depends on action replay determinism."
        ),
        "LIBERO_HDF5_UNAVAILABLE": (
            "No replayable simulator states found in any suite. "
            "Do not start cross-view training; obtain the original LIBERO HDF5 archive first."
        ),
    }.get(decision, "Unknown decision.")

    lines += [decision_text, ""]
    lines += ["## Suite Inventory", ""]

    header = "| Suite | HDF5 files | Demos sampled | Actions | Images | States | Init states | Env args | Frame reset | Replay | Rerender | Notes |"
    sep =    "|-------|-----------|--------------|---------|--------|--------|-------------|----------|-------------|--------|----------|-------|"
    lines += [header, sep]

    def _yn(v: Any) -> str:
        if v is True:
            return "yes"
        if v is False:
            return "no"
        if v == "partial":
            return "partial"
        return "?"

    for s in inventory:
        row = (
            f"| {s['suite_name']} "
            f"| {s['n_hdf5_files']} "
            f"| {s['n_demos_sampled']} "
            f"| {_yn(s['has_actions'])} "
            f"| {_yn(s['has_obs_images'])} "
            f"| {_yn(s['has_states'])} "
            f"| {_yn(s['has_init_states'])} "
            f"| {_yn(s['has_model_xml_or_env_args'])} "
            f"| {_yn(s['exact_frame_reset_feasible'])} "
            f"| {_yn(s['replay_feasible'])} "
            f"| {_yn(s['rerender_feasible'])} "
            f"| {s['notes'][:80]} |"
        )
        lines.append(row)

    lines += ["", "## Per-Suite Detail", ""]

    for s in inventory:
        lines += [f"### {s['suite_name']}", ""]
        lines += [f"- Root: `{s['suite_root']}`"]
        lines += [f"- HDF5 files on disk: `{s['n_hdf5_files']}`"]
        lines += [f"- Demos in sampled files: `{s['n_demos_sampled']}`"]

        if not s["found"]:
            lines += ["- **Status: directory not found — download pending**", ""]
            continue

        for fi in s.get("file_inspections", []):
            lines += [f"  #### `{Path(fi['file']).name}`"]
            if fi["error"]:
                lines += [f"  - **error:** `{fi['error']}`"]
                continue
            lines += [
                f"  - top-level keys: `{fi['top_keys']}`",
                f"  - n_demos: `{fi['n_demos']}`",
                f"  - demo_keys (first demo): `{fi['demo_keys'][:20]}`",
                f"  - action_fields: `{fi['action_fields']}`  shape=`{fi['action_shape']}`",
                f"  - image_fields: `{fi['image_fields'][:5]}`  shape=`{fi['image_shape']}`",
                f"  - per_frame_state_fields: `{fi['per_frame_state_fields']}`  shape=`{fi['state_shape']}`",
                f"  - init_state_fields: `{fi['init_state_fields']}`  shape=`{fi['init_state_shape']}`",
                f"  - asset_fields: `{fi['asset_fields'][:5]}`",
                f"  - robot_state_fields: `{fi['robot_state_fields'][:5]}`",
                f"  - exact_frame_reset_feasible: `{fi['exact_frame_reset_feasible']}`",
                f"  - replay_feasible: `{fi['replay_feasible']}`",
                f"  - rerender_feasible: `{fi['rerender_feasible']}`",
            ]
            if "env_args_keys" in fi:
                lines += [f"  - env_args keys: `{fi['env_args_keys']}`"]
            lines += [""]

        lines += [""]

    lines += [
        "## LIBERO pair-data Consequence",
        "",
        f"- Global decision: `{decision}`",
        "- `LIBERO_RERENDER_PRIMARY`: proceed with `scripts/libero_pair_data/render_libero_multiview_states.py`.",
        "- Render canonical/C1/C2/C3 same-state pairs from original LIBERO states.",
        "- Match the LIBERO-plus camera perturbation categories and category proportions for training-data construction.",
        "- Do not use approximate same-language or same-task pairs in the cross-view primary route.",
        "",
        "## Important Notes on Previously Available Data",
        "",
        "- `data/libero_rlds/` contains OpenVLA RLDS/TFRecord exports (libero_spatial, libero_object,",
        "  libero_goal, libero_10_no_noops). These have RGB images, actions, and robot state only.",
        "  They do NOT have simulator states, model XML, or env_args. Rerender is not possible from these files.",
        "- The local LIBERO-plus LeRobot/RLDS export is useful for evaluation metadata and public camera-task",
        "  definitions, but it does not provide full-trajectory simulator states for direct same-state rerender.",
        "- This audit targets only the original LIBERO HDF5 files (from LIBERO-Project on HuggingFace).",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit original LIBERO HDF5 files for rerender feasibility.")
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=None,
        help="Root directory containing original LIBERO HDF5 suite subdirectories.",
    )
    parser.add_argument(
        "--suite-dirs",
        nargs="*",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"],
        help="Suite subdirectory names to inspect under --libero-root.",
    )
    parser.add_argument(
        "--extra-paths",
        nargs="*",
        default=[],
        help="Additional explicit HDF5 file or directory paths to inspect.",
    )
    parser.add_argument("--max-files-per-suite", type=int, default=3, help="Max HDF5 files to inspect per suite.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    generated_at = now_iso()

    inventory: list[dict] = []

    # Audit each named suite
    if args.libero_root is not None:
        for suite_name in args.suite_dirs:
            suite_root = args.libero_root / suite_name
            print(f"Auditing suite: {suite_name} @ {suite_root}")
            entry = audit_suite(suite_name, suite_root, args.max_files_per_suite)
            inventory.append(entry)
            feas = entry["rerender_feasible"]
            print(f"  -> files={entry['n_hdf5_files']} demos={entry['n_demos_sampled']} rerender={feas}")

    # Audit any extra explicit paths
    for extra in args.extra_paths:
        extra_path = Path(extra)
        suite_name = extra_path.name
        print(f"Auditing extra path: {suite_name} @ {extra_path}")
        entry = audit_suite(suite_name, extra_path, args.max_files_per_suite)
        entry["suite_name"] = str(extra_path)  # use full path as name for clarity
        inventory.append(entry)

    # Handle case where no paths were given
    if not inventory:
        print("No paths to audit. Provide --libero-root or --extra-paths.")
        print("Example:")
        print("  python scripts/libero_pair_data/audit_libero_hdf5_original.py \\")
        print("    --libero-root /path/to/original_libero \\")
        print("    --suite-dirs libero_spatial libero_object libero_goal libero_90 libero_10")
        return

    decision = classify_global(inventory)
    print(f"\nGlobal decision: {decision}")

    # Write compact inventory (strip large file_inspections for the JSON)
    compact_inventory = []
    for entry in inventory:
        compact = {k: v for k, v in entry.items() if k != "file_inspections"}
        compact["file_samples"] = [
            {
                "file": fi["file"],
                "n_demos": fi["n_demos"],
                "has_actions": fi["has_actions"],
                "has_obs_images": fi["has_obs_images"],
                "has_per_frame_states": fi["has_per_frame_states"],
                "has_init_states": fi["has_init_states"],
                "has_model_xml_or_env_args": fi["has_model_xml_or_env_args"],
                "exact_frame_reset_feasible": fi["exact_frame_reset_feasible"],
                "replay_feasible": fi["replay_feasible"],
                "rerender_feasible": fi["rerender_feasible"],
                "per_frame_state_fields": fi["per_frame_state_fields"],
                "init_state_fields": fi["init_state_fields"],
                "asset_fields": fi["asset_fields"][:5],
                "action_shape": fi["action_shape"],
                "state_shape": fi["state_shape"],
                "error": fi["error"],
            }
            for fi in entry.get("file_inspections", [])
        ]
        compact_inventory.append(compact)

    json_payload = {
        "generated_at": generated_at,
        "decision": decision,
        "suites": compact_inventory,
    }
    write_json(output_dir / "libero_pair_data_suite_inventory.json", json_payload)
    print(f"Wrote {output_dir / 'libero_pair_data_suite_inventory.json'}")

    md_text = build_markdown(inventory, decision, generated_at)
    (output_dir / "libero_pair_data_libero_hdf5_audit.md").write_text(md_text, encoding="utf-8")
    print(f"Wrote {output_dir / 'libero_pair_data_libero_hdf5_audit.md'}")


if __name__ == "__main__":
    main()
