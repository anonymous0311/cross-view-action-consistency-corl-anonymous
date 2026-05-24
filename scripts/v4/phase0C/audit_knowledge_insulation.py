"""Audit whether Phase 0B training changed the pi0.5 VLM backbone parameters."""
# ruff: noqa: E402, I001, PERF401

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "openpi/src"))

import flax.traverse_util as traverse_util
import jax.numpy as jnp
import numpy as np
import tqdm

from canonical.eval.libero_plus_eval import restore_params_portable


DEFAULT_BASE_PARAMS = pathlib.Path("gs://openpi-assets/checkpoints/pi05_base/params")
DEFAULT_FINETUNED_PARAMS = (
    REPO_ROOT / "checkpoints/pi05_v4_pair_fm_only/phase0b_fm_only_10k_s42/9999/params"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/v4_routing_audit/knowledge_insulation"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _path_str(path: tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def _has_action_expert_suffix(path: tuple[Any, ...]) -> bool:
    return any(isinstance(part, str) and part.endswith("_1") for part in path)


def _group_for_path(path: tuple[Any, ...]) -> str:
    path_s = _path_str(path)
    if path_s.startswith("PaliGemma/img/"):
        return "vlm_image_encoder"
    if path_s.startswith("PaliGemma/llm/") and not _has_action_expert_suffix(path):
        return "vlm_llm_first_expert"
    if path_s.startswith("PaliGemma/llm/") and _has_action_expert_suffix(path):
        return "action_expert_llm"
    if path_s.startswith(("action_in_proj/", "action_out_proj/", "time_mlp_in/", "time_mlp_out/")):
        return "action_path_projection"
    return "other"


def _compare_group(
    base_flat: dict[tuple[Any, ...], Any],
    tuned_flat: dict[tuple[Any, ...], Any],
    group: str,
) -> dict[str, Any]:
    keys = sorted(
        key
        for key in base_flat
        if _group_for_path(key) == group and key in tuned_flat and hasattr(base_flat[key], "shape")
    )
    sq_diff = 0.0
    sq_base = 0.0
    max_abs = 0.0
    num_params = 0
    num_tensors_changed = 0
    changed_examples: list[str] = []
    for key in tqdm.tqdm(keys, desc=f"[ki] {group}", unit="tensor", dynamic_ncols=True, leave=False):
        base = np.asarray(base_flat[key], dtype=np.float32)
        tuned = np.asarray(tuned_flat[key], dtype=np.float32)
        if base.shape != tuned.shape:
            continue
        diff = tuned - base
        tensor_sq_diff = float(np.sum(np.square(diff)))
        tensor_max = float(np.max(np.abs(diff))) if diff.size else 0.0
        sq_diff += tensor_sq_diff
        sq_base += float(np.sum(np.square(base)))
        max_abs = max(max_abs, tensor_max)
        num_params += int(diff.size)
        if tensor_max > 1e-7:
            num_tensors_changed += 1
            if len(changed_examples) < 10:
                changed_examples.append(_path_str(key))
    l2 = float(np.sqrt(sq_diff))
    base_l2 = float(np.sqrt(sq_base))
    return {
        "group": group,
        "num_tensors": len(keys),
        "num_tensors_changed_gt_1e-7": num_tensors_changed,
        "num_params": num_params,
        "l2_diff": l2,
        "base_l2": base_l2,
        "relative_l2": float(l2 / max(base_l2, 1e-12)),
        "max_abs_diff": max_abs,
        "changed_examples": changed_examples,
    }


def _write_report(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines = [
        "# Knowledge Insulation Audit",
        "",
        f"knowledge_insulation: **{result['summary']['knowledge_insulation']}**",
        "",
        "## Config",
        "",
        f"- base params: `{result['config']['base_params']}`",
        f"- finetuned params: `{result['config']['finetuned_params']}`",
        "",
        "## Parameter Distance",
        "",
        "| group | tensors | changed tensors | params | relative L2 | max abs diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["groups"]:
        lines.append(
            f"| {row['group']} | {row['num_tensors']} | {row['num_tensors_changed_gt_1e-7']} | "
            f"{row['num_params']} | {row['relative_l2']:.6e} | {row['max_abs_diff']:.6e} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "If the VLM groups are near zero while action-path groups move, Phase 0C-early on "
        "`phase0b_fm_only` mostly measures inherent pi0.5/VLM routing. If VLM groups move "
        "materially, the early audit should be interpreted as FM-only-finetuned routing.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=pathlib.Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--finetuned-params", type=pathlib.Path, default=DEFAULT_FINETUNED_PARAMS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-canonical-report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = restore_params_portable(args.base_params, restore_type=np.ndarray, dtype=jnp.bfloat16)
    tuned = restore_params_portable(args.finetuned_params, restore_type=np.ndarray, dtype=jnp.bfloat16)
    base_flat = traverse_util.flatten_dict(base)
    tuned_flat = traverse_util.flatten_dict(tuned)

    groups = [
        _compare_group(base_flat, tuned_flat, group)
        for group in tqdm.tqdm(
            (
                "vlm_image_encoder",
                "vlm_llm_first_expert",
                "action_expert_llm",
                "action_path_projection",
                "other",
            ),
            desc="[ki] groups",
            unit="group",
            dynamic_ncols=True,
        )
    ]
    vlm = [row for row in groups if row["group"].startswith("vlm_")]
    vlm_max_relative = max(row["relative_l2"] for row in vlm)
    vlm_max_abs = max(row["max_abs_diff"] for row in vlm)
    knowledge_insulation = "enabled_or_effectively_unchanged" if vlm_max_relative < 1e-6 and vlm_max_abs < 1e-6 else "vlm_changed"
    result = {
        "config": {
            "base_params": str(args.base_params),
            "finetuned_params": str(args.finetuned_params),
            "vlm_unchanged_threshold": {"relative_l2": 1e-6, "max_abs_diff": 1e-6},
        },
        "summary": {
            "knowledge_insulation": knowledge_insulation,
            "vlm_max_relative_l2": vlm_max_relative,
            "vlm_max_abs_diff": vlm_max_abs,
        },
        "groups": groups,
    }
    (args.output_dir / "knowledge_insulation_audit.json").write_text(json.dumps(result, indent=2, default=_json_default))
    _write_report(args.output_dir / "knowledge_insulation_audit.md", result)
    if args.write_canonical_report or args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        canonical = REPO_ROOT / "results/v4_routing_audit/knowledge_insulation_audit.md"
        canonical.write_text((args.output_dir / "knowledge_insulation_audit.md").read_text())
    print(f"[ki] wrote {args.output_dir / 'knowledge_insulation_audit.md'}")


if __name__ == "__main__":
    main()
