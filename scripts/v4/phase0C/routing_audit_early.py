"""Phase 0C-early routing audit for vanilla pi0.5 on LIBERO multiview pairs.

The audit measures whether contextualized language-prefix hidden states carry
camera/view information before any cross-view consistency training is applied.
It is intentionally offline: no rollout, no parameter updates.
"""
# ruff: noqa: E402, I001, SLF001

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "openpi/src"))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

from canonical.eval.libero_plus_eval import restore_params_portable
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/pi05_v4_pair_fm_only/phase0b_fm_only_10k_s42/9999"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/v4_routing_audit/early/fm_only_10k_s42"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _group_by_demo(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["task_name"], int(row["demo_id"]))].append(row)
    return dict(sorted(groups.items()))


def _build_sample_metadata(
    train_manifest: pathlib.Path,
    val_manifest: pathlib.Path,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Reconstruct LeRobot global sample indices from the exporter ordering."""
    train_groups = _group_by_demo(_read_jsonl(train_manifest))
    val_groups = _group_by_demo(_read_jsonl(val_manifest))
    sample_to_meta: dict[int, dict[str, Any]] = {}
    global_index = 0
    all_groups = list(train_groups.items()) + list(val_groups.items())
    for episode_index, (_key, rows) in enumerate(all_groups):
        for frame_index, row in enumerate(sorted(rows, key=lambda r: int(r["timestep"]))):
            sample_to_meta[global_index] = {
                "sample_index": global_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "pair_id": row["pair_id"],
                "suite": row["suite"],
                "task_name": row["task_name"],
                "demo_id": int(row["demo_id"]),
                "timestep": int(row["timestep"]),
                "language": row["language"],
                "camera_category": row["camera_category"],
                "camera_params_b": row.get("camera_params_b", {}),
            }
            global_index += 1
    return sample_to_meta, len(train_groups)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    denom = np.linalg.norm(a32, axis=-1) * np.linalg.norm(b32, axis=-1)
    denom = np.maximum(denom, 1e-8)
    return 1.0 - np.sum(a32 * b32, axis=-1) / denom


def _normalized_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    return np.linalg.norm(a32 - b32, axis=-1) / np.maximum(np.linalg.norm(a32, axis=-1), 1e-8)


def _unrelated_indices(task_indices: np.ndarray) -> np.ndarray:
    perm = np.roll(np.arange(task_indices.shape[0]), 1)
    for task in np.unique(task_indices):
        idx = np.flatnonzero(task_indices == task)
        if idx.shape[0] >= 2:
            perm[idx] = np.roll(idx, 1)
    return perm


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _std_or_nan(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if abs(denom) > 1e-8 else float("nan")


def _normalized_bypass(same: float, identity: float, unrelated: float) -> float:
    denom = unrelated - identity
    if abs(denom) <= 1e-8:
        return float("nan")
    return float(np.clip((same - identity) / denom, 0.0, 1.5))


def _distance_row(
    *,
    layer: str,
    group: str,
    token_path: str,
    same: np.ndarray,
    identity: np.ndarray,
    unrelated: np.ndarray,
    vision_zero_same: np.ndarray | None = None,
) -> dict[str, Any]:
    same_mean = _mean_or_nan(same)
    identity_mean = _mean_or_nan(identity)
    unrelated_mean = _mean_or_nan(unrelated)
    row = {
        "layer": layer,
        "group": group,
        "token_path": token_path,
        "same_state_diff_view_mean": same_mean,
        "same_state_diff_view_std": _std_or_nan(same),
        "identity_floor_mean": identity_mean,
        "identity_floor_std": _std_or_nan(identity),
        "same_language_unrelated_mean": unrelated_mean,
        "same_language_unrelated_std": _std_or_nan(unrelated),
        "raw_same_over_unrelated": _safe_ratio(same_mean, unrelated_mean),
        "normalized_bypass": _normalized_bypass(same_mean, identity_mean, unrelated_mean),
        "n": int(same.size),
    }
    if vision_zero_same is not None:
        row["vision_zero_same_state_diff_view_mean"] = _mean_or_nan(vision_zero_same)
        row["vision_zero_same_state_diff_view_std"] = _std_or_nan(vision_zero_same)
    return row


def _monotonicity(values: list[float]) -> dict[str, Any]:
    finite = np.array([v for v in values if np.isfinite(v)], dtype=np.float32)
    if finite.size < 2:
        return {"is_monotone_non_decreasing": False, "positive_step_fraction": float("nan")}
    diffs = np.diff(finite)
    return {
        "is_monotone_non_decreasing": bool(np.all(diffs >= -1e-4)),
        "positive_step_fraction": float(np.mean(diffs >= -1e-4)),
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    recalls = []
    for cls in range(num_classes):
        mask = y_true == cls
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _task_split(task_indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tasks = np.array(sorted(np.unique(task_indices)))
    rng.shuffle(tasks)
    n_test = max(1, round(0.2 * len(tasks)))
    test_tasks = {int(x) for x in tasks[:n_test]}
    test = np.array([int(t) in test_tasks for t in task_indices], dtype=bool)
    train = ~test
    return train, test


def _linear_probe(
    x: np.ndarray,
    y: np.ndarray,
    task_indices: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    steps: int,
    lr: float,
    device_name: str,
) -> dict[str, float]:
    train_mask, test_mask = _task_split(task_indices, seed)
    if len(np.unique(y[train_mask])) < num_classes or len(np.unique(y[test_mask])) < num_classes:
        # Fallback to deterministic sample split if task-heldout lacks a class.
        rng = np.random.default_rng(seed)
        order = rng.permutation(x.shape[0])
        n_test = max(num_classes, round(0.2 * x.shape[0]))
        test_mask = np.zeros(x.shape[0], dtype=bool)
        test_mask[order[:n_test]] = True
        train_mask = ~test_mask

    x_train = x[train_mask].astype(np.float32)
    x_test = x[test_mask].astype(np.float32)
    y_train = y[train_mask].astype(np.int64)
    y_test = y[test_mask].astype(np.int64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    x_train = (x_train - mean) / np.maximum(std, 1e-5)
    x_test = (x_test - mean) / np.maximum(std, 1e-5)

    torch.manual_seed(seed)
    device = torch.device(device_name)
    model = torch.nn.Linear(x_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    batch_size = min(1024, x_train.shape[0])
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for _ in range(steps):
        if x_train.shape[0] > batch_size:
            idx = torch.randint(0, x_train.shape[0], (batch_size,), generator=gen, device=device)
            xb = xt[idx]
            yb = yt[idx]
        else:
            xb = xt
            yb = yt
        loss = torch.nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_train = model(xt).argmax(dim=-1).detach().cpu().numpy()
        pred_test = model(torch.from_numpy(x_test).to(device)).argmax(dim=-1).detach().cpu().numpy()

    return {
        "train_balanced_acc": _balanced_accuracy(y_train, pred_train, num_classes),
        "test_balanced_acc": _balanced_accuracy(y_test, pred_test, num_classes),
        "train_n": int(x_train.shape[0]),
        "test_n": int(x_test.shape[0]),
    }


def _late_distance_label(
    distance_rows: list[dict[str, Any]],
    late_layers: set[str],
    language_depth_trend: dict[str, Any],
) -> str:
    late = [
        row
        for row in distance_rows
        if row["token_path"] == "language" and row["group"] == "all" and row["layer"] in late_layers
    ]
    if not late:
        return "unclear"
    values = np.array([row["normalized_bypass"] for row in late], dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "unclear"
    if float(np.mean(finite)) > 0.5:
        if language_depth_trend["is_monotone_non_decreasing"]:
            return "strong_monotone"
        return "strong_layer_specific"
    if float(np.mean(finite)) < 0.2:
        return "weak"
    if not language_depth_trend["is_monotone_non_decreasing"]:
        return "unclear_layer_specific"
    return "unclear"


def _probe_labels(probe_rows: list[dict[str, Any]], late_layers: set[str]) -> dict[str, Any]:
    labels: dict[str, Any] = {
        "perturbation_detection": "not_run",
        "perturbation_typing": "not_run",
        "late_binary_balanced_acc_mean": float("nan"),
        "late_four_way_balanced_acc_mean": float("nan"),
    }
    if not probe_rows:
        return labels
    binary = [
        row["test_balanced_acc"]
        for row in probe_rows
        if row["probe"] == "binary_nominal_vs_perturbed" and row["layer"] in late_layers
    ]
    four_way = [
        row["test_balanced_acc"]
        for row in probe_rows
        if row["probe"] == "four_way_nominal_c1_c2_c3" and row["layer"] in late_layers
    ]
    if binary:
        binary_mean = float(np.mean(binary))
        labels["late_binary_balanced_acc_mean"] = binary_mean
        labels["perturbation_detection"] = "high" if binary_mean >= 0.80 else "low" if binary_mean <= 0.60 else "medium"
    if four_way:
        four_way_mean = float(np.mean(four_way))
        labels["late_four_way_balanced_acc_mean"] = four_way_mean
        labels["perturbation_typing"] = "high" if four_way_mean >= 0.55 else "low" if four_way_mean <= 0.35 else "medium"
    return labels


def _overall_label(distance_label: str, probe_labels: dict[str, Any]) -> str:
    typing = probe_labels["perturbation_typing"]
    detection = probe_labels["perturbation_detection"]
    distance_is_strong = distance_label.startswith("strong")
    if distance_is_strong and typing == "high":
        return "severe"
    if distance_is_strong or typing == "high" or detection == "high":
        return "mild"
    if distance_label == "weak" and detection in {"low", "not_run"} and typing in {"low", "not_run"}:
        return "weak"
    return "unclear"


def _write_report(
    output_path: pathlib.Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    distance_rows: list[dict[str, Any]],
    breakdown_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> None:
    aggregate_lang = [row for row in distance_rows if row["token_path"] == "language" and row["group"] == "all"]
    aggregate_image = [row for row in distance_rows if row["token_path"] == "image" and row["group"] == "all"]
    late_layers = ", ".join(summary["late_layers"])
    lines = [
        "# Phase 0C-early Routing Audit",
        "",
        f"overall bypass: **{summary['overall_bypass']}**",
        f"distance bypass: **{summary['distance_bypass']}**",
        f"perturbation detection: **{summary['perturbation_detection']}**",
        f"perturbation typing: **{summary['perturbation_typing']}**",
        "",
        "## Config",
        "",
        f"- checkpoint: `{config['checkpoint']}`",
        f"- config_name: `{config['config_name']}`",
        f"- max_pairs: `{config['max_pairs']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- layer_indices: `{config['layer_indices']}`",
        "",
        "## Dataset",
        "",
        f"- pairs analyzed: `{summary['num_pairs']}`",
        f"- suites: `{summary['suite_counts']}`",
        f"- camera categories: `{summary['camera_category_counts']}`",
        f"- breakdown rows: `{len(breakdown_rows)}`",
        "",
        "## Hidden Distances",
        "",
        f"Late-layer decision set: `{late_layers}`. `normalized_bypass = "
        "(same_state_diff_view - identity_floor) / (same_language_unrelated - identity_floor)`.",
        "",
        "| layer | lang same | lang identity | lang unrelated | lang normalized bypass | vision-zero same | image normalized bypass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    image_by_layer = {row["layer"]: row for row in aggregate_image}
    for row in aggregate_lang:
        image_row = image_by_layer[row["layer"]]
        lines.append(
            f"| {row['layer']} | {row['same_state_diff_view_mean']:.4f} | {row['identity_floor_mean']:.4f} | "
            f"{row['same_language_unrelated_mean']:.4f} | {row['normalized_bypass']:.3f} | "
            f"{row.get('vision_zero_same_state_diff_view_mean', float('nan')):.4f} | "
            f"{image_row['normalized_bypass']:.3f} |"
        )
    lines += [
        "",
        "## Depth Trend",
        "",
        f"- language normalized-bypass monotone non-decreasing: `{summary['language_depth_trend']['is_monotone_non_decreasing']}`",
        f"- positive step fraction: `{summary['language_depth_trend']['positive_step_fraction']:.3f}`",
        "- non-monotone strong late-layer effects are labeled `strong_layer_specific`, not smooth accumulating bypass.",
        "",
        "## Camera Predictability",
        "",
        "| layer | probe | test balanced acc | train balanced acc | test n |",
        "|---|---|---:|---:|---:|",
    ]
    for row in probe_rows:
        lines.append(
            "| {layer} | {probe} | {test_balanced_acc:.4f} | {train_balanced_acc:.4f} | {test_n} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Breakdown Files",
        "",
        "- `hidden_distance_by_layer.csv`: aggregate language/image normalized bypass by layer.",
        "- `hidden_distance_breakdown.csv`: per-suite, per-category, and suite-by-category language breakdown.",
        "- `probe_by_layer.csv`: binary detection and four-way typing probe results.",
        "- `paper_artifacts.json`: figure-ready data and labels.",
        "",
        "## Interpretation",
        "",
        "Binary probe accuracy measures perturbation detection; four-way probe accuracy measures "
        "C1/C2/C3 perturbation typing. Four-way camera typing is stronger evidence of fine camera leakage "
        "than binary detection alone. This audit measures sensitivity and camera predictability in prefix "
        "hidden states; it is evidence about bypass risk, not causal mediation by itself. Phase 0C-late "
        "must run hidden replacement on the selected Phase 0B model before choosing an architecture-heavy branch.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default="pi05_v4_pair_fm_only")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-train", type=pathlib.Path, default=REPO_ROOT / "results/v4_pair_audit/libero_pair_manifest_train.jsonl")
    parser.add_argument("--manifest-val", type=pathlib.Path, default=REPO_ROOT / "results/v4_pair_audit/libero_pair_manifest_val.jsonl")
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--layer-indices", default="0,4,8,12,16,17")
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--write-canonical-report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = tuple(int(x) for x in args.layer_indices.split(",") if x.strip())
    sample_to_meta, val_episode_start = _build_sample_metadata(args.manifest_train, args.manifest_val)
    val_episodes = range(val_episode_start, val_episode_start + 200)

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=dataclasses.replace(cfg.data, dataset_episodes=tuple(val_episodes)),
    )
    loader = _data_loader.create_data_loader(
        cfg,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        shuffle=not args.no_shuffle,
        num_batches=None,
        skip_norm_stats=False,
    )

    params = restore_params_portable(args.checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
    model = cfg.model.load(params, remove_extra_params=False)
    graphdef, model_state = nnx.split(model)

    @jax.jit
    def _extract_batch(state, obs):
        model_for_batch = nnx.merge(graphdef, state)
        flat_obs = model_for_batch._flatten_pair_observation(obs)
        captures = model_for_batch.capture_prefix_intermediates(flat_obs, layer_indices=layer_indices)
        captures_zero = model_for_batch.capture_prefix_intermediates(
            flat_obs,
            layer_indices=layer_indices,
            zero_image_tokens=True,
        )
        hidden = jnp.concatenate(
            [
                captures["raw_prefix"][None, ...],
                captures["layer_prefix_hidden"],
                captures["prefix_out"][None, ...],
            ],
            axis=0,
        )
        hidden_zero = jnp.concatenate(
            [
                captures_zero["raw_prefix"][None, ...],
                captures_zero["layer_prefix_hidden"],
                captures_zero["prefix_out"][None, ...],
            ],
            axis=0,
        )
        prefix_mask = captures["prefix_mask"]
        n_image_tokens = captures["n_image_tokens"]
        image_mask = prefix_mask[:, :n_image_tokens]
        lang_mask = prefix_mask[:, n_image_tokens:]

        def _masked_mean(x, mask):
            weights = mask[None, :, :, None].astype(x.dtype)
            denom = jnp.maximum(jnp.sum(weights, axis=2), 1.0)
            return jnp.sum(x * weights, axis=2) / denom

        return {
            "lang": _masked_mean(hidden[:, :, n_image_tokens:, :], lang_mask),
            "image": _masked_mean(hidden[:, :, :n_image_tokens, :], image_mask),
            "lang_zero": _masked_mean(hidden_zero[:, :, n_image_tokens:, :], lang_mask),
        }

    layer_names = ["raw_embed", *[f"block{idx:02d}" for idx in layer_indices], "final_norm"]
    lang_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []
    lang_zero_batches: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    task_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    processed = 0
    total_batches = max(1, int(np.ceil(args.max_pairs / args.batch_size)))

    for obs, _actions in tqdm.tqdm(
        loader,
        total=total_batches,
        desc="[early] batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        extracted = _extract_batch(model_state, obs)
        batch_size = int(obs.state.shape[0])
        take = min(batch_size, args.max_pairs - processed)
        lang = np.asarray(extracted["lang"]).reshape(len(layer_names), batch_size, 2, -1)
        image = np.asarray(extracted["image"]).reshape(len(layer_names), batch_size, 2, -1)
        lang_zero = np.asarray(extracted["lang_zero"]).reshape(len(layer_names), batch_size, 2, -1)
        lang_batches.append(np.transpose(lang[:, :take], (1, 2, 0, 3)))
        image_batches.append(np.transpose(image[:, :take], (1, 2, 0, 3)))
        lang_zero_batches.append(np.transpose(lang_zero[:, :take], (1, 2, 0, 3)))
        sample_indices.append(np.asarray(obs.sample_index[:take, 0]))
        task_indices.append(np.asarray(obs.task_index[:take, 0]))
        episode_indices.append(np.asarray(obs.episode_index[:take, 0]))
        processed += take
        tqdm.tqdm.write(f"[early] processed {processed}/{args.max_pairs}")
        if processed >= args.max_pairs:
            break

    lang_features = np.concatenate(lang_batches, axis=0)
    image_features = np.concatenate(image_batches, axis=0)
    lang_zero_features = np.concatenate(lang_zero_batches, axis=0)
    sample_index = np.concatenate(sample_indices, axis=0).astype(np.int64)
    task_index = np.concatenate(task_indices, axis=0).astype(np.int64)
    episode_index = np.concatenate(episode_indices, axis=0).astype(np.int64)
    metadata_rows = [sample_to_meta[int(idx)] for idx in sample_index]
    camera_category = np.array([row["camera_category"] for row in metadata_rows])
    suite = np.array([row["suite"] for row in metadata_rows])
    unrelated = _unrelated_indices(task_index)

    distance_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for layer_id, layer_name in enumerate(
        tqdm.tqdm(layer_names, desc="[early] distances", unit="layer", dynamic_ncols=True)
    ):
        lang_a = lang_features[:, 0, layer_id]
        lang_b = lang_features[:, 1, layer_id]
        lang_zero_a = lang_zero_features[:, 0, layer_id]
        lang_zero_b = lang_zero_features[:, 1, layer_id]
        img_a = image_features[:, 0, layer_id]
        img_b = image_features[:, 1, layer_id]
        lang_same = _cosine_distance(lang_a, lang_b)
        lang_identity = _cosine_distance(lang_a, lang_a.copy())
        lang_unrelated = _cosine_distance(lang_a, lang_b[unrelated])
        lang_zero_same = _cosine_distance(lang_zero_a, lang_zero_b)
        image_same = _cosine_distance(img_a, img_b)
        image_identity = _cosine_distance(img_a, img_a.copy())
        image_unrelated = _cosine_distance(img_a, img_b[unrelated])
        distance_rows.append(
            _distance_row(
                layer=layer_name,
                group="all",
                token_path="language",
                same=lang_same,
                identity=lang_identity,
                unrelated=lang_unrelated,
                vision_zero_same=lang_zero_same,
            )
        )
        distance_rows[-1]["same_state_diff_view_norm_l2_mean"] = float(np.mean(_normalized_l2(lang_a, lang_b)))
        distance_rows.append(
            _distance_row(
                layer=layer_name,
                group="all",
                token_path="image",
                same=image_same,
                identity=image_identity,
                unrelated=image_unrelated,
            )
        )
        for group_name, mask in [
            *[(f"suite={name}", suite == name) for name in sorted(np.unique(suite))],
            *[(f"camera_category={name}", camera_category == name) for name in sorted(np.unique(camera_category))],
            *[
                (f"suite={suite_name}|camera_category={cat}", (suite == suite_name) & (camera_category == cat))
                for suite_name in sorted(np.unique(suite))
                for cat in sorted(np.unique(camera_category))
            ],
        ]:
            if not np.any(mask):
                continue
            breakdown_rows.append(
                _distance_row(
                    layer=layer_name,
                    group=group_name,
                    token_path="language",
                    same=lang_same[mask],
                    identity=lang_identity[mask],
                    unrelated=lang_unrelated[mask],
                    vision_zero_same=lang_zero_same[mask],
                )
            )

    category_to_label = {"C1": 1, "C2": 2, "C3": 3}
    binary_labels = np.concatenate([np.zeros(len(camera_category), dtype=np.int64), np.ones(len(camera_category), dtype=np.int64)])
    four_way_labels = np.concatenate(
        [
            np.zeros(len(camera_category), dtype=np.int64),
            np.array([category_to_label.get(str(cat), 0) for cat in camera_category], dtype=np.int64),
        ]
    )
    probe_task_index = np.concatenate([task_index, task_index], axis=0)
    probe_rows: list[dict[str, Any]] = []
    if not args.skip_probes:
        for layer_id, layer_name in enumerate(
            tqdm.tqdm(layer_names, desc="[early] probes", unit="layer", dynamic_ncols=True)
        ):
            x = lang_features[:, :, layer_id, :].reshape(len(camera_category) * 2, -1)
            for probe_name, labels, num_classes in (
                ("binary_nominal_vs_perturbed", binary_labels, 2),
                ("four_way_nominal_c1_c2_c3", four_way_labels, 4),
            ):
                probe = _linear_probe(
                    x,
                    labels,
                    probe_task_index,
                    num_classes=num_classes,
                    seed=args.seed + layer_id,
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                    device_name=args.probe_device,
                )
                probe_rows.append({"layer": layer_name, "probe": probe_name, **probe})
                print(f"[early] probe {layer_name} {probe_name}: {probe['test_balanced_acc']:.3f}", flush=True)

    requested_late_layers = {"block12", "block16", "block17"}
    late_layers = sorted(requested_late_layers & set(layer_names))
    if not late_layers:
        late_layers = [name for name in layer_names if name.startswith("block")][-3:]
    late_layer_set = set(late_layers)
    probe_label_summary = _probe_labels(probe_rows, late_layer_set)
    language_depth_values = [
        row["normalized_bypass"]
        for row in distance_rows
        if row["token_path"] == "language" and row["group"] == "all" and row["layer"].startswith("block")
    ]
    language_depth_trend = _monotonicity(language_depth_values)
    distance_bypass = _late_distance_label(distance_rows, late_layer_set, language_depth_trend)
    summary = {
        "num_pairs": len(sample_index),
        "suite_counts": {str(k): int(v) for k, v in zip(*np.unique(suite, return_counts=True), strict=True)},
        "camera_category_counts": {
            str(k): int(v) for k, v in zip(*np.unique(camera_category, return_counts=True), strict=True)
        },
        "late_layers": late_layers,
        "distance_bypass": distance_bypass,
        **probe_label_summary,
        "overall_bypass": _overall_label(distance_bypass, probe_label_summary),
        "language_depth_trend": language_depth_trend,
        "state_pathway": "not_available_in_pi05_prefix",
    }
    config = {
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "layer_indices": layer_indices,
        "val_episode_start": val_episode_start,
        "shuffle": not args.no_shuffle,
        "probe_device": args.probe_device,
        "normalized_bypass_thresholds": {"strong": 0.5, "weak": 0.2},
    }
    result = {
        "config": config,
        "summary": summary,
        "distance_by_layer": distance_rows,
        "distance_breakdown": breakdown_rows,
        "probes": probe_rows,
        "paper_deliverables": {
            "figure_a": {
                "name": "depth_by_token_path_bypass_heatmap",
                "source": "hidden_distance_by_layer.csv",
                "x": "layer",
                "y": "token_path",
                "color": "normalized_bypass",
                "note": "pi0.5 has image and language prefix tokens; state tokens are not present in the prefix.",
            },
            "figure_b": {
                "name": "binary_detection_vs_four_way_typing_by_layer",
                "source": "probe_by_layer.csv",
                "x": "layer",
                "y": "test_balanced_acc",
                "series": "probe",
            },
            "table": {
                "name": "bypass_label_by_pathway",
                "language": summary["overall_bypass"],
                "image": "reference_pathway_reported_as_distance_not_bypass_label",
                "state": summary["state_pathway"],
            },
        },
    }
    (args.output_dir / "routing_audit_early.json").write_text(json.dumps(result, indent=2, default=_json_default))
    _write_csv(args.output_dir / "hidden_distance_by_layer.csv", distance_rows)
    _write_csv(args.output_dir / "hidden_distance_breakdown.csv", breakdown_rows)
    _write_csv(args.output_dir / "probe_by_layer.csv", probe_rows)
    (args.output_dir / "paper_artifacts.json").write_text(
        json.dumps(result["paper_deliverables"], indent=2, default=_json_default)
    )
    _write_report(
        args.output_dir / "routing_audit_early.md",
        config=config,
        summary=summary,
        distance_rows=distance_rows,
        breakdown_rows=breakdown_rows,
        probe_rows=probe_rows,
    )
    if args.write_canonical_report or args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        canonical_report = REPO_ROOT / "results/v4_routing_audit/routing_audit_early.md"
        canonical_report.parent.mkdir(parents=True, exist_ok=True)
        canonical_report.write_text((args.output_dir / "routing_audit_early.md").read_text())

    if args.save_features:
        np.savez_compressed(
            args.output_dir / "routing_audit_early_features.npz",
            lang_features=lang_features.astype(np.float16),
            image_features=image_features.astype(np.float16),
            lang_zero_features=lang_zero_features.astype(np.float16),
            sample_index=sample_index,
            task_index=task_index,
            episode_index=episode_index,
            camera_category=camera_category,
            suite=suite,
            layer_names=np.array(layer_names),
        )
    print(f"[early] wrote {args.output_dir / 'routing_audit_early.md'}", flush=True)


if __name__ == "__main__":
    main()
