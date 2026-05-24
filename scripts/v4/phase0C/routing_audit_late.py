"""Phase 0C-late offline mediation audit for trained Phase 0B checkpoints."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "openpi/src"))

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from canonical.eval.libero_plus_eval import restore_params_portable
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pi05_v4_pair_cv010/phase0b_cv010_10k_s42/9999"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/v4_routing_audit/late/cv010_10k_s42"


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
                "camera_category": row["camera_category"],
            }
            global_index += 1
    return sample_to_meta, len(train_groups)


def _first_axis_take(x, index):
    return None if x is None else x[index]


def _select_pair_slot(obs: _model.Observation, slot: int) -> _model.Observation:
    return dataclasses.replace(
        obs,
        images={key: value[:, slot] for key, value in obs.images.items()},
        image_masks={key: value[:, slot] for key, value in obs.image_masks.items()},
        state=obs.state[:, slot],
        tokenized_prompt=_first_axis_take(obs.tokenized_prompt, (slice(None), slot)),
        tokenized_prompt_mask=_first_axis_take(obs.tokenized_prompt_mask, (slice(None), slot)),
        token_ar_mask=_first_axis_take(obs.token_ar_mask, (slice(None), slot)),
        token_loss_mask=_first_axis_take(obs.token_loss_mask, (slice(None), slot)),
        canonical_tokens=_first_axis_take(obs.canonical_tokens, (slice(None), slot)),
        canonical_tokens_neg=_first_axis_take(obs.canonical_tokens_neg, (slice(None), slot)),
        canonical_tokens_mean=_first_axis_take(obs.canonical_tokens_mean, (slice(None), slot)),
        task_index=_first_axis_take(obs.task_index, (slice(None), slot)),
        episode_index=_first_axis_take(obs.episode_index, (slice(None), slot)),
        sample_index=_first_axis_take(obs.sample_index, (slice(None), slot)),
        camera_bin_id=_first_axis_take(obs.camera_bin_id, (slice(None), slot)),
    )


def _permute_batch(obs: _model.Observation, perm: jax.Array) -> _model.Observation:
    return dataclasses.replace(
        obs,
        images={key: value[perm] for key, value in obs.images.items()},
        image_masks={key: value[perm] for key, value in obs.image_masks.items()},
        state=obs.state[perm],
        tokenized_prompt=_first_axis_take(obs.tokenized_prompt, perm),
        tokenized_prompt_mask=_first_axis_take(obs.tokenized_prompt_mask, perm),
        token_ar_mask=_first_axis_take(obs.token_ar_mask, perm),
        token_loss_mask=_first_axis_take(obs.token_loss_mask, perm),
        canonical_tokens=_first_axis_take(obs.canonical_tokens, perm),
        canonical_tokens_neg=_first_axis_take(obs.canonical_tokens_neg, perm),
        canonical_tokens_mean=_first_axis_take(obs.canonical_tokens_mean, perm),
        task_index=_first_axis_take(obs.task_index, perm),
        episode_index=_first_axis_take(obs.episode_index, perm),
        sample_index=_first_axis_take(obs.sample_index, perm),
        camera_bin_id=_first_axis_take(obs.camera_bin_id, perm),
    )


def _same_task_perm(task_index: np.ndarray) -> np.ndarray:
    perm = np.arange(task_index.shape[0])
    for task in np.unique(task_index):
        idx = np.flatnonzero(task_index == task)
        if idx.shape[0] >= 2:
            perm[idx] = np.roll(idx, 1)
    if np.any(perm == np.arange(task_index.shape[0])) and task_index.shape[0] > 1:
        perm = np.roll(np.arange(task_index.shape[0]), 1)
    return perm.astype(np.int32)


def _aggregate(values: list[np.ndarray]) -> dict[str, float]:
    arr = np.concatenate(values, axis=0).astype(np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _classify(summary: dict[str, Any]) -> str:
    lang = summary["modes"]["replace_lang_same_state"]["flow_delta"]["mean"]
    image = summary["modes"]["replace_image_same_state"]["flow_delta"]["mean"]
    both = summary["modes"]["replace_both_same_state"]["flow_delta"]["mean"]
    if lang >= 0.75 * max(image, 1e-8) and lang >= 0.35 * max(both, 1e-8):
        return "strong"
    if lang <= 0.35 * max(image, 1e-8):
        return "weak"
    return "unclear"


def _write_report(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines = [
        "# Phase 0C-late Routing Mediation Audit",
        "",
        f"bypass: **{result['summary']['bypass']}**",
        "",
        "## Config",
        "",
        f"- checkpoint: `{result['config']['checkpoint']}`",
        f"- config_name: `{result['config']['config_name']}`",
        f"- pairs analyzed: `{result['summary']['num_pairs']}`",
        f"- kv_layer_scope: `{result['config']['kv_layer_scope']}`",
        "",
        "## Offline Flow Mediation",
        "",
        "| mode | flow delta mean | flow delta p90 | action delta mean |",
        "|---|---:|---:|---:|",
    ]
    for mode, row in result["summary"]["modes"].items():
        lines.append(
            f"| {mode} | {row['flow_delta']['mean']:.5f} | {row['flow_delta']['p90']:.5f} | "
            f"{row['action_delta']['mean']:.5f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is an offline mediation audit with fixed action noise/time for intervention modes. "
        "The default KV intervention replaces the selected token columns across all prefix-layer "
        "K/V caches, treating the prefix information stream as one mediator. "
        "`same_view_diff_noise_time` is a scale control for flow variation under a different "
        "`x_t,t,eps`; it is not a hidden-route intervention. Rollout "
        "intervention can be added for modes with strong or unclear offline effects.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default="pi05_v4_pair_cv010")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-train", type=pathlib.Path, default=REPO_ROOT / "results/v4_pair_audit/libero_pair_manifest_train.jsonl")
    parser.add_argument("--manifest-val", type=pathlib.Path, default=REPO_ROOT / "results/v4_pair_audit/libero_pair_manifest_val.jsonl")
    parser.add_argument("--max-pairs", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--write-canonical-report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    def _audit_batch(state, obs, actions, rng, unrelated_perm):
        model_for_batch = nnx.merge(graphdef, state)
        obs_a = _select_pair_slot(obs, 0)
        obs_b = _select_pair_slot(obs, 1)
        obs_u = _permute_batch(obs_b, unrelated_perm)
        obs_a = _model.preprocess_observation(None, obs_a, train=False, image_keys=tuple(obs_a.images))
        obs_b = _model.preprocess_observation(None, obs_b, train=False, image_keys=tuple(obs_b.images))
        obs_u = _model.preprocess_observation(None, obs_u, train=False, image_keys=tuple(obs_u.images))

        actions_a = actions[:, 0]
        noise_rng, time_rng = jax.random.split(rng)
        noise2_rng, time2_rng = jax.random.split(jax.random.fold_in(rng, 1))
        noise = jax.random.normal(noise_rng, actions_a.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions_a.shape[:-2]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions_a
        noise2 = jax.random.normal(noise2_rng, actions_a.shape)
        time2 = jax.random.beta(time2_rng, 1.5, 1, actions_a.shape[:-2]) * 0.999 + 0.001
        x_t2 = time2[:, None, None] * noise2 + (1 - time2[:, None, None]) * actions_a

        v_a = model_for_batch.predict_flow(obs_a, x_t, time, train=False)
        v_b = model_for_batch.predict_flow(obs_b, x_t, time, train=False)
        v_a_identity = model_for_batch.predict_flow(obs_a, x_t, time, train=False)
        v_a_diff_noise_time = model_for_batch.predict_flow(obs_a, x_t2, time2, train=False)

        def _prefix_cache(prefix_obs):
            prefix_tokens, prefix_mask, prefix_ar = model_for_batch.embed_prefix(prefix_obs)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = model_for_batch.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )
            n_language_tokens = prefix_obs.tokenized_prompt.shape[-1]
            n_image_tokens = prefix_tokens.shape[1] - n_language_tokens
            return prefix_mask, kv_cache, n_image_tokens

        prefix_mask_a, kv_a, n_image_tokens = _prefix_cache(obs_a)
        _prefix_mask_b, kv_b, _ = _prefix_cache(obs_b)
        _prefix_mask_u, kv_u, _ = _prefix_cache(obs_u)

        def _replace_cache(base_cache, donor_cache, mode):
            base_k, base_v = base_cache
            donor_k, donor_v = donor_cache
            lang_slice = slice(n_image_tokens, base_k.shape[2])
            image_slice = slice(0, n_image_tokens)
            if mode == "lang":
                base_k = base_k.at[:, :, lang_slice, :, :].set(donor_k[:, :, lang_slice, :, :])
                base_v = base_v.at[:, :, lang_slice, :, :].set(donor_v[:, :, lang_slice, :, :])
            elif mode == "image":
                base_k = base_k.at[:, :, image_slice, :, :].set(donor_k[:, :, image_slice, :, :])
                base_v = base_v.at[:, :, image_slice, :, :].set(donor_v[:, :, image_slice, :, :])
            elif mode == "both":
                base_k = donor_k
                base_v = donor_v
            return base_k, base_v

        def _flow_from_cache(kv_cache):
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = model_for_batch.embed_suffix(obs_a, x_t, time)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)
            cross_mask = einops.repeat(prefix_mask_a, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask_a, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = model_for_batch.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                canonical_tokens=obs_a.canonical_tokens,
            )
            return model_for_batch.action_out_proj(suffix_out[:, -model_for_batch.action_horizon :])

        flows = {
            "same_view_same_noise_identity": v_a_identity,
            "same_view_diff_noise_time": v_a_diff_noise_time,
            "normal_b_same_state": v_b,
            "replace_lang_same_state": _flow_from_cache(_replace_cache(kv_a, kv_b, "lang")),
            "replace_image_same_state": _flow_from_cache(_replace_cache(kv_a, kv_b, "image")),
            "replace_both_same_state": _flow_from_cache(_replace_cache(kv_a, kv_b, "both")),
            "replace_lang_same_language_unrelated": _flow_from_cache(_replace_cache(kv_a, kv_u, "lang")),
            "replace_image_same_language_unrelated": _flow_from_cache(_replace_cache(kv_a, kv_u, "image")),
        }

        def _delta(v):
            diff = v_a[:, :, :7] - v[:, :, :7]
            flow_delta = jnp.sqrt(jnp.mean(jnp.square(diff), axis=(1, 2)) + 1e-8)
            action_delta = jnp.mean(jnp.abs(diff), axis=(1, 2))
            return flow_delta, action_delta

        return {mode: _delta(flow) for mode, flow in flows.items()}

    mode_values: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"flow_delta": [], "action_delta": []})
    sample_indices: list[np.ndarray] = []
    categories: list[str] = []
    suites: list[str] = []
    processed = 0
    total_batches = max(1, int(np.ceil(args.max_pairs / args.batch_size)))
    rng = jax.random.key(args.seed)
    for obs, actions in tqdm.tqdm(
        loader,
        total=total_batches,
        desc="[late] batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_size = int(obs.state.shape[0])
        take = min(batch_size, args.max_pairs - processed)
        task_index = np.asarray(obs.task_index[:, 0])
        perm = jnp.asarray(_same_task_perm(task_index))
        rng, batch_rng = jax.random.split(rng)
        batch = _audit_batch(model_state, obs, actions, batch_rng, perm)
        for mode, (flow_delta, action_delta) in batch.items():
            mode_values[mode]["flow_delta"].append(np.asarray(flow_delta)[:take])
            mode_values[mode]["action_delta"].append(np.asarray(action_delta)[:take])
        idx = np.asarray(obs.sample_index[:take, 0]).astype(np.int64)
        sample_indices.append(idx)
        for sample_idx in idx:
            meta = sample_to_meta[int(sample_idx)]
            categories.append(meta["camera_category"])
            suites.append(meta["suite"])
        processed += take
        tqdm.tqdm.write(f"[late] processed {processed}/{args.max_pairs}")
        if processed >= args.max_pairs:
            break

    mode_summary = {
        mode: {
            "flow_delta": _aggregate(values["flow_delta"]),
            "action_delta": _aggregate(values["action_delta"]),
        }
        for mode, values in sorted(mode_values.items())
    }
    sample_index = np.concatenate(sample_indices, axis=0).astype(np.int64)
    summary = {
        "num_pairs": len(sample_index),
        "suite_counts": {
            str(k): int(v) for k, v in zip(*np.unique(np.array(suites), return_counts=True), strict=True)
        },
        "camera_category_counts": {
            str(k): int(v) for k, v in zip(*np.unique(np.array(categories), return_counts=True), strict=True)
        },
        "modes": mode_summary,
    }
    summary["bypass"] = _classify({"modes": mode_summary})
    result = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "config_name": args.config_name,
            "max_pairs": args.max_pairs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "val_episode_start": val_episode_start,
            "shuffle": not args.no_shuffle,
            "kv_layer_scope": "all_prefix_layers",
        },
        "summary": summary,
    }
    (args.output_dir / "routing_audit_late.json").write_text(json.dumps(result, indent=2, default=_json_default))
    _write_report(args.output_dir / "routing_audit_late.md", result)
    if args.write_canonical_report or args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        canonical_report = REPO_ROOT / "results/v4_routing_audit/routing_audit_late.md"
        canonical_report.parent.mkdir(parents=True, exist_ok=True)
        canonical_report.write_text((args.output_dir / "routing_audit_late.md").read_text())
    print(f"[late] wrote {args.output_dir / 'routing_audit_late.md'}", flush=True)


if __name__ == "__main__":
    main()
