"""Resume-aware nominal LIBERO or LIBERO-Plus camera-track evaluation."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from afcv.evaluation import libero_plus_eval as camera
from afcv.evaluation import libero_policy_runner as nominal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--benchmark", choices=["nominal", "camera"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="paired_cv_eval")
    parser.add_argument("--trials", type=int, help="Default: 50 nominal, 3 camera trials per task")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, help="Restrict to a small execution check")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="Short rollout limit for execution checks")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    trials = args.trials if args.trials is not None else (50 if args.benchmark == "nominal" else 3)
    if trials < 1:
        raise ValueError("trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config_path = args.output_dir / "evaluation.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new output directory")
        if json.loads(config_path.read_text()) != spec:
            raise ValueError("Evaluation settings changed during resume")
    config_path.write_text(json.dumps(spec, indent=2) + "\n")
    path = args.output_dir / "trials.jsonl"
    task_dir = args.output_dir / "tasks"
    task_dir.mkdir(exist_ok=True)
    records = [row for file in sorted(task_dir.glob("*.json")) for row in json.loads(file.read_text())]
    completed = {r["task_spec"] for r in records}

    def save_task(rows):
        key = hashlib.sha256(rows[0]["task_spec"].encode()).hexdigest()
        destination = task_dir / f"{key}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows) + "\n")
        temporary.replace(destination)
        records.extend(rows)

    policy = camera.load_pi05_checkpoint(args.checkpoint_dir, config_name=args.config)
    if args.benchmark == "camera":
        evaluator = camera.LIBEROPlusCameraEvaluator(
            policy=policy,
            n_trials_per_task=trials,
            results_dir=args.output_dir,
            base_seed=args.seed,
            image_flip_mode="flipud",
            rollout_config=camera.RolloutConfig(max_steps=args.max_steps or 0),
        )
        tasks = evaluator.camera_tasks[args.shard_index :: args.num_shards][: args.max_tasks]
        for task in tqdm(tasks, desc="Camera tasks", dynamic_ncols=True):
            if task.task_spec in completed:
                continue
            rows = [{**dataclasses.asdict(r), "task_spec": task.task_spec} for r in evaluator.run_single_task(task)]
            save_task(rows)
    else:
        suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        tasks = [(name, i) for name in suites for i in range(nominal.get_task_suite(name).get_num_tasks())]
        if len(tasks) != 40:
            raise ValueError(f"Expected 40 nominal tasks, found {len(tasks)}")
        tasks = tasks[args.shard_index :: args.num_shards][: args.max_tasks]
        for name, i in tqdm(tasks, desc="Nominal tasks", dynamic_ncols=True):
            key = f"{name}:{i}"
            if key in completed:
                continue
            suite = nominal.get_task_suite(name)
            task = suite.get_task(i)
            states = suite.get_task_init_states(i)
            if trials > len(states):
                raise ValueError("Not enough initial states for requested trials")
            env, instruction = nominal.create_libero_env(task, seed=args.seed)
            rows = []
            try:
                for trial in range(trials):
                    env.seed(camera._stable_seed(args.seed, key, trial, 0))
                    result = nominal.rollout_episode(
                        policy,
                        env,
                        instruction,
                        nominal.RolloutConfig(max_steps=args.max_steps or nominal.default_max_steps_for_suite(name)),
                        initial_state=states[trial],
                        image_flip_mode="flipud",
                    )
                    rows.append({"task_spec": key, "trial_index": trial, "success": bool(result["success"])})
            finally:
                env.close()
            save_task(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    expected = len(tasks) * trials
    if len(records) != expected:
        raise ValueError(f"Incomplete or duplicate trials: {len(records)} / {expected}")
    success = sum(r["success"] for r in records)
    summary = {
        "benchmark": args.benchmark,
        "successes": success,
        "trials": len(records),
        "success_rate": success / len(records) if records else None,
        "full_benchmark": args.max_steps is None
        and args.max_tasks is None
        and args.num_shards == 1
        and trials == (50 if args.benchmark == "nominal" else 3),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
