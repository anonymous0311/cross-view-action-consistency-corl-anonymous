"""Step-3 displacement evaluation pipeline for LIBERO Phase-0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import dataclasses
import json
import math
import pathlib
from statistics import NormalDist
from typing import Any

import numpy as np
import tqdm

from canonical.eval.camera_displacement import DISPLACEMENT_CONFIGS
from canonical.eval.camera_displacement import CameraDisplacementWrapper
from canonical.eval import libero_policy_runner

DEFAULT_TASK_SPECS = (
    "libero_spatial:0",
    "libero_spatial:1",
    "libero_object:0",
    "libero_goal:0",
    "libero_10:0",
)

DEFAULT_CONDITIONS = tuple(DISPLACEMENT_CONFIGS.keys())


@dataclasses.dataclass(frozen=True)
class EvalSummaryRow:
    task_spec: str
    condition: str
    displacement_magnitude: float
    n_success: int
    n_total: int
    success_rate: float
    ci_low: float
    ci_high: float
    delta_from_nominal: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_task_spec(task_spec: str) -> tuple[str, int]:
    if ":" not in task_spec:
        raise ValueError(f"Task spec '{task_spec}' must be '<suite_name>:<task_id>'.")
    suite_name, task_id_str = task_spec.split(":", maxsplit=1)
    return suite_name, int(task_id_str)


def task_slug(task_spec: str) -> str:
    return task_spec.replace(":", "_")


def wilson_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate."""
    if n_total <= 0:
        raise ValueError("n_total must be > 0 for Wilson CI.")
    if not 0 <= n_success <= n_total:
        raise ValueError("n_success must satisfy 0 <= n_success <= n_total.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + (z * z) / n_total
    center = (p_hat + (z * z) / (2 * n_total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) / n_total) + ((z * z) / (4 * n_total * n_total))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def displacement_magnitude(config_name: str, rotation_weight: float = 0.01) -> float:
    if config_name not in DISPLACEMENT_CONFIGS:
        known = ", ".join(sorted(DISPLACEMENT_CONFIGS))
        raise ValueError(f"Unknown displacement config '{config_name}'. Known: {known}")
    cfg = DISPLACEMENT_CONFIGS[config_name]
    translation = np.asarray(cfg["translation"], dtype=np.float64)
    rotation_deg = np.asarray(cfg["rotation_deg"], dtype=np.float64)
    return float(np.linalg.norm(translation) + rotation_weight * np.linalg.norm(rotation_deg))


def _collapse_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for x_val, y_val in points:
        grouped[float(x_val)].append(float(y_val))
    return [(x_val, float(np.mean(grouped[x_val]))) for x_val in sorted(grouped)]


def _monotonic_envelope(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    collapsed = _collapse_points(points)
    envelope: list[tuple[float, float]] = []
    running = float("inf")
    for x_val, y_val in collapsed:
        running = min(running, y_val)
        envelope.append((x_val, running))
    return envelope


def _trapz_auc(points: Sequence[tuple[float, float]]) -> float:
    collapsed = _collapse_points(points)
    if len(collapsed) < 2:
        return 0.0
    auc = 0.0
    for idx in range(1, len(collapsed)):
        x0, y0 = collapsed[idx - 1]
        x1, y1 = collapsed[idx]
        auc += 0.5 * (y0 + y1) * (x1 - x0)
    return float(auc)


def _interpolate_threshold(points: Sequence[tuple[float, float]], target: float) -> float | None:
    envelope = _monotonic_envelope(points)
    if not envelope:
        return None
    if envelope[0][1] <= target:
        return envelope[0][0]

    for idx in range(1, len(envelope)):
        x0, y0 = envelope[idx - 1]
        x1, y1 = envelope[idx]
        if y1 > target:
            continue
        if math.isclose(y0, y1):
            return x1
        alpha = (target - y0) / (y1 - y0)
        return float(x0 + alpha * (x1 - x0))
    return None


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


@dataclasses.dataclass(frozen=True)
class DisplacementEvaluatorConfig:
    task_specs: tuple[str, ...] = DEFAULT_TASK_SPECS
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    episodes_per_condition: int = 20
    resolution: int = 256
    seed: int = 7
    rotation_weight: float = 0.01
    output_dir: str = "results/phase0/displacement_eval"
    write_plots: bool = True
    write_videos: bool = False


class DisplacementEvaluator:
    """Systematic camera displacement evaluation with resumable state."""

    def __init__(
        self,
        policy: Any,
        rollout_config: libero_policy_runner.RolloutConfig,
        config: DisplacementEvaluatorConfig,
    ) -> None:
        self.policy = policy
        self.rollout_config = rollout_config
        self.config = config
        self.output_dir = pathlib.Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_task_dir = self.output_dir / "per_task"
        self.per_task_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state_or_init()

    @property
    def state_path(self) -> pathlib.Path:
        return self.output_dir / "displacement_eval_state.json"

    @property
    def rows_jsonl_path(self) -> pathlib.Path:
        return self.output_dir / "displacement_eval_rows.jsonl"

    @property
    def rows_csv_path(self) -> pathlib.Path:
        return self.output_dir / "displacement_eval_rows.csv"

    def _validate_config(self) -> None:
        if self.config.episodes_per_condition <= 0:
            raise ValueError("episodes_per_condition must be > 0.")
        for condition in self.config.conditions:
            if condition not in DISPLACEMENT_CONFIGS:
                known = ", ".join(sorted(DISPLACEMENT_CONFIGS))
                raise ValueError(f"Unknown displacement condition '{condition}'. Known: {known}")
        if "nominal" not in self.config.conditions:
            raise ValueError("conditions must include 'nominal' to compute delta_from_nominal.")

    def _load_state_or_init(self) -> dict[str, Any]:
        self._validate_config()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_state_compat(state)
            return state

        completed = {
            task_spec: {condition: 0 for condition in self.config.conditions}
            for task_spec in self.config.task_specs
        }
        state = {
            "task_specs": list(self.config.task_specs),
            "conditions": list(self.config.conditions),
            "episodes_per_condition": int(self.config.episodes_per_condition),
            "rows_written": 0,
            "seed": int(self.config.seed),
            "resolution": int(self.config.resolution),
            "completed_counts": completed,
        }
        self._save_state(state)
        return state

    def _validate_state_compat(self, state: Mapping[str, Any]) -> None:
        expected_tasks = list(self.config.task_specs)
        expected_conditions = list(self.config.conditions)
        if list(state.get("task_specs", [])) != expected_tasks:
            raise ValueError(
                "Existing run state task_specs mismatch. "
                f"Expected {expected_tasks}, got {state.get('task_specs')}"
            )
        if list(state.get("conditions", [])) != expected_conditions:
            raise ValueError(
                "Existing run state conditions mismatch. "
                f"Expected {expected_conditions}, got {state.get('conditions')}"
            )
        if int(state.get("episodes_per_condition", -1)) != int(self.config.episodes_per_condition):
            raise ValueError(
                "Existing run state episodes_per_condition mismatch. "
                f"Expected {self.config.episodes_per_condition}, got {state.get('episodes_per_condition')}"
            )

    def _save_state(self, state: Mapping[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _append_row(self, row: Mapping[str, Any]) -> None:
        with self.rows_jsonl_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(dict(row), ensure_ascii=True) + "\n")

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        if not self.rows_jsonl_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.rows_jsonl_path.open("r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                rows.append(json.loads(raw_line))
        return rows

    def _write_rows_csv(self) -> None:
        rows = list(self._iter_rows())
        if not rows:
            return
        fieldnames = sorted(rows[0].keys())
        with self.rows_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def evaluate_single_condition(self, task_spec: str, condition: str) -> EvalSummaryRow:
        """Run remaining rollouts for one task-condition pair and return summary."""
        suite_name, task_id = parse_task_spec(task_spec)
        completed = int(self.state["completed_counts"][task_spec][condition])
        if completed >= self.config.episodes_per_condition:
            summaries = self.compute_task_condition_summaries(self._iter_rows())
            for summary in summaries:
                if summary.task_spec == task_spec and summary.condition == condition:
                    return summary
            raise RuntimeError(f"Summary row missing for completed pair {task_spec}/{condition}.")

        suite = libero_policy_runner.get_task_suite(suite_name)
        task = suite.get_task(task_id)
        task_description = str(task.language)
        initial_states = suite.get_task_init_states(task_id)
        if self.config.episodes_per_condition > len(initial_states):
            raise ValueError(
                f"Task {task_spec} has only {len(initial_states)} init states; "
                f"requested {self.config.episodes_per_condition}."
            )

        env, _ = libero_policy_runner.create_libero_env(task, resolution=self.config.resolution, seed=self.config.seed)
        wrapped_env = CameraDisplacementWrapper(env, config_name=condition)
        rollout_cfg = self._rollout_config_for_suite(suite_name)

        ep_bar = tqdm.tqdm(
            range(completed, self.config.episodes_per_condition),
            dynamic_ncols=True,
            leave=False,
            unit="ep",
            desc=f"{task_spec}/{condition}",
        )
        for episode_idx in ep_bar:
            result = libero_policy_runner.rollout_episode(
                self.policy,
                wrapped_env,
                task_description,
                rollout_cfg,
                initial_state=initial_states[episode_idx],
                progress=False,
            )
            row = {
                "task_spec": task_spec,
                "suite_name": suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "condition": condition,
                "episode_idx": int(episode_idx),
                "success": int(result["success"]),
                "done": int(result["done"]),
                "steps_executed": int(result["steps_executed"]),
                "num_steps_wait": int(rollout_cfg.num_steps_wait),
                "replan_steps": int(rollout_cfg.replan_steps),
                "max_steps": int(rollout_cfg.max_steps),
                "seed": int(self.config.seed),
                "resolution": int(self.config.resolution),
            }
            self._append_row(row)
            self.state["rows_written"] = int(self.state["rows_written"]) + 1
            self.state["completed_counts"][task_spec][condition] = int(episode_idx + 1)
            self._save_state(self.state)

        summaries = self.compute_task_condition_summaries(self._iter_rows())
        for summary in summaries:
            if summary.task_spec == task_spec and summary.condition == condition:
                return summary
        raise RuntimeError(f"Summary row missing after evaluation for {task_spec}/{condition}.")

    def run_all(self) -> list[EvalSummaryRow]:
        task_bar = tqdm.tqdm(self.config.task_specs, dynamic_ncols=True, leave=True, unit="task")
        for task_spec in task_bar:
            for condition in self.config.conditions:
                _ = self.evaluate_single_condition(task_spec, condition)
            task_completed = self.state["completed_counts"][task_spec]
            done_str = ",".join(f"{k}:{v}" for k, v in task_completed.items())
            task_bar.set_postfix(done=done_str)
        self._write_rows_csv()
        rows = list(self._iter_rows())
        summaries = self.compute_task_condition_summaries(rows)
        self.write_outputs(rows, summaries)
        return summaries

    def compute_task_condition_summaries(self, rows: Iterable[Mapping[str, Any]]) -> list[EvalSummaryRow]:
        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n_success": 0, "n_total": 0})
        for row in rows:
            task_spec = str(row["task_spec"])
            condition = str(row["condition"])
            grouped[(task_spec, condition)]["n_success"] += int(row["success"])
            grouped[(task_spec, condition)]["n_total"] += 1

        nominal_by_task: dict[str, float] = {}
        for task_spec in self.config.task_specs:
            key = (task_spec, "nominal")
            if key not in grouped or grouped[key]["n_total"] == 0:
                nominal_by_task[task_spec] = float("nan")
                continue
            nominal_by_task[task_spec] = grouped[key]["n_success"] / grouped[key]["n_total"]

        summaries: list[EvalSummaryRow] = []
        for task_spec in self.config.task_specs:
            for condition in self.config.conditions:
                key = (task_spec, condition)
                if key not in grouped:
                    continue
                n_success = grouped[key]["n_success"]
                n_total = grouped[key]["n_total"]
                success_rate = n_success / n_total
                ci_low, ci_high = wilson_ci(n_success, n_total)
                nominal_rate = nominal_by_task.get(task_spec, float("nan"))
                delta = success_rate - nominal_rate if math.isfinite(nominal_rate) else float("nan")
                summaries.append(
                    EvalSummaryRow(
                        task_spec=task_spec,
                        condition=condition,
                        displacement_magnitude=displacement_magnitude(condition, self.config.rotation_weight),
                        n_success=n_success,
                        n_total=n_total,
                        success_rate=success_rate,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        delta_from_nominal=delta,
                    )
                )

        return sorted(summaries, key=lambda item: (item.task_spec, item.displacement_magnitude, item.condition))

    def compute_robustness_metrics(self, summaries: Sequence[EvalSummaryRow]) -> dict[str, dict[str, float | None]]:
        by_task: dict[str, list[EvalSummaryRow]] = defaultdict(list)
        for row in summaries:
            by_task[row.task_spec].append(row)

        metrics: dict[str, dict[str, float | None]] = {}
        for task_spec, task_rows in by_task.items():
            points = [(row.displacement_magnitude, row.success_rate) for row in task_rows]
            collapsed = _collapse_points(points)
            max_mag = collapsed[-1][0] if collapsed else 0.0
            auc = _trapz_auc(collapsed)
            auc_norm = auc / max_mag if max_mag > 0 else (collapsed[0][1] if collapsed else 0.0)
            metrics[task_spec] = {
                "auc": _safe_float(auc),
                "auc_norm": _safe_float(auc_norm),
                "d50": _safe_float(_interpolate_threshold(collapsed, 0.50)),
                "d80": _safe_float(_interpolate_threshold(collapsed, 0.80)),
            }

        aggregate = self.compute_aggregate_condition_summary(summaries)
        points = [(row["displacement_magnitude"], row["macro_success_rate"]) for row in aggregate]
        collapsed = _collapse_points(points)
        max_mag = collapsed[-1][0] if collapsed else 0.0
        auc = _trapz_auc(collapsed)
        auc_norm = auc / max_mag if max_mag > 0 else (collapsed[0][1] if collapsed else 0.0)
        metrics["aggregate"] = {
            "auc": _safe_float(auc),
            "auc_norm": _safe_float(auc_norm),
            "d50": _safe_float(_interpolate_threshold(collapsed, 0.50)),
            "d80": _safe_float(_interpolate_threshold(collapsed, 0.80)),
        }
        return metrics

    def compute_aggregate_condition_summary(self, summaries: Sequence[EvalSummaryRow]) -> list[dict[str, Any]]:
        by_condition: dict[str, list[EvalSummaryRow]] = defaultdict(list)
        for row in summaries:
            by_condition[row.condition].append(row)

        aggregate: list[dict[str, Any]] = []
        nominal_rate = None
        for condition in self.config.conditions:
            rows = by_condition.get(condition, [])
            if not rows:
                continue
            total_success = sum(row.n_success for row in rows)
            total_trials = sum(row.n_total for row in rows)
            pooled_ci_low, pooled_ci_high = wilson_ci(total_success, total_trials)
            macro_rate = float(np.mean([row.success_rate for row in rows]))
            if condition == "nominal":
                nominal_rate = macro_rate
            delta = macro_rate - nominal_rate if nominal_rate is not None else float("nan")
            aggregate.append(
                {
                    "condition": condition,
                    "displacement_magnitude": displacement_magnitude(condition, self.config.rotation_weight),
                    "n_success_total": int(total_success),
                    "n_total_total": int(total_trials),
                    "macro_success_rate": macro_rate,
                    "pooled_ci_low": pooled_ci_low,
                    "pooled_ci_high": pooled_ci_high,
                    "delta_from_nominal": delta,
                }
            )
        return sorted(aggregate, key=lambda item: (item["displacement_magnitude"], item["condition"]))

    def write_outputs(self, rows: Sequence[Mapping[str, Any]], summaries: Sequence[EvalSummaryRow]) -> None:
        summaries_csv = self.output_dir / "task_condition_summary.csv"
        with summaries_csv.open("w", encoding="utf-8", newline="") as file_obj:
            fieldnames = list(EvalSummaryRow.__annotations__.keys())
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row.as_dict())

        aggregate_rows = self.compute_aggregate_condition_summary(summaries)
        aggregate_csv = self.output_dir / "aggregate_condition_summary.csv"
        if aggregate_rows:
            with aggregate_csv.open("w", encoding="utf-8", newline="") as file_obj:
                fieldnames = list(aggregate_rows[0].keys())
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(aggregate_rows)

        metrics = self.compute_robustness_metrics(summaries)
        (self.output_dir / "robustness_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (self.output_dir / "aggregate_results.md").write_text(
            self.generate_results_table_markdown(aggregate_rows),
            encoding="utf-8",
        )
        (self.output_dir / "phase0_summary.md").write_text(
            self.generate_phase0_summary_markdown(summaries, aggregate_rows, metrics),
            encoding="utf-8",
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task_specs": list(self.config.task_specs),
                    "conditions": list(self.config.conditions),
                    "episodes_per_condition": self.config.episodes_per_condition,
                    "rotation_weight": self.config.rotation_weight,
                    "seed": self.config.seed,
                    "resolution": self.config.resolution,
                    "rows_written": len(rows),
                    "write_videos": self.config.write_videos,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.write_per_task_outputs(summaries)
        if self.config.write_plots:
            self._write_plots(summaries, aggregate_rows)

    def write_per_task_outputs(self, summaries: Sequence[EvalSummaryRow]) -> None:
        by_task: dict[str, list[EvalSummaryRow]] = defaultdict(list)
        for row in summaries:
            by_task[row.task_spec].append(row)

        for task_spec, task_rows in by_task.items():
            csv_path = self.per_task_dir / f"{task_slug(task_spec)}_results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
                fieldnames = list(EvalSummaryRow.__annotations__.keys())
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                for row in sorted(task_rows, key=lambda item: (item.displacement_magnitude, item.condition)):
                    writer.writerow(row.as_dict())

    def generate_results_table_markdown(self, aggregate_rows: Sequence[Mapping[str, Any]]) -> str:
        lines = [
            "# Aggregate Displacement Results",
            "",
            "| Condition | Magnitude | Macro Success | 95% Wilson CI (pooled) | Delta vs nominal |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
        for row in aggregate_rows:
            ci_text = f"[{row['pooled_ci_low']:.3f}, {row['pooled_ci_high']:.3f}]"
            lines.append(
                "| {condition} | {mag:.4f} | {rate:.3f} | {ci} | {delta:+.3f} |".format(
                    condition=row["condition"],
                    mag=row["displacement_magnitude"],
                    rate=row["macro_success_rate"],
                    ci=ci_text,
                    delta=row["delta_from_nominal"],
                )
            )
        lines.append("")
        return "\n".join(lines)

    def generate_phase0_summary_markdown(
        self,
        summaries: Sequence[EvalSummaryRow],
        aggregate_rows: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Mapping[str, Any]],
    ) -> str:
        by_task: dict[str, list[EvalSummaryRow]] = defaultdict(list)
        for row in summaries:
            by_task[row.task_spec].append(row)

        lines = [
            "# Phase-0 Step-3 Summary",
            "",
            "## Config",
            f"- task_specs: {', '.join(self.config.task_specs)}",
            f"- conditions: {', '.join(self.config.conditions)}",
            f"- episodes_per_condition: {self.config.episodes_per_condition}",
            f"- seed: {self.config.seed}",
            f"- rotation_weight: {self.config.rotation_weight}",
            "",
            "## Aggregate",
            "",
        ]
        lines.extend(self.generate_results_table_markdown(aggregate_rows).splitlines())
        lines.extend(
            [
                "",
                "## Robustness Metrics",
                "",
                "| Scope | AUC | AUC (normalized) | d50 | d80 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for scope, scope_metrics in metrics.items():
            lines.append(
                "| {scope} | {auc} | {auc_norm} | {d50} | {d80} |".format(
                    scope=scope,
                    auc=self._fmt_metric(scope_metrics.get("auc")),
                    auc_norm=self._fmt_metric(scope_metrics.get("auc_norm")),
                    d50=self._fmt_metric(scope_metrics.get("d50")),
                    d80=self._fmt_metric(scope_metrics.get("d80")),
                )
            )

        lines.extend(["", "## Per-task Snapshot", ""])
        for task_spec in self.config.task_specs:
            rows = sorted(by_task.get(task_spec, []), key=lambda item: (item.displacement_magnitude, item.condition))
            if not rows:
                continue
            lines.append(f"### {task_spec}")
            lines.append("")
            lines.append("| Condition | Magnitude | Success | CI | Delta |")
            lines.append("| --- | ---: | ---: | --- | ---: |")
            for row in rows:
                lines.append(
                    "| {condition} | {mag:.4f} | {rate:.3f} | [{lo:.3f}, {hi:.3f}] | {delta:+.3f} |".format(
                        condition=row.condition,
                        mag=row.displacement_magnitude,
                        rate=row.success_rate,
                        lo=row.ci_low,
                        hi=row.ci_high,
                        delta=row.delta_from_nominal,
                    )
                )
            lines.append("")

        return "\n".join(lines)

    def _fmt_metric(self, value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            f_val = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not math.isfinite(f_val):
            return "N/A"
        return f"{f_val:.4f}"

    def _rollout_config_for_suite(self, suite_name: str) -> libero_policy_runner.RolloutConfig:
        max_steps = int(self.rollout_config.max_steps)
        if max_steps <= 0:
            max_steps = libero_policy_runner.default_max_steps_for_suite(suite_name)
        return libero_policy_runner.RolloutConfig(
            max_steps=max_steps,
            num_steps_wait=int(self.rollout_config.num_steps_wait),
            replan_steps=int(self.rollout_config.replan_steps),
            resize_size=int(self.rollout_config.resize_size),
        )

    def _write_plots(self, summaries: Sequence[EvalSummaryRow], aggregate_rows: Sequence[Mapping[str, Any]]) -> None:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ModuleNotFoundError:
            print("[warn] matplotlib is not installed; skipping plot generation.")
            return

        by_task: dict[str, list[EvalSummaryRow]] = defaultdict(list)
        for row in summaries:
            by_task[row.task_spec].append(row)

        for task_spec, task_rows in by_task.items():
            xs = [row.displacement_magnitude for row in task_rows]
            ys = [row.success_rate for row in task_rows]
            yerr_low = [row.success_rate - row.ci_low for row in task_rows]
            yerr_high = [row.ci_high - row.success_rate for row in task_rows]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], marker="o", linestyle="-", capsize=3)
            ax.set_xlabel("Displacement magnitude")
            ax.set_ylabel("Success rate")
            ax.set_ylim(0.0, 1.05)
            ax.set_title(task_spec)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.per_task_dir / f"{task_slug(task_spec)}_degradation_curve.png", dpi=180)
            plt.close(fig)

        if aggregate_rows:
            xs = [row["displacement_magnitude"] for row in aggregate_rows]
            ys = [row["macro_success_rate"] for row in aggregate_rows]
            yerr_low = [row["macro_success_rate"] - row["pooled_ci_low"] for row in aggregate_rows]
            yerr_high = [row["pooled_ci_high"] - row["macro_success_rate"] for row in aggregate_rows]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], marker="o", linestyle="-", capsize=3)
            ax.set_xlabel("Displacement magnitude")
            ax.set_ylabel("Macro success rate")
            ax.set_ylim(0.0, 1.05)
            ax.set_title("Aggregate degradation curve")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.output_dir / "degradation_curves.png", dpi=200)
            plt.close(fig)
