"""Summarize Phase 0C early/late audit JSON files into one routing note."""

from __future__ import annotations

import argparse
import json
import pathlib


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-json", type=pathlib.Path, default=pathlib.Path("results/v4_routing_audit/early/fm_only_10k_s42/routing_audit_early.json"))
    parser.add_argument("--late-json", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/v4_routing_audit/routing_audit_summary.md"))
    args = parser.parse_args()

    early = _load(args.early_json) if args.early_json.exists() else None
    late = _load(args.late_json) if args.late_json and args.late_json.exists() else None

    lines = ["# Phase 0C Routing Audit Summary", ""]
    if early is None:
        lines += ["- early audit: missing", ""]
    else:
        lines += [
            f"- early audit: `{early['summary'].get('overall_bypass', early['summary'].get('early_bypass_signal', 'unknown'))}`",
            f"- early distance bypass: `{early['summary'].get('distance_bypass', 'unknown')}`",
            f"- early perturbation detection: `{early['summary'].get('perturbation_detection', 'unknown')}`",
            f"- early perturbation typing: `{early['summary'].get('perturbation_typing', 'unknown')}`",
            f"- early pairs: `{early['summary']['num_pairs']}`",
            f"- early checkpoint: `{early['config']['checkpoint']}`",
            "",
        ]
    if late is None:
        lines += ["- late audit: missing or not run yet", ""]
    else:
        lines += [
            f"- late audit: `{late['summary']['bypass']}`",
            f"- late pairs: `{late['summary']['num_pairs']}`",
            f"- late checkpoint: `{late['config']['checkpoint']}`",
            "",
        ]
    lines += [
        "Interpretation rule: use this summary together with Phase 0B matched-vs-clean-wrong results. "
        "Do not choose an architecture-heavy Phase 1 branch from the early audit alone.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
