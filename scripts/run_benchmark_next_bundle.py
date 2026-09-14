#!/usr/bin/env python3
"""Run benchmark, evaluation, and visualization for Next VQA in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def model_tag(model: str) -> str:
    return "".join(ch for ch in model.lower() if ch.isalnum())


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Next benchmark, evaluation, and visualization."
    )
    parser.add_argument("--vqa-dir", required=True, help="Directory with vqa_next_*.jsonl")
    parser.add_argument("--gt-dir", default=None, help="Directory with gt_next_*.jsonl")
    parser.add_argument("--output-root", required=True, help="Root dir for predictions/metrics/viz")
    parser.add_argument("--model", required=True, help="Model name for run_benchmark_next.py")
    parser.add_argument("--tasks", default="a1,b1,a2,b2",
                        help="Comma-separated tasks to run, e.g. a1,b1,a2,b2")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Optional max questions per task for smoke runs")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--goal-hop", type=int, default=1,
                        help="Routing eval/visualization goal hop tolerance")
    parser.add_argument("--goal-dist-m", type=float, default=0.0)
    parser.add_argument("--weak-connectivity", action="store_true")
    parser.add_argument("--restrict-to-predictions", action="store_true",
                        help="Restrict evaluation/visualization to predicted question_ids. "
                             "Recommended for smoke runs.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks requested")

    vqa_dir = Path(args.vqa_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    out_root = Path(args.output_root)
    pred_dir = out_root / "predictions"
    metrics_dir = out_root / "metrics"
    viz_dir = out_root / "viz"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    auto_restrict = args.restrict_to_predictions or (args.max_questions is not None)
    tag = model_tag(args.model)
    summary: list[dict] = []

    for task in tasks:
        vqa_path = vqa_dir / f"vqa_next_{task}.jsonl"
        if not vqa_path.exists():
            print(f"[skip] {task}: missing {vqa_path}")
            continue

        pred_path = pred_dir / f"preds_{task}_{tag}.jsonl"
        metric_path = metrics_dir / f"metrics_{task}_{tag}.json"
        task_viz_dir = viz_dir / task
        gt_path = gt_dir / f"gt_next_{task}.jsonl" if gt_dir else None

        cmd = [
            sys.executable,
            str(ROOT / "run_benchmark_next.py"),
            "--vqa", str(vqa_path),
            "--model", args.model,
            "--output", str(pred_path),
            "--timeout-s", str(args.timeout_s),
            "--max-retries", str(args.max_retries),
        ]
        if args.max_questions is not None:
            cmd.extend(["--max-questions", str(args.max_questions)])
        if args.no_resume:
            cmd.append("--no-resume")
        run_cmd(cmd)

        if not args.skip_evaluate:
            cmd = [
                sys.executable,
                str(ROOT / "evaluate_next.py"),
                "--vqa", str(vqa_path),
                "--predictions", str(pred_path),
                "--output", str(metric_path),
            ]
            if auto_restrict:
                cmd.append("--restrict-to-predictions")
            if task in ("a2", "b2"):
                if args.goal_hop > 0:
                    cmd.extend(["--goal-hop", str(args.goal_hop)])
                if args.goal_dist_m > 0.0:
                    cmd.extend(["--goal-dist-m", str(args.goal_dist_m)])
                if args.weak_connectivity:
                    cmd.append("--weak-connectivity")
            run_cmd(cmd)

        if not args.skip_visualize:
            cmd = [
                sys.executable,
                str(ROOT / "visualize_next_results.py"),
                "--vqa", str(vqa_path),
                "--predictions", str(pred_path),
                "--output", str(task_viz_dir),
            ]
            if auto_restrict:
                cmd.append("--restrict-to-predictions")
            if task in ("a2", "b2") and gt_path and gt_path.exists():
                cmd.extend(["--gt", str(gt_path)])
                if args.goal_hop > 0:
                    cmd.extend(["--goal-hop", str(args.goal_hop)])
                if args.goal_dist_m > 0.0:
                    cmd.extend(["--goal-dist-m", str(args.goal_dist_m)])
                if args.weak_connectivity:
                    cmd.append("--weak-connectivity")
            run_cmd(cmd)

        rec = {
            "task": task,
            "vqa": str(vqa_path),
            "predictions": str(pred_path),
            "metrics": str(metric_path) if not args.skip_evaluate else None,
            "visualization": str(task_viz_dir) if not args.skip_visualize else None,
        }
        if metric_path.exists():
            rec["metric_json"] = json.loads(metric_path.read_text())
        summary.append(rec)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
