#!/usr/bin/env python3
"""Run LLaMA-Factory NavBench3D prediction and evaluation across tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = ("a1", "b1", "a2", "b2", "c")
DEFAULT_RELEASE_ROOT = Path("published/navbench3d_release_20260417_rebuild5/data/release")
DEFAULT_MODEL = "/mnt/data/omnimodel/weights/base/Qwen3.5-4B"
DEFAULT_LLAMAFACTORY_ROOT = "/mnt/data/wuchanglin/LLaMA-Factory"


class NavbenchEvalError(RuntimeError):
    """Raised when the local SFT evaluation configuration is invalid."""


def parse_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise NavbenchEvalError(f"unknown tasks: {unknown}")
    return tasks


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise NavbenchEvalError(f"expected JSON object: {path}")
    return payload


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def prediction_command(args: argparse.Namespace, task: str, output_path: Path) -> list[str]:
    repo_root = Path(args.repo_root).resolve()
    release_root = Path(args.release_root)
    vqa_path = release_root / args.split / "vqa" / f"vqa_next_{task}.jsonl"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_llamafactory_navbench_predict.py"),
        "--vqa",
        str(vqa_path),
        "--output",
        str(output_path),
        "--repo-root",
        str(repo_root),
        "--llamafactory-root",
        args.llamafactory_root,
        "--model-name-or-path",
        args.model_name_or_path,
        "--template",
        args.template,
        "--image-max-pixels",
        str(args.image_max_pixels),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--request-sleep-s",
        str(args.request_sleep_s),
    ]
    if args.adapter_name_or_path:
        command.extend(["--adapter-name-or-path", args.adapter_name_or_path])
    if args.max_questions is not None:
        command.extend(["--max-questions", str(args.max_questions)])
    if args.resume:
        command.append("--resume")
    if args.enable_thinking:
        command.append("--enable-thinking")
    return command


def evaluation_command(args: argparse.Namespace, task: str, predictions_path: Path, metrics_path: Path) -> list[str]:
    repo_root = Path(args.repo_root).resolve()
    release_root = Path(args.release_root)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_next.py"),
        "--vqa",
        str(release_root / args.split / "vqa" / f"vqa_next_{task}.jsonl"),
        "--gt",
        str(release_root / args.split / "gt" / f"gt_next_{task}.jsonl"),
        "--predictions",
        str(predictions_path),
        "--output",
        str(metrics_path),
    ]
    if args.restrict_to_predictions or args.max_questions is not None:
        command.append("--restrict-to-predictions")
    if args.goal_hop:
        command.extend(["--goal-hop", str(args.goal_hop)])
    if args.goal_dist_m:
        command.extend(["--goal-dist-m", str(args.goal_dist_m)])
    if args.weak_connectivity:
        command.append("--weak-connectivity")
    return command


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print(json.dumps({"dry_run": command}, ensure_ascii=True))
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def task_metric_summary(metrics_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    planning_tasks = [task for task in ("a2", "b2", "c") if task in metrics_by_task]
    planning_keys = ("success_rate", "valid_path_rate", "valid_id_rate", "spl_all")
    planning_mean: dict[str, float] = {}
    for key in planning_keys:
        values = [
            float(metrics_by_task[task]["metrics"][key])
            for task in planning_tasks
            if key in metrics_by_task[task].get("metrics", {})
        ]
        if values:
            planning_mean[key] = round(sum(values) / len(values), 4)

    classification_tasks = [task for task in ("a1", "b1") if task in metrics_by_task]
    classification_keys = ("f1", "balanced_accuracy", "accuracy")
    classification_mean: dict[str, float] = {}
    for key in classification_keys:
        values = [
            float(metrics_by_task[task]["metrics"][key])
            for task in classification_tasks
            if key in metrics_by_task[task].get("metrics", {})
        ]
        if values:
            classification_mean[key] = round(sum(values) / len(values), 4)

    return {
        "classification_mean": classification_mean,
        "planning_mean": planning_mean,
    }


def write_summary(args: argparse.Namespace, out_dir: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    metrics_by_task: dict[str, dict[str, Any]] = {}
    prediction_counts: dict[str, int] = {}
    for task in tasks:
        metrics_path = out_dir / "metrics" / f"{task}.json"
        if metrics_path.exists():
            metrics_by_task[task] = read_json(metrics_path)
        prediction_counts[task] = count_jsonl(out_dir / "predictions" / f"{task}.jsonl")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/run_llamafactory_navbench_eval.py",
        "release_root": args.release_root,
        "split": args.split,
        "tasks": list(tasks),
        "model_name_or_path": args.model_name_or_path,
        "adapter_name_or_path": args.adapter_name_or_path,
        "prediction_config": {
            "template": args.template,
            "image_max_pixels": args.image_max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "enable_thinking": args.enable_thinking,
        },
        "prediction_counts": prediction_counts,
        "metrics_by_task": metrics_by_task,
        "aggregate": task_metric_summary(metrics_by_task),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LLaMA-Factory NavBench3D evaluation.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/llamafactory_eval/qwen35_4b")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--split", default="val")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--llamafactory-root", default=DEFAULT_LLAMAFACTORY_ROOT)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-name-or-path", default=None)
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--request-sleep-s", type=float, default=0.0)
    parser.add_argument("--goal-hop", type=int, default=0)
    parser.add_argument("--goal-dist-m", type=float, default=0.0)
    parser.add_argument("--weak-connectivity", action="store_true")
    parser.add_argument("--restrict-to-predictions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_root) / args.run_name
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    for task in tasks:
        predictions_path = out_dir / "predictions" / f"{task}.jsonl"
        metrics_path = out_dir / "metrics" / f"{task}.json"
        if not args.skip_predict:
            run_command(prediction_command(args, task, predictions_path), cwd=repo_root, dry_run=args.dry_run)
        run_command(evaluation_command(args, task, predictions_path, metrics_path), cwd=repo_root, dry_run=args.dry_run)

    if not args.dry_run:
        summary = write_summary(args, out_dir, tasks)
        print(json.dumps(summary["aggregate"], ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
