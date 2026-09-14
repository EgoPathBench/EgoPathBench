#!/usr/bin/env python3
"""Extract final JSON lists from CoT predictions and re-run NavBench3D metrics."""

from __future__ import annotations

import ast
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


THINK_CLOSE = "</think>"
JSON_LIST_RE = re.compile(r"\[[\s\S]*?\]")


class PostprocessError(RuntimeError):
    """Raised when CoT post-processing cannot safely continue."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise PostprocessError(f"expected_json_object:{path}")
    return payload


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


def parse_json_list(text: str) -> list[int]:
    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    if not isinstance(parsed, list):
        raise PostprocessError("final_json_is_not_list")
    try:
        return [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise PostprocessError("final_json_list_not_ints") from exc


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_final_json(text: str) -> list[int]:
    stripped = text.strip()
    if THINK_CLOSE in stripped:
        stripped = stripped.rsplit(THINK_CLOSE, 1)[1].strip()

    candidates = [strip_code_fences(stripped), stripped]
    for candidate in candidates:
        try:
            return parse_json_list(candidate)
        except PostprocessError:
            pass

    for match in JSON_LIST_RE.finditer(stripped):
        try:
            return parse_json_list(match.group(0))
        except PostprocessError:
            continue

    raise PostprocessError("missing_trailing_final_json")


def iter_task_files(predictions_dir: Path) -> Iterable[Path]:
    for path in sorted(predictions_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def postprocess_task_file(src: Path, dst: Path, *, overwrite: bool) -> dict[str, Any]:
    rows = load_jsonl(src)
    if not rows:
        raise PostprocessError(f"empty_predictions:{src}")
    if dst.exists() and overwrite:
        dst.unlink()
    elif dst.exists() and not overwrite:
        raise PostprocessError(f"output_exists:{dst}")

    extract_ok = 0
    extract_fail = 0
    for row in rows:
        output = str(row.get("output") or "")
        try:
            final_json = extract_final_json(output)
            append_jsonl(
                dst,
                {
                    **row,
                    "raw_output": output,
                    "output": final_json,
                    "final_json_extract_success": True,
                },
            )
            extract_ok += 1
        except Exception as exc:
            append_jsonl(
                dst,
                {
                    **row,
                    "raw_output": output,
                    "output": None,
                    "final_json_extract_success": False,
                    "error": str(exc),
                },
            )
            extract_fail += 1

    return {"extract_ok": extract_ok, "extract_fail": extract_fail}


def run_eval(eval_script: Path, *, vqa: Path, gt: Path, predictions: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--vqa",
        str(vqa),
        "--gt",
        str(gt),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process CoT benchmark predictions.")
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--output-root", default="results/llamafactory_eval/qwen35_4b")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--rewrite-summary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_root = Path(args.output_root) / args.source_run_name
    output_root = Path(args.output_root) / args.output_run_name
    vqa_root = repo_root / "published/navbench3d_release_20260417_rebuild5/data/release" / args.split / "vqa"
    gt_root = repo_root / "published/navbench3d_release_20260417_rebuild5/data/release" / args.split / "gt"

    predictions_dir = source_root / "predictions"
    output_predictions_dir = output_root / "predictions_final_json"
    output_metrics_dir = output_root / "metrics_final_json"
    output_predictions_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    task_summaries: dict[str, dict[str, int]] = {}
    metrics_by_task: dict[str, dict[str, Any]] = {}
    prediction_counts: dict[str, int] = {}
    for src in iter_task_files(predictions_dir):
        task = src.stem
        dst = output_predictions_dir / f"{task}.jsonl"
        task_summaries[task] = postprocess_task_file(src, dst, overwrite=args.overwrite)
        metrics_path = output_metrics_dir / f"{task}.json"
        run_eval(
            repo_root / "scripts" / "evaluate_next.py",
            vqa=vqa_root / f"vqa_next_{task}.jsonl",
            gt=gt_root / f"gt_next_{task}.jsonl",
            predictions=dst,
            output=metrics_path,
        )
        metrics_by_task[task] = read_json(metrics_path)
        prediction_counts[task] = sum(1 for _ in dst.open())

    summary = {
        "aggregate": task_metric_summary(metrics_by_task),
        "extraction": task_summaries,
        "metrics_by_task": metrics_by_task,
        "note": "Post-hoc evaluation using the final JSON list extracted after </think>; model predictions were not rerun.",
        "prediction_counts": prediction_counts,
        "source_run": str(source_root),
        "output_run": str(output_root),
    }
    write_json(output_root / "summary_final_json_extracted.json", summary)

    if args.rewrite_summary:
        raw_summary = source_root / "summary.json"
        if raw_summary.exists():
            shutil.copy2(raw_summary, output_root / "summary_raw.json")

    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
