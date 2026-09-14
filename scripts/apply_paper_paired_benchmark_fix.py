#!/usr/bin/env python3
"""Apply the paper-facing paired A2/B2 benchmark fix and recompute metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DROP_A2_QID = "arkitscenes__Training__47430327_v3_routing_v3_c004_a2"

TASKS = ("a1", "b1", "a2", "b2", "c")

API_RUNS = (
    "claude_opus48_micu",
    "gemini31_rightcode",
    "gpt55_gateway",
    "grok43_fast_ld_20260618",
    "kimi26_dashscope",
    "llama4_maverick_nim",
    "minimax_m3_micu",
    "mistral_large3_nim",
    "qwen36_dashscope",
)

EXTRA_API_SYNC_RUNS = (
    "mistral_medium35_nim",
    "step37_flash_nim",
)

SFT_RUNS = (
    "base_benchmark_max8192",
    "v4seg_full_new_gpt_cot_lora_base2_checkpoint_3000_benchmark_max8192",
)

ORACLE_RUNS = (
    "results/benchmark_paper_oracle_20260420",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=False) + "\n")


def count_jsonl(path: Path) -> int:
    return len(load_jsonl(path))


def filter_qid(path: Path, qid: str) -> tuple[int, int]:
    rows = load_jsonl(path)
    kept = [row for row in rows if row.get("question_id") != qid]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
    return len(rows), len(kept)


def set_path(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> bool:
    node: Any = data
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        return False
    node[keys[-1]] = value
    return True


def update_manifest_counts(path: Path) -> None:
    data = read_json(path)
    changed = False
    if set_path(data, ("splits", "benchmark", "question_counts_by_task", "a2"), 309):
        changed = True
    if set_path(data, ("splits", "benchmark", "questions_total"), 1111):
        changed = True
    if set_path(data, ("splits", "benchmark", "reporting", "question_variants"), 1111):
        changed = True
    if set_path(data, ("row_counts_by_task", "a2"), 309):
        changed = True
    if set_path(data, ("sidecar_counts_by_task", "a2"), 294):
        changed = True
    if set_path(data, ("audits", "nonempty_acceptable_goal_rows"), 819):
        changed = True
    if set_path(data, ("audits", "reference_path_legal_rows"), 819):
        changed = True
    if set_path(data, ("audits", "reference_endpoint_ok_rows"), 819):
        changed = True
    if isinstance(data.get("notes"), str):
        old = data["notes"]
        new = old.replace(
            "Dropped 1 invalid b2 benchmark row",
            "Dropped 1 invalid b2 benchmark row and its paired a2 row",
        ).replace(
            "because the embodied direct-segment graph disconnected start from goal ring under the final benchmark protocol.",
            "because the embodied direct-segment graph disconnected start from goal ring under the final benchmark protocol; the paired a2 route unit is removed for the paper-facing paired A2/B2 benchmark.",
        )
        if new != old:
            data["notes"] = new
            changed = True
    if changed:
        write_json(path, data)


def run_eval(
    repo: Path,
    *,
    task: str,
    vqa: Path,
    gt: Path | None,
    predictions: Path,
    output: Path,
    restrict_to_predictions: bool = False,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "evaluate_next.py"),
        "--vqa",
        str(vqa),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
    ]
    if gt is not None and gt.exists():
        cmd.extend(["--gt", str(gt)])
    if restrict_to_predictions:
        cmd.append("--restrict-to-predictions")
    subprocess.run(cmd, cwd=str(repo), check=True, stdout=subprocess.DEVNULL)
    return read_json(output)


def update_list_summary(path: Path, repo: Path) -> None:
    if not path.exists():
        return
    summary = read_json(path)
    if not isinstance(summary, list):
        return
    for row in summary:
        metrics_path = Path(row.get("metrics") or "")
        if not metrics_path.is_absolute():
            repo_relative = repo / metrics_path
            metrics_path = repo_relative if repo_relative.exists() else path.parent / metrics_path
        if metrics_path.exists():
            row["metric_json"] = read_json(metrics_path)
    write_json(path, summary)


def aggregate_metrics(metrics_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    planning_tasks = [task for task in ("a2", "b2", "c") if task in metrics_by_task]
    planning_mean: dict[str, float] = {}
    for key in ("success_rate", "valid_path_rate", "valid_id_rate", "spl_all"):
        values = [
            float(metrics_by_task[task]["metrics"][key])
            for task in planning_tasks
            if key in metrics_by_task[task].get("metrics", {})
        ]
        if values:
            planning_mean[key] = round(sum(values) / len(values), 4)

    classification_tasks = [task for task in ("a1", "b1") if task in metrics_by_task]
    classification_mean: dict[str, float] = {}
    for key in ("f1", "balanced_accuracy", "accuracy"):
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


def update_sft_summary(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        return
    metrics_by_task: dict[str, dict[str, Any]] = {}
    prediction_counts: dict[str, int] = {}
    for task in TASKS:
        metrics_path = run_dir / "metrics" / f"{task}.json"
        if metrics_path.exists():
            metrics_by_task[task] = read_json(metrics_path)
        pred_path = run_dir / "predictions" / f"{task}.jsonl"
        if pred_path.exists():
            prediction_counts[task] = count_jsonl(pred_path)
    summary["metrics_by_task"] = metrics_by_task
    summary["prediction_counts"] = prediction_counts
    summary["aggregate"] = aggregate_metrics(metrics_by_task)
    write_json(summary_path, summary)


def filter_paper_predictions(run_dir: Path, task: str, qid: str) -> None:
    for subdir in ("predictions", "predictions_final_json"):
        pred_dir = run_dir / subdir
        if not pred_dir.exists():
            continue
        for path in pred_dir.glob(f"{task}.jsonl"):
            filter_qid(path, qid)
        for path in pred_dir.glob(f"preds_{task}_*.jsonl"):
            filter_qid(path, qid)


def rewrite_text(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text()
    new = text
    for old, replacement in replacements.items():
        new = new.replace(old, replacement)
    deduped: list[str] = []
    duplicate_sensitive_lines = {
        f"- `{DROP_A2_QID}`",
    }
    for line in new.splitlines():
        if line in duplicate_sensitive_lines and deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    new = "\n".join(deduped) + ("\n" if text.endswith("\n") else "")
    if new != text:
        path.write_text(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    release = repo / "published" / "navbench3d_release_20260417_rebuild5"
    release_bench = release / "data" / "release" / "benchmark"
    evalfix_bench = release / "data" / "benchmark_evalfix" / "benchmark"

    touched_filter: list[tuple[str, int, int]] = []
    for root in (release_bench, evalfix_bench):
        for subdir in ("vqa", "gt"):
            path = root / subdir / "vqa_next_a2.jsonl" if subdir == "vqa" else root / subdir / "gt_next_a2.jsonl"
            before, after = filter_qid(path, DROP_A2_QID)
            touched_filter.append((str(path.relative_to(repo)), before, after))

    update_manifest_counts(release / "data" / "release" / "release_manifest.json")
    update_manifest_counts(release / "data" / "benchmark_evalfix" / "release_manifest.json")
    update_manifest_counts(release / "data" / "benchmark_evalfix" / "summary.json")

    rewrite_text(
        release / "README.md",
        {
            "- `a2 = 310`": "- `a2 = 309`",
            "发布时额外剔除了 `1` 条不满足最终 benchmark 直线段协议的 `b2` 坏题：": "发布时额外剔除了 `1` 个不满足最终 benchmark 直线段协议的 `a2/b2` 成对 route unit：",
            "- `arkitscenes__Training__47430327_v3_routing_v3_c004_b2`": "- `arkitscenes__Training__47430327_v3_routing_v3_c004_a2`\n- `arkitscenes__Training__47430327_v3_routing_v3_c004_b2`",
            "- benchmark 额外删除了上述 `1` 条 `b2` 坏题；": "- benchmark 额外删除了上述 `1` 个 `a2/b2` 成对 route unit；",
            "- `benchmark = 1112`": "- `benchmark = 1111`",
            "benchmark = 1112": "benchmark = 1111",
        },
    )

    replacements = {
        "a2=310、b2=309": "a2=309、b2=309",
        "a2 310": "a2 309",
        "| a2 | 310 |": "| a2 | 309 |",
        "310 A2, 309 B2": "309 A2, 309 B2",
        "& 310 \\\\": "& 309 \\\\",
        "1,112": "1,111",
        "1112": "1111",
        "820/820": "819/819",
        "820 / 820": "819 / 819",
        "acceptable goal 820": "acceptable goal 819",
        "legal reference path 820": "legal reference path 819",
        "valid endpoint 820": "valid endpoint 819",
        "nonempty acceptable goal rows: 820": "nonempty acceptable goal rows: 819",
        "legal reference path rows: 820": "legal reference path rows: 819",
        "valid reference endpoint rows: 820": "valid reference endpoint rows: 819",
        "820 acceptable-goal rows": "819 acceptable-goal rows",
        "820 legal reference paths": "819 legal reference paths",
        "820 valid endpoints": "819 valid endpoints",
        "820 rows with nonempty acceptable goals": "819 rows with nonempty acceptable goals",
        "820 navigation rows": "819 navigation rows",
        "820 rows": "819 rows",
        "Benchmark & 255 & 1111 & 146 & 146 & 310 & 309 & 201": "Benchmark & 255 & 1111 & 146 & 146 & 309 & 309 & 201",
    }
    for text_path in (
        repo / "AGENTS.md",
        repo / "PAPER_PLAN.md",
        repo / "docs" / "paper_artifacts" / "evidence_index.md",
        release / "docs" / "SOURCE_PATHS.md",
        repo / "paper" / "sections" / "0_abstract.tex",
        repo / "paper" / "sections" / "3_dataset_benchmark.tex",
        repo / "paper" / "scripts" / "build_paper_assets.py",
        repo / "paper" / "tables" / "table_task_taxonomy.tex",
        repo / "paper" / "tables" / "table_dataset_stats.tex",
    ):
        rewrite_text(text_path, replacements)

    for run_name in (*API_RUNS, *EXTRA_API_SYNC_RUNS):
        run_dir = repo / "results" / "main_model_runs_20260528" / "native_default_full" / "runs" / run_name
        if not run_dir.exists():
            continue
        filter_paper_predictions(run_dir, "a2", DROP_A2_QID)
        for task in TASKS:
            pred_matches = sorted((run_dir / "predictions").glob(f"preds_{task}_*.jsonl"))
            if not pred_matches:
                continue
            metric_matches = sorted((run_dir / "metrics").glob(f"metrics_{task}_*.json"))
            if metric_matches:
                metric_path = metric_matches[0]
            else:
                tag = pred_matches[0].stem.split(f"preds_{task}_", 1)[-1]
                metric_path = run_dir / "metrics" / f"metrics_{task}_{tag}.json"
            run_eval(
                repo,
                task=task,
                vqa=evalfix_bench / "vqa" / f"vqa_next_{task}.jsonl",
                gt=evalfix_bench / "gt" / f"gt_next_{task}.jsonl",
                predictions=pred_matches[0],
                output=metric_path,
            )
        update_list_summary(run_dir / "summary.json", repo)

    for sft_name in SFT_RUNS:
        run_dir = repo / "results" / "llamafactory_eval" / "qwen35_4b" / sft_name
        if not run_dir.exists():
            continue
        filter_paper_predictions(run_dir, "a2", DROP_A2_QID)
        for task in TASKS:
            pred_path = run_dir / "predictions" / f"{task}.jsonl"
            if pred_path.exists():
                run_eval(
                    repo,
                    task=task,
                    vqa=release_bench / "vqa" / f"vqa_next_{task}.jsonl",
                    gt=release_bench / "gt" / f"gt_next_{task}.jsonl",
                    predictions=pred_path,
                    output=run_dir / "metrics" / f"{task}.json",
                )
        update_sft_summary(run_dir)

    for run_root in ORACLE_RUNS:
        run_dir = repo / run_root
        if not run_dir.exists():
            continue
        filter_paper_predictions(run_dir, "a2", DROP_A2_QID)
        for task in TASKS:
            pred_matches = sorted((run_dir / "predictions").glob(f"preds_{task}_*.jsonl"))
            if not pred_matches:
                continue
            metric_matches = sorted((run_dir / "metrics").glob(f"metrics_{task}_*.json"))
            metric_path = metric_matches[0] if metric_matches else run_dir / "metrics" / f"metrics_{task}.json"
            run_eval(
                repo,
                task=task,
                vqa=evalfix_bench / "vqa" / f"vqa_next_{task}.jsonl",
                gt=evalfix_bench / "gt" / f"gt_next_{task}.jsonl",
                predictions=pred_matches[0],
                output=metric_path,
            )
        update_list_summary(run_dir / "summary.json", repo)

    report = {
        "drop_a2_question_id": DROP_A2_QID,
        "filtered_files": touched_filter,
        "benchmark_counts": {
            task: count_jsonl(evalfix_bench / "vqa" / f"vqa_next_{task}.jsonl")
            for task in TASKS
        },
        "api_runs_recomputed": list(API_RUNS),
        "extra_api_runs_synchronized": list(EXTRA_API_SYNC_RUNS),
        "sft_runs_recomputed": list(SFT_RUNS),
        "oracle_runs_recomputed": list(ORACLE_RUNS),
    }
    report_path = repo / "docs" / "paper_artifacts" / "paired_benchmark_fix_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
