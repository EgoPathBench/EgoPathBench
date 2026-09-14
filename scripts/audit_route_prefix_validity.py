#!/usr/bin/env python3
"""Audit first-edge and full-route legality on the frozen main predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_next import (
    load_direct_pairs,
    load_jsonl,
    load_visible_payload,
    merge_vqa_with_gt,
    parse_ids,
    prediction_output,
)


TASKS = ("a2", "b2", "c")
MODEL_RUNS = {
    "claude_opus48_micu": "Claude Opus 4.8",
    "gemini31_rightcode": "Gemini 3.1 Pro",
    "gpt55_gateway": "GPT-5.5",
    "grok43_fast_ld_20260618": "Grok 4.3",
    "kimi26_dashscope": "Kimi K2.6",
    "llama4_maverick_nim": "Llama 4",
    "minimax_m3_micu": "MiniMax M3",
    "mistral_large3_nim": "Mistral L3",
    "qwen36_dashscope": "Qwen 3.6",
}


def prediction_file(run_dir: Path, task: str) -> Path:
    matches = sorted((run_dir / "predictions").glob(f"preds_{task}_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one prediction file for {run_dir.name}/{task}, found {matches}")
    return matches[0]


def metrics_file(run_dir: Path, task: str) -> Path:
    matches = sorted((run_dir / "metrics").glob(f"metrics_{task}_*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one metrics file for {run_dir.name}/{task}, found {matches}")
    return matches[0]


def direct_pairs_path(benchmark_root: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    candidates = (
        benchmark_root / path,
        benchmark_root.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve direct-pairs sidecar: {ref}")


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def audit_task(
    benchmark_root: Path,
    vqa_rows: list[dict],
    predictions: dict[str, dict],
) -> dict:
    counts = {
        "total": 0,
        "eligible": 0,
        "first_edge_legal": 0,
        "full_route_legal": 0,
        "legal_first_illegal_suffix": 0,
        "goal_hit": 0,
        "full_success": 0,
    }

    for question in vqa_rows:
        counts["total"] += 1
        qid = str(question["question_id"])
        prediction = predictions.get(qid, {})
        pred_ids = parse_ids(prediction_output(prediction))
        if pred_ids is None or len(pred_ids) < 2:
            continue

        visible_payload = load_visible_payload(Path(question["visible_waypoints_path"]))
        visible_waypoints = list(visible_payload.get("visible_waypoints", []))
        visible_ids = {
            int(waypoint["display_id"])
            for waypoint in visible_waypoints
            if waypoint.get("display_id") is not None
        }
        ground_truth = question.get("ground_truth", {})
        start_id = ground_truth.get("start_id")
        if any(pred_id not in visible_ids for pred_id in pred_ids):
            continue
        if start_id is None or int(pred_ids[0]) != int(start_id):
            continue

        counts["eligible"] += 1
        sidecar = direct_pairs_path(benchmark_root, str(ground_truth["direct_pairs_ref"]))
        adjacency = load_direct_pairs(sidecar, visible_waypoints)
        edge_checks = [right in adjacency.get(left, {}) for left, right in zip(pred_ids[:-1], pred_ids[1:])]
        first_edge_legal = bool(edge_checks and edge_checks[0])
        full_route_legal = bool(edge_checks and all(edge_checks))
        goal_ids = {int(value) for value in ground_truth.get("acceptable_goal_ids", [])}
        goal_hit = bool(goal_ids and int(pred_ids[-1]) in goal_ids)

        counts["first_edge_legal"] += int(first_edge_legal)
        counts["full_route_legal"] += int(full_route_legal)
        counts["legal_first_illegal_suffix"] += int(first_edge_legal and not full_route_legal)
        counts["goal_hit"] += int(goal_hit)
        counts["full_success"] += int(full_route_legal and goal_hit)

    total = counts["total"]
    eligible = counts["eligible"]
    first_edge_legal = counts["first_edge_legal"]
    return {
        **counts,
        "eligible_rate": rate(eligible, total),
        "first_edge_legal_rate": rate(first_edge_legal, total),
        "full_route_legal_rate": rate(counts["full_route_legal"], total),
        "goal_hit_rate": rate(counts["goal_hit"], total),
        "full_success_rate": rate(counts["full_success"], total),
        "first_edge_legal_given_eligible": rate(first_edge_legal, eligible),
        "full_route_legal_given_eligible": rate(counts["full_route_legal"], eligible),
        "suffix_failure_given_legal_first": rate(counts["legal_first_illegal_suffix"], first_edge_legal),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("published/navbench3d_release_20260417_rebuild5/data/benchmark_evalfix/benchmark"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/main_model_runs_20260528/native_default_full/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/route_prefix_validity_audit_20260718"),
    )
    args = parser.parse_args()

    task_rows: dict[str, list[dict]] = {}
    for task in TASKS:
        vqa_path = args.benchmark_root / "vqa" / f"vqa_next_{task}.jsonl"
        gt_path = args.benchmark_root / "gt" / f"gt_next_{task}.jsonl"
        vqa_rows = merge_vqa_with_gt(load_jsonl(vqa_path), load_jsonl(gt_path))
        task_rows[task] = vqa_rows

    rows = []
    for run_name, model_name in MODEL_RUNS.items():
        run_dir = args.runs_root / run_name
        for task in TASKS:
            predictions = {
                str(row["question_id"]): row
                for row in load_jsonl(prediction_file(run_dir, task))
            }
            metrics = audit_task(args.benchmark_root, task_rows[task], predictions)
            official = json.loads(metrics_file(run_dir, task).read_text())["metrics"]
            official_valid_path = float(official["valid_path_rate"])
            if abs(metrics["full_route_legal_rate"] - official_valid_path) > 5e-5:
                raise ValueError(
                    f"full-route mismatch for {run_name}/{task}: "
                    f"audit={metrics['full_route_legal_rate']:.6f}, official={official_valid_path:.6f}"
                )
            rows.append({
                "run": run_name,
                "model": model_name,
                "task": task,
                **metrics,
            })

    pooled_rows = []
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        pooled_counts = {
            key: sum(int(row[key]) for row in selected)
            for key in (
                "total",
                "eligible",
                "first_edge_legal",
                "full_route_legal",
                "legal_first_illegal_suffix",
                "goal_hit",
                "full_success",
            )
        }
        total = pooled_counts["total"]
        eligible = pooled_counts["eligible"]
        first_edge_legal = pooled_counts["first_edge_legal"]
        pooled_rows.append({
            "task": task,
            **pooled_counts,
            "eligible_rate": rate(eligible, total),
            "first_edge_legal_rate": rate(first_edge_legal, total),
            "full_route_legal_rate": rate(pooled_counts["full_route_legal"], total),
            "goal_hit_rate": rate(pooled_counts["goal_hit"], total),
            "full_success_rate": rate(pooled_counts["full_success"], total),
            "first_edge_legal_given_eligible": rate(first_edge_legal, eligible),
            "full_route_legal_given_eligible": rate(pooled_counts["full_route_legal"], eligible),
            "suffix_failure_given_legal_first": rate(
                pooled_counts["legal_first_illegal_suffix"], first_edge_legal
            ),
            "model_rank_spearman_first_vs_full": spearman(
                [float(row["first_edge_legal_rate"]) for row in selected],
                [float(row["full_route_legal_rate"]) for row in selected],
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "by_model_task.csv", rows)
    write_csv(args.output_dir / "pooled_by_task.csv", pooled_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "contract": {
            "eligible": "parseable nonempty route, visible IDs, required start, at least one selected edge",
            "first_edge_legal": "the first selected edge is present in the formal direct-pair graph",
            "full_route_legal": "every selected edge is present in the formal direct-pair graph",
            "suffix_failure": "the first edge is legal but at least one later selected edge is illegal",
        },
        "pooled_by_task": pooled_rows,
        "by_model_task": rows,
    }, indent=2))
    print(json.dumps(pooled_rows, indent=2))


if __name__ == "__main__":
    main()
