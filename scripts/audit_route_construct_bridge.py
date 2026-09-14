#!/usr/bin/env python3
"""Audit route-level failures after observable interface conditions are met."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from audit_route_prefix_validity import (
    MODEL_RUNS,
    direct_pairs_path,
    prediction_file,
    rate,
)
from evaluate_next import (
    load_direct_pairs,
    load_jsonl,
    load_visible_payload,
    merge_vqa_with_gt,
    parse_ids,
    prediction_output,
)


TASKS = ("a2", "b2", "c")
TASK_NAMES = {
    "a2": "Point Path",
    "b2": "Embodied Path",
    "c": "Intent Path",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_key(question: dict) -> tuple[str, int, str]:
    identity = question.get("ground_truth", {}).get("immutable_identity", {})
    routing_id = identity.get("routing_id", question.get("routing_id"))
    return str(question["scene_id"]), int(question["view_id"]), str(routing_id)


def prepare_questions(benchmark_root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    task_rows = {}
    records = {}
    for task in TASKS:
        vqa_path = benchmark_root / "vqa" / f"vqa_next_{task}.jsonl"
        gt_path = benchmark_root / "gt" / f"gt_next_{task}.jsonl"
        rows = merge_vqa_with_gt(load_jsonl(vqa_path), load_jsonl(gt_path))
        task_rows[task] = rows
        for question in rows:
            payload = load_visible_payload(Path(question["visible_waypoints_path"]))
            waypoints = list(payload.get("visible_waypoints", []))
            ground_truth = question.get("ground_truth", {})
            records[str(question["question_id"])] = {
                "question": question,
                "visible_waypoints": waypoints,
                "visible_ids": {
                    int(waypoint["display_id"])
                    for waypoint in waypoints
                    if waypoint.get("display_id") is not None
                },
                "display_signature": tuple(
                    sorted(
                        (
                            int(waypoint["display_id"]),
                            tuple(float(value) for value in waypoint["image_xy"]),
                        )
                        for waypoint in waypoints
                        if waypoint.get("display_id") is not None
                    )
                ),
                "candidate_count": len(waypoints),
            }

    thresholds = {}
    for task in TASKS:
        selected = [records[str(question["question_id"])] for question in task_rows[task]]
        thresholds[task] = {
            "candidate_count_max": median(item["candidate_count"] for item in selected),
        }
        for item in selected:
            threshold = thresholds[task]
            item["lower_marker_load"] = (
                item["candidate_count"] <= threshold["candidate_count_max"]
            )

    explicit_pairs = {
        task: {pair_key(question): question for question in task_rows[task]}
        for task in ("a2", "b2")
    }
    if set(explicit_pairs["a2"]) != set(explicit_pairs["b2"]):
        raise ValueError("Point and Embodied explicit tasks do not share the same pair keys")
    for key, point_question in explicit_pairs["a2"].items():
        body_question = explicit_pairs["b2"][key]
        point_record = records[str(point_question["question_id"])]
        body_record = records[str(body_question["question_id"])]
        point_target = point_question.get("ground_truth", {}).get("target_id")
        body_target = body_question.get("ground_truth", {}).get("target_id")
        if point_target != body_target:
            raise ValueError(f"paired target mismatch for {key}")
        if point_record["display_signature"] != body_record["display_signature"]:
            raise ValueError(f"paired display mismatch for {key}")

    return task_rows, records, thresholds


def prediction_flags(
    benchmark_root: Path,
    record: dict,
    prediction: dict,
    adjacency_cache: dict[str, dict],
) -> dict:
    question = record["question"]
    ground_truth = question.get("ground_truth", {})
    pred_ids = parse_ids(prediction_output(prediction))
    flags = {
        "eligible": False,
        "first_edge_legal": False,
        "full_route_legal": False,
        "goal_hit": False,
        "full_success": False,
        "anchored": False,
        "anchored_suffix_failure": False,
    }
    if pred_ids is None or len(pred_ids) < 2:
        return flags
    if any(pred_id not in record["visible_ids"] for pred_id in pred_ids):
        return flags
    start_id = ground_truth.get("start_id")
    if start_id is None or int(pred_ids[0]) != int(start_id):
        return flags

    flags["eligible"] = True
    sidecar = direct_pairs_path(benchmark_root, str(ground_truth["direct_pairs_ref"]))
    cache_key = str(sidecar)
    if cache_key not in adjacency_cache:
        adjacency_cache[cache_key] = load_direct_pairs(sidecar, record["visible_waypoints"])
    adjacency = adjacency_cache[cache_key]
    edge_checks = [
        right in adjacency.get(left, {})
        for left, right in zip(pred_ids[:-1], pred_ids[1:])
    ]
    flags["first_edge_legal"] = bool(edge_checks and edge_checks[0])
    flags["full_route_legal"] = bool(edge_checks and all(edge_checks))
    goal_ids = {int(value) for value in ground_truth.get("acceptable_goal_ids", [])}
    flags["goal_hit"] = bool(goal_ids and int(pred_ids[-1]) in goal_ids)
    flags["full_success"] = flags["full_route_legal"] and flags["goal_hit"]
    flags["anchored"] = flags["first_edge_legal"] and flags["goal_hit"]
    flags["anchored_suffix_failure"] = flags["anchored"] and not flags["full_route_legal"]
    return flags


def summarize(rows: list[dict]) -> dict:
    counts = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "eligible",
            "first_edge_legal",
            "full_route_legal",
            "goal_hit",
            "full_success",
            "anchored",
            "anchored_suffix_failure",
        )
    }
    counts["predictions"] = len(rows)
    counts["anchored_suffix_failure_rate"] = rate(
        counts["anchored_suffix_failure"], counts["anchored"]
    )
    counts["full_success_rate"] = rate(counts["full_success"], counts["predictions"])
    return counts


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
        default=Path("results/route_construct_bridge_audit_20260719"),
    )
    parser.add_argument(
        "--paper-data",
        type=Path,
        default=Path("paper/data/route_construct_bridge.csv"),
    )
    args = parser.parse_args()

    task_rows, records, thresholds = prepare_questions(args.benchmark_root)
    adjacency_cache = {}
    prediction_rows = []
    for run_name, model_name in MODEL_RUNS.items():
        run_dir = args.runs_root / run_name
        for task in TASKS:
            predictions = {
                str(row["question_id"]): row
                for row in load_jsonl(prediction_file(run_dir, task))
            }
            for question in task_rows[task]:
                question_id = str(question["question_id"])
                record = records[question_id]
                prediction_rows.append({
                    "run": run_name,
                    "model": model_name,
                    "task": task,
                    "question_id": question_id,
                    "pair_key": pair_key(question),
                    "lower_marker_load": record["lower_marker_load"],
                    **prediction_flags(
                        args.benchmark_root,
                        record,
                        predictions.get(question_id, {}),
                        adjacency_cache,
                    ),
                })

    pooled_rows = []
    for task in TASKS:
        task_predictions = [row for row in prediction_rows if row["task"] == task]
        low_predictions = [row for row in task_predictions if row["lower_marker_load"]]
        task_questions = [records[str(question["question_id"])] for question in task_rows[task]]
        pooled_rows.append({
            "task": TASK_NAMES[task],
            "questions": len(task_rows[task]),
            "lower_marker_questions": sum(
                int(record["lower_marker_load"]) for record in task_questions
            ),
            "all_anchored": summarize(task_predictions)["anchored"],
            "all_anchored_suffix_failures": summarize(task_predictions)[
                "anchored_suffix_failure"
            ],
            "all_anchored_suffix_failure_rate": summarize(task_predictions)[
                "anchored_suffix_failure_rate"
            ],
            "lower_marker_anchored": summarize(low_predictions)["anchored"],
            "lower_marker_anchored_suffix_failures": summarize(low_predictions)[
                "anchored_suffix_failure"
            ],
            "lower_marker_anchored_suffix_failure_rate": summarize(low_predictions)[
                "anchored_suffix_failure_rate"
            ],
        })

    paired_by_model = []
    grouped = defaultdict(dict)
    for row in prediction_rows:
        if row["task"] in ("a2", "b2"):
            grouped[(row["run"], row["pair_key"])][row["task"]] = row
    for run_name, model_name in MODEL_RUNS.items():
        pairs = [
            pair
            for (run, _), pair in grouped.items()
            if run == run_name and set(pair) == {"a2", "b2"}
        ]
        both_eligible = [pair for pair in pairs if pair["a2"]["eligible"] and pair["b2"]["eligible"]]
        point_legal = [pair for pair in both_eligible if pair["a2"]["full_route_legal"]]
        point_legal_body_illegal = [
            pair for pair in point_legal if not pair["b2"]["full_route_legal"]
        ]
        point_success = [pair for pair in both_eligible if pair["a2"]["full_success"]]
        point_success_body_edge_failure = [
            pair for pair in point_success if not pair["b2"]["full_route_legal"]
        ]
        paired_by_model.append({
            "run": run_name,
            "model": model_name,
            "pairs": len(pairs),
            "both_eligible": len(both_eligible),
            "point_legal": len(point_legal),
            "point_legal_body_illegal": len(point_legal_body_illegal),
            "point_legal_body_illegal_rate": rate(
                len(point_legal_body_illegal), len(point_legal)
            ),
            "point_success": len(point_success),
            "point_success_body_edge_failure": len(point_success_body_edge_failure),
            "point_success_body_edge_failure_rate": rate(
                len(point_success_body_edge_failure), len(point_success)
            ),
        })

    pair_totals = {
        key: sum(int(row[key]) for row in paired_by_model)
        for key in (
            "pairs",
            "both_eligible",
            "point_legal",
            "point_legal_body_illegal",
            "point_success",
            "point_success_body_edge_failure",
        )
    }
    pair_totals.update({
        "point_legal_body_illegal_rate": rate(
            pair_totals["point_legal_body_illegal"], pair_totals["point_legal"]
        ),
        "point_success_body_edge_failure_rate": rate(
            pair_totals["point_success_body_edge_failure"], pair_totals["point_success"]
        ),
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pooled_by_task.csv", pooled_rows)
    write_csv(args.output_dir / "paired_by_model.csv", paired_by_model)
    write_csv(args.paper_data, pooled_rows)
    summary = {
        "contract": {
            "anchored": "prediction has a legal first edge and ends at an acceptable goal",
            "anchored_suffix_failure": "an anchored prediction contains at least one illegal later edge",
            "lower_marker_load": (
                "within-task visible-candidate count at or below the median; target identification "
                "is controlled separately by conditioning on an acceptable predicted endpoint"
            ),
            "paired_explicit_tasks": (
                "Point and Embodied Path share scene, view, target, displayed IDs, and marker coordinates"
            ),
        },
        "thresholds": thresholds,
        "pooled_by_task": pooled_rows,
        "paired_explicit_tasks": pair_totals,
        "paired_by_model": paired_by_model,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
