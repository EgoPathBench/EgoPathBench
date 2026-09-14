#!/usr/bin/env python3
"""Audit LLaMA-Factory NavBench3D SFT results against the CoT plan gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLANNING_KEYS = ("spl_all", "success_rate", "valid_path_rate")
CLASSIFICATION_TASKS = ("a1", "b1")
PLANNING_TASKS = ("a2", "b2", "c")


class AcceptanceAuditError(RuntimeError):
    """Raised when acceptance-audit inputs are invalid."""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AcceptanceAuditError(f"missing summary: {path}")
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise AcceptanceAuditError(f"expected JSON object: {path}")
    return payload


def get_nested_float(payload: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    try:
        return float(cursor)
    except (TypeError, ValueError):
        return default


def metric_delta(after: dict[str, Any], before: dict[str, Any], keys: tuple[str, ...]) -> float:
    return round(get_nested_float(after, keys) - get_nested_float(before, keys), 6)


def planning_mean(summary: dict[str, Any], key: str) -> float:
    return get_nested_float(summary, ("aggregate", "planning_mean", key))


def task_metric(summary: dict[str, Any], task: str, key: str) -> float:
    return get_nested_float(summary, ("metrics_by_task", task, "metrics", key))


def direct_vs_base_gate(args: argparse.Namespace, base: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        key: metric_delta(direct, base, ("aggregate", "planning_mean", key))
        for key in PLANNING_KEYS
    }
    passing_metrics = [key for key, value in deltas.items() if value >= args.direct_min_planning_gain]
    return {
        "status": "pass" if passing_metrics else "fail",
        "threshold": args.direct_min_planning_gain,
        "planning_mean_deltas": deltas,
        "passing_metrics": passing_metrics,
    }


def cot_vs_direct_gate(args: argparse.Namespace, direct: dict[str, Any], cot: dict[str, Any]) -> dict[str, Any]:
    planning_deltas = {
        key: metric_delta(cot, direct, ("aggregate", "planning_mean", key))
        for key in PLANNING_KEYS
    }
    direct_invalid_or_illegal = max(0.0, 1.0 - planning_mean(direct, "valid_path_rate"))
    cot_invalid_or_illegal = max(0.0, 1.0 - planning_mean(cot, "valid_path_rate"))
    if direct_invalid_or_illegal > 0:
        invalid_relative_drop = round(
            (direct_invalid_or_illegal - cot_invalid_or_illegal) / direct_invalid_or_illegal, 6
        )
    else:
        invalid_relative_drop = 0.0

    c_deltas = {
        key: metric_delta(cot, direct, ("metrics_by_task", "c", "metrics", key))
        for key in ("success_rate", "valid_path_rate")
    }
    pass_reasons: list[str] = []
    if planning_deltas["spl_all"] >= args.cot_min_spl_gain:
        pass_reasons.append("planning_spl_all_gain")
    if invalid_relative_drop >= args.cot_min_invalid_relative_drop:
        pass_reasons.append("invalid_or_illegal_relative_drop")
    if any(value >= args.clear_c_gain for value in c_deltas.values()):
        pass_reasons.append("c_task_clear_gain")

    task_balanced_accuracy_deltas = {
        task: metric_delta(cot, direct, ("metrics_by_task", task, "metrics", "balanced_accuracy"))
        for task in CLASSIFICATION_TASKS
    }
    classification_ok = all(
        value >= -args.max_classification_balanced_accuracy_drop
        for value in task_balanced_accuracy_deltas.values()
    )

    return {
        "status": "pass" if pass_reasons and classification_ok else "fail",
        "pass_reasons": pass_reasons,
        "planning_mean_deltas": planning_deltas,
        "invalid_or_illegal_path_rate": {
            "direct": round(direct_invalid_or_illegal, 6),
            "cot": round(cot_invalid_or_illegal, 6),
            "relative_drop": invalid_relative_drop,
            "threshold": args.cot_min_invalid_relative_drop,
        },
        "c_task_deltas": c_deltas,
        "classification_guard": {
            "status": "pass" if classification_ok else "fail",
            "max_allowed_drop": args.max_classification_balanced_accuracy_drop,
            "task_balanced_accuracy_deltas": task_balanced_accuracy_deltas,
        },
        "thresholds": {
            "cot_min_spl_gain": args.cot_min_spl_gain,
            "clear_c_gain": args.clear_c_gain,
        },
    }


def benchmark_transfer_gate(base: dict[str, Any], cot: dict[str, Any]) -> dict[str, Any]:
    planning_deltas = {
        key: metric_delta(cot, base, ("aggregate", "planning_mean", key))
        for key in PLANNING_KEYS
    }
    classification_delta = metric_delta(cot, base, ("aggregate", "classification_mean", "balanced_accuracy"))
    improved_metrics = [key for key, value in planning_deltas.items() if value > 0]
    if classification_delta > 0:
        improved_metrics.append("classification_balanced_accuracy")
    return {
        "status": "pass" if improved_metrics else "fail",
        "planning_mean_deltas": planning_deltas,
        "classification_balanced_accuracy_delta": classification_delta,
        "improved_metrics": improved_metrics,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    base_val = read_json(Path(args.base_val))
    direct_val = read_json(Path(args.direct_val))
    cot_val = read_json(Path(args.cot_val))

    direct_gate = direct_vs_base_gate(args, base_val, direct_val)
    cot_gate = cot_vs_direct_gate(args, direct_val, cot_val)
    benchmark_gate: dict[str, Any]
    missing_benchmark = []
    if args.base_benchmark and args.cot_benchmark:
        base_benchmark_path = Path(args.base_benchmark)
        cot_benchmark_path = Path(args.cot_benchmark)
        if base_benchmark_path.exists() and cot_benchmark_path.exists():
            benchmark_gate = benchmark_transfer_gate(read_json(base_benchmark_path), read_json(cot_benchmark_path))
        else:
            if not base_benchmark_path.exists():
                missing_benchmark.append(str(base_benchmark_path))
            if not cot_benchmark_path.exists():
                missing_benchmark.append(str(cot_benchmark_path))
            benchmark_gate = {"status": "missing", "missing": missing_benchmark}
    else:
        benchmark_gate = {"status": "missing", "missing": ["base_benchmark", "cot_benchmark"]}

    gate_statuses = [direct_gate["status"], cot_gate["status"], benchmark_gate["status"]]
    if any(status == "missing" for status in gate_statuses):
        overall_status = "incomplete"
    elif all(status == "pass" for status in gate_statuses):
        overall_status = "pass"
    else:
        overall_status = "fail"

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/audit_llamafactory_navbench_sft.py",
        "overall_status": overall_status,
        "inputs": {
            "base_val": args.base_val,
            "direct_val": args.direct_val,
            "cot_val": args.cot_val,
            "base_benchmark": args.base_benchmark,
            "cot_benchmark": args.cot_benchmark,
        },
        "direct_vs_base": direct_gate,
        "cot_vs_direct": cot_gate,
        "benchmark_transfer": benchmark_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NavBench3D LLaMA-Factory SFT acceptance gates.")
    parser.add_argument("--base-val", default="results/llamafactory_eval/qwen35_4b/base_val/summary.json")
    parser.add_argument(
        "--direct-val",
        default="results/llamafactory_eval/qwen35_4b/direct_full_seed42_val/summary.json",
    )
    parser.add_argument(
        "--cot-val",
        default="results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_val/summary.json",
    )
    parser.add_argument(
        "--base-benchmark",
        default="results/llamafactory_eval/qwen35_4b/base_benchmark/summary.json",
    )
    parser.add_argument(
        "--cot-benchmark",
        default="results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_benchmark/summary.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--direct-min-planning-gain", type=float, default=0.03)
    parser.add_argument("--cot-min-spl-gain", type=float, default=0.02)
    parser.add_argument("--cot-min-invalid-relative-drop", type=float, default=0.10)
    parser.add_argument("--clear-c-gain", type=float, default=0.01)
    parser.add_argument("--max-classification-balanced-accuracy-drop", type=float, default=0.01)
    args = parser.parse_args()

    audit = build_audit(args)
    payload = json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
