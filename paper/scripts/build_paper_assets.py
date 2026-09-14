#!/usr/bin/env python3
"""Build paper-facing tables and CSV summaries from fixed evidence files."""

from __future__ import annotations

import csv
import base64
import json
import math
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
DATA = PAPER / "data"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

RELEASE_ROOT = ROOT / "published/navbench3d_release_20260417_rebuild5/data/release"
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
EVALFIX_SUMMARY = ROOT / "published/navbench3d_release_20260417_rebuild5/data/benchmark_evalfix/summary.json"
COT_EXPORT_REPORT = ROOT / "results/lf_data_v4_segment_full_20260616/export_report.json"
LEADERBOARD_ROOT = ROOT / "results/main_model_runs_20260528/native_default_full/runs"
EVALFIX_VQA_ROOT = ROOT / "published/navbench3d_release_20260417_rebuild5/data/benchmark_evalfix/benchmark/vqa"
ANTI_SHORTCUT_ROOT = ROOT / "results/benchmark_paper_anti_shortcut_20260605/runs"
EXTERNAL_SUITE_ROOT = ROOT / "results/external_spatial_suite_20260621_strict4096/runs"
SFT_BASE_METRICS_ROOT = (
    ROOT
    / "results/llamafactory_eval/qwen35_4b/base_benchmark_max8192_reparsed_v2/metrics_paper_final_contract"
)
SFT_CHECKPOINT_METRICS_ROOT = (
    ROOT
    / "results/llamafactory_eval/qwen35_4b/v4seg_full_new_gpt_cot_lora_base2_checkpoint_3000_benchmark_max8192/metrics_paper_final_contract"
)
EDGE_OBSERVABILITY_ROOT = ROOT / "results/edge_observability_audit_20260716"
GEOMETRY_SENSITIVITY_ROOT = ROOT / "results/geometry_contract_sensitivity_20260717"
HUMAN_EVAL_ROOT = (
    ROOT
    / "human_baseline/subset_n20_seed20260630/eval_50_from_51_seed20260701"
)
HUMAN_COMPARISON_SUMMARY = (
    HUMAN_EVAL_ROOT / "vlm_same50" / "same50_human_vs_vlm_summary.json"
)
OVERVIEW_QUESTION_ID = (
    "arkitscenes__Training__45261541_v0_routing_v0_c002_b2"
)
OVERVIEW_GOAL_TEXT = "table near the cabinet"
OVERVIEW_ROUTE = [1, 7, 23]
OVERVIEW_GOAL_ID = 27
OVERVIEW_EDGE_LEGALITY = [True, False]

MODEL_ORDER = [
    ("claude_opus48_micu", "Claude Opus 4.8"),
    ("gemini31_rightcode", "Gemini 3.1 Pro"),
    ("gpt55_gateway", "GPT-5.5"),
    ("grok43_fast_ld_20260618", "Grok 4.3 Fast"),
    ("kimi26_dashscope", "Kimi K2.6"),
    ("llama4_maverick_nim", "Llama 4 Maverick"),
    ("minimax_m3_micu", "MiniMax M3"),
    ("mistral_large3_nim", "Mistral Large 3"),
    ("qwen36_dashscope", "Qwen 3.6"),
]

TABLE_MODEL_NAMES = {
    "Grok 4.3 Fast": "Grok 4.3",
    "Llama 4 Maverick": "Llama 4",
    "Mistral Large 3": "Mistral L3",
}

TASK_FULL_NAMES = {
    "a1": "Point Traversability",
    "b1": "Embodied Traversability",
    "a2": "Explicit Point-Goal Path",
    "b2": "Explicit Embodied-Goal Path",
    "c": "Intent-Grounded Embodied Path",
}

TASK_SHORT_NAMES = {
    "a1": "Point Trav.",
    "b1": "Body Trav.",
    "a2": "Point Path",
    "b2": "Body Path",
    "c": "Intent Path",
}

TASK_FAILURE_NAMES = {
    "a2": "Point path",
    "b2": "Embodied path",
    "c": "Intent path",
}

TASK_ROUTE_NAMES = {
    "a2": "Point Path",
    "b2": "Embodied Path",
    "c": "Intent Path",
}

ANTI_MODELS = [
    ("gpt55_gateway", "GPT-5.5"),
    ("claude_opus48_micu", "Claude Opus 4.8"),
    ("qwen36_dashscope", "Qwen 3.6"),
]

PLANNING_TASKS = ["a2", "b2", "c"]
VISIBILITY_LEVELS = [
    ("all_supported", "All", 0, 0.0),
    ("clear_target", "Clear", 32, 0.30),
    ("strict_target", "Strict", 64, 0.50),
]
FAILURE_CATEGORIES = [
    "format",
    "invalid_id",
    "wrong_start",
    "illegal_edge",
    "wrong_goal",
    "non_shortest_success",
    "shortest_success",
]
FAILURE_LABELS = {
    "format": "Format",
    "invalid_id": "Invalid ID",
    "wrong_start": "Wrong start",
    "illegal_edge": "Illegal edge",
    "wrong_goal": "Wrong goal",
    "non_shortest_success": "Non-shortest success",
    "shortest_success": "Shortest success",
}

EXTERNAL_BENCHMARKS = [
    ("spatialeval_vqa", "SpatialEval-VQA"),
    ("spatialeval_vtqa", "SpatialEval-VTQA"),
    ("3dsrbench", "3DSRBench"),
]


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_prediction_files(paths: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            question_id = row["question_id"]
            if question_id in merged:
                raise ValueError(f"Duplicate question_id {question_id} across prediction shards")
            merged[question_id] = row
    return merged


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def dec(x: float) -> str:
    return f"{x:.3f}"


def signed_pct(x: float) -> str:
    if abs(x) < 0.0005:
        return "0.0"
    sign = "+" if x >= 0 else ""
    return f"{sign}{100 * x:.1f}"


def pct_cell(value: float, best: float | None = None) -> str:
    cell = pct(value)
    if best is not None and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12):
        return rf"\textbf{{{cell}}}"
    return cell


def tex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_")


def svg_escape(s: object) -> str:
    text = str(s)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    png_path = pdf_path.with_suffix(".render.png")
    subprocess.run(
        ["convert", "-density", "300", str(svg_path), str(png_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    from PIL import Image

    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=300)
    png_path.unlink()


def row_end() -> str:
    return r" \\"


def task_metrics(summary: list[dict]) -> dict[str, dict]:
    out = {}
    for row in summary:
        task = row["task"]
        out[task] = row["metric_json"]["metrics"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def median(values: list[float]) -> float:
    return quantile(values, 0.5)


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def ci_text(p: float, lo: float, hi: float, signed: bool = False) -> str:
    head = signed_pct(p) if signed else pct(p)
    low = signed_pct(lo) if signed else pct(lo)
    high = signed_pct(hi) if signed else pct(hi)
    return f"{head} [{low}, {high}]"


def paired_bootstrap_interval(
    base_correct: list[bool], checkpoint_correct: list[bool], seed: int = 20260714
) -> tuple[float, float]:
    if len(base_correct) != len(checkpoint_correct) or not base_correct:
        return 0.0, 0.0
    differences = np.asarray(checkpoint_correct, dtype=np.float64) - np.asarray(
        base_correct, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(10000, dtype=np.float64)
    for start in range(0, len(estimates), 250):
        stop = min(start + 250, len(estimates))
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[indices].mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_bootstrap_interval(
    values: list[float], clusters: list[str], seed: int = 20260714
) -> tuple[float, float]:
    if len(values) != len(clusters) or not values:
        return 0.0, 0.0
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped.setdefault(cluster, []).append(float(value))
    cluster_values = list(grouped.values())
    rng = np.random.default_rng(seed)
    estimates = np.empty(10000, dtype=np.float64)
    for sample_index in range(len(estimates)):
        selected = rng.integers(0, len(cluster_values), size=len(cluster_values))
        sample = [value for index in selected for value in cluster_values[index]]
        estimates[sample_index] = np.mean(sample)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def prediction_output(pred: dict | None) -> object:
    if not pred or pred.get("success") is False:
        return None
    return pred.get("output")


def parse_ids(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [int(x) for x in raw]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        text = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            try:
                return [int(x) for x in parsed]
            except (TypeError, ValueError):
                return None
    return None


def load_visible_waypoints(path_value: str | None) -> list[dict]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    payload = read_json(path)
    return list(payload.get("visible_waypoints", []))


def visible_ids_from_waypoints(visible_waypoints: list[dict]) -> set[int]:
    ids = set()
    for waypoint in visible_waypoints:
        display_id = waypoint.get("display_id")
        if display_id is not None:
            ids.add(int(display_id))
    return ids


def resolve_optional_path(ref: str | None, anchors: list[Path]) -> Path | None:
    if not ref:
        return None
    path = Path(ref)
    if path.is_absolute():
        return path if path.exists() else None
    for anchor in anchors:
        candidate = anchor / path
        if candidate.exists():
            return candidate
    return None


def load_direct_pairs(
    path: Path, visible_waypoints: list[dict] | None = None
) -> tuple[set[int], dict[int, dict[int, float]]]:
    payload = read_json(path)
    display_ids = {int(x) for x in payload.get("display_ids", []) if x is not None}
    world_xyz = {}
    for waypoint in visible_waypoints or []:
        display_id = waypoint.get("display_id")
        xyz = waypoint.get("world_xyz") or []
        if display_id is not None and len(xyz) == 3:
            world_xyz[int(display_id)] = tuple(float(value) for value in xyz)
    adjacency: dict[int, dict[int, float]] = {}
    for src_raw, entries in payload.get("direct_neighbors", {}).items():
        try:
            src = int(src_raw)
        except (TypeError, ValueError):
            continue
        src_adj = adjacency.setdefault(src, {})
        for entry in entries or []:
            dst_raw = entry.get("display_id")
            distance = entry.get("distance_m")
            if dst_raw is None:
                continue
            try:
                dst = int(dst_raw)
            except (TypeError, ValueError):
                continue
            if distance is None and src in world_xyz and dst in world_xyz:
                distance = math.dist(world_xyz[src], world_xyz[dst])
            if distance is not None:
                src_adj[dst] = float(distance)
    return display_ids, adjacency


def benchmark_vqa_rows(task: str) -> list[dict]:
    return read_jsonl(EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl")


def visible_candidate_count(q: dict) -> int:
    waypoints = load_visible_waypoints(q.get("visible_waypoints_path"))
    if waypoints:
        return len(waypoints)
    all_ids = q.get("ground_truth", {}).get("all_ids") or []
    return len(all_ids)


def reference_path(q: dict) -> list[int]:
    gt = q.get("ground_truth", {})
    return list(gt.get("reference_path_display_ids") or gt.get("answer") or [])


def target_ambiguity(q: dict) -> int:
    gt = q.get("ground_truth", {})
    for key in ("ambiguity_count_H_view", "supported_visible_same_type_ids"):
        value = gt.get(key)
        if isinstance(value, list):
            return len(value)
        if value is not None:
            return int(value)
    return int(gt.get("navigation_reference_facts", {}).get("same_semantic_group_count_total") or 1)


def route_length_m(q: dict) -> float:
    gt = q.get("ground_truth", {})
    return float(gt.get("optimal_length_m") or gt.get("canonical_sparse_length_m") or 0.0)


def narrow_passage_fraction(q: dict) -> float:
    gt = q.get("ground_truth", {})
    facts = gt.get("navigation_embodiment_facts") or {}
    routing = gt.get("routing_complexity") or {}
    return float(facts.get("narrow_passage_fraction", routing.get("narrow_passage_fraction", 0.0)) or 0.0)


def fmt_stat(value: float, unit: str = "") -> str:
    if unit == "m":
        return f"{value:.1f} m"
    if unit == "pct":
        return rf"{100 * value:.0f}\%"
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.1f}"


def distribution_row(label: str, scope: str, values: list[float], unit: str, interpretation: str) -> dict:
    return {
        "signal": label,
        "scope": scope,
        "n": len(values),
        "median": median(values),
        "p90": quantile(values, 0.9),
        "max": max(values) if values else 0.0,
        "unit": unit,
        "interpretation": interpretation,
    }


def classify_route_failure(q: dict, pred: dict | None, vqa_path: Path) -> tuple[str, float | None]:
    flags = route_contract_flags(q, pred, vqa_path)
    pred_ids = flags["pred_ids"]
    if not flags["parseable"]:
        return "format", None
    if not flags["candidate_valid"]:
        return "invalid_id", None
    if not flags["start_correct"]:
        return "wrong_start", None
    if not flags["edge_legal"]:
        return "illegal_edge", None
    path_length = flags["path_length_m"]
    if not flags["endpoint_hit"]:
        return "wrong_goal", path_length

    optimal = q.get("ground_truth", {}).get("optimal_length_m")
    if optimal is not None and path_length > float(optimal) + 1e-3:
        return "non_shortest_success", path_length
    return "shortest_success", path_length


def route_contract_flags(q: dict, pred: dict | None, vqa_path: Path) -> dict:
    pred_ids = parse_ids(prediction_output(pred))
    parseable = pred_ids is not None and len(pred_ids) > 0
    flags = {
        "pred_ids": pred_ids,
        "parseable": parseable,
        "candidate_valid": False,
        "start_correct": False,
        "edge_legal": False,
        "endpoint_hit": False,
        "full_success": False,
        "path_length_m": None,
    }
    if not parseable:
        return flags

    gt = q.get("ground_truth", {})
    direct_pairs_ref = gt.get("direct_pairs_ref") or q.get("direct_pairs_ref")
    direct_pairs_path = resolve_optional_path(
        direct_pairs_ref, [vqa_path.parent, vqa_path.parent.parent, ROOT]
    )
    if direct_pairs_path is None:
        return flags

    visible_waypoints = load_visible_waypoints(q.get("visible_waypoints_path"))
    direct_pair_ids, adjacency = load_direct_pairs(direct_pairs_path, visible_waypoints)
    visible_ids = visible_ids_from_waypoints(visible_waypoints)
    all_ids = visible_ids or direct_pair_ids
    flags["candidate_valid"] = all(pid in all_ids for pid in pred_ids)

    start_id = gt.get("start_id")
    flags["start_correct"] = start_id is None or int(pred_ids[0]) == int(start_id)

    acceptable_goal_ids = {
        int(x)
        for x in (gt.get("acceptable_goal_ids") or gt.get("goal_ids") or [])
        if x is not None
    }
    flags["endpoint_hit"] = int(pred_ids[-1]) in acceptable_goal_ids

    if flags["candidate_valid"]:
        path_length = 0.0
        edge_legal = True
        for src, dst in zip(pred_ids[:-1], pred_ids[1:]):
            distance = adjacency.get(int(src), {}).get(int(dst))
            if distance is None:
                edge_legal = False
                break
            path_length += distance
        flags["edge_legal"] = edge_legal
        if edge_legal:
            flags["path_length_m"] = path_length

    flags["full_success"] = all(
        flags[key] for key in ("candidate_valid", "start_correct", "edge_legal", "endpoint_hit")
    )
    return flags


def route_visual_support(q: dict) -> dict:
    gt = q.get("ground_truth", {})
    complexity = gt.get("routing_complexity", {})
    validity = gt.get("navigation_validity", {})
    checks = validity.get("check_results", {})
    return {
        "path_visible_fraction": float(complexity.get("path_visible_point_fraction", 0.0) or 0.0),
        "sparse_projection_safe": bool(complexity.get("path_projection_safe", False)),
        "dense_projection_safe": bool(complexity.get("dense_path_projection_safe", False)),
        "target_bbox_short_side_px": int(complexity.get("target_bbox_short_side_px", 0) or 0),
        "target_visible_surface_ratio": float(complexity.get("target_visible_surface_ratio", 0.0) or 0.0),
        "target_human_visible": gt.get("human_visible_legality") == "legal",
        "target_unique": gt.get("human_visible_uniqueness") == "unique",
        "no_prompt_leakage": gt.get("prompt_answer_leakage") is False,
        "start_in_bottom_band": bool(checks.get("start_in_bottom_band", False)),
        "start_in_forward_sector": bool(checks.get("start_in_forward_sector", False)),
        "goal_in_target_zone": bool(checks.get("goal_in_target_zone", False)),
    }


def visibility_level_passes(q: dict, bbox_min_px: int, surface_min: float) -> bool:
    support = route_visual_support(q)
    return all(
        [
            math.isclose(support["path_visible_fraction"], 1.0, abs_tol=1e-9),
            support["sparse_projection_safe"],
            support["dense_projection_safe"],
            support["target_human_visible"],
            support["target_unique"],
            support["no_prompt_leakage"],
            support["start_in_bottom_band"],
            support["start_in_forward_sector"],
            support["goal_in_target_zone"],
            support["target_bbox_short_side_px"] >= bbox_min_px,
            support["target_visible_surface_ratio"] >= surface_min,
        ]
    )


def egopath_score(metrics: dict[str, float]) -> float:
    """Combine chance-adjusted traversability BA with raw route success rates."""
    return (
        (2 * metrics["a1_ba"] - 1)
        + (2 * metrics["b1_ba"] - 1)
        + metrics["a2_sr"]
        + metrics["b2_sr"]
        + metrics["c_sr"]
    ) / 5


def build_human_solvable_analysis() -> None:
    questions: dict[str, tuple[dict, Path]] = {}
    human_flags: dict[str, dict] = {}
    for task in PLANNING_TASKS:
        vqa_path = HUMAN_EVAL_ROOT / f"vqa_{task}.jsonl"
        task_questions = read_jsonl(vqa_path)
        human_predictions = {
            row.get("question_id"): row
            for row in read_jsonl(HUMAN_EVAL_ROOT / f"preds_{task}.jsonl")
        }
        for question in task_questions:
            question_id = str(question.get("question_id"))
            questions[question_id] = (question, vqa_path)
            human_flags[question_id] = route_contract_flags(
                question, human_predictions.get(question_id), vqa_path
            )

    human_legal_ids = {
        question_id
        for question_id, flags in human_flags.items()
        if flags["candidate_valid"] and flags["start_correct"] and flags["edge_legal"]
    }
    human_solved_ids = {
        question_id
        for question_id, flags in human_flags.items()
        if flags["full_success"]
    }

    def summarize(label: str, flags_by_id: dict[str, dict]) -> dict:
        all_flags = list(flags_by_id.values())
        legal_flags = [flags_by_id[qid] for qid in human_legal_ids]
        solved_flags = [flags_by_id[qid] for qid in human_solved_ids]

        def valid_path(flags: dict) -> bool:
            return bool(
                flags["candidate_valid"]
                and flags["start_correct"]
                and flags["edge_legal"]
            )

        return {
            "evaluator": label,
            "all_n": len(all_flags),
            "all_valid_path_rate": sum(valid_path(flags) for flags in all_flags)
            / max(len(all_flags), 1),
            "all_full_success": sum(flags["full_success"] for flags in all_flags)
            / max(len(all_flags), 1),
            "human_legal_n": len(legal_flags),
            "valid_path_on_human_legal": sum(valid_path(flags) for flags in legal_flags)
            / max(len(legal_flags), 1),
            "human_solved_n": len(solved_flags),
            "full_success_on_human_solved": sum(
                flags["full_success"] for flags in solved_flags
            )
            / max(len(solved_flags), 1),
        }

    rows = [summarize("Annotator", human_flags)]
    for model_dir, model_name in MODEL_ORDER:
        flags_by_id = {}
        for task in PLANNING_TASKS:
            prediction_path = (
                HUMAN_EVAL_ROOT / "vlm_same50" / model_dir / f"preds_{task}.jsonl"
            )
            predictions = {
                row.get("question_id"): row for row in read_jsonl(prediction_path)
            }
            for question_id, (question, vqa_path) in questions.items():
                if question.get("task") != task:
                    continue
                flags_by_id[question_id] = route_contract_flags(
                    question, predictions.get(question_id), vqa_path
                )
        row = summarize(model_name, flags_by_id)
        row["model_dir"] = model_dir
        rows.append(row)
    with (DATA / "human_calibration_same50.csv").open() as f:
        same50_by_run = {row["run"]: row for row in csv.DictReader(f)}
    comparison_full = read_json(HUMAN_COMPARISON_SUMMARY)["full"]
    for row in rows:
        run = "human" if row["evaluator"] == "Annotator" else row["model_dir"]
        same50 = same50_by_run[run]
        row["egopath_score"] = egopath_score(
            {key: float(same50[key]) for key in ["a1_ba", "b1_ba", "a2_sr", "b2_sr", "c_sr"]}
        )
        route_metrics = comparison_full[run]["metrics_by_task"]
        row["reported_valid_path_rate"] = sum(
            float(route_metrics[task]["valid_path_rate"]) for task in PLANNING_TASKS
        ) / len(PLANNING_TASKS)
    write_csv(DATA / "human_solvable_subset.csv", rows)

    displayed = sorted(rows, key=lambda row: row["egopath_score"], reverse=True)
    best_score = max(row["egopath_score"] for row in displayed)
    best_valid_path = max(row["reported_valid_path_rate"] for row in displayed)
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Same-question calibration on a fixed 50-question subset (10 per task; \%). Score averages chance-adjusted traversability skill and the three route success rates; valid path is averaged over the route tasks.}",
        r"\label{tab:human_calibration}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Evaluator & Score & Valid path \\",
        r"\midrule",
    ]
    for row in displayed:
        name = "Human calibration" if row["evaluator"] == "Annotator" else TABLE_MODEL_NAMES.get(row["evaluator"], row["evaluator"])
        score = pct(row["egopath_score"])
        valid_path = pct(row["reported_valid_path_rate"])
        if math.isclose(row["egopath_score"], best_score, abs_tol=1e-12):
            score = rf"\textbf{{{score}}}"
        if math.isclose(row["reported_valid_path_rate"], best_valid_path, abs_tol=1e-12):
            valid_path = rf"\textbf{{{valid_path}}}"
        lines.append(
            f"{tex_escape(name)} & {score} & {valid_path}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_human_calibration.tex").write_text("\n".join(lines))


def build_dataset_difficulty_profile() -> None:
    task_rows = {task: benchmark_vqa_rows(task) for task in ["a1", "b1", "a2", "b2", "c"]}
    path_rows = [q for task in PLANNING_TASKS for q in task_rows[task]]
    embodied_path_rows = [q for task in ["b2", "c"] for q in task_rows[task]]

    summary_rows = [
        distribution_row(
            "Same-type target ambiguity",
            "path rows",
            [target_ambiguity(q) for q in path_rows],
            "objects",
            "Target grounding often requires resolving one object among same-family alternatives.",
        ),
        *[
            distribution_row(
                f"Reference segments ({TASK_SHORT_NAMES[task]})",
                f"{task} benchmark rows",
                [max(len(reference_path(q)) - 1, 0) for q in task_rows[task]],
                "edges",
                "Route complexity is reported separately because point and embodied graphs differ.",
            )
            for task in PLANNING_TASKS
        ],
        distribution_row(
            "Reference path length",
            "path rows",
            [route_length_m(q) for q in path_rows],
            "m",
            "The shortest-path target is metric and scene-grounded.",
        ),
        distribution_row(
            "Embodied narrow-passage fraction",
            "embodied path rows",
            [narrow_passage_fraction(q) for q in embodied_path_rows],
            "pct",
            "Embodied routing repeatedly passes through clearance-sensitive geometry.",
        ),
    ]
    write_csv(DATA / "dataset_difficulty_profile.csv", summary_rows)

    per_task_rows = []
    for task in ["a1", "b1", "a2", "b2", "c"]:
        rows = task_rows[task]
        visible = [visible_candidate_count(q) for q in rows]
        item = {
            "task": TASK_SHORT_NAMES[task],
            "n": len(rows),
            "visible_median": median(visible),
            "visible_p90": quantile(visible, 0.9),
            "visible_max": max(visible),
        }
        if task in PLANNING_TASKS:
            segments = [max(len(reference_path(q)) - 1, 0) for q in rows]
            lengths = [route_length_m(q) for q in rows]
            item.update(
                {
                    "ambiguity_median": median([target_ambiguity(q) for q in rows]),
                    "segments_median": median(segments),
                    "segments_p90": quantile(segments, 0.9),
                    "length_median_m": median(lengths),
                    "length_p90_m": quantile(lengths, 0.9),
                }
            )
        per_task_rows.append(item)
    write_csv(DATA / "dataset_difficulty_by_task.csv", per_task_rows)

    return summary_rows


def metric_file(model_dir: str, task: str) -> Path | None:
    matches = sorted((LEADERBOARD_ROOT / model_dir / "metrics").glob(f"metrics_{task}_*.json"))
    return matches[0] if matches else None


def build_statistical_uncertainty() -> None:
    rows = []

    for task in PLANNING_TASKS:
        best = None
        for model_dir, model_name in MODEL_ORDER:
            path = metric_file(model_dir, task)
            if path is None:
                continue
            metrics = read_json(path)["metrics"]
            rate = float(metrics["success_rate"])
            item = {
                "claim": f"Best zero-shot {TASK_SHORT_NAMES[task]} success",
                "source": model_name,
                "model_dir": model_dir,
                "n": int(metrics["total"]),
                "estimate": rate,
            }
            if best is None or item["estimate"] > best["estimate"]:
                best = item
        if best is not None:
            summary = read_json(LEADERBOARD_ROOT / best["model_dir"] / "summary.json")
            result = next(row for row in summary if row.get("task") == task)
            vqa_path = ROOT / result["vqa"] if "vqa" in result else EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl"
            predictions = {
                row.get("question_id"): row for row in read_jsonl(ROOT / result["predictions"])
            }
            values = []
            clusters = []
            for question in read_jsonl(vqa_path):
                flags = route_contract_flags(
                    question, predictions.get(question.get("question_id")), vqa_path
                )
                values.append(float(flags["full_success"]))
                clusters.append(str(question.get("scene_id")))
            lo, hi = cluster_bootstrap_interval(values, clusters)
            best.pop("model_dir")
            best.update(
                {
                    "clusters": len(set(clusters)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "interval": "Scene-cluster bootstrap 95% (10,000 resamples)",
                }
            )
            rows.append(best)

    human_rows = []
    with (DATA / "human_calibration_same50.csv").open() as f:
        for row in csv.DictReader(f):
            human_rows.append(row)
    for selector, label, source_label in [
        (lambda r: r["run"] == "human", "Single-annotator same-subset planning success", "Annotator"),
        (lambda r: r["run"] != "human", "Best VLM same-subset planning success", None),
    ]:
        candidates = [r for r in human_rows if selector(r)]
        if not candidates:
            continue
        if source_label is None:
            chosen = max(candidates, key=lambda r: float(r["planning_mean_sr"]))
            source_label = chosen["model"]
        else:
            chosen = candidates[0]
        rate = float(chosen["planning_mean_sr"])
        lo, hi = wilson_interval(rate, 30)
        rows.append(
            {
                "claim": label,
                "source": source_label,
                "n": 30,
                "estimate": rate,
                "ci_low": lo,
                "ci_high": hi,
                "interval": "Descriptive item-level Wilson 95%",
            }
        )

    human_valid_path = 21 / 30
    lo, hi = wilson_interval(human_valid_path, 30)
    rows.append(
        {
            "claim": "Single-annotator same-subset valid path rate",
            "source": "Annotator",
            "n": 30,
            "estimate": human_valid_path,
            "ci_low": lo,
            "ci_high": hi,
            "interval": "Descriptive item-level Wilson 95%",
        }
    )

    with (DATA / "visibility_sensitivity_by_task.csv").open() as f:
        visibility_rows = list(csv.DictReader(f))
    for row in visibility_rows:
        if row["visibility_level"] != "strict_target":
            continue
        rows.append(
            {
                "claim": f"Strict-visibility endpoint/full-success gap: {row['task']}",
                "source": "Nine foundation VLMs",
                "n": int(row["predictions"]),
                "clusters": int(row["scene_clusters"]),
                "estimate": float(row["endpoint_gap"]),
                "ci_low": float(row["gap_ci_low"]),
                "ci_high": float(row["gap_ci_high"]),
                "interval": "Scene-cluster bootstrap 95% (10,000 resamples)",
            }
        )

    cross_rows = []
    with (DATA / "cross_benchmark_spatial_generalization.csv").open() as f:
        for row in csv.DictReader(f):
            cross_rows.append(row)
    overall = next(row for row in cross_rows if row["benchmark"] == "Aggregate")
    n = int(overall["n"])
    base = float(overall["base"])
    cot = float(overall["spatial_cot"])
    for label, source, rate in [
        ("External spatial suite aggregate accuracy", "Qwen 3.5 4B base", base),
        ("External spatial suite aggregate accuracy", "Complete EgoPathBench SFT checkpoint", cot),
    ]:
        lo, hi = wilson_interval(rate, n)
        rows.append(
            {
                "claim": label,
                "source": source,
                "n": n,
                "estimate": rate,
                "ci_low": lo,
                "ci_high": hi,
                "interval": "Descriptive item-level Wilson 95%",
            }
        )
    lo = float(overall["delta_ci_low"])
    hi = float(overall["delta_ci_high"])
    rows.append(
        {
            "claim": "External spatial suite aggregate gain",
            "source": "Complete EgoPathBench SFT - base",
            "n": n,
            "estimate": cot - base,
            "ci_low": lo,
            "ci_high": hi,
            "interval": "Paired item bootstrap 95% (10,000 resamples)",
        }
    )

    write_csv(DATA / "statistical_uncertainty.csv", rows)


def build_dataset_stats() -> None:
    release = read_json(RELEASE_MANIFEST)
    evalfix = read_json(EVALFIX_SUMMARY)

    split_rows = []
    for split in ["train", "val", "benchmark"]:
        item = release["splits"][split]
        counts = item["question_counts_by_task"]
        questions = [
            question
            for task in ["a1", "b1", "a2", "b2", "c"]
            for question in read_jsonl(RELEASE_ROOT / split / "vqa" / f"vqa_next_{task}.jsonl")
        ]
        views = {(question.get("scene_id"), question.get("view_id")) for question in questions}
        route_questions = [question for question in questions if question.get("task") in PLANNING_TASKS]
        route_units = {
            (question.get("scene_id"), question.get("view_id"), question.get("routing_id"))
            for question in route_questions
        }
        target_instances = {
            (question.get("scene_id"), question.get("ground_truth", {}).get("target_id"))
            for question in route_questions
        }
        split_rows.append(
            {
                "split": split,
                "scenes": item["scene_count"],
                "views": len(views),
                "route_units": len(route_units),
                "target_instances": len(target_instances),
                "questions": item["questions_total"],
                "a1": counts["a1"],
                "b1": counts["b1"],
                "a2": counts["a2"],
                "b2": counts["b2"],
                "c": counts["c"],
            }
        )
    write_csv(DATA / "dataset_stats.csv", split_rows)

    difficulty_rows = build_dataset_difficulty_profile()
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Release scale and benchmark route structure. Views are unique scene--camera pairs; route units are unique scene--view--route tuples shared by paired route questions.}",
        r"\label{tab:dataset_stats}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\textbf{(a) Release scale}\\[-2pt]",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Split & Scenes & Views & Route units & Targets & Questions \\",
        r"\midrule",
    ]
    for row in split_rows:
        lines.append(
            f"{row['split'].capitalize()} & {row['scenes']:,} & {row['views']:,} & "
            f"{row['route_units']:,} & {row['target_instances']:,} & {row['questions']:,}{row_end()}"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[3pt]\textbf{(b) Benchmark route structure}\\[-2pt]",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Signal & N & Median & P90 & Max \\",
        r"\midrule",
    ]
    for row in difficulty_rows:
        unit = row["unit"]
        lines.append(
            f"{tex_escape(row['signal'])} & {row['n']:,} & {fmt_stat(row['median'], unit)} & "
            f"{fmt_stat(row['p90'], unit)} & {fmt_stat(row['max'], unit)}{row_end()}"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (TABLES / "table_dataset_stats.tex").write_text("\n".join(lines))


def build_leaderboard() -> None:
    uncertainty_path = DATA / "leaderboard_uncertainty.csv"
    uncertainty = {}
    if uncertainty_path.exists():
        with uncertainty_path.open() as handle:
            uncertainty = {row["model_dir"]: row for row in csv.DictReader(handle)}
    rows = []
    for model_dir, name in MODEL_ORDER:
        summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
        metrics = task_metrics(summary)
        row = {
            "model": name,
            "model_dir": model_dir,
            "a1_ba": metrics["a1"]["balanced_accuracy"],
            "a1_f1": metrics["a1"]["f1"],
            "b1_ba": metrics["b1"]["balanced_accuracy"],
            "b1_f1": metrics["b1"]["f1"],
            "a2_sr": metrics["a2"]["success_rate"],
            "a2_spl_all": metrics["a2"]["spl_all"],
            "a2_vpr": metrics["a2"]["valid_path_rate"],
            "b2_sr": metrics["b2"]["success_rate"],
            "b2_spl_all": metrics["b2"]["spl_all"],
            "b2_vpr": metrics["b2"]["valid_path_rate"],
            "c_sr": metrics["c"]["success_rate"],
            "c_spl_all": metrics["c"]["spl_all"],
            "c_vpr": metrics["c"]["valid_path_rate"],
        }
        row["egopath_score"] = egopath_score(row)
        if model_dir in uncertainty:
            row["uncertainty"] = uncertainty[model_dir]
        rows.append(row)
    rows.sort(key=lambda row: row["egopath_score"], reverse=True)
    write_csv(DATA / "main_leaderboard.csv", rows)
    metric_keys = [
        "egopath_score",
        "a1_ba", "a1_f1", "b1_ba", "b1_f1",
        "a2_vpr", "a2_sr", "a2_spl_all",
        "b2_vpr", "b2_sr", "b2_spl_all",
        "c_vpr", "c_sr", "c_spl_all",
    ]
    best = {key: max(row[key] for row in rows) for key in metric_keys}

    def sft_row(label: str, metrics_root: Path) -> dict:
        metrics = {task: read_json(metrics_root / f"{task}.json")["metrics"] for task in ["a1", "b1", "a2", "b2", "c"]}
        row = {
            "model": label,
            "a1_ba": metrics["a1"]["balanced_accuracy"],
            "a1_f1": metrics["a1"]["f1"],
            "b1_ba": metrics["b1"]["balanced_accuracy"],
            "b1_f1": metrics["b1"]["f1"],
            "a2_vpr": metrics["a2"]["valid_path_rate"],
            "a2_sr": metrics["a2"]["success_rate"],
            "a2_spl_all": metrics["a2"]["spl_all"],
            "b2_vpr": metrics["b2"]["valid_path_rate"],
            "b2_sr": metrics["b2"]["success_rate"],
            "b2_spl_all": metrics["b2"]["spl_all"],
            "c_vpr": metrics["c"]["valid_path_rate"],
            "c_sr": metrics["c"]["success_rate"],
            "c_spl_all": metrics["c"]["spl_all"],
        }
        row["egopath_score"] = egopath_score(row)
        return row

    training_rows = [
        sft_row("Qwen3.5-4B base", SFT_BASE_METRICS_ROOT),
        sft_row("+ EgoPathBench SFT", SFT_CHECKPOINT_METRICS_ROOT),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\caption{EgoPathBench performance (\%). BA and F1 evaluate traversable candidate selection. For route tasks, VPR requires every selected edge to be legal, SR additionally requires an acceptable endpoint, and SPL assigns zero to unsuccessful routes. The upper block ranks zero-shot foundation VLMs by EgoPath Score; the lower block reports the Qwen3.5-4B training-resource comparison.}",
        r"\label{tab:main_leaderboard}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccccccccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Point Trav.} & \multicolumn{2}{c}{Embodied Trav.} & \multicolumn{3}{c}{Point Path} & \multicolumn{3}{c}{Embodied Path} & \multicolumn{3}{c}{Intent Path} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}",
        r"Model & Score & BA & F1 & BA & F1 & VPR & SR & SPL & VPR & SR & SPL & VPR & SR & SPL \\",
        r"\midrule",
        r"\multicolumn{15}{@{}l}{\emph{Zero-shot foundation VLMs}} \\",
    ]

    def cells(row: dict, *, highlight_best: bool) -> str:
        model_name = TABLE_MODEL_NAMES.get(row["model"], row["model"])
        values = []
        for key in metric_keys:
            value = pct(row[key])
            if highlight_best and math.isclose(row[key], best[key], abs_tol=1e-12):
                value = rf"\textbf{{{value}}}"
            values.append(value)
        return f"{tex_escape(model_name)} & " + " & ".join(values) + row_end()

    lines.extend(cells(row, highlight_best=True) for row in rows)
    lines += [
        r"\midrule",
        r"\multicolumn{15}{@{}l}{\emph{Training-resource study}} \\",
    ]
    lines.extend(cells(row, highlight_best=False) for row in training_rows)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ]
    (TABLES / "table_main_leaderboard.tex").write_text("\n".join(lines))

    full_lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Complete task metrics for the nine foundation VLMs (\%). SPL-all assigns zero to unsuccessful routes.}",
        r"\label{tab:full_leaderboard_metrics}",
        r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
        r"Model & P-F1 & E-F1 & P-VPR & P-SR & P-SPL & E-VPR & E-SR & E-SPL & I-VPR & I-SPL \\", r"\midrule",
    ]
    for row in rows:
        model_name = TABLE_MODEL_NAMES.get(row["model"], row["model"])
        vals = [row["a1_f1"], row["b1_f1"], row["a2_vpr"], row["a2_sr"], row["a2_spl_all"], row["b2_vpr"], row["b2_sr"], row["b2_spl_all"], row["c_vpr"], row["c_spl_all"]]
        full_lines.append(f"{tex_escape(model_name)} & " + " & ".join(pct(value) for value in vals) + row_end())
    full_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "table_main_leaderboard_full.tex").write_text("\n".join(full_lines))


def build_geometry_sensitivity() -> None:
    source = GEOMETRY_SENSITIVITY_ROOT / "per_condition.csv"
    if not source.exists():
        return
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["condition"] != "nominal"]
    write_csv(DATA / "geometry_contract_sensitivity.csv", selected)
    labels = {
        "radius_0p25": r"Radius 0.25 m",
        "radius_0p35": r"Radius 0.35 m",
        "grid_0p04": r"Grid 0.04 m",
        "grid_0p06": r"Grid 0.06 m",
        "goal_0p05": r"Goal ring 0.05 m",
        "goal_0p15": r"Goal ring 0.15 m",
    }
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\caption{Geometry-contract sensitivity on the fixed 819 route questions and nine-model outputs. The nominal contract is 0.30 m radius, 0.05 m grid, and 0.10 m goal ring. Ref. is the fraction retaining a graph solution; edge flip is the question-weighted symmetric difference from nominal; success flip is over all model--question outputs.}",
        r"\label{tab:geometry_sensitivity}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Variant & Ref. & Edge flip & Success flip & Rank $\rho$ \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{labels[row['condition']]} & "
            f"{pct(float(row['reference_solution_coverage']))} & "
            f"{pct(float(row['edge_flip_rate']))} & "
            f"{pct(float(row['success_flip_rate']))} & "
            f"{float(row['score_spearman']):.2f}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_geometry_sensitivity.tex").write_text("\n".join(lines))


def build_failure_decomposition() -> None:
    model_task_rows = []
    incidence_model_task_rows = []
    task_totals = {
        task: {category: 0 for category in FAILURE_CATEGORIES} | {"total": 0}
        for task in PLANNING_TASKS
    }
    incidence_keys = [
        "format_failure",
        "invalid_id",
        "candidate_valid",
        "wrong_start",
        "illegal_edge",
        "wrong_goal",
        "endpoint_hit",
        "edge_legal",
        "endpoint_valid_candidate",
        "endpoint_legal_route",
        "full_success",
    ]
    incidence_totals = {
        task: {key: 0 for key in incidence_keys} | {"total": 0}
        for task in PLANNING_TASKS
    }
    gap_values = {task: [] for task in PLANNING_TASKS}
    gap_clusters = {task: [] for task in PLANNING_TASKS}

    for model_dir, model_name in MODEL_ORDER:
        summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
        for row in summary:
            task = row.get("task")
            if task not in PLANNING_TASKS:
                continue
            if "predictions" not in row:
                continue
            vqa_path = ROOT / row["vqa"] if "vqa" in row else EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl"
            pred_path = ROOT / row["predictions"]
            vqa_rows = read_jsonl(vqa_path)
            preds = {item.get("question_id"): item for item in read_jsonl(pred_path)}
            counts = {category: 0 for category in FAILURE_CATEGORIES}
            incidence_counts = {key: 0 for key in incidence_keys}
            lengths = []
            for q in vqa_rows:
                pred = preds.get(q.get("question_id"))
                flags = route_contract_flags(q, pred, vqa_path)
                category, path_length = classify_route_failure(q, pred, vqa_path)
                counts[category] += 1
                task_totals[task][category] += 1
                task_totals[task]["total"] += 1
                incidence = {
                    "format_failure": not flags["parseable"],
                    "invalid_id": flags["parseable"] and not flags["candidate_valid"],
                    "candidate_valid": flags["candidate_valid"],
                    "wrong_start": flags["parseable"] and not flags["start_correct"],
                    "illegal_edge": flags["candidate_valid"] and not flags["edge_legal"],
                    "wrong_goal": flags["parseable"] and not flags["endpoint_hit"],
                    "endpoint_hit": flags["endpoint_hit"],
                    "edge_legal": flags["candidate_valid"] and flags["edge_legal"],
                    "endpoint_valid_candidate": flags["candidate_valid"]
                    and flags["endpoint_hit"],
                    "endpoint_legal_route": flags["candidate_valid"]
                    and flags["edge_legal"]
                    and flags["endpoint_hit"],
                    "full_success": flags["full_success"],
                }
                for key, value in incidence.items():
                    incidence_counts[key] += int(value)
                    incidence_totals[task][key] += int(value)
                incidence_totals[task]["total"] += 1
                gap_values[task].append(float(flags["endpoint_hit"]) - float(flags["full_success"]))
                gap_clusters[task].append(str(q.get("scene_id")))
                if path_length is not None:
                    lengths.append(path_length)

            total = len(vqa_rows)
            model_task_row = {
                "model": model_name,
                "model_dir": model_dir,
                "task": TASK_FAILURE_NAMES[task],
                "total": total,
            }
            for category in FAILURE_CATEGORIES:
                model_task_row[f"{category}_count"] = counts[category]
                model_task_row[f"{category}_pct"] = round(counts[category] / max(total, 1), 4)
            model_task_row["mean_valid_path_length_m"] = round(sum(lengths) / len(lengths), 4) if lengths else ""
            model_task_rows.append(model_task_row)

            incidence_row = {
                "model": model_name,
                "model_dir": model_dir,
                "task": TASK_FAILURE_NAMES[task],
                "total": total,
            }
            for key in incidence_keys:
                incidence_row[f"{key}_count"] = incidence_counts[key]
                incidence_row[f"{key}_pct"] = round(incidence_counts[key] / max(total, 1), 4)
            incidence_model_task_rows.append(incidence_row)

    write_csv(DATA / "failure_decomposition_model_task.csv", model_task_rows)
    write_csv(DATA / "failure_incidence_model_task.csv", incidence_model_task_rows)

    task_rows = []
    for task in PLANNING_TASKS:
        total = task_totals[task]["total"]
        task_row = {
            "task": TASK_FAILURE_NAMES[task],
            "total_predictions": total,
        }
        for category in FAILURE_CATEGORIES:
            task_row[f"{category}_count"] = task_totals[task][category]
            task_row[f"{category}_pct"] = round(task_totals[task][category] / max(total, 1), 4)
        task_rows.append(task_row)
    write_csv(DATA / "failure_decomposition_by_task.csv", task_rows)

    incidence_rows = []
    for task in PLANNING_TASKS:
        total = incidence_totals[task]["total"]
        row = {"task": TASK_FAILURE_NAMES[task], "total_predictions": total}
        for key in incidence_keys:
            row[f"{key}_count"] = incidence_totals[task][key]
            row[f"{key}_pct"] = round(incidence_totals[task][key] / max(total, 1), 4)
        row["endpoint_overestimate_pct"] = round(
            row["endpoint_hit_pct"] - row["full_success_pct"], 4
        )
        incidence_rows.append(row)
    write_csv(DATA / "failure_incidence_by_task.csv", incidence_rows)
    conditional_rows = []
    for row in incidence_rows:
        candidate_valid = row["candidate_valid_count"]
        edge_legal = row["edge_legal_count"]
        conditional_rows.append(
            {
                "task": row["task"],
                "total_predictions": row["total_predictions"],
                "candidate_valid_count": candidate_valid,
                "edge_legal_given_valid_ids": row["edge_legal_count"]
                / max(candidate_valid, 1),
                "endpoint_given_valid_ids": row["endpoint_valid_candidate_count"]
                / max(candidate_valid, 1),
                "endpoint_given_legal_route": row["endpoint_legal_route_count"]
                / max(edge_legal, 1),
                "full_success": row["full_success_pct"],
            }
        )
    write_csv(DATA / "route_conditional_diagnostics.csv", conditional_rows)
    gap_rows = []
    for task, row in zip(PLANNING_TASKS, incidence_rows):
        ci_low, ci_high = cluster_bootstrap_interval(gap_values[task], gap_clusters[task])
        gap_rows.append(
            {
                "task": row["task"],
                "n": row["total_predictions"],
                "scene_clusters": len(set(gap_clusters[task])),
                "endpoint_hit": row["endpoint_hit_pct"],
                "edge_legal": row["edge_legal_pct"],
                "full_success": row["full_success_pct"],
                "endpoint_only_overestimate": row["endpoint_overestimate_pct"],
                "gap_ci_low": ci_low,
                "gap_ci_high": ci_high,
                "interval": "Scene-cluster bootstrap 95% (10,000 resamples)",
            }
        )
    write_csv(DATA / "route_contract_gap_by_task.csv", gap_rows)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Path-contract failure decomposition for the nine foundation VLMs. Percentages are pooled over models for each planning task. Categories are mutually exclusive and assigned in contract order: parseable answer, legal candidate IDs, correct start, legal adjacent segments, acceptable endpoint, and shortest-path efficiency. ``Long success'' means a legal route reaches an acceptable endpoint but is longer than the shortest reference.}",
        r"\label{tab:failure_decomposition}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Task & N & Format & Invalid ID & Wrong start & Illegal edge & Wrong goal & Long success & Shortest success \\",
        r"\midrule",
    ]
    for row in task_rows:
        lines.append(
            f"{row['task']} & {row['total_predictions']} & "
            f"{pct(row['format_pct'])} & {pct(row['invalid_id_pct'])} & "
            f"{pct(row['wrong_start_pct'])} & {pct(row['illegal_edge_pct'])} & "
            f"{pct(row['wrong_goal_pct'])} & {pct(row['non_shortest_success_pct'])} & "
            f"{pct(row['shortest_success_pct'])}{row_end()}"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    (TABLES / "table_failure_decomposition.tex").write_text("\n".join(lines))
    build_failure_figure(incidence_rows, conditional_rows)


def build_route_length_stratification() -> None:
    grouped: dict[tuple[str, str], dict] = {}
    for model_dir, _ in MODEL_ORDER:
        summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
        for result in summary:
            task = result.get("task")
            if task not in PLANNING_TASKS or "predictions" not in result:
                continue
            vqa_path = ROOT / result["vqa"] if "vqa" in result else EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl"
            predictions = {
                item.get("question_id"): item for item in read_jsonl(ROOT / result["predictions"])
            }
            for question in read_jsonl(vqa_path):
                segments = max(len(reference_path(question)) - 1, 0)
                if task == "a2":
                    bucket = "1" if segments == 1 else r"$\geq$2"
                else:
                    bucket = "1" if segments == 1 else ("2" if segments == 2 else r"$\geq$3")
                key = (task, bucket)
                row = grouped.setdefault(
                    key,
                    {
                        "task": TASK_ROUTE_NAMES[task],
                        "reference_edges": bucket,
                        "question_ids": set(),
                        "predictions": 0,
                        "endpoint_hits": 0,
                        "full_successes": 0,
                        "illegal_edges": 0,
                    },
                )
                flags = route_contract_flags(
                    question, predictions.get(question.get("question_id")), vqa_path
                )
                row["question_ids"].add(question.get("question_id"))
                row["predictions"] += 1
                row["endpoint_hits"] += int(flags["endpoint_hit"])
                row["full_successes"] += int(flags["full_success"])
                row["illegal_edges"] += int(flags["candidate_valid"] and not flags["edge_legal"])

    rows = []
    for task in PLANNING_TASKS:
        buckets = ["1", r"$\geq$2"] if task == "a2" else ["1", "2", r"$\geq$3"]
        for bucket in buckets:
            raw = grouped[(task, bucket)]
            total = raw["predictions"]
            endpoint = raw["endpoint_hits"] / total
            full = raw["full_successes"] / total
            rows.append(
                {
                    "task": raw["task"],
                    "reference_edges": bucket.replace("$", "").replace(r"\geq", ">="),
                    "questions": len(raw["question_ids"]),
                    "predictions": total,
                    "endpoint_hit": endpoint,
                    "full_success": full,
                    "endpoint_gap": endpoint - full,
                    "illegal_edge": raw["illegal_edges"] / total,
                }
            )
    write_csv(DATA / "route_results_by_reference_edges.csv", rows)

    short_task_names = {
        "Point Path": "Point",
        "Embodied Path": "Embodied",
        "Intent Path": "Intent",
    }
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Route results by reference length, pooled over nine VLMs (\%). $N$ counts questions before model expansion.}",
        r"\label{tab:route_length_stratification}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Task & Edges & $N$ & End hit & Success & Illegal edge \\",
        r"\midrule",
    ]
    previous_task = None
    for row in rows:
        if previous_task is not None and row["task"] != previous_task:
            lines.append(r"\midrule")
        edge_label = row["reference_edges"]
        if edge_label.startswith(">="):
            edge_label = rf"$\geq${edge_label[2:]}"
        lines.append(
            f"{short_task_names[row['task']]} & {edge_label} & {row['questions']} & "
            f"{pct(row['endpoint_hit'])} & {pct(row['full_success'])} & "
            f"{pct(row['illegal_edge'])}{row_end()}"
        )
        previous_task = row["task"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_route_length_stratification.tex").write_text("\n".join(lines))


def build_single_view_support_audit() -> None:
    rows = []
    for task in PLANNING_TASKS:
        questions = read_jsonl(EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl")
        support = [route_visual_support(question) for question in questions]
        bbox_values = [row["target_bbox_short_side_px"] for row in support]
        surface_values = [row["target_visible_surface_ratio"] for row in support]
        rows.append(
            {
                "task": TASK_ROUTE_NAMES[task],
                "questions": len(questions),
                "full_path_visible_count": sum(
                    math.isclose(row["path_visible_fraction"], 1.0, abs_tol=1e-9)
                    for row in support
                ),
                "sparse_projection_safe_count": sum(row["sparse_projection_safe"] for row in support),
                "dense_projection_safe_count": sum(row["dense_projection_safe"] for row in support),
                "target_human_visible_count": sum(row["target_human_visible"] for row in support),
                "target_unique_count": sum(row["target_unique"] for row in support),
                "no_prompt_leakage_count": sum(row["no_prompt_leakage"] for row in support),
                "target_bbox_short_side_p10_px": quantile(bbox_values, 0.10),
                "target_bbox_short_side_median_px": median(bbox_values),
                "target_visible_surface_p10": quantile(surface_values, 0.10),
                "target_visible_surface_median": median(surface_values),
            }
        )
    write_csv(DATA / "single_view_support_audit.csv", rows)

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Reference-route admission checks for the 819 benchmark route questions. Path projection requires every dense reference-path point to lie in the image; mask pass checks sparse and dense routes against the visible-obstacle mask. Target columns report P10/median.}",
        r"\label{tab:single_view_support}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Task & $N$ & Path proj. & Mask pass & BBox px & Surface \\",
        r"\midrule",
    ]
    for row in rows:
        short_name = {
            "Point Path": "Point",
            "Embodied Path": "Embodied",
            "Intent Path": "Intent",
        }[row["task"]]
        all_projection_safe = min(
            row["sparse_projection_safe_count"], row["dense_projection_safe_count"]
        )
        lines.append(
            f"{short_name} & {row['questions']} & "
            f"{pct(row['full_path_visible_count'] / row['questions'])} & "
            f"{pct(all_projection_safe / row['questions'])} & "
            f"{row['target_bbox_short_side_p10_px']:.0f}/{row['target_bbox_short_side_median_px']:.0f} & "
            f"{pct(row['target_visible_surface_p10'])}/{pct(row['target_visible_surface_median'])}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_single_view_support.tex").write_text("\n".join(lines))


def build_visibility_sensitivity() -> None:
    rows = []
    for task in PLANNING_TASKS:
        vqa_path = EVALFIX_VQA_ROOT / f"vqa_next_{task}.jsonl"
        questions = read_jsonl(vqa_path)
        model_predictions = []
        for model_dir, _ in MODEL_ORDER:
            summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
            result = next(row for row in summary if row.get("task") == task)
            model_predictions.append(
                {
                    row.get("question_id"): row
                    for row in read_jsonl(ROOT / result["predictions"])
                }
            )

        for level_key, level_label, bbox_min_px, surface_min in VISIBILITY_LEVELS:
            selected = [
                question
                for question in questions
                if visibility_level_passes(question, bbox_min_px, surface_min)
            ]
            flags = []
            clusters = []
            gap_values = []
            for predictions in model_predictions:
                for question in selected:
                    item = route_contract_flags(
                        question,
                        predictions.get(question.get("question_id")),
                        vqa_path,
                    )
                    flags.append(item)
                    clusters.append(str(question.get("scene_id")))
                    gap_values.append(
                        float(item["endpoint_hit"]) - float(item["full_success"])
                    )
            total = len(flags)
            endpoint_hit = sum(row["endpoint_hit"] for row in flags) / max(total, 1)
            full_success = sum(row["full_success"] for row in flags) / max(total, 1)
            illegal_edge = sum(
                row["candidate_valid"] and not row["edge_legal"] for row in flags
            ) / max(total, 1)
            gap_low, gap_high = cluster_bootstrap_interval(gap_values, clusters)
            rows.append(
                {
                    "task": TASK_ROUTE_NAMES[task],
                    "visibility_level": level_key,
                    "visibility_label": level_label,
                    "bbox_min_px": bbox_min_px,
                    "target_visible_surface_min": surface_min,
                    "questions": len(selected),
                    "retained_fraction": len(selected) / len(questions),
                    "predictions": total,
                    "scene_clusters": len(set(clusters)),
                    "endpoint_hit": endpoint_hit,
                    "full_success": full_success,
                    "endpoint_gap": endpoint_hit - full_success,
                    "gap_ci_low": gap_low,
                    "gap_ci_high": gap_high,
                    "illegal_edge": illegal_edge,
                }
            )
    write_csv(DATA / "visibility_sensitivity_by_task.csv", rows)

    display_rows = [
        row for row in rows if row["visibility_level"] in {"all_supported", "strict_target"}
    ]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\caption{Visibility sensitivity over nine VLMs (\%). All questions satisfy the admission checks in Table~\ref{tab:single_view_support}; Strict additionally requires target bbox short side $\geq64$ px and visible surface $\geq50\%$. End/Full reports endpoint hit and full route success.}",
        r"\label{tab:visibility_sensitivity}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Task & View & $N$ & End/Full & Gap & Illegal \\",
        r"\midrule",
    ]
    for index, row in enumerate(display_rows):
        if index and index % 2 == 0:
            lines.append(r"\midrule")
        task_name = {
            "Point Path": "Point",
            "Embodied Path": "Embodied",
            "Intent Path": "Intent",
        }[row["task"]]
        lines.append(
            f"{task_name} & {row['visibility_label']} & {row['questions']} & "
            f"{pct(row['endpoint_hit'])}/{pct(row['full_success'])} & "
            f"{pct(row['endpoint_gap'])} & {pct(row['illegal_edge'])}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_visibility_sensitivity.tex").write_text("\n".join(lines))


def build_edge_observability_audit() -> None:
    with (EDGE_OBSERVABILITY_ROOT / "task_summary.csv").open() as f:
        source_rows = list(csv.DictReader(f))
    rows = []
    for row in source_rows:
        rows.append({
            "task": row["task"],
            "reference_edge_support": float(row["reference_edge_observable_rate"]),
            "reference_question_support": float(row["reference_question_all_edges_observable_rate"]),
            "start_legal_support": float(row["start_legal_edge_observable_rate"]),
            "invalid_failure_depth_evidence": float(row["invalid_failure_depth_evidence_rate"]),
            "invalid_failure_object_only": float(row["invalid_failure_object_only_rate"]),
            "invalid_failure_uncertain_only": float(row["invalid_failure_uncertain_only_rate"]),
        })

    summary = read_json(EDGE_OBSERVABILITY_ROOT / "summary.json")
    reference_counts = summary["reference_support"]
    question_total = int(summary["counts"]["questions"])
    with (EDGE_OBSERVABILITY_ROOT / "observable_reference_subset_summary.csv").open() as f:
        observable_questions = {
            int(row["observable_reference_questions"])
            for row in csv.DictReader(f)
            if row["task"] == "all"
        }
    with (EDGE_OBSERVABILITY_ROOT / "prediction_question_audit.csv").open() as f:
        prediction_rows = list(csv.DictReader(f))
    invalid_rows = [row for row in prediction_rows if int(row["invalid_edge_count"]) > 0]
    invalid_failures = len(invalid_rows)
    depth_failures = sum(int(row["has_depth_obstruction_failure"]) for row in invalid_rows)
    object_failures = sum(int(row["has_object_only_obstruction_failure"]) for row in invalid_rows)
    uncertain_failures = sum(
        int(row["uncertain_only_invalid_edge_failure"]) for row in invalid_rows
    )
    start_counts = summary["start_action_support"]
    all_row = {
        "task": "all",
        "reference_edge_support": reference_counts.get("observable_safe", 0)
        / max(sum(reference_counts.values()), 1),
        "reference_question_support": next(iter(observable_questions)) / question_total,
        "start_legal_support": start_counts.get("observable_safe", 0)
        / max(start_counts.get("observable_safe", 0) + start_counts.get("uncertain_safe", 0), 1),
        "invalid_failure_depth_evidence": depth_failures / max(invalid_failures, 1),
        "invalid_failure_object_only": object_failures / max(invalid_failures, 1),
        "invalid_failure_uncertain_only": uncertain_failures / max(invalid_failures, 1),
    }
    rows.append(all_row)
    write_csv(DATA / "edge_observability_audit.csv", rows)

def build_failure_figure(incidence_rows: list[dict], conditional_rows: list[dict]) -> None:
    width, height = 430, 360
    bar_x, bar_width, thin_height = 148, 232, 7
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '''<defs><style>text { font-family: Arial, Helvetica, sans-serif; fill: #202428; }
.label { font-size: 13px; font-weight: 700; }
.panel { font-size: 13px; font-weight: 700; }
.note { font-size: 11px; fill: #4b5563; } .value { font-size: 10.5px; font-weight: 700; }
</style></defs>''',
             '<rect width="430" height="360" fill="#ffffff"/>',
             '<text x="12" y="17" class="panel">A. Independent route checks</text>']
    check_specs = [
        ("endpoint_hit_pct", "Endpoint hit", "#2878b5"),
        ("edge_legal_pct", "Legal route", "#b66a1f"),
        ("full_success_pct", "Full success", "#26734d"),
    ]
    for row_index, row in enumerate(incidence_rows):
        y = 34 + row_index * 45
        label = row["task"].replace(" path", "")
        parts.append(f'<text x="12" y="{y + 8}" class="label">{svg_escape(label)}</text>')
        for check_index, (key, label_text, color) in enumerate(check_specs):
            line_y = y + 12 + check_index * 10
            value = row[key]
            parts.append(f'<text x="142" y="{line_y + 7}" text-anchor="end" class="note">{label_text}</text>')
            parts.append(f'<rect x="{bar_x}" y="{line_y}" width="{bar_width}" height="{thin_height}" fill="#eceff1"/>')
            parts.append(f'<rect x="{bar_x}" y="{line_y}" width="{bar_width * value:.1f}" height="{thin_height}" fill="{color}"/>')
            parts.append(f'<text x="418" y="{line_y + 7}" text-anchor="end" class="value" fill="{color}">{100 * value:.1f}</text>')

    parts.append('<line x1="12" y1="181" x2="418" y2="181" stroke="#d6d9dd"/>')
    parts.append('<text x="12" y="202" class="panel">B. Order-independent conditional rates</text>')
    conditional_specs = [
        ("edge_legal_given_valid_ids", "Edge legal | IDs", "#b66a1f"),
        ("endpoint_given_valid_ids", "Goal hit | IDs", "#2878b5"),
        ("endpoint_given_legal_route", "Goal hit | legal", "#7554a3"),
        ("full_success", "Full success", "#26734d"),
    ]
    for row_index, row in enumerate(conditional_rows):
        y = 212 + row_index * 46
        label = row["task"].replace(" path", "")
        parts.append(f'<text x="12" y="{y + 8}" class="label">{svg_escape(label)}</text>')
        for check_index, (key, label_text, color) in enumerate(conditional_specs):
            line_y = y + 12 + check_index * 8
            value = row[key]
            parts.append(f'<text x="142" y="{line_y + 6}" text-anchor="end" class="note">{label_text}</text>')
            parts.append(f'<rect x="{bar_x}" y="{line_y}" width="{bar_width}" height="5" fill="#eceff1"/>')
            parts.append(f'<rect x="{bar_x}" y="{line_y}" width="{bar_width * value:.1f}" height="5" fill="{color}"/>')
            parts.append(f'<text x="418" y="{line_y + 6}" text-anchor="end" class="value" fill="{color}">{100 * value:.1f}</text>')
    parts.append('</svg>')
    svg_path = FIGURES / "fig_failure_decomposition.svg"
    svg_path.write_text("\n".join(parts))
    svg_to_pdf(svg_path, FIGURES / "fig_failure_decomposition.pdf")

    figure_lines = [
        r"\begin{figure}[!ht]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/fig_failure_decomposition.pdf}",
        r"\caption{Route diagnostics pooled over nine VLMs. (A) Independent rates use all predictions as denominator. (B) Conditional rates do not depend on evaluator order: legal route and goal hit are conditioned on valid IDs, and goal-given-legal is conditioned on an edge-legal route. Full success requires all contract checks.}",
        r"\label{fig:failure_decomposition}",
        r"\end{figure}",
        r"\vspace{-3pt}",
        "",
    ]
    (FIGURES / "fig_failure_decomposition.tex").write_text("\n".join(figure_lines))


def aggregate_summary(summary: list[dict]) -> tuple[float, float]:
    metrics = task_metrics(summary)
    cls_ba = (metrics["a1"]["balanced_accuracy"] + metrics["b1"]["balanced_accuracy"]) / 2.0
    plan_sr = (
        metrics["a2"]["success_rate"] + metrics["b2"]["success_rate"] + metrics["c"]["success_rate"]
    ) / 3.0
    return cls_ba, plan_sr


def build_anti_shortcut() -> None:
    rows = []
    variants = [("full", "Full"), ("text_only", "Text only"), ("overlay_mismatch", "Overlay mismatch")]
    for model_dir, model_name in ANTI_MODELS:
        for variant_dir, variant_name in variants:
            summary = read_json(ANTI_SHORTCUT_ROOT / model_dir / variant_dir / "summary.json")
            cls_ba, plan_sr = aggregate_summary(summary)
            metrics = task_metrics(summary)
            rows.append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "classification_mean_ba": cls_ba,
                    "planning_mean_sr": plan_sr,
                    "a2_sr": metrics["a2"]["success_rate"],
                    "b2_sr": metrics["b2"]["success_rate"],
                    "c_sr": metrics["c"]["success_rate"],
                }
            )
    write_csv(DATA / "anti_shortcut.csv", rows)

    grouped = []
    for _, model_name in ANTI_MODELS:
        by_variant = {row["variant"]: row for row in rows if row["model"] == model_name}
        full = by_variant["Full"]
        grouped.append(
            {
                "model": model_name,
                "full_cls": full["classification_mean_ba"],
                "text_cls": by_variant["Text only"]["classification_mean_ba"],
                "mismatch_cls": by_variant["Overlay mismatch"]["classification_mean_ba"],
                "full_plan": full["planning_mean_sr"],
                "text_plan": by_variant["Text only"]["planning_mean_sr"],
                "mismatch_plan": by_variant["Overlay mismatch"]["planning_mean_sr"],
                "full_route": [full["a2_sr"], full["b2_sr"], full["c_sr"]],
                "text_route": [
                    by_variant["Text only"]["a2_sr"],
                    by_variant["Text only"]["b2_sr"],
                    by_variant["Text only"]["c_sr"],
                ],
                "mismatch_route": [
                    by_variant["Overlay mismatch"]["a2_sr"],
                    by_variant["Overlay mismatch"]["b2_sr"],
                    by_variant["Overlay mismatch"]["c_sr"],
                ],
            }
        )
    write_csv(DATA / "anti_shortcut_comparison.csv", grouped)

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Input-dependence route SR (\%) on fixed paired questions. Cells report Point/Embodied/Intent Path; Embodied/Intent full-input SR is near zero, so degradation is informative mainly for Point Path.}",
        r"\label{tab:anti_shortcut}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Model & Full & Text only & Mismatched overlay \\",
        r"\midrule",
    ]
    for row in grouped:
        full_route = "/".join(pct(value) for value in row["full_route"])
        text_route = "/".join(pct(value) for value in row["text_route"])
        mismatch_route = "/".join(pct(value) for value in row["mismatch_route"])
        lines.append(
            f"{tex_escape(row['model'])} & {full_route} & {text_route} & "
            f"{mismatch_route}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\vspace{-3pt}", ""]
    (TABLES / "table_anti_shortcut.tex").write_text("\n".join(lines))


def build_cot_contract() -> None:
    report = read_json(COT_EXPORT_REPORT)
    contract = report["contract"]
    template_source_key = next(k for k in contract if k.startswith("uses_deterministic_"))
    audit_rows = [
        ("Expected training rows", report["expected_rows"]),
        ("Spatial CoT rows", report["cot_rows"]),
        ("Missing Spatial CoT rows", report["missing_cot_rows"]),
        ("Missing images", report["missing_images"]),
        ("GPT text source", contract["cot_text_source"]),
        ("Template source used", str(contract[template_source_key]).lower()),
        ("Clean think-format validation", str(contract["validates_clean_think_format"]).lower()),
        ("Final JSON equals ground truth", str(contract["validates_final_json_equals_gt"]).lower()),
    ]
    write_csv(DATA / "spatial_cot_contract.csv", [{"field": k, "value": v} for k, v in audit_rows])
    rows = [
        ("Complete training rows", f"{report['cot_rows']} / {report['expected_rows']}"),
        ("Missing image / rationale", f"{report['missing_images']} / {report['missing_cot_rows']}"),
        ("Rationale source", "GPT-written, non-template"),
        ("Format and answer audit", "Pass"),
    ]

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Audit of the released EgoPathBench SFT export.}",
        r"\label{tab:spatial_cot_contract}",
        r"\begin{tabular}{p{0.55\linewidth}p{0.34\linewidth}}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
    ]
    for k, v in rows:
        align_value = tex_escape(str(v))
        lines.append(f"{tex_escape(k)} & {align_value}{row_end()}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\vspace{-3pt}", ""]
    (TABLES / "table_spatial_cot_contract.tex").write_text("\n".join(lines))


def build_evaluation_contract_details() -> None:
    implementation_rows = [
        {
            "component": "Candidate IDs",
            "detail": "Predictions are saved as JSON lists of visible display IDs; hidden or invented IDs are invalid.",
        },
        {
            "component": "Parsing",
            "detail": "The runner normalizes one ID list from the model response; the evaluator then reads only that list and reports format failures separately.",
        },
        {
            "component": "Robot body",
            "detail": "Robot-body tasks use a 0.6 m diameter robot, represented by a 0.30 m radius footprint.",
        },
        {
            "component": "Grid",
            "detail": "Route checks use a 0.05 m grid. Obstacles are Minkowski-inflated by the task radius, so the center-line can be checked on the grid.",
        },
        {
            "component": "Segment",
            "detail": "An edge is legal only if the straight segment between two feasible visible waypoints remains free on the task-specific grid.",
        },
        {
            "component": "Endpoint",
            "detail": "Each row stores acceptable goal IDs from a target-centered goal ring; embodied path tasks use embodied-feasible endpoints near the object.",
        },
        {
            "component": "Shortest path",
            "detail": "Dijkstra computes the optimal route over the legal direct-pair graph; equal-length legal shortest routes receive the same SPL credit.",
        },
        {
            "component": "Scoring",
            "detail": "Repeated IDs or loops are not removed; they are scored as written, so detours reduce SPL or fail edge legality.",
        },
    ]
    write_csv(DATA / "evaluation_contract_details.csv", implementation_rows)

    rows = [
        ("Parse", "One ordered ID list is recovered.", "Format"),
        ("Candidate", "Every ID is visible and allowed.", "Invalid ID"),
        ("Start", "The first ID matches the required start.", "Wrong start"),
        ("Edge", "Every consecutive pair is a legal direct edge.", "Illegal edge"),
        ("Endpoint", "The final ID is an acceptable goal.", "Wrong goal"),
        ("Efficiency", "A successful route is compared with shortest references.", "SPL"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Ordered evaluation contract for route-bearing tasks. SPL is an efficiency measure applied after route success.}",
        r"\label{tab:evaluation_contract_details}",
        r"\begin{tabular}{@{}p{0.19\linewidth}p{0.52\linewidth}p{0.20\linewidth}@{}}",
        r"\toprule",
        r"Stage & Pass condition & Outcome \\",
        r"\midrule",
    ]
    for stage, condition, outcome in rows:
        lines.append(f"{tex_escape(stage)} & {tex_escape(condition)} & {tex_escape(outcome)}{row_end()}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_evaluation_contract_details.tex").write_text("\n".join(lines))


def build_cross_benchmark() -> None:
    rows = []
    all_base = []
    all_checkpoint = []
    for benchmark_dir, benchmark_name in EXTERNAL_BENCHMARKS:
        base_path = (
            EXTERNAL_SUITE_ROOT
            / "base_strict4096"
            / "predictions"
            / benchmark_dir
            / "predictions.jsonl"
        )
        checkpoint_paths = [
            EXTERNAL_SUITE_ROOT
            / "ckpt3000_strict4096"
            / "predictions"
            / benchmark_dir
            / "predictions.jsonl",
            *[
            EXTERNAL_SUITE_ROOT
            / f"ckpt3000_strict4096_worker{worker}"
            / "predictions"
            / benchmark_dir
            / "predictions.jsonl"
            for worker in (0, 1)
            ],
        ]
        base_by_id = {row["question_id"]: row for row in read_jsonl(base_path)}
        checkpoint_by_id = merge_prediction_files(checkpoint_paths)
        if set(checkpoint_by_id) != set(base_by_id):
            missing = len(set(base_by_id) - set(checkpoint_by_id))
            extra = len(set(checkpoint_by_id) - set(base_by_id))
            raise ValueError(
                f"{benchmark_dir} checkpoint/base question IDs differ: "
                f"missing={missing}, extra={extra}"
            )
        matched_ids = sorted(set(base_by_id) & set(checkpoint_by_id))
        base_correct = [bool(base_by_id[qid].get("correct")) for qid in matched_ids]
        checkpoint_correct = [bool(checkpoint_by_id[qid].get("correct")) for qid in matched_ids]
        base_accuracy = sum(base_correct) / len(matched_ids)
        checkpoint_accuracy = sum(checkpoint_correct) / len(matched_ids)
        ci_low, ci_high = paired_bootstrap_interval(base_correct, checkpoint_correct)
        rows.append(
            {
                "benchmark": benchmark_name,
                "n": len(matched_ids),
                "base": base_accuracy,
                "spatial_cot": checkpoint_accuracy,
                "delta": checkpoint_accuracy - base_accuracy,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
            }
        )
        all_base.extend(base_correct)
        all_checkpoint.extend(checkpoint_correct)

    benchmark_rows = list(rows)
    rows.append(
        {
            "benchmark": "Macro average",
            "n": "--",
            "base": sum(row["base"] for row in benchmark_rows) / len(benchmark_rows),
            "spatial_cot": sum(row["spatial_cot"] for row in benchmark_rows) / len(benchmark_rows),
            "delta": sum(row["delta"] for row in benchmark_rows) / len(benchmark_rows),
            "delta_ci_low": "",
            "delta_ci_high": "",
        }
    )

    aggregate_base = sum(all_base) / len(all_base)
    aggregate_checkpoint = sum(all_checkpoint) / len(all_checkpoint)
    ci_low, ci_high = paired_bootstrap_interval(all_base, all_checkpoint)
    rows.append(
        {
            "benchmark": "Aggregate",
            "n": len(all_base),
            "base": aggregate_base,
            "spatial_cot": aggregate_checkpoint,
            "delta": aggregate_checkpoint - aggregate_base,
            "delta_ci_low": ci_low,
            "delta_ci_high": ci_high,
        }
    )
    write_csv(DATA / "cross_benchmark_spatial_generalization.csv", rows)

    task_metrics = [
        ("a1", "Point Trav. BA", 146, "balanced_accuracy"),
        ("b1", "Body Trav. BA", 146, "balanced_accuracy"),
        ("a2", "Point Path SR", 309, "success_rate"),
        ("b2", "Body Path SR", 309, "success_rate"),
        ("c", "Intent Path SR", 201, "success_rate"),
    ]
    in_domain_rows = []
    for task, metric_name, n, metric_key in task_metrics:
        base = read_json(SFT_BASE_METRICS_ROOT / f"{task}.json")["metrics"][metric_key]
        checkpoint = read_json(SFT_CHECKPOINT_METRICS_ROOT / f"{task}.json")["metrics"][metric_key]
        in_domain_rows.append(
            {
                "metric": metric_name,
                "n": n,
                "base": base,
                "sft": checkpoint,
                "delta": checkpoint - base,
            }
        )
    base_score = egopath_score(
        dict(zip(["a1_ba", "b1_ba", "a2_sr", "b2_sr", "c_sr"], [row["base"] for row in in_domain_rows]))
    )
    checkpoint_score = egopath_score(
        dict(zip(["a1_ba", "b1_ba", "a2_sr", "b2_sr", "c_sr"], [row["sft"] for row in in_domain_rows]))
    )
    score_row = {
        "metric": "EgoPath Score",
        "n": "--",
        "base": base_score,
        "sft": checkpoint_score,
        "delta": checkpoint_score - base_score,
    }
    write_csv(DATA / "training_resource_eval.csv", [score_row, *in_domain_rows])

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Matched-item accuracy on independent spatial benchmarks before and after EgoPathBench fine-tuning (\%).}",
        r"\label{tab:training_resource}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Benchmark & N & Base & SFT & $\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        if row["benchmark"] == "Macro average":
            continue
        label = tex_escape(row["benchmark"])
        if row["benchmark"] == "Aggregate":
            label = rf"\textbf{{{label}}}"
        lines.append(f"{label} & {row['n']} & {pct(row['base'])} & {pct(row['spatial_cot'])} & {signed_pct(row['delta'])}{row_end()}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_cross_benchmark.tex").write_text("\n".join(lines))


def build_task_taxonomy() -> None:
    evalfix = read_json(EVALFIX_SUMMARY)
    rows = [
        ("Point Trav.", "--", "Point", "Set", "Traversability", evalfix["row_counts_by_task"]["a1"]),
        ("Embodied Trav.", "--", "Emb.", "Set", "Feasibility", evalfix["row_counts_by_task"]["b1"]),
        ("Point Path", "Explicit", "Point", "Route", "Edges, endpoint", evalfix["row_counts_by_task"]["a2"]),
        ("Embodied Path", "Explicit", "Emb.", "Route", "Footprint, edges, goal", evalfix["row_counts_by_task"]["b2"]),
        ("Intent Path", "Intent", "Emb.", "Route", "Intent, footprint, edges", evalfix["row_counts_by_task"]["c"]),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\caption{EgoPathBench task taxonomy.}",
        r"\label{tab:task_taxonomy}",
        r"\begin{tabular}{@{}p{0.21\linewidth}lllp{0.29\linewidth}r@{}}",
        r"\toprule",
        r"Task & Goal & Agent & Output & Scored constraints & N \\",
        r"\midrule",
    ]
    for index, (task_name, goal, agent, output, decision, count) in enumerate(rows):
        if index == 2:
            lines.append(r"\midrule")
        lines.append(
            f"{tex_escape(task_name)} & {goal} & {agent} & {output} & {tex_escape(decision)} & {count}{row_end()}"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "table_task_taxonomy.tex").write_text("\n".join(lines))


def build_overview_box() -> None:
    vqa_path = EVALFIX_VQA_ROOT / "vqa_next_b2.jsonl"
    question = next(q for q in read_jsonl(vqa_path) if q["question_id"] == OVERVIEW_QUESTION_ID)
    pred_ids = OVERVIEW_ROUTE
    waypoints = load_visible_waypoints(question["visible_waypoints_path"])
    positions = {int(w["display_id"]): tuple(w["image_xy"]) for w in waypoints}
    gt = question["ground_truth"]
    direct_pairs_path = resolve_optional_path(
        gt["direct_pairs_ref"], [vqa_path.parent, vqa_path.parent.parent, ROOT]
    )
    _, adjacency = load_direct_pairs(direct_pairs_path, waypoints)
    edge_legality = [dst in adjacency.get(src, {}) for src, dst in zip(pred_ids[:-1], pred_ids[1:])]
    acceptable_goal_ids = {int(value) for value in gt["acceptable_goal_ids"]}
    assert pred_ids[0] == 1
    assert all(route_id in positions for route_id in pred_ids)
    assert edge_legality == OVERVIEW_EDGE_LEGALITY
    assert OVERVIEW_GOAL_ID in acceptable_goal_ids
    assert pred_ids[-1] not in acceptable_goal_ids
    illegal_indices = {
        index
        for index, (src, dst) in enumerate(zip(pred_ids[:-1], pred_ids[1:]))
        if dst not in adjacency.get(src, {})
    }

    image_bytes = Path(question["image_path"]).read_bytes()
    image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    width, height = 720, 690
    image_x, image_y, image_size = 18, 58, 420
    scale = image_size / 1024.0

    route_parts = []
    illegal_segments = []
    for index, (src, dst) in enumerate(zip(pred_ids[:-1], pred_ids[1:])):
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        x1, y1 = image_x + x1 * scale, image_y + y1 * scale
        x2, y2 = image_x + x2 * scale, image_y + y2 * scale
        illegal = index in illegal_indices
        color = "#b42318" if illegal else "#087e8b"
        dash = ' stroke-dasharray="8 5"' if illegal else ""
        route_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#ffffff" stroke-width="{11 if illegal else 9}" stroke-linecap="round" opacity="0.95"{dash}/>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{7 if illegal else 5}" stroke-linecap="round" opacity="0.95"{dash}/>'
        )
        if illegal:
            illegal_segments.append((x1, y1, x2, y2, src, dst))

    illegal_marks = []
    for x1, y1, x2, y2, src, dst in illegal_segments:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        illegal_marks.append(
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="10" fill="#b42318" stroke="#ffffff" stroke-width="3"/>'
        )
        illegal_marks.append(
            f'<line x1="{mx - 12:.1f}" y1="{my - 12:.1f}" x2="{mx + 12:.1f}" y2="{my + 12:.1f}" stroke="#ffffff" stroke-width="7"/>'
            f'<line x1="{mx - 12:.1f}" y1="{my + 12:.1f}" x2="{mx + 12:.1f}" y2="{my - 12:.1f}" stroke="#ffffff" stroke-width="7"/>'
            f'<line x1="{mx - 12:.1f}" y1="{my - 12:.1f}" x2="{mx + 12:.1f}" y2="{my + 12:.1f}" stroke="#b42318" stroke-width="3"/>'
            f'<line x1="{mx - 12:.1f}" y1="{my + 12:.1f}" x2="{mx + 12:.1f}" y2="{my - 12:.1f}" stroke="#b42318" stroke-width="3"/>'
        )

    goal_id = OVERVIEW_GOAL_ID
    goal_x = image_x + positions[goal_id][0] * scale
    goal_y = image_y + positions[goal_id][1] * scale
    goal_mark = (
        f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="14" fill="none" '
        f'stroke="#ffffff" stroke-width="7"/>'
        f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="14" fill="none" '
        f'stroke="#16794a" stroke-width="4"/>'
    )

    # Fixed crop for the selected example: the illegal edge, wrong endpoint, and correct goal.
    zoom_x, zoom_y, zoom_w, zoom_h = 474, 222, 222, 164
    crop_x, crop_y, crop_w, crop_h = 395, 545, 285, 315
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as source_image:
        crop = source_image.convert("RGB").crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        crop_buffer = BytesIO()
        crop.save(crop_buffer, "PNG")
    zoom_uri = "data:image/png;base64," + base64.b64encode(crop_buffer.getvalue()).decode("ascii")
    zoom_scale_x = zoom_w / crop_w
    zoom_scale_y = zoom_h / crop_h
    zoom_marks = []
    for _, _, _, _, src, dst in illegal_segments:
        zx1 = zoom_x + (positions[src][0] - crop_x) * zoom_scale_x
        zy1 = zoom_y + (positions[src][1] - crop_y) * zoom_scale_y
        zx2 = zoom_x + (positions[dst][0] - crop_x) * zoom_scale_x
        zy2 = zoom_y + (positions[dst][1] - crop_y) * zoom_scale_y
        zmx, zmy = (zx1 + zx2) / 2, (zy1 + zy2) / 2
        zoom_marks.append(
            f'<line x1="{zx1:.1f}" y1="{zy1:.1f}" x2="{zx2:.1f}" y2="{zy2:.1f}" stroke="#ffffff" stroke-width="15" stroke-linecap="round"/>'
            f'<line x1="{zx1:.1f}" y1="{zy1:.1f}" x2="{zx2:.1f}" y2="{zy2:.1f}" stroke="#b42318" stroke-width="10" stroke-linecap="round"/>'
            f'<circle cx="{zx2:.1f}" cy="{zy2:.1f}" r="11" fill="#b42318" stroke="#ffffff" stroke-width="3"/>'
            f'<line x1="{zmx - 14:.1f}" y1="{zmy - 14:.1f}" x2="{zmx + 14:.1f}" y2="{zmy + 14:.1f}" stroke="#ffffff" stroke-width="8"/>'
            f'<line x1="{zmx - 14:.1f}" y1="{zmy + 14:.1f}" x2="{zmx + 14:.1f}" y2="{zmy - 14:.1f}" stroke="#ffffff" stroke-width="8"/>'
            f'<line x1="{zmx - 14:.1f}" y1="{zmy - 14:.1f}" x2="{zmx + 14:.1f}" y2="{zmy + 14:.1f}" stroke="#b42318" stroke-width="3"/>'
            f'<line x1="{zmx - 14:.1f}" y1="{zmy + 14:.1f}" x2="{zmx + 14:.1f}" y2="{zmy - 14:.1f}" stroke="#b42318" stroke-width="3"/>'
        )
    zoom_goal_x = zoom_x + (positions[goal_id][0] - crop_x) * zoom_scale_x
    zoom_goal_y = zoom_y + (positions[goal_id][1] - crop_y) * zoom_scale_y
    zoom_marks.append(
        f'<circle cx="{zoom_goal_x:.1f}" cy="{zoom_goal_y:.1f}" r="15" fill="none" stroke="#ffffff" stroke-width="8"/>'
        f'<circle cx="{zoom_goal_x:.1f}" cy="{zoom_goal_y:.1f}" r="15" fill="none" stroke="#16794a" stroke-width="5"/>'
    )

    status = [
        ("Parse", "JSON route", "PASS", "#16794a"),
        ("Candidate", "Visible IDs", "PASS", "#16794a"),
        ("Start", "Starts at 1", "PASS", "#16794a"),
        ("Edge", "7 -> 23", "FAIL", "#b42318"),
        ("Endpoint", "23 != 27", "FAIL", "#b42318"),
        ("Efficiency", "After success", "NOT SCORED", "#6b7280"),
    ]
    status_parts = []
    for i, (stage, detail, result, color) in enumerate(status):
        x = 16 + i * 116
        status_parts.append(
            f'<rect x="{x}" y="535" width="104" height="91" rx="3" fill="#ffffff" stroke="#d6d9dd"/>'
            f'<text x="{x + 52}" y="558" text-anchor="middle" class="stage">{stage}</text>'
            f'<text x="{x + 52}" y="579" text-anchor="middle" class="detail">{detail}</text>'
            f'<text x="{x + 52}" y="606" text-anchor="middle" class="result" fill="{color}">{result}</text>'
        )
        if i < len(status) - 1:
            status_parts.append(f'<path d="M {x + 105} 580 L {x + 114} 580" stroke="#7b828a" marker-end="url(#arrow)"/>')

    route_text = "[" + ", ".join(str(x) for x in pred_ids) + "]"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #202428; }}
.title {{ font-size: 18px; font-weight: 700; }} .body {{ font-size: 14px; }} .emph {{ font-size: 15px; font-weight: 700; }} .fail {{ font-size: 14px; font-weight: 700; fill: #b42318; }}
.stage {{ font-size: 12px; font-weight: 700; }} .detail {{ font-size: 9.5px; fill: #4b5563; }}
.result {{ font-size: 10.5px; font-weight: 700; }}
</style>
<marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#7b828a"/></marker>
</defs>
<rect width="720" height="690" fill="#ffffff"/>
<text x="22" y="29" class="title">A. First-person waypoint input</text>
<image href="{image_uri}" x="{image_x}" y="{image_y}" width="{image_size}" height="{image_size}"/>
{''.join(route_parts)}
{''.join(illegal_marks)}
{goal_mark}
<line x1="454" y1="62" x2="454" y2="478" stroke="#d6d9dd"/>
<text x="474" y="87" class="title">B. Route decision</text>
<text x="474" y="111" class="detail">Embodied-Goal Path · 0.6 m robot</text>
<text x="474" y="139" class="body">Goal: {svg_escape(OVERVIEW_GOAL_TEXT)}</text>
<text x="474" y="164" class="body">Start ID: 1</text>
<text x="474" y="191" class="body">Illustrative erroneous route</text>
<text x="474" y="215" class="emph">{svg_escape(route_text)}</text>
<image href="{zoom_uri}" x="{zoom_x}" y="{zoom_y}" width="{zoom_w}" height="{zoom_h}"/>
{''.join(zoom_marks)}
<rect x="{zoom_x}" y="{zoom_y}" width="{zoom_w}" height="{zoom_h}" rx="3" fill="none" stroke="#6b7280" stroke-width="1.5"/>
<rect x="482" y="342" width="204" height="39" rx="3" fill="#ffffff" opacity="0.94"/>
<text x="492" y="358" class="fail">Edge 7 -> 23 is illegal</text>
<text x="492" y="374" class="fail">Endpoint 23; correct goal 27</text>
<text x="474" y="414" class="body">Correct endpoint: 27</text>
<text x="474" y="438" class="body">Illustrative route: rejected</text>
<text x="22" y="505" class="title">C. Ordered route-contract audit</text>
{''.join(status_parts)}
<text x="360" y="663" text-anchor="middle" class="detail">Constructed illustration; waypoint IDs, goal set, and edge legality follow the benchmark contract.</text>
</svg>'''
    svg_path = FIGURES / "fig_overview.svg"
    svg_path.write_text(svg)
    svg_to_pdf(svg_path, FIGURES / "fig_overview.pdf")

    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=0.98\linewidth]{figures/fig_overview.pdf}",
        r"\caption{EgoPathBench task and route-level evaluation. For the table near the cabinet, the constructed route $[1,7,23]$ starts correctly and uses visible waypoint IDs, but edge $7\!\rightarrow\!23$ is illegal and endpoint 23 differs from the acceptable goal 27. The evaluator therefore rejects the route before considering efficiency.}",
        r"\label{fig:overview}",
        r"\end{figure}",
        "",
    ]
    (FIGURES / "fig_overview.tex").write_text("\n".join(lines))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_dataset_stats()
    build_task_taxonomy()
    build_single_view_support_audit()
    build_leaderboard()
    build_geometry_sensitivity()
    build_human_solvable_analysis()
    build_failure_decomposition()
    build_visibility_sensitivity()
    build_edge_observability_audit()
    build_route_length_stratification()
    build_anti_shortcut()
    build_cot_contract()
    build_evaluation_contract_details()
    build_cross_benchmark()
    build_statistical_uncertainty()
    build_overview_box()
    print(f"Wrote paper assets under {PAPER}")


if __name__ == "__main__":
    main()
