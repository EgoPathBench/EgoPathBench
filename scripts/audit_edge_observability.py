#!/usr/bin/env python3
"""Audit observation support for reference, start-action, and predicted route edges."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_task_outputs import load_depth_image, load_object_index_image
from evaluate_next import parse_ids, prediction_output


DEFAULT_BENCHMARK_ROOT = Path(
    "published/navbench3d_release_20260417_rebuild5/data/benchmark_evalfix/benchmark"
)
DEFAULT_RUNS_ROOT = Path("results/main_model_runs_20260528/native_default_full/runs")
DEFAULT_OUTPUT_DIR = Path("results/edge_observability_audit_20260718_radius030")
DEFAULT_EMBODIED_RADIUS_M = 0.30
AUDIT_DATE = "2026-07-18"
DEFAULT_MODEL_RUNS = (
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_view_calibration(waypoints: list[dict]) -> dict:
    usable = [
        waypoint
        for waypoint in waypoints
        if len(waypoint.get("world_xyz") or []) == 3
        and len(waypoint.get("image_xy") or []) == 2
        and float(waypoint.get("depth", 0.0)) > 0.0
    ]
    if len(usable) < 4:
        raise ValueError("at least four waypoint correspondences are required")

    world_xy = np.asarray([waypoint["world_xyz"][:2] for waypoint in usable], dtype=np.float32)
    image_xy = np.asarray([waypoint["image_xy"] for waypoint in usable], dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(world_xy, image_xy, cv2.RANSAC, 2.0)
    if homography is None:
        raise ValueError("ground-plane homography fit failed")

    projected = cv2.perspectiveTransform(world_xy[:, None, :], homography)[:, 0, :]
    reprojection_error = np.linalg.norm(projected - image_xy, axis=1)
    depth_design = np.column_stack([world_xy, np.ones(len(world_xy), dtype=np.float32)])
    depth_values = np.asarray([float(waypoint["depth"]) for waypoint in usable], dtype=np.float64)
    depth_coefficients, _, _, _ = np.linalg.lstsq(depth_design, depth_values, rcond=None)
    depth_error = np.abs(depth_design @ depth_coefficients - depth_values)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else len(usable)

    return {
        "homography": homography,
        "depth_coefficients": depth_coefficients,
        "waypoint_count": len(usable),
        "homography_inlier_fraction": inliers / len(usable),
        "reprojection_error_median_px": float(np.median(reprojection_error)),
        "reprojection_error_p95_px": float(np.percentile(reprojection_error, 95)),
        "depth_fit_error_p95_m": float(np.percentile(depth_error, 95)),
    }


def project_ground_point(calibration: dict, world_xy: tuple[float, float]) -> tuple[float, float, float]:
    point = np.asarray([[[float(world_xy[0]), float(world_xy[1])]]], dtype=np.float32)
    image_xy = cv2.perspectiveTransform(point, calibration["homography"])[0, 0]
    coefficients = calibration["depth_coefficients"]
    depth = float(coefficients[0] * world_xy[0] + coefficients[1] * world_xy[1] + coefficients[2])
    return float(image_xy[0]), float(image_xy[1]), depth


def sample_segment(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    step_m: float,
) -> list[tuple[float, float]]:
    distance = math.dist(start_xy, end_xy)
    steps = max(1, int(math.ceil(distance / step_m)))
    return [
        (
            start_xy[0] + (end_xy[0] - start_xy[0]) * idx / steps,
            start_xy[1] + (end_xy[1] - start_xy[1]) * idx / steps,
        )
        for idx in range(steps + 1)
    ]


def sample_corridor(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    step_m: float,
    radius_m: float,
) -> list[tuple[float, float]]:
    centerline = sample_segment(start_xy, end_xy, step_m)
    if radius_m <= 0.0:
        return centerline
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-8:
        return centerline
    nx, ny = -dy / norm, dx / norm
    offsets = np.linspace(-radius_m, radius_m, 5)
    return [
        (point[0] + float(offset) * nx, point[1] + float(offset) * ny)
        for point in centerline
        for offset in offsets
    ]


def inspect_ground_samples(
    samples: list[tuple[float, float]],
    calibration: dict,
    depth_map: np.ndarray,
    object_index: np.ndarray | None,
    depth_eps_m: float,
) -> dict:
    height, width = depth_map.shape[:2]
    total = len(samples)
    if not samples:
        return {
            "sample_count": 0, "in_frame_count": 0, "visually_free_count": 0,
            "foreground_count": 0, "object_pixel_count": 0,
            "in_frame_fraction": 0.0, "visually_free_fraction": 0.0, "foreground_fraction": 0.0,
        }
    world_xy = np.asarray(samples, dtype=np.float32)
    image_xy = cv2.perspectiveTransform(world_xy[:, None, :], calibration["homography"])[:, 0, :]
    coefficients = calibration["depth_coefficients"]
    expected_depth = world_xy @ coefficients[:2] + coefficients[2]
    pixels = np.rint(image_xy).astype(np.int64)
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        & (expected_depth > 0.0)
    )
    valid_pixels = pixels[valid]
    valid_expected = expected_depth[valid]
    observed = depth_map[valid_pixels[:, 1], valid_pixels[:, 0]] if len(valid_pixels) else np.asarray([])
    foreground_mask = (observed > 0.0) & (observed + depth_eps_m < valid_expected)
    in_frame = int(valid.sum())
    foreground = int(foreground_mask.sum())
    visually_free = in_frame - foreground
    object_pixels = 0
    if object_index is not None and len(valid_pixels):
        object_pixels = int((object_index[valid_pixels[:, 1], valid_pixels[:, 0]] > 0).sum())
    return {
        "sample_count": total,
        "in_frame_count": in_frame,
        "visually_free_count": visually_free,
        "foreground_count": foreground,
        "object_pixel_count": object_pixels,
        "in_frame_fraction": in_frame / total if total else 0.0,
        "visually_free_fraction": visually_free / total if total else 0.0,
        "foreground_fraction": foreground / total if total else 0.0,
    }


def classify_edge_support(
    legal: bool,
    center_stats: dict,
    corridor_stats: dict,
    embodied: bool,
    center_support_threshold: float,
    corridor_support_threshold: float,
) -> str:
    support_stats = corridor_stats if embodied else center_stats
    if legal:
        if (
            center_stats["in_frame_fraction"] >= center_support_threshold
            and center_stats["visually_free_fraction"] >= center_support_threshold
            and support_stats["in_frame_fraction"] >= corridor_support_threshold
            and support_stats["visually_free_fraction"] >= corridor_support_threshold
        ):
            return "observable_safe"
        return "uncertain_safe"
    minimum_foreground_samples = max(1, int(math.ceil(0.02 * support_stats["in_frame_count"])))
    if support_stats["foreground_count"] >= minimum_foreground_samples:
        return "depth_obstruction_evidence"
    if support_stats["object_pixel_count"] > 0:
        return "object_only_obstruction_cue"
    return "uncertain_blocked"


def edge_key(task: str, scene_id: str, view_id: int, src: int, dst: int) -> tuple:
    lo, hi = sorted((int(src), int(dst)))
    return task, scene_id, int(view_id), lo, hi


def direct_edge_set(sidecar: dict) -> set[tuple[int, int]]:
    edges = set()
    for src_raw, entries in sidecar.get("direct_neighbors", {}).items():
        src = int(src_raw)
        for entry in entries or []:
            dst = int(entry["display_id"])
            edges.add(tuple(sorted((src, dst))))
    return edges


def resolve_render_view(row: dict) -> Path:
    image_path = Path(row["image_path"])
    release_root = image_path.parents[3]
    return release_root / "renders" / str(row["scene_id"]) / f"view_{int(row['view_id'])}"


def load_model_predictions(
    runs_root: Path,
    tasks: tuple[str, ...],
    model_runs: tuple[str, ...],
) -> dict[str, dict[str, dict]]:
    predictions = {}
    for model_run in model_runs:
        run_dir = runs_root / model_run
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        task_predictions = {}
        complete = True
        for task in tasks:
            matches = sorted((run_dir / "predictions").glob(f"preds_{task}_*.jsonl"))
            if not matches:
                complete = False
                break
            task_predictions[task] = {
                str(row["question_id"]): row for row in load_jsonl(matches[0])
            }
        if complete:
            predictions[run_dir.name] = task_predictions
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", default="a2,b2,c")
    parser.add_argument("--step-m", type=float, default=0.05)
    parser.add_argument(
        "--embodied-radius-m",
        type=float,
        default=DEFAULT_EMBODIED_RADIUS_M,
    )
    parser.add_argument("--depth-eps-m", type=float, default=0.08)
    parser.add_argument("--center-support-threshold", type=float, default=0.95)
    parser.add_argument("--corridor-support-threshold", type=float, default=0.80)
    parser.add_argument("--model-runs", default=",".join(DEFAULT_MODEL_RUNS))
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    rows = []
    for task in tasks:
        task_rows = load_jsonl(args.benchmark_root / "gt" / f"gt_next_{task}.jsonl")
        if args.limit_questions is not None:
            task_rows = task_rows[: args.limit_questions]
        for row in task_rows:
            row["task"] = task
            rows.append(row)
    model_runs = tuple(value.strip() for value in args.model_runs.split(",") if value.strip())
    predictions = load_model_predictions(args.runs_root, tasks, model_runs)

    view_cache = {}
    sidecar_cache = {}
    edge_audit_cache = {}
    calibration_rows = []

    def load_view(row: dict) -> dict:
        key = (str(row["scene_id"]), int(row["view_id"]), str(row["task"]))
        if key in view_cache:
            return view_cache[key]
        visible_path = Path(row["visible_waypoints_path"])
        waypoints = json.loads(visible_path.read_text())["visible_waypoints"]
        render_view = resolve_render_view(row)
        depth_map = load_depth_image(render_view / "depth.png")
        if depth_map is None:
            raise FileNotFoundError(render_view / "depth.png")
        map_path = render_view.parent / "object_index_map.json"
        max_pass_id = max((int(value) for value in json.loads(map_path.read_text()).keys()), default=None)
        object_index = load_object_index_image(render_view / "object_index.png", max_pass_id=max_pass_id)
        calibration = fit_view_calibration(waypoints)
        image_size = Image.open(render_view / "rgb.png").size
        payload = {
            "waypoints": {int(waypoint["display_id"]): waypoint for waypoint in waypoints},
            "calibration": calibration,
            "depth_map": depth_map,
            "object_index": object_index,
            "image_size": image_size,
        }
        view_cache[key] = payload
        calibration_rows.append({
            "task": key[2],
            "scene_id": key[0],
            "view_id": key[1],
            "waypoint_count": calibration["waypoint_count"],
            "homography_inlier_fraction": round(calibration["homography_inlier_fraction"], 6),
            "reprojection_error_median_px": round(calibration["reprojection_error_median_px"], 6),
            "reprojection_error_p95_px": round(calibration["reprojection_error_p95_px"], 6),
            "depth_fit_error_p95_m": round(calibration["depth_fit_error_p95_m"], 6),
        })
        return payload

    def load_sidecar(row: dict) -> dict:
        ref = str(row["direct_pairs_ref"])
        key = (str(row["task"]), str(row["scene_id"]), int(row["view_id"]), ref)
        if key not in sidecar_cache:
            payload = json.loads((args.benchmark_root / ref).read_text())
            sidecar_cache[key] = {
                "payload": payload,
                "legal_edges": direct_edge_set(payload),
            }
        return sidecar_cache[key]

    def audit_edge(row: dict, src: int, dst: int) -> dict:
        key = edge_key(str(row["task"]), str(row["scene_id"]), int(row["view_id"]), src, dst)
        if key in edge_audit_cache:
            return edge_audit_cache[key]
        view = load_view(row)
        sidecar = load_sidecar(row)
        waypoints = view["waypoints"]
        src_waypoint = waypoints.get(int(src))
        dst_waypoint = waypoints.get(int(dst))
        legal_edges = sidecar["legal_edges"]
        legal = tuple(sorted((int(src), int(dst)))) in legal_edges and int(src) != int(dst)
        if src_waypoint is None or dst_waypoint is None:
            result = {
                "task": row["task"], "scene_id": row["scene_id"], "view_id": row["view_id"],
                "src": int(src), "dst": int(dst), "legal": False,
                "support_class": "invalid_display_id",
                "center_in_frame_fraction": 0.0, "center_visually_free_fraction": 0.0,
                "center_foreground_fraction": 0.0, "corridor_in_frame_fraction": 0.0,
                "corridor_visually_free_fraction": 0.0, "corridor_foreground_fraction": 0.0,
                "corridor_object_pixel_count": 0,
            }
            edge_audit_cache[key] = result
            return result

        start_xy = tuple(float(value) for value in src_waypoint["world_xyz"][:2])
        end_xy = tuple(float(value) for value in dst_waypoint["world_xyz"][:2])
        center_samples = sample_segment(start_xy, end_xy, args.step_m)
        embodied = str(row["task"]) in {"b2", "c"}
        corridor_samples = sample_corridor(
            start_xy,
            end_xy,
            args.step_m,
            args.embodied_radius_m if embodied else 0.0,
        )
        center_stats = inspect_ground_samples(
            center_samples, view["calibration"], view["depth_map"], view["object_index"], args.depth_eps_m
        )
        corridor_stats = inspect_ground_samples(
            corridor_samples, view["calibration"], view["depth_map"], view["object_index"], args.depth_eps_m
        )
        support_class = classify_edge_support(
            legal,
            center_stats,
            corridor_stats,
            embodied,
            args.center_support_threshold,
            args.corridor_support_threshold,
        )
        result = {
            "task": row["task"], "scene_id": row["scene_id"], "view_id": int(row["view_id"]),
            "src": int(src), "dst": int(dst), "legal": bool(legal), "support_class": support_class,
            "center_in_frame_fraction": round(center_stats["in_frame_fraction"], 6),
            "center_visually_free_fraction": round(center_stats["visually_free_fraction"], 6),
            "center_foreground_fraction": round(center_stats["foreground_fraction"], 6),
            "corridor_in_frame_fraction": round(corridor_stats["in_frame_fraction"], 6),
            "corridor_visually_free_fraction": round(corridor_stats["visually_free_fraction"], 6),
            "corridor_foreground_fraction": round(corridor_stats["foreground_fraction"], 6),
            "corridor_object_pixel_count": int(corridor_stats["object_pixel_count"]),
        }
        edge_audit_cache[key] = result
        return result

    reference_occurrences = []
    start_action_occurrences = []
    prediction_occurrences = []
    question_rows = []

    for row_index, row in enumerate(rows, start=1):
        sidecar = load_sidecar(row)
        display_ids = [int(value) for value in sidecar["payload"].get("display_ids", [])]
        reference = [int(value) for value in row.get("reference_path_display_ids", [])]
        for src, dst in zip(reference[:-1], reference[1:]):
            reference_occurrences.append({"question_id": row["question_id"], **audit_edge(row, src, dst)})
        start_id = int(row["start_id"])
        for dst in display_ids:
            if dst != start_id:
                start_action_occurrences.append({"question_id": row["question_id"], **audit_edge(row, start_id, dst)})

        for model, task_predictions in predictions.items():
            prediction = task_predictions[row["task"]].get(str(row["question_id"]), {})
            predicted_ids = parse_ids(prediction_output(prediction))
            visible_ids = set(load_view(row)["waypoints"])
            legal_edges = sidecar["legal_edges"]
            acceptable_goals = {int(value) for value in row.get("acceptable_goal_ids", [])}
            route_status = "no_output"
            invalid_edges = []
            if predicted_ids is not None:
                if not predicted_ids or any(value not in visible_ids for value in predicted_ids):
                    route_status = "invalid_id"
                elif len(predicted_ids) < 2 or predicted_ids[0] != start_id:
                    route_status = "invalid_start"
                else:
                    invalid_edges = [
                        (src, dst)
                        for src, dst in zip(predicted_ids[:-1], predicted_ids[1:])
                        if src == dst or tuple(sorted((src, dst))) not in legal_edges
                    ]
                    if invalid_edges:
                        route_status = "invalid_edge"
                    elif predicted_ids[-1] not in acceptable_goals:
                        route_status = "wrong_goal"
                    else:
                        route_status = "success"

                for edge_index, (src, dst) in enumerate(zip(predicted_ids[:-1], predicted_ids[1:])):
                    prediction_occurrences.append({
                        "model": model,
                        "question_id": row["question_id"],
                        "edge_index": edge_index,
                        "route_status": route_status,
                        **audit_edge(row, src, dst),
                    })

            invalid_edge_support = [audit_edge(row, src, dst)["support_class"] for src, dst in invalid_edges]
            question_rows.append({
                "model": model,
                "question_id": row["question_id"],
                "task": row["task"],
                "route_status": route_status,
                "predicted_edge_count": max(0, len(predicted_ids or []) - 1),
                "invalid_edge_count": len(invalid_edges),
                "has_depth_obstruction_failure": int("depth_obstruction_evidence" in invalid_edge_support),
                "has_object_only_obstruction_failure": int(
                    "depth_obstruction_evidence" not in invalid_edge_support
                    and "object_only_obstruction_cue" in invalid_edge_support
                ),
                "uncertain_only_invalid_edge_failure": int(
                    bool(invalid_edge_support)
                    and all(value == "uncertain_blocked" for value in invalid_edge_support)
                ),
            })
        if row_index % 50 == 0:
            print(f"audited {row_index}/{len(rows)} questions; unique_edges={len(edge_audit_cache)}", flush=True)

    model_summary = []
    rows_by_model = defaultdict(list)
    edges_by_model = defaultdict(list)
    for row in question_rows:
        rows_by_model[row["model"]].append(row)
    for row in prediction_occurrences:
        edges_by_model[row["model"]].append(row)
    for model in sorted(rows_by_model):
        model_rows = rows_by_model[model]
        model_edges = edges_by_model[model]
        status_counts = Counter(row["route_status"] for row in model_rows)
        support_counts = Counter(row["support_class"] for row in model_edges)
        invalid_failures = status_counts["invalid_edge"]
        depth_failures = sum(row["has_depth_obstruction_failure"] for row in model_rows)
        object_only_failures = sum(row["has_object_only_obstruction_failure"] for row in model_rows)
        uncertain_failures = sum(row["uncertain_only_invalid_edge_failure"] for row in model_rows)
        model_summary.append({
            "model": model,
            "questions": len(model_rows),
            "success_rate": round(status_counts["success"] / len(model_rows), 6),
            "invalid_edge_failure_rate": round(invalid_failures / len(model_rows), 6),
            "depth_obstruction_failure_rate": round(depth_failures / len(model_rows), 6),
            "object_only_obstruction_failure_rate": round(object_only_failures / len(model_rows), 6),
            "uncertain_only_invalid_edge_failure_rate": round(uncertain_failures / len(model_rows), 6),
            "uncertain_share_of_invalid_edge_failures": round(
                uncertain_failures / invalid_failures, 6
            ) if invalid_failures else 0.0,
            "predicted_edges": len(model_edges),
            "observable_safe_edges": support_counts["observable_safe"],
            "uncertain_safe_edges": support_counts["uncertain_safe"],
            "depth_obstruction_edges": support_counts["depth_obstruction_evidence"],
            "object_only_obstruction_edges": support_counts["object_only_obstruction_cue"],
            "uncertain_blocked_edges": support_counts["uncertain_blocked"],
        })

    reference_by_question = defaultdict(list)
    for occurrence in reference_occurrences:
        reference_by_question[(occurrence["task"], occurrence["question_id"])].append(
            occurrence["support_class"]
        )
    task_summary = []
    for task in tasks:
        task_reference = [row for row in reference_occurrences if row["task"] == task]
        task_start = [row for row in start_action_occurrences if row["task"] == task]
        task_questions = [row for row in question_rows if row["task"] == task]
        reference_counts = Counter(row["support_class"] for row in task_reference)
        start_counts = Counter(row["support_class"] for row in task_start)
        legal_start = start_counts["observable_safe"] + start_counts["uncertain_safe"]
        illegal_start = (
            start_counts["depth_obstruction_evidence"]
            + start_counts["object_only_obstruction_cue"]
            + start_counts["uncertain_blocked"]
        )
        invalid_failures = [row for row in task_questions if row["route_status"] == "invalid_edge"]
        depth_failures = sum(row["has_depth_obstruction_failure"] for row in invalid_failures)
        object_only_failures = sum(row["has_object_only_obstruction_failure"] for row in invalid_failures)
        uncertain_failures = sum(row["uncertain_only_invalid_edge_failure"] for row in invalid_failures)
        task_reference_questions = [
            classes for (row_task, _), classes in reference_by_question.items() if row_task == task
        ]
        all_observable_questions = sum(
            all(value == "observable_safe" for value in classes)
            for classes in task_reference_questions
        )
        task_summary.append({
            "task": task,
            "questions": len(task_reference_questions),
            "reference_edges": len(task_reference),
            "reference_edge_observable_rate": round(
                reference_counts["observable_safe"] / len(task_reference), 6
            ) if task_reference else 0.0,
            "reference_question_all_edges_observable_rate": round(
                all_observable_questions / len(task_reference_questions), 6
            ) if task_reference_questions else 0.0,
            "start_legal_edges": legal_start,
            "start_legal_edge_observable_rate": round(
                start_counts["observable_safe"] / legal_start, 6
            ) if legal_start else 0.0,
            "start_illegal_edges": illegal_start,
            "start_illegal_depth_evidence_rate": round(
                start_counts["depth_obstruction_evidence"] / illegal_start, 6
            ) if illegal_start else 0.0,
            "start_illegal_object_only_rate": round(
                start_counts["object_only_obstruction_cue"] / illegal_start, 6
            ) if illegal_start else 0.0,
            "start_illegal_uncertain_rate": round(
                start_counts["uncertain_blocked"] / illegal_start, 6
            ) if illegal_start else 0.0,
            "model_question_occurrences": len(task_questions),
            "invalid_edge_failures": len(invalid_failures),
            "invalid_failure_depth_evidence_rate": round(
                depth_failures / len(invalid_failures), 6
            ) if invalid_failures else 0.0,
            "invalid_failure_object_only_rate": round(
                object_only_failures / len(invalid_failures), 6
            ) if invalid_failures else 0.0,
            "invalid_failure_uncertain_only_rate": round(
                uncertain_failures / len(invalid_failures), 6
            ) if invalid_failures else 0.0,
        })

    calibration_values = {
        key: [float(row[key]) for row in calibration_rows]
        for key in (
            "homography_inlier_fraction",
            "reprojection_error_median_px",
            "reprojection_error_p95_px",
            "depth_fit_error_p95_m",
        )
    }
    calibration_summary = {
        key: {
            "min": min(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }
        for key, values in calibration_values.items()
    }

    sensitivity_profiles = (
        ("lenient", 0.90, 0.70, 0.01),
        ("default", args.center_support_threshold, args.corridor_support_threshold, 0.02),
        ("strict", 0.98, 0.90, 0.05),
    )
    threshold_sensitivity = []
    for profile, center_threshold, corridor_threshold, depth_threshold in sensitivity_profiles:
        for task in (*tasks, "all"):
            task_reference = [
                row for row in reference_occurrences if task == "all" or row["task"] == task
            ]
            observable_reference = 0
            for row in task_reference:
                row_embodied = row["task"] in {"b2", "c"}
                observable = (
                    float(row["center_in_frame_fraction"]) >= center_threshold
                    and float(row["center_visually_free_fraction"]) >= center_threshold
                )
                if row_embodied:
                    observable = (
                        observable
                        and float(row["corridor_in_frame_fraction"]) >= corridor_threshold
                        and float(row["corridor_visually_free_fraction"]) >= corridor_threshold
                    )
                observable_reference += int(observable)

            invalid_groups = defaultdict(list)
            for row in prediction_occurrences:
                if row["route_status"] != "invalid_edge" or bool(row["legal"]):
                    continue
                if task != "all" and row["task"] != task:
                    continue
                invalid_groups[(row["model"], row["question_id"])].append(row)
            failures_with_depth = sum(
                any(float(row["corridor_foreground_fraction"]) >= depth_threshold for row in group)
                for group in invalid_groups.values()
            )
            threshold_sensitivity.append({
                "profile": profile,
                "task": task,
                "center_support_threshold": center_threshold,
                "corridor_support_threshold": corridor_threshold,
                "depth_foreground_threshold": depth_threshold,
                "reference_edges": len(task_reference),
                "reference_edge_observable_rate": round(
                    observable_reference / len(task_reference), 6
                ) if task_reference else 0.0,
                "invalid_edge_failures": len(invalid_groups),
                "invalid_failure_depth_evidence_rate": round(
                    failures_with_depth / len(invalid_groups), 6
                ) if invalid_groups else 0.0,
            })

    observable_reference_questions = {
        question_id
        for (_, question_id), classes in reference_by_question.items()
        if all(value == "observable_safe" for value in classes)
    }
    observable_subset_summary = []
    for model in sorted(rows_by_model):
        for task in (*tasks, "all"):
            full_rows = [
                row for row in rows_by_model[model] if task == "all" or row["task"] == task
            ]
            subset_rows = [
                row for row in full_rows if row["question_id"] in observable_reference_questions
            ]
            full_success_rate = (
                sum(row["route_status"] == "success" for row in full_rows) / len(full_rows)
                if full_rows else 0.0
            )
            subset_success_rate = (
                sum(row["route_status"] == "success" for row in subset_rows) / len(subset_rows)
                if subset_rows else 0.0
            )
            observable_subset_summary.append({
                "model": model,
                "task": task,
                "full_questions": len(full_rows),
                "full_success_rate": round(full_success_rate, 6),
                "observable_reference_questions": len(subset_rows),
                "observable_reference_success_rate": round(subset_success_rate, 6),
                "success_rate_delta": round(subset_success_rate - full_success_rate, 6),
            })

    protocol = {
        "schema_version": "egopath_edge_observability_audit_v1",
        "date": AUDIT_DATE,
        "role": "construct-validity supplement; does not replace the full benchmark leaderboard",
        "audited_edge_sets": ["reference_route", "all_start_actions", "actual_model_predictions"],
        "parameters": {
            "step_m": args.step_m,
            "embodied_radius_m": args.embodied_radius_m,
            "depth_eps_m": args.depth_eps_m,
            "center_support_threshold": args.center_support_threshold,
            "corridor_support_threshold": args.corridor_support_threshold,
        },
        "support_classes": {
            "observable_safe": "legal edge with strong current-view support over its centerline/corridor",
            "uncertain_safe": "legal edge whose full centerline/corridor is not strongly supported",
            "depth_obstruction_evidence": "illegal edge with repeated foreground-depth evidence in its projected centerline/corridor",
            "object_only_obstruction_cue": "illegal edge intersecting visible object pixels without sufficient foreground-depth support",
            "uncertain_blocked": "illegal edge without a visible obstruction witness under this certificate",
        },
        "limitations": [
            "The certificate measures rendered geometric support, not whether a human or VLM can infer metric depth perfectly.",
            "Depth obstruction is a conservative image-space witness and is not used to replace the static geometry label.",
            "The audit covers reference edges, all first actions, and actual model predictions rather than every possible pair.",
        ],
    }
    summary = {
        **protocol,
        "counts": {
            "questions": len(rows),
            "models": len(predictions),
            "unique_task_views": len(view_cache),
            "unique_scene_views": len({(str(row["scene_id"]), int(row["view_id"])) for row in rows}),
            "unique_edges_audited": len(edge_audit_cache),
            "reference_edge_occurrences": len(reference_occurrences),
            "start_action_occurrences": len(start_action_occurrences),
            "prediction_edge_occurrences": len(prediction_occurrences),
        },
        "reference_support": dict(Counter(row["support_class"] for row in reference_occurrences)),
        "start_action_support": dict(Counter(row["support_class"] for row in start_action_occurrences)),
        "prediction_support": dict(Counter(row["support_class"] for row in prediction_occurrences)),
        "calibration_summary": calibration_summary,
        "task_summary": task_summary,
        "threshold_sensitivity": threshold_sensitivity,
        "observable_reference_subset_summary": observable_subset_summary,
        "model_summary": model_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "view_calibration.csv", calibration_rows)
    write_csv(args.output_dir / "unique_edge_audit.csv", list(edge_audit_cache.values()))
    write_csv(args.output_dir / "reference_edge_audit.csv", reference_occurrences)
    write_csv(args.output_dir / "start_action_audit.csv", start_action_occurrences)
    write_csv(args.output_dir / "prediction_edge_audit.csv", prediction_occurrences)
    write_csv(args.output_dir / "prediction_question_audit.csv", question_rows)
    write_csv(args.output_dir / "task_summary.csv", task_summary)
    write_csv(args.output_dir / "threshold_sensitivity.csv", threshold_sensitivity)
    write_csv(args.output_dir / "observable_reference_subset_summary.csv", observable_subset_summary)
    write_csv(args.output_dir / "model_summary.csv", model_summary)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
