#!/usr/bin/env python3
"""Re-evaluate fixed EgoPathBench predictions under geometry-contract variants."""

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "published/navbench3d_release_20260417_rebuild5"
EVALFIX = RELEASE / "data/benchmark_evalfix/benchmark"
BUILD_SCRIPTS = RELEASE / "scripts"
ROOT_SCRIPTS = ROOT / "scripts"
SCENES_ROOT = Path(
    "/mnt/data/zhaoyang/navbench3d-data/scenes_v3_4src_relaxed_refilter_post_objaverse_20260325_124913"
)
CONFIG_PATH = ROOT / "configs/internscene_local.yaml"
LEADERBOARD_ROOT = ROOT / "results/main_model_runs_20260528/native_default_full/runs"
DEFAULT_OUTPUT = ROOT / "results/geometry_contract_sensitivity_20260717"

for scripts_dir in (ROOT_SCRIPTS, BUILD_SCRIPTS):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

from build_navigation_gt import build_occupancy_grid  # noqa: E402


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
ROUTE_TASKS = ("a2", "b2", "c")
CONDITIONS = {
    "nominal": {"resolution": 0.05, "effective_radius": 0.30, "goal_tolerance": 0.10},
    "radius_0p25": {"resolution": 0.05, "effective_radius": 0.25, "goal_tolerance": 0.10},
    "radius_0p35": {"resolution": 0.05, "effective_radius": 0.35, "goal_tolerance": 0.10},
    "grid_0p04": {"resolution": 0.04, "effective_radius": 0.30, "goal_tolerance": 0.10},
    "grid_0p06": {"resolution": 0.06, "effective_radius": 0.30, "goal_tolerance": 0.10},
    "goal_0p05": {"resolution": 0.05, "effective_radius": 0.30, "goal_tolerance": 0.05},
    "goal_0p15": {"resolution": 0.05, "effective_radius": 0.30, "goal_tolerance": 0.15},
}


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw: str | None) -> Path:
    if not raw:
        raise ValueError("missing path")
    path = Path(raw)
    candidates = [path, ROOT / path, EVALFIX / path, RELEASE / "data/release" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def parse_ids(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [int(value) for value in raw]
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        for key in ("path", "route", "waypoints", "ids", "answer"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return None
    try:
        return [int(value) for value in payload]
    except (TypeError, ValueError):
        return None


def extract_layout_objects(payload) -> list[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("objects", "layout", "instances"):
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key]]
    raise ValueError("unsupported layout payload")


def is_direct_segment_traversable(nav_grid, start_xy, end_xy) -> bool:
    gx0, gy0 = nav_grid.world_to_grid(float(start_xy[0]), float(start_xy[1]))
    gx1, gy1 = nav_grid.world_to_grid(float(end_xy[0]), float(end_xy[1]))
    dx, dy = abs(gx1 - gx0), abs(gy1 - gy0)
    sx, sy = (1 if gx0 < gx1 else -1), (1 if gy0 < gy1 else -1)
    err = dx - dy
    gx, gy = gx0, gy0
    while True:
        if not nav_grid.is_free(gx, gy):
            return False
        if gx == gx1 and gy == gy1:
            return True
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            gx += sx
        if err2 < dx:
            err += dx
            gy += sy


def build_direct_pairs_sidecar(visible_waypoints, nav_grid, feasible_key, **_) -> dict:
    feasible = []
    for waypoint in visible_waypoints:
        xyz = waypoint.get("world_xyz") or []
        if waypoint.get("display_id") is None or len(xyz) != 3 or not waypoint.get(feasible_key, False):
            continue
        feasible.append(
            {
                "display_id": int(waypoint["display_id"]),
                "world_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            }
        )
    feasible.sort(key=lambda item: item["display_id"])
    neighbors = {str(item["display_id"]): [] for item in feasible}
    for index, src in enumerate(feasible):
        for dst in feasible[index + 1 :]:
            if not is_direct_segment_traversable(nav_grid, src["world_xyz"], dst["world_xyz"]):
                continue
            distance = math.hypot(
                dst["world_xyz"][0] - src["world_xyz"][0],
                dst["world_xyz"][1] - src["world_xyz"][1],
            )
            neighbors[str(src["display_id"])].append(
                {"display_id": dst["display_id"], "distance_m": round(distance, 4)}
            )
            neighbors[str(dst["display_id"])].append(
                {"display_id": src["display_id"], "distance_m": round(distance, 4)}
            )
    for entries in neighbors.values():
        entries.sort(key=lambda item: item["display_id"])
    return {"direct_neighbors": neighbors}


def distance_to_oriented_bbox_2d(px: float, py: float, bbox: list[float]) -> float:
    if len(bbox) < 5:
        return float("inf")
    cx, cy = float(bbox[0]), float(bbox[1])
    sx, sy = float(bbox[3]), float(bbox[4])
    rz = float(bbox[8]) if len(bbox) > 8 else (float(bbox[6]) if len(bbox) > 6 else 0.0)
    dx, dy = px - cx, py - cy
    cosr, sinr = math.cos(-rz), math.sin(-rz)
    lx, ly = dx * cosr - dy * sinr, dx * sinr + dy * cosr
    return math.hypot(max(abs(lx) - sx / 2.0, 0.0), max(abs(ly) - sy / 2.0, 0.0))


def build_goal_ring_info(visible_waypoints, bbox, ring_tolerance, feasible_key) -> dict:
    candidates = []
    min_distance = float("inf")
    for waypoint in visible_waypoints:
        if not waypoint.get(feasible_key, False):
            continue
        xyz = waypoint.get("world_xyz") or []
        if waypoint.get("display_id") is None or len(xyz) != 3:
            continue
        raw_distance = distance_to_oriented_bbox_2d(float(xyz[0]), float(xyz[1]), bbox)
        rounded_distance = round(raw_distance, 4)
        candidates.append((int(waypoint["display_id"]), rounded_distance, waypoint))
        min_distance = min(min_distance, raw_distance)
    if not candidates:
        raise ValueError("empty acceptable_goal_ids")
    threshold = min_distance + float(ring_tolerance)
    if feasible_key == "pointmass_walkable":
        embodied = [distance for _, distance, item in candidates if item.get("embodied_feasible", False)]
        if embodied:
            threshold = max(threshold, min(embodied))
    goals = sorted(display_id for display_id, distance, _ in candidates if distance <= threshold + 1e-9)
    if not goals:
        raise ValueError("empty acceptable_goal_ids")
    return {"acceptable_goal_ids": goals}


def shortest_reference_path(start_id, acceptable_goal_ids, direct_neighbors) -> list[int]:
    adjacency = {
        int(src): {int(item["display_id"]): float(item["distance_m"]) for item in entries}
        for src, entries in direct_neighbors.items()
    }
    goals = {int(value) for value in acceptable_goal_ids}
    if int(start_id) not in adjacency or not goals:
        return []
    heap = [(0.0, int(start_id))]
    distance = {int(start_id): 0.0}
    previous = {int(start_id): None}
    found = None
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > distance.get(node, float("inf")):
            continue
        if node in goals:
            found = node
            break
        for neighbor, weight in adjacency.get(node, {}).items():
            candidate = current_distance + weight
            if candidate >= distance.get(neighbor, float("inf")):
                continue
            distance[neighbor] = candidate
            previous[neighbor] = node
            heapq.heappush(heap, (candidate, neighbor))
    if found is None:
        return []
    path = [found]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    return list(reversed(path))


def prediction_ids(prediction: dict | None) -> list[int] | None:
    if not prediction or prediction.get("success") is False:
        return None
    return parse_ids(prediction.get("output"))


def edge_set(neighbors: dict[str, list[dict]]) -> set[tuple[int, int]]:
    edges = set()
    for src, entries in neighbors.items():
        for item in entries:
            a, b = int(src), int(item["display_id"])
            edges.add((min(a, b), max(a, b)))
    return edges


def evaluate_route(question: dict, pred_ids: list[int] | None, contract: dict) -> dict:
    flags = {
        "parseable": bool(pred_ids),
        "candidate_valid": False,
        "start_correct": False,
        "edge_legal": False,
        "endpoint_hit": False,
        "valid_path": False,
        "success": False,
    }
    if not pred_ids:
        return flags
    visible_ids = {int(value) for value in contract["visible_ids"]}
    flags["candidate_valid"] = all(int(value) in visible_ids for value in pred_ids)
    flags["start_correct"] = int(pred_ids[0]) == int(question["ground_truth"]["start_id"])
    goals = {int(value) for value in contract["acceptable_goal_ids"]}
    flags["endpoint_hit"] = int(pred_ids[-1]) in goals
    adjacency = {
        int(src): {int(item["display_id"]) for item in entries}
        for src, entries in contract["direct_neighbors"].items()
    }
    if flags["candidate_valid"]:
        flags["edge_legal"] = all(
            int(dst) in adjacency.get(int(src), set())
            for src, dst in zip(pred_ids[:-1], pred_ids[1:])
        )
    flags["valid_path"] = all(
        flags[key] for key in ("candidate_valid", "start_correct", "edge_legal")
    )
    flags["success"] = flags["valid_path"] and flags["endpoint_hit"]
    return flags


def grid_config(base_config: dict, resolution: float, effective_radius: float) -> dict:
    config = copy.deepcopy(base_config)
    config["navigation"]["grid_resolution"] = float(resolution)
    config["navigation"]["agent_radius"] = float(effective_radius)
    config["navigation"]["passable_margin"] = 0.0
    return config


def fixed_contract_waypoints(visible_waypoints: list[dict], feasible_key: str) -> list[dict]:
    updated = []
    for waypoint in visible_waypoints:
        item = dict(waypoint)
        item["sensitivity_feasible"] = bool(item.get(feasible_key, False))
        updated.append(item)
    return updated


def official_contract(question: dict) -> dict:
    gt = question["ground_truth"]
    visible = read_json(resolve_path(question["visible_waypoints_path"]))["visible_waypoints"]
    sidecar = read_json(resolve_path(gt["direct_pairs_ref"]))
    return {
        "visible_ids": [int(item["display_id"]) for item in visible if item.get("display_id") is not None],
        "direct_neighbors": sidecar["direct_neighbors"],
        "acceptable_goal_ids": [int(value) for value in gt["acceptable_goal_ids"]],
        "reference_path_exists": bool(gt.get("reference_path_display_ids")),
    }


def rebuilt_contract(
    question: dict,
    visible_waypoints: list[dict],
    target_bbox: list[float],
    nav_grid,
    goal_tolerance: float,
) -> dict:
    feasible_key = "pointmass_walkable" if question["task"] == "a2" else "embodied_feasible"
    updated = fixed_contract_waypoints(visible_waypoints, feasible_key)
    sidecar = build_direct_pairs_sidecar(
        visible_waypoints=updated,
        nav_grid=nav_grid,
        feasible_key="sensitivity_feasible",
        task=question["task"],
        scene_id=question["scene_id"],
        view_id=int(question["view_id"]),
    )
    try:
        goal_info = build_goal_ring_info(
            visible_waypoints=updated,
            bbox=target_bbox,
            ring_tolerance=float(goal_tolerance),
            feasible_key=feasible_key,
        )
        goals = goal_info["acceptable_goal_ids"]
    except ValueError:
        goals = []
    reference = shortest_reference_path(
        start_id=int(question["ground_truth"]["start_id"]),
        acceptable_goal_ids=goals,
        direct_neighbors=sidecar["direct_neighbors"],
    )
    return {
        "visible_ids": [int(item["display_id"]) for item in visible_waypoints if item.get("display_id") is not None],
        "direct_neighbors": sidecar["direct_neighbors"],
        "acceptable_goal_ids": [int(value) for value in goals],
        "reference_path_exists": bool(reference),
    }


def goal_only_contract(question: dict, goal_tolerance: float) -> dict:
    contract = official_contract(question)
    visible = read_json(resolve_path(question["visible_waypoints_path"]))["visible_waypoints"]
    feasible_key = "pointmass_walkable" if question["task"] == "a2" else "embodied_feasible"
    layout = extract_layout_objects(read_json(SCENES_ROOT / question["scene_id"] / "layout.json"))
    target = next(item for item in layout if int(item["id"]) == int(question["ground_truth"]["target_id"]))
    try:
        goal_info = build_goal_ring_info(
            visible_waypoints=visible,
            bbox=target["bbox"],
            ring_tolerance=float(goal_tolerance),
            feasible_key=feasible_key,
        )
        goals = goal_info["acceptable_goal_ids"]
    except ValueError:
        goals = []
    reference = shortest_reference_path(
        start_id=int(question["ground_truth"]["start_id"]),
        acceptable_goal_ids=goals,
        direct_neighbors=contract["direct_neighbors"],
    )
    contract["acceptable_goal_ids"] = [int(value) for value in goals]
    contract["reference_path_exists"] = bool(reference)
    return contract


def build_scene_shard(scene_id: str, questions: list[dict], config_path: str, output_path: str) -> str:
    output = Path(output_path)
    if output.exists():
        return str(output)
    base_config = yaml.safe_load(Path(config_path).read_text())
    layout = extract_layout_objects(read_json(SCENES_ROOT / scene_id / "layout.json"))
    targets = {int(item["id"]): item for item in layout if item.get("id") is not None}
    grids = {}
    for resolution, radius in [
        (0.05, 0.0),
        (0.05, 0.25),
        (0.05, 0.30),
        (0.05, 0.35),
        (0.04, 0.0),
        (0.04, 0.30),
        (0.06, 0.0),
        (0.06, 0.30),
    ]:
        grids[(resolution, radius)] = build_occupancy_grid(
            layout,
            grid_config(base_config, resolution, radius),
            scene_path=SCENES_ROOT / scene_id,
        )

    records = []
    for question in questions:
        visible = read_json(resolve_path(question["visible_waypoints_path"]))["visible_waypoints"]
        target = targets[int(question["ground_truth"]["target_id"])]
        official = official_contract(question)
        contracts = {"nominal": official}
        for condition in ("radius_0p25", "radius_0p35", "grid_0p04", "grid_0p06"):
            spec = CONDITIONS[condition]
            radius = 0.0 if question["task"] == "a2" else spec["effective_radius"]
            if condition.startswith("radius_") and question["task"] == "a2":
                contracts[condition] = official
                continue
            contracts[condition] = rebuilt_contract(
                question,
                visible,
                target["bbox"],
                grids[(spec["resolution"], radius)],
                spec["goal_tolerance"],
            )
        contracts["goal_0p05"] = goal_only_contract(question, 0.05)
        contracts["goal_0p15"] = goal_only_contract(question, 0.15)

        nominal_rebuilt = rebuilt_contract(
            question,
            visible,
            target["bbox"],
            grids[(0.05, 0.0 if question["task"] == "a2" else 0.30)],
            0.10,
        )
        records.append(
            {
                "question_id": question["question_id"],
                "task": question["task"],
                "scene_id": scene_id,
                "view_id": int(question["view_id"]),
                "nominal_rebuild_edge_exact": edge_set(official["direct_neighbors"])
                == edge_set(nominal_rebuilt["direct_neighbors"]),
                "nominal_rebuild_goal_exact": set(official["acceptable_goal_ids"])
                == set(nominal_rebuilt["acceptable_goal_ids"]),
                "contracts": contracts,
            }
        )
    write_json(output, {"scene_id": scene_id, "records": records})
    return str(output)


def load_questions() -> list[dict]:
    rows = []
    for task in ROUTE_TASKS:
        for question in read_jsonl(EVALFIX / "vqa" / f"vqa_next_{task}.jsonl"):
            question["task"] = task
            rows.append(question)
    return rows


def load_predictions() -> dict[str, dict[str, dict[str, dict]]]:
    predictions = {}
    for model_dir, _ in MODEL_ORDER:
        summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
        predictions[model_dir] = {}
        for task in ROUTE_TASKS:
            result = next(item for item in summary if item["task"] == task)
            predictions[model_dir][task] = {
                str(item["question_id"]): item
                for item in read_jsonl(resolve_path(result["predictions"]))
            }
    return predictions


def traversal_scores() -> dict[str, tuple[float, float]]:
    scores = {}
    for model_dir, _ in MODEL_ORDER:
        summary = read_json(LEADERBOARD_ROOT / model_dir / "summary.json")
        by_task = {item["task"]: item["metric_json"]["metrics"] for item in summary}
        scores[model_dir] = (
            float(by_task["a1"]["balanced_accuracy"]),
            float(by_task["b1"]["balanced_accuracy"]),
        )
    return scores


def refresh_goal_contracts(records: list[dict], question_by_id: dict[str, dict]) -> None:
    layout_cache = {}
    visible_cache = {}
    for record in records:
        qid = str(record["question_id"])
        question = question_by_id[qid]
        scene_id = str(question["scene_id"])
        if scene_id not in layout_cache:
            layout = extract_layout_objects(read_json(SCENES_ROOT / scene_id / "layout.json"))
            layout_cache[scene_id] = {
                int(item["id"]): item for item in layout if item.get("id") is not None
            }
        visible_path = str(resolve_path(question["visible_waypoints_path"]))
        if visible_path not in visible_cache:
            visible_cache[visible_path] = read_json(Path(visible_path))["visible_waypoints"]
        visible = visible_cache[visible_path]
        target = layout_cache[scene_id][int(question["ground_truth"]["target_id"])]
        feasible_key = "pointmass_walkable" if question["task"] == "a2" else "embodied_feasible"

        rebuilt_nominal_goals = build_goal_ring_info(
            visible,
            target["bbox"],
            CONDITIONS["nominal"]["goal_tolerance"],
            feasible_key,
        )["acceptable_goal_ids"]
        record["nominal_rebuild_goal_exact"] = set(rebuilt_nominal_goals) == set(
            record["contracts"]["nominal"]["acceptable_goal_ids"]
        )
        for condition, spec in CONDITIONS.items():
            if condition == "nominal":
                continue
            try:
                goals = build_goal_ring_info(
                    visible,
                    target["bbox"],
                    spec["goal_tolerance"],
                    feasible_key,
                )["acceptable_goal_ids"]
            except ValueError:
                goals = []
            contract = record["contracts"][condition]
            contract["acceptable_goal_ids"] = [int(value) for value in goals]
            contract["reference_path_exists"] = bool(
                shortest_reference_path(
                    int(question["ground_truth"]["start_id"]),
                    goals,
                    contract["direct_neighbors"],
                )
            )


def summarize(records: list[dict], questions: list[dict], output_root: Path) -> dict:
    question_by_id = {str(item["question_id"]): item for item in questions}
    refresh_goal_contracts(records, question_by_id)
    predictions = load_predictions()
    traverse = traversal_scores()
    nominal_outcomes = {}
    model_rows = []
    outcome_rows = []

    for condition in CONDITIONS:
        for model_dir, model_name in MODEL_ORDER:
            task_values = {task: [] for task in ROUTE_TASKS}
            for record in records:
                qid = str(record["question_id"])
                task = record["task"]
                question = question_by_id[qid]
                pred = predictions[model_dir][task].get(qid)
                flags = evaluate_route(question, prediction_ids(pred), record["contracts"][condition])
                task_values[task].append(flags)
                key = (model_dir, qid)
                if condition == "nominal":
                    nominal_outcomes[key] = flags
                outcome_rows.append(
                    {
                        "condition": condition,
                        "model_dir": model_dir,
                        "question_id": qid,
                        "task": task,
                        "scene_id": record["scene_id"],
                        **flags,
                    }
                )
            route_sr = {
                task: float(np.mean([row["success"] for row in task_values[task]]))
                for task in ROUTE_TASKS
            }
            valid_path = {
                task: float(np.mean([row["valid_path"] for row in task_values[task]]))
                for task in ROUTE_TASKS
            }
            a1_ba, b1_ba = traverse[model_dir]
            score = (a1_ba + b1_ba + sum(route_sr.values())) / 5.0
            model_rows.append(
                {
                    "condition": condition,
                    "model_dir": model_dir,
                    "model": model_name,
                    "egopath_score": score,
                    "a2_sr": route_sr["a2"],
                    "b2_sr": route_sr["b2"],
                    "c_sr": route_sr["c"],
                    "a2_valid_path_rate": valid_path["a2"],
                    "b2_valid_path_rate": valid_path["b2"],
                    "c_valid_path_rate": valid_path["c"],
                }
            )

    for row in outcome_rows:
        nominal = nominal_outcomes[(row["model_dir"], row["question_id"])]
        row["success_flip"] = bool(row["success"] != nominal["success"])
        row["valid_path_flip"] = bool(row["valid_path"] != nominal["valid_path"])

    nominal_scores = {
        row["model_dir"]: row["egopath_score"] for row in model_rows if row["condition"] == "nominal"
    }
    nominal_order = sorted(nominal_scores, key=nominal_scores.get, reverse=True)
    condition_rows = []
    for condition, spec in CONDITIONS.items():
        contracts = [record["contracts"][condition] for record in records]
        ref_coverage = float(np.mean([contract["reference_path_exists"] for contract in contracts]))
        edge_removed = edge_added = edge_union = 0
        for record in records:
            nominal_edges = edge_set(record["contracts"]["nominal"]["direct_neighbors"])
            variant_edges = edge_set(record["contracts"][condition]["direct_neighbors"])
            edge_removed += len(nominal_edges - variant_edges)
            edge_added += len(variant_edges - nominal_edges)
            edge_union += len(nominal_edges | variant_edges)
        condition_scores = {
            row["model_dir"]: row["egopath_score"]
            for row in model_rows
            if row["condition"] == condition
        }
        score_nominal = [nominal_scores[model] for model in nominal_order]
        score_variant = [condition_scores[model] for model in nominal_order]
        success_rows = [row for row in outcome_rows if row["condition"] == condition]
        condition_rows.append(
            {
                "condition": condition,
                "grid_resolution_m": spec["resolution"],
                "effective_radius_m": spec["effective_radius"],
                "goal_tolerance_m": spec["goal_tolerance"],
                "reference_solution_coverage": ref_coverage,
                "edge_removed": edge_removed,
                "edge_added": edge_added,
                "edge_flip_rate": (edge_removed + edge_added) / max(edge_union, 1),
                "success_flip_rate": float(np.mean([row["success_flip"] for row in success_rows])),
                "valid_path_flip_rate": float(np.mean([row["valid_path_flip"] for row in success_rows])),
                "score_spearman": float(spearmanr(score_nominal, score_variant).statistic),
                "score_kendall": float(kendalltau(score_nominal, score_variant).statistic),
                "top_model": max(condition_scores, key=condition_scores.get),
            }
        )

    write_csv(output_root / "per_model_condition.csv", model_rows)
    write_csv(output_root / "per_condition.csv", condition_rows)
    with (output_root / "question_outcomes.jsonl").open("w") as handle:
        for row in outcome_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = {
        "schema_version": "egopath_geometry_contract_sensitivity_v1",
        "fixed_question_count": len(records),
        "model_count": len(MODEL_ORDER),
        "conditions": CONDITIONS,
        "nominal_rebuild_edge_exact_rate": float(
            np.mean([record["nominal_rebuild_edge_exact"] for record in records])
        ),
        "nominal_rebuild_goal_exact_rate": float(
            np.mean([record["nominal_rebuild_goal_exact"] for record in records])
        ),
        "per_condition": condition_rows,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    by_scene = {}
    for question in questions:
        by_scene.setdefault(str(question["scene_id"]), []).append(question)
    scene_ids = sorted(by_scene)
    if args.max_scenes is not None:
        scene_ids = scene_ids[: args.max_scenes]
        questions = [question for question in questions if question["scene_id"] in set(scene_ids)]

    shard_root = args.output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    if args.force:
        for scene_id in scene_ids:
            path = shard_root / f"{scene_id}.json"
            if path.exists():
                path.unlink()

    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for scene_id in scene_ids:
            path = shard_root / f"{scene_id}.json"
            future = executor.submit(
                build_scene_shard,
                scene_id,
                by_scene[scene_id],
                str(CONFIG_PATH),
                str(path),
            )
            futures[future] = scene_id
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"[{completed}/{len(futures)}] {futures[future]}", flush=True)

    records = []
    for scene_id in scene_ids:
        records.extend(read_json(shard_root / f"{scene_id}.json")["records"])
    summary = summarize(records, questions, args.output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
